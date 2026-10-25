"""Pipeline definition model + pure helpers for the runner.

A :class:`Pipeline` is an ordered list of :class:`Step` s. Each step runs one
framework ``stage`` (a name in ``STAGE_MAPPING``) across a *batch* of items
(scenarios). The batch's items are partitioned **round-robin and balanced** into
the step's number of *instances*, and each instance becomes a work unit launched
by an :class:`~executors.Executor`.

This module is pure (no subprocess / SLURM / Ray) so it is easy to unit-test.

Parallelism model
-----------------
* ``per_item``   -> one instance per item (data parallelism). K items -> K units,
  each unit holds exactly one item.
* ``instances: N`` -> N instances. K items are round-robin-balanced into N shards;
  each unit holds its shard (model loads once per unit, processes the shard).
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Parallelism:
    """How a step is parallelised across instances.

    mode:
        ``per_item``   one instance per item; ``K`` items -> ``K`` units.
        ``instances``  a fixed ``instances`` count; ``K`` items -> balanced shards.
    """

    mode: str = "per_item"  # "per_item" | "instances"
    instances: int | None = None  # required when mode == "instances"

    def unit_count(self, n_items: int) -> int:
        if self.mode == "per_item":
            return n_items
        if self.mode == "instances":
            if not self.instances or self.instances < 1:
                raise ValueError("parallelism mode 'instances' requires instances >= 1")
            return self.instances
        raise ValueError(f"unknown parallelism mode: {self.mode!r}")


@dataclass
class Step:
    """One pipeline step: a stage run across the batch with a parallelism policy."""

    stage: str
    parallelism: Parallelism = field(default_factory=Parallelism)
    batch_size: int | None = None  # forwarded to the stage as run.batch_size (micro-batch)
    interpreter: str | None = None  # python interpreter/venv to run the stage in
    overrides: list[str] = field(default_factory=list)  # extra hydra overrides
    env: dict = field(default_factory=dict)  # extra env for the unit (e.g. CUDA_VISIBLE_DEVICES)
    slurm: dict | None = None  # carried for SlurmExecutor (command + opaque options)
    ray: dict | None = None  # carried for RayExecutor (num_gpus / num_cpus, etc.)


@dataclass
class Pipeline:
    """An ordered set of steps plus the run-level knobs that drive them."""

    name: str
    steps: list[Step]
    executor: str = "local"  # local | slurm | ray
    batch: dict = field(default_factory=dict)  # one of {dirs, manifest, glob}
    default_interpreter: str | None = None  # fallback interpreter for steps that omit one
    default_cwd: str | None = None  # working dir for the launched stage subprocesses


def resolve_batch(batch: dict) -> list[str]:
    """Resolve a batch source to a list of absolute output-dir paths.

    ``batch`` must set exactly one of:
        ``dirs``     a list, or a comma-separated string, of output dirs
        ``manifest`` a file with one output dir per line
        ``glob``     a glob pattern matching the output dirs
    """
    sources = [k for k in ("dirs", "manifest", "glob") if batch.get(k)]
    if len(sources) != 1:
        found = ", ".join(sources) or "none"
        raise ValueError(f"Provide exactly one of dirs/manifest/glob (found: {found})")

    src = sources[0]
    if src == "dirs":
        val = batch["dirs"]
        if isinstance(val, (list, tuple)):
            parts = [str(x) for x in val]
        else:
            parts = [p.strip() for p in str(val).split(",")]
        dirs = [d for d in parts if d.strip()]
    elif src == "manifest":
        lines = Path(str(batch["manifest"])).read_text().splitlines()
        dirs = [d.strip() for d in lines if d.strip()]
    else:  # glob
        dirs = sorted(glob.glob(str(batch["glob"])))

    if not dirs:
        raise ValueError(f"Batch resolved to zero items via {src!r}.")
    return [str(Path(d).resolve()) for d in dirs]


def partition_round_robin(items: list, n: int) -> list[list]:
    """Partition ``items`` into ``n`` balanced shards, round-robin.

    ``shard[i]`` receives ``items[i], items[i+n], items[i+2n], ...`` so shards
    differ in size by at most one item. ``n == len(items)`` yields one item each.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    shards: list[list] = [[] for _ in range(n)]
    for idx, item in enumerate(items):
        shards[idx % n].append(item)
    return shards

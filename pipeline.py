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
class StopGate:
    """Optional early-exit gate for a :class:`Loop`.

    After each pass, ``stage`` (a name in ``Pipeline.stages``) runs across the
    batch. It signals a per-item "done" by writing ``key: True`` into that
    item's shared context (see :mod:`context`). When every item has signalled,
    the loop stops (bounded by the loop's ``max_iterations``). Default ``key``
    is ``"done"``.
    """

    stage: str
    key: str = "done"


@dataclass
class Loop:
    """Repeat a set of stage steps, in order, across the same batch.

    ``steps``          -- stage names (keys of ``Pipeline.stages``), run in order each pass.
    ``count``          -- fixed number of passes (required unless ``stop`` is set).
    ``stop``           -- optional gate; stop early once every item signals ``stop.key``.
    ``max_iterations`` -- safety cap when using ``stop`` (default 100).

    Each sub-step runs exactly like a normal step (modular / reusable); the 0-based
    pass index is exposed to stages via ``run.loop_iteration``.
    """

    steps: list[str]
    count: int | None = None
    stop: StopGate | None = None
    max_iterations: int | None = None


@dataclass
class StageDef:
    """Definition of one named stage: which stage class, and how to run it.

    The name (the key in ``Pipeline.stages``) must be a key of ``STAGE_MAPPING``;
    one definition per stage.
    """

    stage: str  # STAGE_MAPPING name; equals the key in Pipeline.stages
    always: bool = False  # run even if a prior step failed (finally-semantics, e.g. cleanup)
    parallelism: Parallelism = field(default_factory=Parallelism)
    batch_size: int | None = None  # forwarded to the stage as run.batch_size (micro-batch)
    interpreter: str | None = None  # python interpreter/venv to run the stage in
    overrides: list[str] = field(default_factory=list)  # extra hydra overrides
    env: dict = field(default_factory=dict)  # extra env for the unit (e.g. CUDA_VISIBLE_DEVICES)
    slurm: dict | None = None  # carried for SlurmExecutor (command + opaque options)
    ray: dict | None = None  # carried for RayExecutor (num_gpus / num_cpus, etc.)


@dataclass
class Step:
    """One pipeline step: a reference to a named stage, OR a loop of stage names.

    Exactly one of ``name`` / ``loop`` is set.
    """

    name: str | None = None  # key into Pipeline.stages
    loop: Loop | None = None


@dataclass
class Pipeline:
    """Stage definitions, an ordered list of steps, and the run-level knobs."""

    name: str
    stages: dict[str, StageDef] = field(default_factory=dict)  # name -> definition
    steps: list[Step] = field(default_factory=list)  # execution order
    executor: str = "local"  # local | slurm | ray
    batch: dict = field(default_factory=dict)  # one of {dirs, manifest, glob}
    default_interpreter: str | None = None  # fallback interpreter for defs that omit one
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

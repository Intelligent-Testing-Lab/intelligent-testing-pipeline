"""Pipeline runner: resolve the batch, partition per step, dispatch via an executor.

The runner is backend-agnostic: it computes *what* to run (steps, sharding,
interpreter, args) and delegates *how* to launch to an :class:`~executors.Executor`.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import yaml

from executors import Executor, LocalExecutor, WorkUnit
from pipeline import (
    Parallelism,
    Pipeline,
    Step,
    partition_round_robin,
    resolve_batch,
)

logger = logging.getLogger("pipeline.runner")


class PipelineRunError(RuntimeError):
    """Raised when a step (one or more of its units) fails."""


def _resolve_interpreter(step: Step, pipeline: Pipeline) -> str:
    return step.interpreter or pipeline.default_interpreter or _python()


def _python() -> str:
    import sys

    return sys.executable


def _write_manifest(items: list[str]) -> str:
    fd, path = tempfile.mkstemp(prefix="pipeline_batch_", suffix=".txt")
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(items) + "\n")
    return path


def _unit_argv(step: Step, shard: list[str]) -> tuple[list[str], list[str]]:
    argv = [f"stage={step.stage}"]
    tmpfiles: list[str] = []
    if len(shard) == 1:
        argv.append(f"run.output_dir={shard[0]}")
    else:
        manifest = _write_manifest(shard)
        tmpfiles.append(manifest)
        argv.append(f"batch.manifest={manifest}")
    if step.batch_size:
        argv.append(f"run.batch_size={int(step.batch_size)}")
    argv += list(step.overrides)
    return argv, tmpfiles


def _cleanup_units(units: list[WorkUnit]) -> None:
    """Remove per-shard manifest temp files created for these units."""
    for u in units:
        for path in u.tmpfiles:
            try:
                os.unlink(path)
            except OSError:
                pass


def build_units(step: Step, items: list[str], pipeline: Pipeline) -> list[WorkUnit]:
    """Partition the batch into the step's instances and build one unit each."""
    n = step.parallelism.unit_count(len(items))
    shards = partition_round_robin(items, n)
    interpreter = _resolve_interpreter(step, pipeline)
    cwd = pipeline.default_cwd
    units = []
    for i, shard in enumerate(shards):
        argv, tmpfiles = _unit_argv(step, shard)
        units.append(
            WorkUnit(
                name=f"{step.stage}_{i}",
                interpreter=interpreter,
                argv=argv,
                env=dict(step.env or {}),
                cwd=cwd,
                tmpfiles=tmpfiles,
            )
        )
    return units


def get_executor(name: str) -> Executor:
    name = (name or "local").lower()
    if name == "local":
        return LocalExecutor()
    if name == "slurm":
        from slurm_executor import SlurmExecutor

        return SlurmExecutor()
    if name == "ray":
        from ray_executor import RayExecutor

        return RayExecutor()
    raise ValueError(f"unknown executor: {name!r} (expected local|slurm|ray)")


def run_pipeline(pipeline: Pipeline, executor: Executor | None = None) -> None:
    """Run the pipeline's steps in order; each step's units run in parallel.

    A step is skipped if it declares a ``metric`` that is not in
    ``pipeline.metrics``. Any failed unit fails the pipeline (current behaviour).
    The batch is resolved lazily so a fully-skipped pipeline does not require it.
    """
    executor = executor or get_executor(pipeline.executor)
    items: list[str] | None = None


    # again, refactor this. Metrics are cosmos-transfer specific
    for step in pipeline.steps:
        if step.metric is not None:
            if pipeline.metrics is not None and step.metric not in pipeline.metrics:
                logger.info("Skipping step %s (metric %s not selected)", step.stage, step.metric)
                continue

        if items is None:
            items = resolve_batch(pipeline.batch)
            logger.info(
                "Pipeline %s: %d item(s); executor=%s; %d step(s)",
                pipeline.name, len(items), pipeline.executor, len(pipeline.steps),
            )

        units = build_units(step, items, pipeline)
        logger.info(
            "Step %s: %d unit(s) [mode=%s%s]",
            step.stage, len(units), step.parallelism.mode,
            f", batch_size={step.batch_size}" if step.batch_size else "",
        )
        try:
            results = executor.run_units(step, units)
        finally:
            _cleanup_units(units)
        failed = [r for r in results if not r.ok]
        if failed:
            for r in failed:
                logger.error(
                    "Unit %s failed:\nSTDOUT:\n%s\nSTDERR:\n%s",
                    r.unit.name, r.stdout, r.stderr,
                )
            raise PipelineRunError(
                f"step '{step.stage}' failed: {len(failed)}/{len(results)} unit(s) failed"
            )

    logger.info("Pipeline %s complete.", pipeline.name)


def _step_from_dict(s: dict) -> Step:
    par = s.get("parallelism", "per_item")
    if isinstance(par, dict):
        parallelism = Parallelism(
            mode=par.get("mode", "instances"),
            instances=par.get("instances"),
        )
    else:
        parallelism = Parallelism(mode=str(par))
    return Step(
        stage=s["stage"],
        parallelism=parallelism,
        batch_size=s.get("batch_size"),
        interpreter=s.get("interpreter"),
        metric=s.get("metric"),
        overrides=list(s.get("overrides", []) or []),
        env=dict(s.get("env", {}) or {}),
        slurm=s.get("slurm"),
        ray=s.get("ray"),
    )


def load_pipeline(path: str | Path) -> Pipeline:
    """Load a pipeline definition (a YAML of steps + optional run-level knobs)."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    steps = [_step_from_dict(s) for s in data.get("steps", [])]
    return Pipeline(
        name=data.get("name", Path(path).stem),
        steps=steps,
        executor=data.get("executor", "local"),
        batch=dict(data.get("batch", {}) or {}),
        metrics=data.get("metrics"),
        default_interpreter=data.get("interpreter"),
        default_cwd=data.get("cwd"),
    )

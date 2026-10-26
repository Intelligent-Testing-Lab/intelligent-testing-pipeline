"""Pipeline runner: resolve the batch, partition per step, dispatch via an executor.

A pipeline has two sections (mirroring the YAML):
* ``stages`` -- named definitions (stage class + how to run it: interpreter,
  parallelism, overrides, env, slurm/ray, ``always``).
* ``steps``  -- execution order: stage names, or ``loop`` entries repeating a
  list of stage names across the same batch.

The runner is backend-agnostic: it computes *what* to run (stages, sharding,
interpreter, args) and delegates *how* to launch to an :class:`~executors.Executor`.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import yaml
from omegaconf import OmegaConf

import context
from executors import Executor, LocalExecutor, WorkUnit
from pipeline import (
    Loop,
    Parallelism,
    Pipeline,
    StageDef,
    Step,
    StopGate,
    partition_round_robin,
    resolve_batch,
)

logger = logging.getLogger("pipeline.runner")


class PipelineRunError(RuntimeError):
    """Raised when a step (one or more of its units) fails."""


def _python() -> str:
    import sys

    return sys.executable


def _known_stage_names() -> dict:
    from intelligent_pipeline import STAGE_MAPPING

    return STAGE_MAPPING


def _resolve_interpreter(defn: StageDef, pipeline: Pipeline) -> str:
    return defn.interpreter or pipeline.default_interpreter or _python()


def _write_manifest(items: list[str]) -> str:
    fd, path = tempfile.mkstemp(prefix="pipeline_batch_", suffix=".txt")
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(items) + "\n")
    return path


def _unit_argv(
    defn: StageDef,
    shard: list[str],
    extra_overrides: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    argv = [f"stage={defn.stage}"]
    tmpfiles: list[str] = []
    if len(shard) == 1:
        argv.append(f"run.output_dir={shard[0]}")
    else:
        manifest = _write_manifest(shard)
        tmpfiles.append(manifest)
        argv.append(f"batch.manifest={manifest}")
    if defn.batch_size:
        argv.append(f"run.batch_size={int(defn.batch_size)}")
    argv += list(defn.overrides)
    argv += list(extra_overrides or [])
    return argv, tmpfiles


def _cleanup_units(units: list[WorkUnit]) -> None:
    """Remove per-shard manifest temp files created for these units."""
    for u in units:
        for path in u.tmpfiles:
            try:
                os.unlink(path)
            except OSError:
                pass


def _stage_def(name: str, pipeline: Pipeline) -> StageDef:
    try:
        return pipeline.stages[name]
    except KeyError:
        known = ", ".join(sorted(pipeline.stages)) or "none"
        raise ValueError(f"unknown stage name {name!r} in pipeline {pipeline.name!r}. Defined: {known}") from None


def build_units(
    defn: StageDef,
    items: list[str],
    pipeline: Pipeline,
    extra_overrides: list[str] | None = None,
) -> list[WorkUnit]:
    """Partition the batch into the stage's instances and build one unit each."""
    n = defn.parallelism.unit_count(len(items))
    shards = partition_round_robin(items, n)
    interpreter = _resolve_interpreter(defn, pipeline)
    cwd = pipeline.default_cwd
    units = []
    for i, shard in enumerate(shards):
        argv, tmpfiles = _unit_argv(defn, shard, extra_overrides)
        units.append(
            WorkUnit(
                name=f"{defn.stage}_{i}",
                interpreter=interpreter,
                argv=argv,
                env=dict(defn.env or {}),
                cwd=cwd,
                tmpfiles=tmpfiles,
            )
        )
    return units


def _run_stage_step(
    name: str,
    items: list[str],
    pipeline: Pipeline,
    executor: Executor,
    extra_overrides: list[str] | None = None,
) -> None:
    """Run one named stage across the batch; raise on any failed unit."""
    defn = _stage_def(name, pipeline)
    units = build_units(defn, items, pipeline, extra_overrides)
    logger.info(
        "Step %s: %d unit(s) [mode=%s%s]",
        name, len(units), defn.parallelism.mode,
        f", batch_size={defn.batch_size}" if defn.batch_size else "",
    )
    try:
        results = executor.run_units(defn, units)
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
            f"stage '{name}' failed: {len(failed)}/{len(results)} unit(s) failed"
        )


def _run_loop(
    loop: Loop,
    items: list[str],
    pipeline: Pipeline,
    executor: Executor,
) -> None:
    """Run a loop: each pass runs the sub-stages in order, then the stop gate.

    ``max_iter = loop.count`` when set, else ``loop.max_iterations or 100``.
    """
    if loop.count is None and loop.stop is None:
        raise ValueError("loop requires 'count' or 'stop' (or both)")
    max_iter = loop.count if loop.count is not None else (loop.max_iterations or 100)
    logger.info(
        "Loop: up to %d iteration(s); %d sub-step(s)%s",
        max_iter, len(loop.steps),
        f", stop-gate={loop.stop.stage}" if loop.stop else "",
    )
    for it in range(max_iter):
        logger.info("Loop iteration %d/%d", it, max_iter - 1)
        for name in loop.steps:
            _run_stage_step(name, items, pipeline, executor,
                            extra_overrides=[f"run.loop_iteration={it}"])
        if loop.stop and _all_items_stopped(items, loop.stop, pipeline, executor,
                                            extra_overrides=[f"run.loop_iteration={it}"]):
            logger.info("Loop: stop gate satisfied after iteration %d", it)
            break


def _all_items_stopped(
    items: list[str],
    gate: StopGate,
    pipeline: Pipeline,
    executor: Executor,
    extra_overrides: list[str] | None = None,
) -> bool:
    """Run the gate stage across the batch, then check every item's context."""
    _run_stage_step(gate.stage, items, pipeline, executor, extra_overrides=extra_overrides)
    for item in items:
        data = context.get(OmegaConf.create({"output_dir": item}))
        if not data.get(gate.key):
            return False
    return True


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

    Steps are stage names or loops. If any step fails the pipeline halts,
    EXCEPT that stages defined with ``always: true`` still run (finally-
    semantics, e.g. ``cleanup_carla``); the original error is re-raised.
    The batch is resolved lazily so an empty pipeline does not require it.
    """
    executor = executor or get_executor(pipeline.executor)
    items: list[str] | None = None

    def get_items() -> list[str]:
        nonlocal items
        if items is None:
            items = resolve_batch(pipeline.batch)
            logger.info(
                "Pipeline %s: %d item(s); executor=%s; %d step(s)",
                pipeline.name, len(items), pipeline.executor, len(pipeline.steps),
            )
        return items

    def exec_step(index: int) -> None:
        step = pipeline.steps[index]
        if step.loop is not None:
            _run_loop(step.loop, get_items(), pipeline, executor)
        else:
            _run_stage_step(step.name, get_items(), pipeline, executor)

    ran: set[int] = set()
    try:
        for i in range(len(pipeline.steps)):
            ran.add(i)
            exec_step(i)
    except PipelineRunError as first:
        # finally-semantics: run any not-yet-run 'always' stages, then re-raise.
        for i, step in enumerate(pipeline.steps):
            if i in ran or step.loop is not None or step.name is None:
                continue
            defn = _stage_def(step.name, pipeline)
            if not defn.always:
                continue
            logger.warning("A prior stage failed; running 'always' stage %s", step.name)
            ran.add(i)
            try:
                exec_step(i)
            except Exception as e:  # noqa: BLE001 - an always-stage must not mask the original error
                logger.error("'always' stage %s also failed: %s", step.name, e)
        raise first

    logger.info("Pipeline %s complete.", pipeline.name)


def _stage_def_from_dict(name: str, d: dict | None, known: dict) -> StageDef:
    d = dict(d or {})
    if name not in known:
        known_names = ", ".join(sorted(known))
        raise ValueError(f"stages.{name}: unknown stage. Known stages: {known_names}")
    inner = d.get("stage")
    if inner is not None and inner != name:
        raise ValueError(f"stages.{name}: inner 'stage' must match the key (got {inner!r})")
    par = d.get("parallelism", "per_item")
    if isinstance(par, dict):
        parallelism = Parallelism(mode=par.get("mode", "instances"), instances=par.get("instances"))
    else:
        parallelism = Parallelism(mode=str(par))
    batch_size = d.get("batch_size")
    if batch_size is not None and (not isinstance(batch_size, int) or isinstance(batch_size, bool)):
        raise ValueError(f"stages.{name}: 'batch_size' must be an integer")
    return StageDef(
        stage=name,
        always=bool(d.get("always", False)),
        parallelism=parallelism,
        batch_size=batch_size,
        interpreter=d.get("interpreter"),
        overrides=list(d.get("overrides", []) or []),
        env=dict(d.get("env", {}) or {}),
        slurm=d.get("slurm"),
        ray=d.get("ray"),
    )


def _loop_from_dict(d: dict, names: dict) -> Loop:
    d = dict(d or {})
    raw_steps = d.get("steps")
    if not raw_steps or not isinstance(raw_steps, (list, tuple)):
        raise ValueError("loop requires a non-empty 'steps' list of stage names")
    steps = []
    for s in raw_steps:
        if not isinstance(s, str) or s not in names:
            known = ", ".join(sorted(names))
            raise ValueError(f"loop step {s!r} is not a defined stage. Defined: {known}")
        steps.append(s)

    stop = d.get("stop")
    gate = None
    if stop:
        if not isinstance(stop, dict) or stop.get("stage") not in names:
            known = ", ".join(sorted(names))
            raise ValueError(f"loop 'stop' must be {{stage: <defined stage>, key: <context key>}} (got {stop!r}). Defined: {known}")
        gate = StopGate(stage=stop["stage"], key=stop.get("key", "done"))

    count = d.get("count")
    if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 1):
        raise ValueError("loop 'count' must be a positive integer")
    max_iterations = d.get("max_iterations")
    if max_iterations is not None and (
        not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1
    ):
        raise ValueError("loop 'max_iterations' must be a positive integer")
    if count is None and gate is None:
        raise ValueError("loop requires 'count' or 'stop' (or both)")
    return Loop(steps=steps, count=count, stop=gate, max_iterations=max_iterations)


def _step_from_dict(s, names: dict) -> Step:
    if isinstance(s, str):
        if s not in names:
            known = ", ".join(sorted(names))
            raise ValueError(f"step {s!r} is not a defined stage. Defined: {known}")
        return Step(name=s)
    if isinstance(s, dict) and "loop" in s:
        if s.get("stage") is not None or s.get("name") is not None:
            raise ValueError("a step is either a stage name or a 'loop' entry, not both")
        return Step(loop=_loop_from_dict(s["loop"], names))
    raise ValueError(f"invalid step: {s!r} (expected a stage name or a 'loop' entry)")


def load_pipeline(path: str | Path, known_stages: dict | None = None) -> Pipeline:
    """Load a pipeline definition (stage definitions + step order) from YAML.

    ``known_stages`` defaults to the built-in ``STAGE_MAPPING``; override it to
    validate against a custom stage registry (used by tests).
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    known = known_stages if known_stages is not None else _known_stage_names()

    raw_stages = data.get("stages")
    if not raw_stages or not isinstance(raw_stages, dict):
        raise ValueError("pipeline requires a non-empty 'stages:' map")
    stages = {name: _stage_def_from_dict(name, d, known) for name, d in raw_stages.items()}

    raw_steps = data.get("steps")
    if not raw_steps or not isinstance(raw_steps, (list, tuple)):
        raise ValueError("pipeline requires a non-empty 'steps:' list")
    steps = [_step_from_dict(s, stages) for s in raw_steps]

    return Pipeline(
        name=data.get("name", Path(path).stem),
        stages=stages,
        steps=steps,
        executor=data.get("executor", "local"),
        batch=dict(data.get("batch", {}) or {}),
        default_interpreter=data.get("interpreter"),
        default_cwd=data.get("cwd"),
    )

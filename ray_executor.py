"""Ray executor: dispatch a step's work as Ray tasks with bounded concurrency.

Model (core Ray only -- tasks + scheduler):
  * ``per_item``    -> one task per item (``K`` items -> ``K`` tasks, each 1 item).
  * ``instances: N`` -> ``N`` concurrent workers; the item pool is split into
    batches of ``step.batch_size`` (default ``ceil(K / N)``) and dispatched so at
    most ``N`` run at a time, refilling as each completes (a bounded thread pool).

Each batch runs the stage ONCE, in-process, via :func:`stage_runner.run_stage_in_process`
-- so a stage's model is loaded once per batch and reused across that batch's items.
GPU/CPU placement comes from ``step.ray = {num_gpus, num_cpus, runtime_env?}``;
``step.env`` (e.g. ``CUDA_VISIBLE_DEVICES``) is forwarded as worker env vars.

Ray is imported lazily, so this module (and the dispatch logic) is importable and
unit-testable without Ray installed: tests inject a fake ``task_runner``. Real Ray
task submission is validated on a local cluster and, for GPUs/SLURM-scheduled
Ray, on the user's cluster.

Notes:
  * Worker stdout/stderr does NOT come back to the driver (Ray routes it to the
    raylet logs, e.g. ``/tmp/ray/session_*/logs``). A failed batch surfaces its
    exception in ``UnitResult.stderr``; per-item stage logs must be read there.
  * ``per_item`` runs one task per item with concurrency capped by
    ``MAX_INFLIGHT`` (or ``step.ray.max_inflight``) for backpressure.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from executors import Executor, UnitResult, WorkUnit
from pipeline import Step

logger = logging.getLogger("pipeline.ray")

# Upper bound on concurrent in-flight tasks (backpressure). `instances: N` is
# already bounded by N; this caps `per_item` so a large item pool does not spawn
# one blocked driver thread per item. Override per step via step.ray.max_inflight.
MAX_INFLIGHT = 64

try:  # keep the module importable without ray (enables the fake-backend tests)
    import ray
except ImportError:  # pragma: no cover - depends on environment
    ray = None

if ray is not None:  # pragma: no cover - exercised only when ray is present

    @ray.remote
    def _run_stage_task(argv: list, project_root: str) -> str:
        from stage_runner import run_stage_in_process

        run_stage_in_process(list(argv), project_root)
        return "ok"
else:  # pragma: no cover
    _run_stage_task = None


def _project_root() -> str:
    return str(Path(os.getenv("COSMOS_ROOT", os.getcwd())))


def _unit_items(unit: WorkUnit) -> list[str]:
    """Recover the item list a unit was built for (single dir or a manifest shard)."""
    for tok in unit.argv:
        if tok.startswith("run.output_dir="):
            return [tok.split("=", 1)[1]]
        if tok.startswith("batch.manifest="):
            path = tok.split("=", 1)[1]
            return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    return []


def _split(items: list, batch_size: int) -> list[list]:
    size = max(1, batch_size)
    return [items[i : i + size] for i in range(0, len(items), size)]


def _ensure_ray() -> None:
    if ray is None or _run_stage_task is None:
        raise RuntimeError("ray is not installed in this environment")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)


def _real_task_runner(argv: list, step: Step) -> tuple[bool, str, str]:
    """Submit one batch as a Ray task (in-process stage run) and await it."""
    _ensure_ray()
    ray_cfg = step.ray or {}
    options: dict = {}
    if ray_cfg.get("num_gpus"):
        options["num_gpus"] = ray_cfg["num_gpus"]
    if ray_cfg.get("num_cpus") is not None:
        options["num_cpus"] = ray_cfg["num_cpus"]
    runtime_env = dict(ray_cfg.get("runtime_env") or {})
    python_path = list(runtime_env.get("python_path") or [])
    if _project_root() not in python_path:
        python_path.insert(0, _project_root())  # make this repo importable in the worker
    runtime_env["python_path"] = python_path
    # Ray's runtime_env.env_vars requires string values; coerce (e.g. {GPU: 0} -> {GPU: "0"}).
    env_vars = {k: str(v) for k, v in (runtime_env.get("env_vars") or {}).items()}
    env_vars.update({k: str(v) for k, v in (step.env or {}).items()})
    env_vars.setdefault("COSMOS_ROOT", _project_root())
    runtime_env["env_vars"] = env_vars
    options["runtime_env"] = runtime_env
    try:
        ray.get(_run_stage_task.options(**options).remote(list(argv), _project_root()))
    except Exception as exc:  # noqa: BLE001 - surface as a per-batch failure
        logger.error("[ray] batch task failed: %r", exc)
        return (False, "", repr(exc))
    return (True, "", "")


def _dispatch(
    batches: list[list], workers: int, runner, step: Step, interpreter: str
) -> list[UnitResult]:
    """Run each batch via ``runner`` with at most ``workers`` in flight, in order."""
    from runner import _unit_argv

    def run(item) -> UnitResult:
        idx, shard = item
        argv, tmpfiles = _unit_argv(step, shard)
        unit = WorkUnit(
            name=f"ray_{idx}", interpreter=interpreter, argv=argv, env={}, cwd=None, tmpfiles=tmpfiles
        )
        try:
            ok, out, err = runner(argv, step)
        except Exception as exc:  # noqa: BLE001 - surface as a per-batch failure
            ok, out, err = False, "", repr(exc)
        finally:
            for path in tmpfiles:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        return UnitResult(unit=unit, ok=ok, stdout=out, stderr=err)

    workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, enumerate(batches)))


class RayExecutor(Executor):
    """Dispatch a step's units as Ray tasks with bounded concurrency."""

    def __init__(self, task_runner=None):
        # ``task_runner``: (argv, step) -> (ok, stdout, stderr). None => real Ray.
        self._task_runner = task_runner

    def run_units(self, step: Step, units: list[WorkUnit]) -> list[UnitResult]:
        items = [it for u in units for it in _unit_items(u)]
        if not items:
            return []
        if step.parallelism.mode == "per_item":
            # One task per item; concurrency is capped (backpressure) but every
            # item still gets its own task.
            max_inflight = (step.ray or {}).get("max_inflight", MAX_INFLIGHT)
            workers, batch_size = min(len(items), int(max_inflight)), 1
        else:
            workers = step.parallelism.unit_count(len(items))
            batch_size = step.batch_size or max(1, -(-len(items) // workers))
        # NB: the runner's round-robin sharding of `units` is intentionally
        # re-batched here -- Ray owns the batch size + concurrency, so the input
        # unit partition is only used to recover the full item pool.
        batches = _split(items, batch_size)
        logger.info(
            "[ray] %d item(s) -> %d batch(es) of <= %d, %d concurrent",
            len(items), len(batches), batch_size, workers,
        )
        runner = self._task_runner or _real_task_runner
        if self._task_runner is None:
            _ensure_ray()
        return _dispatch(batches, workers, runner, step, step.interpreter or "")

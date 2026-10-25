"""Executors: how a step's work units are launched in parallel.

The runner partitions a step's items into work units (see :mod:`pipeline`) and
hands them to an :class:`Executor`. An executor launches the units in parallel
and reports per-unit success. The executor is the single place a backend's
launch mechanism lives, so Local / SLURM / Ray stay swappable behind one
interface.
"""
from __future__ import annotations

import logging
import os
import subprocess
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from pipeline import Step

logger = logging.getLogger(__name__)


@dataclass
class WorkUnit:
    """One launchable unit of work: a stage run over a shard of items."""

    name: str
    interpreter: str
    argv: list[str]  # hydra-style args, e.g. ["stage=X", "run.output_dir=..."]
    env: dict = field(default_factory=dict)  # extra env for the subprocess (e.g. CUDA_*)
    cwd: str | None = None  # working dir (where `intelligent_pipeline` is importable)
    tmpfiles: list[str] = field(default_factory=list)  # temp files the runner must clean up


@dataclass
class UnitResult:
    unit: WorkUnit
    ok: bool
    stdout: str = ""
    stderr: str = ""


class Executor(ABC):
    """Launch a step's work units in parallel and await them."""

    @abstractmethod
    def run_units(self, step: Step, units: list[WorkUnit]) -> list[UnitResult]:
        ...


class LocalExecutor(Executor):
    """Run each unit as a subprocess in a local thread pool (default / test backend).

    One worker per unit; a unit is
        ``<interpreter> -m intelligent_pipeline <argv...>``
    A non-zero exit marks the unit as failed (the runner then halts the pipeline).
    """

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers

    def run_units(self, step: Step, units: list[WorkUnit]) -> list[UnitResult]:
        def run(u: WorkUnit) -> UnitResult:
            cmd = [u.interpreter, "-m", "intelligent_pipeline", *u.argv]
            env = dict(os.environ)
            env.update(u.env or {})
            logger.info("[local] %s %s", " ".join(cmd), f"(cwd={u.cwd or os.getcwd()})")
            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    cwd=(u.cwd or os.getcwd()),
                    capture_output=True,
                    text=True,
                )
            except OSError as exc:  # interpreter missing / not executable -> a unit failure
                logger.error("[local] unit %s failed to launch: %r", u.name, exc)
                return UnitResult(unit=u, ok=False, stderr=f"failed to launch interpreter: {exc!r}")
            if proc.returncode != 0:
                logger.error("[local] unit %s exited %d", u.name, proc.returncode)
            return UnitResult(
                unit=u,
                ok=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

        workers = self.max_workers or max(1, len(units))
        results: list[UnitResult] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, u): u for u in units}
            for fut in futures:
                results.append(fut.result())

        order = {u.name: i for i, u in enumerate(units)}
        results.sort(key=lambda r: order[r.unit.name])
        return results

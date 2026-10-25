"""SLURM executor: launch a step's work units as SLURM jobs.

Each :class:`~executors.WorkUnit` becomes one SLURM launch. Steps stay sequential
(the runner awaits a step before starting the next); a step's units launch in
parallel. The launch mechanism comes from ``step.slurm``:

    slurm:
      command: srun          # "srun" (blocking, default) or "sbatch" (submit + poll)
      options:               # opaque flag->value mapping, rendered to SLURM flags
        --gres: "gpu:1"
        --cpus-per-task: 16
        --time: "02:00:00"
      poll_interval: 1.0     # sbatch only: seconds between sacct polls
      poll_timeout: null     # sbatch only: give up after this many seconds (null = forever)

``srun`` runs ``[srun, --opt=..., <interpreter> -m intelligent_pipeline, *argv]``
directly and uses its exit code. ``sbatch`` writes a per-unit script
(``#SBATCH --opt=...`` lines + the command), submits it, and polls
``sacct -j <id> --format=State,ExitCode`` to a terminal state.

GPUs can be requested via ``--gres=gpu:N`` in ``options`` or by setting
``CUDA_VISIBLE_DEVICES`` on a unit's ``env``. This module spawns subprocesses and
never imports SLURM, so it is importable (and unit-testable) on a machine with
no SLURM installed -- tests drive it with fake ``srun``/``sbatch``/``sacct`` on
``PATH``.
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from executors import Executor, UnitResult, WorkUnit
from pipeline import Step

logger = logging.getLogger("pipeline.slurm")

# sacct job states that mean "finished, do not poll again".
TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
    "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "PREEMPTED",
}

# Non-real states we emit when sacct itself fails / returns nothing. These stop
# polling immediately (a definitive stop) rather than looping on a broken query.
STOP_STATES = TERMINAL_STATES | {"ERROR", "UNKNOWN"}

# Every state sacct may report (terminal + common transient). Used to *locate* the
# State column robustly -- its position varies by SLURM version and the extra
# columns requested, so we do not assume a fixed index.
KNOWN_STATES = TERMINAL_STATES | {
    "PENDING", "RUNNING", "RESIZING", "REVOKED", "SUSPENDED",
    "CONFIGURING", "COMPLETING", "REQUEUED", "CANCELLED",
}


def render_options(options) -> list[str]:
    """Render a step's ``options`` into SLURM flag tokens.

    * mapping ``{--gres: "gpu:1"}``  -> ``["--gres=gpu:1"]`` (``True`` -> bare flag,
      ``False``/``None`` -> omitted)
    * list/tuple                    -> strings, as-is (raw pre-formatted flags)
    * string                        -> whitespace-split (raw pre-formatted flags)
    """
    if not options:
        return []
    if isinstance(options, str):
        return shlex.split(options)
    if isinstance(options, (list, tuple)):
        return [str(x) for x in options]
    tokens: list[str] = []
    for flag, value in options.items():
        if value is True:
            tokens.append(str(flag))
        elif value is False or value is None:
            continue
        else:
            tokens.append(f"{flag}={value}")
    return tokens


def _command_line(unit: WorkUnit) -> list[str]:
    """The interpreter invocation a unit runs (no SLURM wrapper)."""
    return [unit.interpreter, "-m", "intelligent_pipeline", *unit.argv]


def _map_parallel(items, fn) -> list:
    """Apply ``fn`` to each item in a thread pool, preserving input order."""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        return list(pool.map(fn, items))


def _write_sbatch_script(unit: WorkUnit, opts: list[str]) -> str:
    lines = ["#!/bin/bash"]
    lines += [f"#SBATCH {o}" for o in opts]
    if unit.cwd:
        lines.append(f"cd {shlex.quote(unit.cwd)}")
    lines.append(" ".join(shlex.quote(t) for t in _command_line(unit)))
    fd, path = tempfile.mkstemp(prefix="slurm_job_", suffix=".sh")
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return path


def _parse_job_id(output: str) -> str | None:
    m = re.search(r"Submitted batch job (\d+)", output)
    if m:
        return m.group(1)
    tokens = output.split()
    if tokens and tokens[-1].isdigit():
        return tokens[-1]
    return None


class SlurmExecutor(Executor):
    """Launch a step's units via SLURM (``srun`` or ``sbatch`` per ``step.slurm``)."""

    def __init__(self, poll_interval: float = 1.0, poll_timeout: float | None = None):
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout

    def run_units(self, step: Step, units: list[WorkUnit]) -> list[UnitResult]:
        spec = step.slurm or {}
        command = str(spec.get("command", "srun"))
        if command == "srun":
            return self._run_srun(units, render_options(spec.get("options")))
        if command == "sbatch":
            return self._run_sbatch(units, spec)
        raise ValueError(f"unsupported slurm command: {command!r} (expected srun|sbatch)")

    # -- srun (blocking) ------------------------------------------------------
    def _run_srun(self, units: list[WorkUnit], opts: list[str]) -> list[UnitResult]:
        def run(u: WorkUnit) -> UnitResult:
            cmd = ["srun", *opts, *_command_line(u)]
            env = dict(os.environ)
            env.update(u.env or {})
            logger.info("[slurm/srun] %s (cwd=%s)", " ".join(cmd), u.cwd or os.getcwd())
            try:
                proc = subprocess.run(
                    cmd, env=env, cwd=(u.cwd or os.getcwd()), capture_output=True, text=True
                )
            except (OSError, subprocess.SubprocessError) as exc:
                logger.error("[slurm/srun] unit %s failed to launch: %r", u.name, exc)
                return UnitResult(unit=u, ok=False, stderr=f"slurm launch failed: {exc!r}")
            if proc.returncode != 0:
                logger.error("[slurm/srun] unit %s exited %d", u.name, proc.returncode)
            return UnitResult(unit=u, ok=proc.returncode == 0, stdout=proc.stdout, stderr=proc.stderr)

        return _map_parallel(units, run)

    # -- sbatch (submit + poll) -----------------------------------------------
    def _run_sbatch(self, units: list[WorkUnit], spec: dict) -> list[UnitResult]:
        opts = render_options(spec.get("options"))
        poll_interval = float(spec.get("poll_interval", self.poll_interval))
        poll_timeout = spec.get("poll_timeout", self.poll_timeout)
        env_base = dict(os.environ)

        def submit(pair) -> UnitResult:
            u, script = pair
            env = dict(env_base)
            env.update(u.env or {})
            cwd = u.cwd or os.getcwd()
            try:
                proc = subprocess.run(
                    ["sbatch", script], env=env, cwd=cwd, capture_output=True, text=True, timeout=60
                )
            except (OSError, subprocess.SubprocessError) as exc:
                logger.error("[slurm/sbatch] unit %s submit failed: %r", u.name, exc)
                return UnitResult(unit=u, ok=False, stderr=f"slurm submit failed: {exc!r}")
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            job_id = _parse_job_id(out)
            if job_id is None:
                detail = out.strip() or f"sbatch exited {proc.returncode}"
                logger.error("[slurm/sbatch] unit %s submit failed: %s", u.name, detail)
                return UnitResult(unit=u, ok=False, stderr=detail)

            try:
                state, exit_code = self._poll_job(job_id, env, cwd, poll_interval, poll_timeout)
            except TimeoutError as exc:
                return UnitResult(unit=u, ok=False, stderr=str(exc))

            ok = state == "COMPLETED"
            detail = f"slurm job {job_id}: state={state} exit={exit_code}"
            if not ok:
                logger.error("[slurm/sbatch] unit %s failed: %s", u.name, detail)
            return UnitResult(unit=u, ok=ok, stderr=detail)

        try:
            scripts = [_write_sbatch_script(u, opts) for u in units]
            return _map_parallel(list(zip(units, scripts)), submit)
        finally:
            for path in scripts:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _poll_job(
        self, job_id: str, env: dict, cwd: str, poll_interval: float, poll_timeout: float | None
    ) -> tuple[str, str | None]:
        """Poll ``sacct`` until the job reaches a terminal state; return (state, exit_code)."""
        start = time.monotonic()
        while True:
            state, exit_code = self._sacct_state(job_id, env, cwd)
            if state in STOP_STATES:
                return state, exit_code
            if poll_timeout is not None and (time.monotonic() - start) > poll_timeout:
                raise TimeoutError(f"SLURM job {job_id} still {state!r} after {poll_timeout}s")
            time.sleep(poll_interval)

    def _sacct_state(self, job_id: str, env: dict, cwd: str) -> tuple[str, str | None]:
        cmd = ["sacct", f"-j={job_id}", "--noheader", "--format=State,ExitCode"]
        try:
            proc = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            return "ERROR", repr(exc)
        for line in (proc.stdout or "").splitlines():
            parts = line.split()
            if not parts:
                continue
            # Locate the State token (position varies by SLURM version, e.g. a JobId
            # column can lead); the ExitCode (e.g. "0:0") is the following token.
            state = parts[0]
            exit_code = parts[1] if len(parts) > 1 else None
            for i, tok in enumerate(parts):
                if tok in KNOWN_STATES:
                    state = tok
                    exit_code = parts[i + 1] if i + 1 < len(parts) else None
                    break
            return state, exit_code
        return "UNKNOWN", None

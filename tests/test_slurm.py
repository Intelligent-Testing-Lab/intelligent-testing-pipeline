"""Tests for the SLURM executor (no cluster required).

Run from the project root:
    .venv/bin/python tests/test_slurm.py

Covers:
  * render_options        -- mapping / bool / None / list / string
  * srun                  -- exact command line, success, and failure (fake `srun`)
  * sbatch                -- script content, submit+poll success and failure
                             (fake `sbatch` + `sacct`)
  * unknown command       -- rejected

The fake executables record their arguments to $SLURM_TEST_LOG and honour
$SLURM_TEST_EXIT / $SLURM_TEST_SACCT_STATE so we can assert behaviour without a
real SLURM.
"""
import os
import shlex
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from executors import WorkUnit  # noqa: E402
from pipeline import Parallelism, Pipeline, Step  # noqa: E402
from slurm_executor import SlurmExecutor, render_options  # noqa: E402


# ---------------------------------------------------------------------------
# render_options
# ---------------------------------------------------------------------------
def test_render_options_mapping():
    assert render_options({"--gres": "gpu:1", "--cpus-per-task": 16, "--time": "02:00:00"}) == [
        "--gres=gpu:1",
        "--cpus-per-task=16",
        "--time=02:00:00",
    ]


def test_render_options_bool_and_none():
    assert render_options({"--requeue": True, "--hold": False, "--x": None}) == ["--requeue"]


def test_render_options_list_and_string():
    assert render_options(["--a", "--b=1"]) == ["--a", "--b=1"]
    assert render_options("--a --b=1") == ["--a", "--b=1"]
    assert render_options(None) == []
    assert render_options({}) == []


# ---------------------------------------------------------------------------
# fake SLURM binaries
# ---------------------------------------------------------------------------
def _install_fakes(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "srun").write_text(
        "#!/bin/sh\n"
        'if [ -n "$SLURM_TEST_LOG" ]; then\n'
        "  { printf '%s' \"$0\"; for a in \"$@\"; do printf ' %s' \"$a\"; done; printf '\\n'; } >> \"$SLURM_TEST_LOG\"\n"
        'fi\n'
        'exit "${SLURM_TEST_EXIT:-0}"\n'
    )
    (bin_dir / "sbatch").write_text(
        "#!/bin/sh\n"
        'if [ -n "$SLURM_TEST_LOG" ]; then\n'
        '  printf "sbatch <script>\\n" >> "$SLURM_TEST_LOG"\n'
        '  cat "$1" >> "$SLURM_TEST_LOG"\n'
        'fi\n'
        'echo "Submitted batch job 4242"\n'
        'exit 0\n'
    )
    (bin_dir / "sacct").write_text(
        "#!/bin/sh\n"
        'if [ "${SLURM_TEST_SACCT_STATE:-}" = "EMPTY" ]; then exit 0; fi\n'
        'printf "%s\\n" "${SLURM_TEST_SACCT_STATE:-COMPLETED 0:0}"\n'
        'exit 0\n'
    )
    for name in ("srun", "sbatch", "sacct"):
        os.chmod(bin_dir / name, 0o755)


class _Env:
    """Context manager that points PATH at the fake bin dir + sets test knobs."""

    def __init__(self, bin_dir: Path, log: Path, exit_code: int = 0, sacct: str = "COMPLETED 0:0"):
        self.bin_dir = bin_dir
        self.log = log
        self.exit_code = exit_code
        self.sacct = sacct
        self.saved = {}

    def __enter__(self):
        self.saved = {
            k: os.environ.get(k)
            for k in ("PATH", "SLURM_TEST_LOG", "SLURM_TEST_EXIT", "SLURM_TEST_SACCT_STATE")
        }
        os.environ["PATH"] = f"{self.bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        os.environ["SLURM_TEST_LOG"] = str(self.log)
        os.environ["SLURM_TEST_EXIT"] = str(self.exit_code)
        os.environ["SLURM_TEST_SACCT_STATE"] = self.sacct
        self.log.write_text("")
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def _unit(argv=None, cwd=None) -> WorkUnit:
    return WorkUnit(
        name="u",
        interpreter="/venvs/cosmos/bin/python",
        argv=argv or ["stage=inference", "run.batch_size=8", "run.output_dir=/a"],
        env={},
        cwd=cwd,
    )


def _step(slurm) -> Step:
    return Step(stage="inference", slurm=slurm)


# ---------------------------------------------------------------------------
# srun
# ---------------------------------------------------------------------------
def test_srun_command_line_and_success():
    tmp = Path(tempfile.mkdtemp())
    try:
        bin_dir = tmp / "bin"
        log = tmp / "log"
        _install_fakes(bin_dir)
        with _Env(bin_dir, log):
            units = [_unit(), _unit(argv=["stage=edge", "run.output_dir=/b"])]
            results = SlurmExecutor().run_units(
                _step({"command": "srun", "options": {"--gres": "gpu:1", "--cpus-per-task": 16}}),
                units,
            )
        assert all(r.ok for r in results), [r.stderr for r in results]
        lines = [ln.rstrip() for ln in log.read_text().splitlines() if ln.strip()]
        # The two units launch on parallel threads, so log order is non-deterministic
        # -- match each unit's full command line as a set rather than by position.
        tail_infer = (
            "/venvs/cosmos/bin/python -m intelligent_pipeline"
            " stage=inference run.batch_size=8 run.output_dir=/a"
        )
        infer_lines = [ln for ln in lines if ln.endswith(tail_infer)]
        assert len(infer_lines) == 1, lines
        line = infer_lines[0]
        assert line.split()[0].endswith("srun"), line
        assert "--gres=gpu:1" in line and "--cpus-per-task=16" in line, line
        # SLURM options must precede the interpreter invocation
        assert line.index("--gres=gpu:1") < line.index("/venvs/cosmos/bin/python"), line
        assert len([ln for ln in lines if "stage=edge" in ln and "run.output_dir=/b" in ln]) == 1, lines
        assert len(lines) == 2, lines
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def test_srun_failure():
    tmp = Path(tempfile.mkdtemp())
    try:
        bin_dir = tmp / "bin"
        log = tmp / "log"
        _install_fakes(bin_dir)
        with _Env(bin_dir, log, exit_code=3):
            results = SlurmExecutor().run_units(_step({"command": "srun"}), [_unit()])
        assert len(results) == 1
        assert not results[0].ok
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# sbatch
# ---------------------------------------------------------------------------
def test_sbatch_script_submit_and_poll_success():
    tmp = Path(tempfile.mkdtemp())
    try:
        bin_dir = tmp / "bin"
        log = tmp / "log"
        _install_fakes(bin_dir)
        with _Env(bin_dir, log, sacct="COMPLETED 0:0"):
            results = SlurmExecutor(poll_interval=0.01).run_units(
                _step(
                    {
                        "command": "sbatch",
                        "options": {"--gres": "gpu:1", "--cpus-per-task": 16},
                    }
                ),
                [_unit(cwd=str(PROJECT_ROOT))],
            )
        assert all(r.ok for r in results), [r.stderr for r in results]
        assert "COMPLETED" in results[0].stderr
        first, _, script = log.read_text().partition("\n")
        assert "sbatch <script>" in first, first
        assert script.startswith("#!/bin/bash\n"), script
        assert f"cd {shlex.quote(str(PROJECT_ROOT))}" in script, script
        assert "#SBATCH --gres=gpu:1" in script, script
        assert "#SBATCH --cpus-per-task=16" in script, script
        assert "stage=inference" in script and "-m intelligent_pipeline" in script, script
        # directive order: #SBATCH block, then cd, then the interpreter invocation
        assert (
            script.index("#SBATCH --gres=gpu:1")
            < script.index("cd ")
            < script.index("-m intelligent_pipeline")
        ), script
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def test_sbatch_failure_state():
    tmp = Path(tempfile.mkdtemp())
    try:
        bin_dir = tmp / "bin"
        log = tmp / "log"
        _install_fakes(bin_dir)
        with _Env(bin_dir, log, sacct="FAILED 1:0"):
            results = SlurmExecutor(poll_interval=0.01).run_units(
                _step({"command": "sbatch"}), [_unit()]
            )
        assert len(results) == 1
        assert not results[0].ok
        assert "FAILED" in results[0].stderr
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def test_unknown_command_rejected():
    try:
        SlurmExecutor().run_units(_step({"command": "spull"}), [_unit()])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown slurm command")


def test_sbatch_sacct_empty_fails_fast():
    """sacct returning nothing -> UNKNOWN, a stop state (no infinite polling)."""
    tmp = Path(tempfile.mkdtemp())
    try:
        bin_dir = tmp / "bin"
        log = tmp / "log"
        _install_fakes(bin_dir)
        with _Env(bin_dir, log, sacct="EMPTY"):
            results = SlurmExecutor(poll_interval=0.01).run_units(
                _step({"command": "sbatch"}), [_unit()]
            )
        assert len(results) == 1
        assert not results[0].ok
        assert "UNKNOWN" in results[0].stderr
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def test_step_env_reaches_unit():
    """A step's ``env`` (e.g. CUDA_VISIBLE_DEVICES) is forwarded to the work unit."""
    from runner import build_units

    step = Step(
        stage="seg_mask",
        parallelism=Parallelism(mode="per_item"),
        env={"CUDA_VISIBLE_DEVICES": "3"},
    )
    pl = Pipeline(name="t", steps=[], executor="local", batch={}, metrics=None)
    units = build_units(step, ["/a"], pl)
    assert units[0].env == {"CUDA_VISIBLE_DEVICES": "3"}, units[0].env


def test_runner_slurm_integration():
    """run_pipeline -> get_executor('slurm') -> SlurmExecutor -> fake srun."""
    from runner import get_executor, run_pipeline

    tmp = Path(tempfile.mkdtemp())
    try:
        bin_dir = tmp / "bin"
        log = tmp / "log"
        _install_fakes(bin_dir)
        with _Env(bin_dir, log):
            assert isinstance(get_executor("slurm"), SlurmExecutor)
            pl = Pipeline(
                name="t",
                executor="slurm",
                batch={"dirs": ["/fake/a", "/fake/b"]},
                steps=[
                    Step(
                        stage="seg_mask",
                        parallelism=Parallelism(mode="per_item"),
                        slurm={"command": "srun", "options": {"--cpus-per-task": 4}},
                    )
                ],
            )
            run_pipeline(pl)
        lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2, lines
        assert all("--cpus-per-task=4" in ln for ln in lines), lines
        assert all("stage=seg_mask" in ln for ln in lines), lines
        assert any("run.output_dir=/fake/a" in ln for ln in lines), lines
        assert any("run.output_dir=/fake/b" in ln for ln in lines), lines
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def main():
    tests = [
        ("render_options.mapping", test_render_options_mapping),
        ("render_options.bool_none", test_render_options_bool_and_none),
        ("render_options.list_string", test_render_options_list_and_string),
        ("srun.command_line_success", test_srun_command_line_and_success),
        ("srun.failure", test_srun_failure),
        ("sbatch.script_submit_poll_success", test_sbatch_script_submit_and_poll_success),
        ("sbatch.failure_state", test_sbatch_failure_state),
        ("sbatch.unknown_command", test_unknown_command_rejected),
        ("sbatch.sacct_empty_fails_fast", test_sbatch_sacct_empty_fails_fast),
        ("step.env_reaches_unit", test_step_env_reaches_unit),
        ("integration.runner_slurm", test_runner_slurm_integration),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

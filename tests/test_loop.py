"""Tests for the runner's loop construct, stage definitions, and ``always`` semantics.

Run with:  .venv/bin/python tests/test_loop.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from executors import Executor, UnitResult  # noqa: E402
from pipeline import Loop, Pipeline, StageDef, Step, StopGate  # noqa: E402
from runner import PipelineRunError, _step_from_dict, load_pipeline, run_pipeline  # noqa: E402


class RecordingExecutor(Executor):
    """Records each unit's (stage, argv) and optionally forces failures."""

    def __init__(self, fail_stages: frozenset[str] = frozenset()):
        self.calls: list[tuple[str, list[str]]] = []
        self.fail_stages = set(fail_stages)

    def run_units(self, step, units):
        results = []
        for u in units:
            self.calls.append((step.stage, list(u.argv)))
            ok = step.stage not in self.fail_stages
            results.append(
                UnitResult(
                    unit=u,
                    ok=ok,
                    stderr="" if ok else "forced failure",
                )
            )
        return results


def make_items(n: int = 1) -> list[str]:
    tmp = Path(tempfile.mkdtemp(prefix="loop_test_"))
    return [str(tmp / f"item{i}") for i in range(n)]


def expected_seq(stages: list[str], n: int) -> list[str]:
    """Each step runs one unit per batch item (per_item), so repeat each stage n times."""
    out: list[str] = []
    for s in stages:
        out.extend([s] * n)
    return out


def make_pipeline(
    items: list[str],
    loop: Loop,
    names=("launch_carla", "mutate", "scenario", "evaluate", "gate", "cleanup_carla"),
) -> Pipeline:
    stages = {n: StageDef(stage=n) for n in names}
    steps = [Step(name="launch_carla"), Step(loop=loop), Step(name="cleanup_carla")]
    return Pipeline(name="loop_test", stages=stages, steps=steps, batch={"dirs": items})


def test_count_loop_ordering() -> None:
    items = make_items(2)
    loop = Loop(steps=["mutate", "scenario", "evaluate"], count=3)
    ex = RecordingExecutor()
    run_pipeline(make_pipeline(items, loop), ex)
    stages = [c[0] for c in ex.calls]
    assert stages == expected_seq(
        ["launch_carla"] + ["mutate", "scenario", "evaluate"] * 3 + ["cleanup_carla"], 2
    ), stages


def test_loop_iteration_override() -> None:
    loop = Loop(steps=["mutate", "evaluate"], count=2)
    ex = RecordingExecutor()
    run_pipeline(make_pipeline(make_items(), loop), ex)
    loop_calls = [argv for stage, argv in ex.calls if stage in ("mutate", "evaluate")]
    expected = [f"run.loop_iteration={i}" for i in (0, 0, 1, 1)]
    actual = [next(a for a in argv if a.startswith("run.loop_iteration=")) for argv in loop_calls]
    assert actual == expected, (actual, expected)


def test_fail_fast() -> None:
    loop = Loop(steps=["mutate", "scenario", "evaluate"], count=5)
    ex = RecordingExecutor(fail_stages=frozenset({"scenario"}))
    try:
        run_pipeline(make_pipeline(make_items(), loop), ex)
        raise AssertionError("expected PipelineRunError")
    except PipelineRunError:
        pass
    stages = [c[0] for c in ex.calls]
    assert stages == ["launch_carla", "mutate", "scenario"], stages


def test_stop_gate_early_exit() -> None:
    items = make_items(2)
    # Pre-signal every item as done -> the gate must stop the loop after 1 pass.
    for item in items:
        Path(item).mkdir(parents=True, exist_ok=True)
        (Path(item) / "pipeline_context.json").write_text(json.dumps({"done": True}))
    loop = Loop(steps=["mutate"], stop=StopGate(stage="gate", key="done"), max_iterations=5)
    ex = RecordingExecutor()
    run_pipeline(make_pipeline(items, loop), ex)
    stages = [c[0] for c in ex.calls]
    assert stages == expected_seq(["launch_carla", "mutate", "gate", "cleanup_carla"], 2), stages


def test_stop_gate_not_triggered() -> None:
    items = make_items()  # no context file -> gate never satisfied
    loop = Loop(
        steps=["mutate"],
        count=3,
        stop=StopGate(stage="gate", key="done"),
        max_iterations=5,
    )
    ex = RecordingExecutor()
    run_pipeline(make_pipeline(items, loop), ex)
    stages = [c[0] for c in ex.calls]
    assert stages == expected_seq(["launch_carla"] + ["mutate", "gate"] * 3 + ["cleanup_carla"], 1), stages


def test_stop_only_max_iterations_caps() -> None:
    items = make_items()  # no context file -> gate never satisfied
    loop = Loop(steps=["mutate"], stop=StopGate(stage="gate", key="done"), max_iterations=2)
    ex = RecordingExecutor()
    run_pipeline(make_pipeline(items, loop), ex)
    stages = [c[0] for c in ex.calls]
    assert stages == ["launch_carla", "mutate", "gate", "mutate", "gate", "cleanup_carla"], stages


def test_gate_receives_loop_iteration() -> None:
    loop = Loop(steps=["mutate"], stop=StopGate(stage="gate", key="done"), max_iterations=2)
    ex = RecordingExecutor()
    run_pipeline(make_pipeline(make_items(), loop), ex)
    gate_calls = [argv for stage, argv in ex.calls if stage == "gate"]
    expected = ["run.loop_iteration=0", "run.loop_iteration=1"]
    actual = [next(a for a in argv if a.startswith("run.loop_iteration=")) for argv in gate_calls]
    assert actual == expected, (actual, expected)


def test_always_runs_after_failure() -> None:
    items = make_items()
    stages = {n: StageDef(stage=n) for n in ("a", "b", "c")}
    stages["c"].always = True
    pipeline = Pipeline(
        name="always_test",
        stages=stages,
        steps=[Step(name="a"), Step(name="b"), Step(name="c")],
        batch={"dirs": items},
    )
    ex = RecordingExecutor(fail_stages=frozenset({"b"}))
    try:
        run_pipeline(pipeline, ex)
        raise AssertionError("expected PipelineRunError")
    except PipelineRunError:
        pass
    assert [c[0] for c in ex.calls] == ["a", "b", "c"], ex.calls


def test_always_not_rerun_if_it_ran_before_failure() -> None:
    items = make_items()
    stages = {n: StageDef(stage=n) for n in ("a", "b", "cleanup")}
    stages["cleanup"].always = True
    pipeline = Pipeline(
        name="always_test",
        stages=stages,
        steps=[Step(name="cleanup"), Step(name="a"), Step(name="b")],
        batch={"dirs": items},
    )
    ex = RecordingExecutor(fail_stages=frozenset({"b"}))
    try:
        run_pipeline(pipeline, ex)
        raise AssertionError("expected PipelineRunError")
    except PipelineRunError:
        pass
    # cleanup runs once, in normal order, before the failure; it is NOT re-run.
    assert [c[0] for c in ex.calls] == ["cleanup", "a", "b"], ex.calls


def test_always_failure_does_not_mask_original() -> None:
    items = make_items()
    stages = {n: StageDef(stage=n) for n in ("a", "cleanup")}
    stages["cleanup"].always = True
    pipeline = Pipeline(
        name="always_test",
        stages=stages,
        steps=[Step(name="a"), Step(name="cleanup")],
        batch={"dirs": items},
    )
    ex = RecordingExecutor(fail_stages=frozenset({"a", "cleanup"}))
    try:
        run_pipeline(pipeline, ex)
        raise AssertionError("expected PipelineRunError")
    except PipelineRunError as e:
        assert "stage 'a' failed" in str(e), str(e)
    assert [c[0] for c in ex.calls] == ["a", "cleanup"], ex.calls


KNOWN = ["launch_carla", "cleanup_carla", "mutate", "scenario", "evaluate", "gate"]


def test_pipeline_yaml_parses() -> None:
    yaml_text = """
name: scenario_search
executor: local
batch:
  dirs: [/tmp/a, /tmp/b]
stages:
  launch_carla:
    interpreter: /venvs/carla/bin/python
  mutate:
    parallelism: per_item
  scenario:
    interpreter: /venvs/carla/bin/python
  evaluate:
    parallelism: {mode: instances, instances: 2}
    interpreter: /venvs/gpu/bin/python
    ray: {num_gpus: 1}
  gate:
    interpreter: /venvs/gpu/bin/python
    overrides: [stage.mode=gate]
  cleanup_carla:
    always: true
steps:
  - launch_carla
  - loop:
      stop: {stage: gate, key: converged}
      max_iterations: 10
      steps: [mutate, scenario, evaluate]
  - cleanup_carla
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pipeline.yaml"
        path.write_text(yaml_text)
        p = load_pipeline(path, known_stages=KNOWN)
    assert p.stages["launch_carla"].interpreter == "/venvs/carla/bin/python"
    assert p.stages["evaluate"].parallelism.instances == 2
    assert p.stages["gate"].overrides == ["stage.mode=gate"]
    assert p.stages["cleanup_carla"].always is True
    assert p.steps[0].name == "launch_carla"
    loop = p.steps[1].loop
    assert isinstance(loop, Loop)
    assert loop.steps == ["mutate", "scenario", "evaluate"]
    assert loop.stop is not None and loop.stop.stage == "gate" and loop.stop.key == "converged"
    assert loop.max_iterations == 10 and loop.count is None
    assert p.steps[2].name == "cleanup_carla"


def test_parse_errors() -> None:
    cases = {
        "both name and loop": (
            {"name": "mutate", "loop": {"count": 1, "steps": ["mutate"]}},
            KNOWN,
        ),
        "undefined stage in steps": (
            {"stage": "nope"},
            KNOWN,
        ),
        "undefined loop step": (
            {"loop": {"count": 1, "steps": ["nope"]}},
            KNOWN,
        ),
        "undefined stop stage": (
            {"loop": {"stop": {"stage": "nope"}, "steps": ["mutate"]}},
            KNOWN,
        ),
        "loop count zero": (
            {"loop": {"count": 0, "steps": ["mutate"]}},
            KNOWN,
        ),
        "loop count string": (
            {"loop": {"count": "3", "steps": ["mutate"]}},
            KNOWN,
        ),
        "loop empty steps": ({"loop": {"count": 3}}, KNOWN),
        "stage not in STAGE_MAPPING": ({"stage": "not_a_stage"}, KNOWN),
        "inner stage mismatch": (
            {"stage": "evaluate", "loop": None},
            KNOWN,
        ),
    }
    for label, (step, known) in cases.items():
        try:
            _step_from_dict(step, {n: StageDef(stage=n) for n in known})
            raise AssertionError(f"expected ValueError for {label!r}")
        except ValueError:
            pass


def test_stage_def_key_must_match_known() -> None:
    try:
        import runner

        runner._stage_def_from_dict("nope", {}, KNOWN)
        raise AssertionError("expected ValueError for unknown stage key")
    except ValueError:
        pass


def test_missing_stages_or_steps_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for label, text in [
            ("missing stages", "steps: [scenario]"),
            ("missing steps", "stages:\n  scenario: {}"),
            ("empty steps", "stages:\n  scenario: {}\nsteps: []"),
        ]:
            path = Path(tmp) / "p.yaml"
            path.write_text(text)
            try:
                load_pipeline(path, known_stages=KNOWN)
                raise AssertionError(f"expected ValueError for {label!r}")
            except ValueError:
                pass


def main() -> int:
    tests = [
        test_count_loop_ordering,
        test_loop_iteration_override,
        test_fail_fast,
        test_stop_gate_early_exit,
        test_stop_gate_not_triggered,
        test_stop_only_max_iterations_caps,
        test_gate_receives_loop_iteration,
        test_always_runs_after_failure,
        test_always_not_rerun_if_it_ran_before_failure,
        test_always_failure_does_not_mask_original,
        test_pipeline_yaml_parses,
        test_parse_errors,
        test_stage_def_key_must_match_known,
        test_missing_stages_or_steps_rejected,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

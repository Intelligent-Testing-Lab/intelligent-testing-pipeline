"""Tests for the Ray executor + the in-process stage runner.

Run from the project root:
    .venv/bin/python tests/test_ray.py

Covers (without needing a GPU / remote cluster):
  * stage_runner.parse_argv / build_stage_config   -- pure
  * stage_runner.run_stage_in_process              -- real in-process `seg_mask` run
  * ray_executor._split / _unit_items              -- pure
  * RayExecutor dispatch (fake task_runner)        -- per_item / instances / batch_size,
                                                       bounded concurrency, failure
  * run_pipeline integration (RayExecutor + fake)  -- failure -> PipelineRunError
  * RayExecutor REAL local run of `seg_mask`        -- full ray.init + task + in-process stage

The real-Ray test is skipped automatically if ray is not installed.
"""
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import Parallelism, Pipeline, Step  # noqa: E402
from ray_executor import RayExecutor, _split, _unit_items  # noqa: E402
from runner import PipelineRunError, build_units, run_pipeline  # noqa: E402
from stage_runner import build_stage_config, parse_argv, run_stage_in_process  # noqa: E402


def _pl(**kw):
    base = dict(name="t", steps=[], executor="local", batch={}, metrics=None)
    base.update(kw)
    return Pipeline(**base)


# ---------------------------------------------------------------------------
# stage_runner (pure)
# ---------------------------------------------------------------------------
def test_parse_argv():
    stage, overrides = parse_argv(["stage=inference", "run.batch_size=8", "run.output_dir=/a"])
    assert stage == "inference"
    assert overrides == [("run.batch_size", "8"), ("run.output_dir", "/a")]
    stage2, ov2 = parse_argv(["stage=seg_mask", "batch.manifest=/m.txt"])
    assert stage2 == "seg_mask" and ov2 == [("batch.manifest", "/m.txt")]
    assert parse_argv(["nope"]) == (None, [])


def test_build_stage_config():
    config = build_stage_config(
        "seg_mask", [("run.output_dir", "/some/dir"), ("run.batch_size", "4")], str(PROJECT_ROOT)
    )
    assert config.stage.name == "seg_mask"
    assert str(config.run.output_dir) == "/some/dir"
    # run.* interpolations resolve against the overridden output_dir
    assert str(config.run.rgb_input) == "/some/dir/sensor_recordings/front_rgb/front_rgb.mp4"
    assert str(config.run.batch_size) == "4"


def test_run_stage_in_process_seg_mask():
    import numpy as np

    work = Path(tempfile.mkdtemp())
    try:
        d = work / "a"
        sem = d / "sensor_recordings" / "front_semantic_id"
        sem.mkdir(parents=True)
        np.save(str(sem / "raw_frames.npy"), np.random.randint(0, 30, (3, 32, 48), dtype=np.uint8))
        run_stage_in_process(["stage=seg_mask", f"run.output_dir={d}"], str(PROJECT_ROOT))
        assert (d / "seg_mask.mp4").exists()
        assert (d / "edge_mask.mp4").exists()
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# ray_executor helpers (pure)
# ---------------------------------------------------------------------------
def test_split_and_unit_items():
    assert _split([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert _split([1, 2, 3], 5) == [[1, 2, 3]]
    from executors import WorkUnit

    assert _unit_items(WorkUnit(name="u", interpreter="py", argv=["stage=x", "run.output_dir=/a"])) == ["/a"]


def _manifest_unit(items):
    import tempfile as tf

    from executors import WorkUnit

    fd, path = tf.mkstemp(suffix=".txt")
    with open(fd, "w") as fh:
        fh.write("\n".join(items))
    return WorkUnit(name="u", interpreter="py", argv=["stage=x", f"batch.manifest={path}"])


def test_unit_items_manifest():
    u = _manifest_unit(["/a", "/b"])
    assert _unit_items(u) == ["/a", "/b"]
    Path([a for a in u.argv if a.startswith("batch.manifest=")][0].split("=", 1)[1]).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# RayExecutor dispatch (fake task_runner)
# ---------------------------------------------------------------------------
class FakeRunner:
    def __init__(self, fail_on=None):
        self.dispatched = []
        self.fail_on = fail_on

    def __call__(self, argv, step):
        self.dispatched.append(list(argv))
        if self.fail_on and any(self.fail_on in a for a in argv):
            return (False, "", "forced failure")
        return (True, "", "")


def test_ray_per_item_dispatch():
    step = Step(stage="seg_mask", parallelism=Parallelism(mode="per_item"))
    units = build_units(step, ["/a", "/b", "/c"], _pl())
    fake = FakeRunner()
    results = RayExecutor(task_runner=fake).run_units(step, units)
    assert len(results) == 3 and all(r.ok for r in results)
    assert len(fake.dispatched) == 3
    assert all(any(a.startswith("run.output_dir=") for a in argv) for argv in fake.dispatched)
    assert all("stage=seg_mask" in argv for argv in fake.dispatched)


def test_ray_instances_batch_size():
    step = Step(
        stage="inference", parallelism=Parallelism(mode="instances", instances=3), batch_size=2
    )
    units = build_units(step, ["/a", "/b", "/c", "/d", "/e", "/f"], _pl())
    fake = FakeRunner()
    results = RayExecutor(task_runner=fake).run_units(step, units)
    assert len(fake.dispatched) == 3  # 6 items / batch_size 2
    assert all("run.batch_size=2" in argv for argv in fake.dispatched)
    assert all(any(a.startswith("batch.manifest=") for a in argv) for argv in fake.dispatched)
    assert len(results) == 3 and all(r.ok for r in results)


class ConcurrencyTracker:
    def __init__(self, delay=0.03):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __call__(self, argv, step):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1
        return (True, "", "")


def test_ray_bounded_concurrency():
    step = Step(
        stage="inference", parallelism=Parallelism(mode="instances", instances=3), batch_size=1
    )
    units = build_units(step, [f"/i{k}" for k in range(9)], _pl())  # 9 batches, 3 workers
    tracker = ConcurrencyTracker()
    results = RayExecutor(task_runner=tracker).run_units(step, units)
    assert len(results) == 9
    assert tracker.max_active <= 3, f"expected <=3 in flight, got {tracker.max_active}"
    assert tracker.max_active >= 2, f"expected concurrency, got {tracker.max_active}"


def test_ray_failure_in_results():
    step = Step(stage="seg_mask", parallelism=Parallelism(mode="per_item"))
    units = build_units(step, ["/a", "/b"], _pl())
    fake = FakeRunner(fail_on="/b")
    results = RayExecutor(task_runner=fake).run_units(step, units)
    assert len(results) == 2
    assert any(not r.ok for r in results)
    assert any(r.ok for r in results)


def test_ray_pipeline_failure_raises():
    pl = _pl(
        batch={"dirs": ["/a", "/b"]},
        steps=[Step(stage="seg_mask", parallelism=Parallelism(mode="per_item"))],
    )
    fake = FakeRunner(fail_on="/b")
    try:
        run_pipeline(pl, RayExecutor(task_runner=fake))
    except PipelineRunError:
        pass
    else:
        raise AssertionError("expected PipelineRunError when a ray batch fails")


# ---------------------------------------------------------------------------
# real local Ray (skipped if ray missing)
# ---------------------------------------------------------------------------
def test_ray_real_local_seg_mask():
    try:
        import ray  # noqa: F401
    except ImportError:
        print("  SKIP  ray.real_local_seg_mask (ray not installed)")
        return
    import numpy as np

    work = Path(tempfile.mkdtemp())
    try:
        d = work / "a"
        sem = d / "sensor_recordings" / "front_semantic_id"
        sem.mkdir(parents=True)
        np.save(str(sem / "raw_frames.npy"), np.random.randint(0, 30, (3, 32, 48), dtype=np.uint8))
        step = Step(
            stage="seg_mask",
            parallelism=Parallelism(mode="per_item"),
            ray={"num_gpus": 0, "num_cpus": 1},
        )
        units = build_units(step, [str(d)], _pl())
        results = RayExecutor().run_units(step, units)  # real ray.init + task + in-process stage
        assert all(r.ok for r in results), [r.stderr for r in results]
        assert (d / "seg_mask.mp4").exists()
        assert (d / "edge_mask.mp4").exists()
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if ray.is_initialized():
            ray.shutdown()


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def main():
    tests = [
        ("stage_runner.parse_argv", test_parse_argv),
        ("stage_runner.build_stage_config", test_build_stage_config),
        ("stage_runner.in_process_seg_mask", test_run_stage_in_process_seg_mask),
        ("ray.split_unit_items", test_split_and_unit_items),
        ("ray.unit_items_manifest", test_unit_items_manifest),
        ("ray.per_item_dispatch", test_ray_per_item_dispatch),
        ("ray.instances_batch_size", test_ray_instances_batch_size),
        ("ray.bounded_concurrency", test_ray_bounded_concurrency),
        ("ray.failure_in_results", test_ray_failure_in_results),
        ("ray.pipeline_failure", test_ray_pipeline_failure_raises),
        ("ray.real_local_seg_mask", test_ray_real_local_seg_mask),
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

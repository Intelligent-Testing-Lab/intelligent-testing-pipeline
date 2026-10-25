"""Tests for the pipeline runner, model helpers, and the LocalExecutor.

Run from the project root:
    .venv/bin/python tests/test_runner.py

Covers:
  * resolve_batch          -- dirs (list + comma string), manifest, glob, exactly-one
  * partition_round_robin  -- per_item / instances, balance, uneven splits
  * unit_argv (via build_units) -- single item vs multi-item shard, batch_size, overrides
  * runner                 -- metric-subset skip, failure propagation (RecordingExecutor)
  * LocalExecutor          -- real per_item run of the light `seg_mask` stage (synthetic)
"""
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from executors import Executor, LocalExecutor, UnitResult  # noqa: E402
from pipeline import (  # noqa: E402
    Parallelism,
    Pipeline,
    Step,
    partition_round_robin,
    resolve_batch,
)
from runner import PipelineRunError, build_units, run_pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# resolve_batch
# ---------------------------------------------------------------------------
def test_resolve_batch_dirs_list():
    assert resolve_batch({"dirs": ["/a", "/b", "/c"]}) == ["/a", "/b", "/c"]


def test_resolve_batch_dirs_comma_string():
    assert resolve_batch({"dirs": "/a, /b ,/c"}) == ["/a", "/b", "/c"]


def test_resolve_batch_manifest():
    tmp = Path(tempfile.mkdtemp())
    try:
        mf = tmp / "m.txt"
        mf.write_text("/x/one\n/x/two\n\n")
        assert resolve_batch({"manifest": str(mf)}) == ["/x/one", "/x/two"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_batch_glob():
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "b").mkdir()
        (tmp / "a").mkdir()
        out = resolve_batch({"glob": str(tmp / "*")})
        assert sorted(Path(p).name for p in out) == ["a", "b"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_batch_exactly_one():
    for bad in ({}, {"dirs": "/a", "glob": "x/*"}):
        try:
            resolve_batch(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


# ---------------------------------------------------------------------------
# partition_round_robin
# ---------------------------------------------------------------------------
def test_partition_per_item_is_one_each():
    items = [1, 2, 3, 4]
    assert partition_round_robin(items, 4) == [[1], [2], [3], [4]]


def test_partition_instances_balanced():
    assert partition_round_robin([1, 2, 3, 4], 2) == [[1, 3], [2, 4]]


def test_partition_instances_uneven():
    shards = partition_round_robin(list(range(7)), 3)
    sizes = sorted(len(s) for s in shards)
    assert sizes == [2, 2, 3]  # differ by at most one
    assert sorted(x for s in shards for x in s) == list(range(7))  # no item lost


# ---------------------------------------------------------------------------
# build_units / argv (pure -- no subprocess)
# ---------------------------------------------------------------------------
def _pl(**kw):
    base = dict(name="t", steps=[], executor="local", batch={}, metrics=None)
    base.update(kw)
    return Pipeline(**base)


def test_build_units_per_item_single():
    step = Step(stage="seg_mask", parallelism=Parallelism(mode="per_item"))
    units = build_units(step, ["/a", "/b"], _pl())
    assert len(units) == 2
    assert "stage=seg_mask" in units[0].argv
    assert "run.output_dir=/a" in units[0].argv
    assert "run.output_dir=/b" in units[1].argv
    assert not any("batch.manifest" in a for u in units for a in u.argv)


def test_build_units_instances_multi_shard():
    step = Step(stage="inference", parallelism=Parallelism(mode="instances", instances=2))
    units = build_units(step, ["/a", "/b", "/c", "/d"], _pl())
    assert len(units) == 2
    # 2-item shards -> batch.manifest, not run.output_dir
    assert all(any("batch.manifest=" in a for a in u.argv) for u in units)
    assert not any("run.output_dir" in a for u in units for a in u.argv)
    # each manifest holds 2 items
    for u in units:
        mf = [a for a in u.argv if a.startswith("batch.manifest=")][0].split("=", 1)[1]
        assert len(Path(mf).read_text().splitlines()) == 2


def test_build_units_batch_size_and_overrides():
    step = Step(
        stage="inference",
        parallelism=Parallelism(mode="instances", instances=1),
        batch_size=8,
        overrides=["run.results_file=/r.json"],
    )
    units = build_units(step, ["/a"], _pl())
    argv = units[0].argv
    assert "run.batch_size=8" in argv
    assert "run.results_file=/r.json" in argv


# ---------------------------------------------------------------------------
# runner logic (RecordingExecutor -- no subprocess)
# ---------------------------------------------------------------------------
class RecordingExecutor(Executor):
    def __init__(self):
        self.calls: list[tuple[str, list[list[str]]]] = []
        self.fail_stages: set[str] = set()

    def run_units(self, step, units):
        self.calls.append((step.stage, [list(u.argv) for u in units]))
        if step.stage in self.fail_stages:
            return [UnitResult(unit=u, ok=False, stderr="boom") for u in units]
        return [UnitResult(unit=u, ok=True) for u in units]


def test_runner_skips_unselected_metric():
    pl = _pl(
        batch={"dirs": ["/a"]},
        metrics=["ssim"],
        steps=[
            Step(stage="seg_mask"),
            Step(stage="eval_lpips", metric="lpips"),
            Step(stage="eval_ssim", metric="ssim"),
        ],
    )
    ex = RecordingExecutor()
    run_pipeline(pl, ex)
    ran = [s for s, _ in ex.calls]
    assert ran == ["seg_mask", "eval_ssim"], ran  # eval_lpips skipped (not selected)


def test_runner_runs_all_when_metrics_none():
    pl = _pl(
        batch={"dirs": ["/a"]},
        steps=[Step(stage="eval_lpips", metric="lpips"), Step(stage="eval_ssim", metric="ssim")],
    )
    ex = RecordingExecutor()
    run_pipeline(pl, ex)
    assert [s for s, _ in ex.calls] == ["eval_lpips", "eval_ssim"]


def test_runner_failure_raises():
    pl = _pl(
        batch={"dirs": ["/a", "/b"]},
        steps=[Step(stage="seg_mask"), Step(stage="edge")],
    )
    ex = RecordingExecutor()
    ex.fail_stages.add("edge")
    try:
        run_pipeline(pl, ex)
    except PipelineRunError:
        pass
    else:
        raise AssertionError("expected PipelineRunError when a step fails")
    # the failing step's units were the last thing attempted
    assert [s for s, _ in ex.calls][-1] == "edge"


# ---------------------------------------------------------------------------
# LocalExecutor -- real per_item run of the light `seg_mask` stage
# ---------------------------------------------------------------------------
def test_localexecutor_per_item_seg_mask():
    import numpy as np

    work = Path(tempfile.mkdtemp())
    try:
        items = []
        for name in ("a", "b", "c"):
            d = work / name
            sem = d / "sensor_recordings" / "front_semantic_id"
            sem.mkdir(parents=True)
            np.save(str(sem / "raw_frames.npy"), np.random.randint(0, 30, (4, 64, 96), dtype=np.uint8))
            items.append(str(d))

        venv_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
        pl = Pipeline(
            name="test_local",
            executor="local",
            batch={"dirs": items},
            default_interpreter=venv_python,
            default_cwd=str(PROJECT_ROOT),
            steps=[Step(stage="seg_mask", parallelism=Parallelism(mode="per_item"))],
        )
        run_pipeline(pl, LocalExecutor())

        for d in items:
            assert (Path(d) / "seg_mask.mp4").exists(), f"missing seg_mask.mp4 in {d}"
            assert (Path(d) / "edge_mask.mp4").exists(), f"missing edge_mask.mp4 in {d}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def main():
    tests = [
        ("resolve_batch.dirs_list", test_resolve_batch_dirs_list),
        ("resolve_batch.dirs_comma", test_resolve_batch_dirs_comma_string),
        ("resolve_batch.manifest", test_resolve_batch_manifest),
        ("resolve_batch.glob", test_resolve_batch_glob),
        ("resolve_batch.exactly_one", test_resolve_batch_exactly_one),
        ("partition.per_item", test_partition_per_item_is_one_each),
        ("partition.instances_balanced", test_partition_instances_balanced),
        ("partition.instances_uneven", test_partition_instances_uneven),
        ("build_units.per_item", test_build_units_per_item_single),
        ("build_units.instances", test_build_units_instances_multi_shard),
        ("build_units.batch_size_overrides", test_build_units_batch_size_and_overrides),
        ("runner.skip_metric", test_runner_skips_unselected_metric),
        ("runner.all_metrics", test_runner_runs_all_when_metrics_none),
        ("runner.failure", test_runner_failure_raises),
        ("local.executor_seg_mask", test_localexecutor_per_item_seg_mask),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

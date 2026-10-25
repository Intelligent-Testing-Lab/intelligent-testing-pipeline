### intelligent pipeline

An internal library for building multi-stage data pipelines. Each stage is a
`Stage` subclass with its own config (via [hydra-core]()); a **pipeline runner**
partitions a *batch* of items across the stages and dispatches them to a
swappable **executor** -- **local** (subprocesses), **SLURM** (`srun`/`sbatch`),
or **Ray** (tasks with bounded concurrency) -- with **per-stage interpreters/venvs**
and **per-stage GPU** selection. It handles logging, performance timing, and
optional completion verification per stage.

### Dependencies
- UV
- Python 3.10+
- Ubuntu 24.04+
- `ray` (driver) -- only needed for the Ray executor

### Usage

Install the framework requirements (the light set; heavy stage deps live in the
venv you point each stage at):
```bash
uv venv --python 3.10
uv pip install -r requirements.txt
```

## Writing a stage

Inherit `Stage`, override `_run()` (required) and, optionally, `verify_completion()`:

```python
from stage import Stage
from omegaconf import DictConfig

class EdgeDetector(Stage):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, stage="edge")

    # required
    def _run(self) -> None:
        # stage-specific config is under self.config.stage
        input_video = self.config.stage.input_dir
        # ... do the work, write artifacts ...

    # optional: return False if the stage did not complete correctly
    def verify_completion(self) -> bool:
        return True
```

- Stage-specific parameters live in `configs/stage/<name>.yaml` and are read via
  `self.config.stage`.
- The shared per-item context lives in `configs/config.yaml` under `run:` and is
  read via `self.config.run` (e.g. `run.output_dir`, `run.rgb_input`).
- Register the stage in `STAGE_MAPPING` in `intelligent_pipeline.py`
  (`"edge": ("stages.edge", "EdgeDetector")`) -- the name becomes the
  `stage=` / step selector.

### Single-stage mode

Run one stage directly (handy for development and debugging):
```bash
python -m intelligent_pipeline stage=edge run.output_dir=/data/.../my_run
# any config key can be overridden (hydra):
python -m intelligent_pipeline stage=edge stage.guidance=8 run.results_file=/shared/results.json
# the inference stage takes a batch of output dirs:
python -m intelligent_pipeline stage=inference batch.manifest=/path/to/dirs.txt
```

## Pipeline mode

A pipeline is a YAML of ordered steps in `configs/pipelines/<name>.yaml`, run via:
```bash
python -m intelligent_pipeline pipeline=cosmos_transfer \
  batch.glob=/data/.../test_3/* \
  executor=local \
  metrics=lpips,ssim
```

| Knob (CLI / `configs/config.yaml`) | Meaning | Default |
|---|---|---|
| `pipeline` | pipeline name (`configs/pipelines/<name>.yaml`) | `null` (single-stage mode) |
| `batch.dirs` / `batch.manifest` / `batch.glob` | the batch of output dirs (exactly one) | required to run |
| `executor` | `local` \| `slurm` \| `ray` | the pipeline's, else `local` |
| `metrics` | subset of eval steps to run (e.g. `lpips,ssim`) | `null` = all |
| `interpreter` | default interpreter for steps that omit one | current python |

### Steps

Each step runs one registered stage across the batch:
```yaml
steps:
  - stage: prompt            # required
    parallelism: per_item    # per_item | {instances: N}
    batch_size: 8            # optional: forwarded to the stage as run.batch_size
    interpreter: /venvs/light/bin/python   # venv this stage runs in
    env: {CUDA_VISIBLE_DEVICES: "0"}       # optional extra env for the unit
    metric: lpips            # optional: step runs only if `lpips` is in `metrics`
    overrides: [run.results_file=/shared/results.json]  # optional extra hydra overrides
    slurm: {command: srun, options: {--cpus-per-task: 8}}  # for the SLURM executor
    ray:   {num_gpus: 1, num_cpus: 16}                      # for the Ray executor
```

**Parallelism** (per step):
- `per_item` -- one work unit per item (data parallelism).
- `instances: N` -- `N` work units; the items are round-robin-balanced into `N`
  shards (sizes differ by at most one). This is what keeps a resident model (e.g.
  inference) loaded once per unit instead of once per item.

### Executors

One executor is chosen per pipeline run (`executor=`). Each step's work units are
launched in parallel; a step that fails halts the pipeline.

- **local** (default) -- each unit runs as
  `<interpreter> -m intelligent_pipeline <args>` in a thread pool. Great for
  development and CI.
- **slurm** -- one SLURM launch per unit. `srun` (blocking) by default, or
  `sbatch` (submit a per-unit `#SBATCH` script, then poll `sacct` to a terminal
  state). `step.slurm.options` are opaque flags (`{--gres: "gpu:1", --cpus-per-task: 16}`).
  GPUs via `--gres=gpu:N` or `CUDA_VISIBLE_DEVICES` in `env`.
- **ray** -- batches dispatched as Ray tasks with bounded concurrency. `per_item`
  runs one task per item; `instances: N` keeps at most `N` batches in flight
  (each batch runs the stage once in-process, so its model loads once per batch).
  Resources from `step.ray = {num_gpus, num_cpus, runtime_env?, max_inflight?}`;
  the stage's venv is reached via `runtime_env` (e.g. `python_path` / a virtualenv).

> The runner is backend-agnostic: it decides *what* to run (steps, sharding,
> interpreter, args); the executor decides *how* to launch it. All three are
> unit-tested -- local and Ray against a real local cluster, SLURM against
> stubbed `srun`/`sbatch`/`sacct`. GPU/SLURM/Ray cluster behaviour is validated
> on a real cluster.

## Example pipeline: Cosmos-transfer data generation + quality evaluation

This repo ships a concrete pipeline (`configs/pipelines/cosmos_transfer.yaml`)
that re-implements the Cosmos-transfer data-generation + quality-evaluation
workflow. Each stage is its own file in `stages/` with its own config in
`configs/stage/` (named to match), all registered in `STAGE_MAPPING`.

| Stage | What it does | Heavy deps |
|---|---|---|
| `prompt` | Build a text prompt for the scenario | openai |
| `seg_mask` | Build the segmentation control mask + edge ROI mask from raw CARLA semantic IDs | opencv |
| `edge` | Edge-detect the RGB video | opencv |
| `inference` | Guided cosmos generation (resident model, batched) | cosmos + torch |
| `eval_lpips` | Per-frame LPIPS (original vs generated) | torch + pyiqa |
| `eval_clipqa` | CLIPIQA+ on a single (user-chosen) video | torch + pyiqa |
| `eval_ssim` | Per-frame SSIM (original vs generated) | skimage |
| `eval_iou` | Per-class semantic IoU | ultralytics |
| `eval_iou_temporal` | Optical-flow (temporal) per-class IoU | ultralytics |

Run it (local executor, a subset of metrics):
```bash
python -m intelligent_pipeline pipeline=cosmos_transfer \
  batch.glob=/data/cosmos/outputs/scratch/test_3/* \
  executor=local \
  metrics=lpips,ssim,iou
```
Or on SLURM / Ray:
```bash
python -m intelligent_pipeline pipeline=cosmos_transfer batch.glob=... executor=slurm
python -m intelligent_pipeline pipeline=cosmos_transfer batch.glob=... executor=ray
```

Each eval stage **appends** its metric to the shared results file (default
`<output_dir>/results.json`, set `run.results_file` to share one), so one file
accumulates every metric for an item. `eval_clipqa` scores a single video -- set
`run.clipqa_video` to the generated or the original RGB, your choice.

## Tests

```bash
.venv/bin/python tests/test_runner.py   # runner + model + LocalExecutor
.venv/bin/python tests/test_slurm.py    # SLURM executor (stubbed sbatch/srun/sacct)
.venv/bin/python tests/test_ray.py      # Ray executor + in-process stage (real local Ray)
uv run ruff check .
```

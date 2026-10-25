import importlib
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(os.getenv("COSMOS_ROOT", os.getcwd()))  # defaults to the docker container root

# Stage name -> (module path, class name).
#
# Stored as strings and resolved lazily (see ``load_stage``) so importing the
# entrypoint does *not* import every stage's dependencies. This is what allows
# each stage to run under its own Python interpreter / venv -- the executors
# launch each unit with the stage's interpreter -- instead of forcing every
# dependency into one environment.
#
# NOTE: add a new entry here after implementing each stage.
STAGE_MAPPING: dict[str, tuple[str, str]] = {
    "prompt": ("stages.prompt", "PromptStage"),
    "seg_mask": ("stages.seg_mask", "SegMaskStage"),
    "edge": ("stages.edge", "EdgeDetection"),
    "inference": ("stages.inference", "InferenceStage"),
    "eval_lpips": ("stages.eval_lpips", "LPIPSMetric"),
    "eval_clipqa": ("stages.eval_clipqa", "CLIPIQAMetric"),
    "eval_ssim": ("stages.eval_ssim", "SSIMMetric"),
    "eval_iou": ("stages.eval_iou", "IoUMetric"),
    "eval_iou_temporal": ("stages.eval_iou_temporal", "IoUTemporalMetric"),
}


def load_stage(name: str):
    """Lazily import and return the ``Stage`` class registered under ``name``."""
    try:
        module_name, class_name = STAGE_MAPPING[name]
    except KeyError as exc:
        known = ", ".join(sorted(STAGE_MAPPING))
        raise KeyError(f"Unknown stage {name!r}. Known stages: {known}") from exc
    return getattr(importlib.import_module(module_name), class_name)


def _parse_metrics(metrics) -> list[str]:
    if isinstance(metrics, str):
        return [m.strip() for m in metrics.split(",") if m.strip()]
    return [str(m) for m in metrics if str(m).strip()]


def _run_pipeline_mode(name: str, cfg: DictConfig) -> None:
    """Load the named pipeline and run it (steps come from the pipeline YAML;
    run-level knobs come from the global config / CLI overrides)."""
    from runner import load_pipeline, run_pipeline

    pipelines_dir = PROJECT_ROOT / "configs" / "pipelines"
    pipeline_path = pipelines_dir / f"{name}.yaml"
    if not pipeline_path.exists():
        available = sorted(p.stem for p in pipelines_dir.glob("*.yaml"))
        raise FileNotFoundError(f"Pipeline {name!r} not found. Available: {available}")

    pipeline = load_pipeline(pipeline_path)

    # Run-level knobs from the global config (CLI-overridable) take precedence.
    if cfg.get("executor"):
        pipeline.executor = str(cfg.executor)
    # refactor this - metrics are specific to Cosmos-transfer
    metrics = cfg.get("metrics")
    if metrics is not None:
        pipeline.metrics = _parse_metrics(metrics)
    if cfg.get("batch"):
        batch = OmegaConf.to_container(cfg.batch, resolve=True) or {}
        batch = {k: v for k, v in batch.items() if v is not None}
        if batch:
            pipeline.batch = batch
    if cfg.get("interpreter"):
        pipeline.default_interpreter = str(cfg.interpreter)

    run_pipeline(pipeline)


@hydra.main(
    config_path=str((PROJECT_ROOT / "configs").resolve()),
    config_name="config",
)
def main(cfg: DictConfig):
    pipeline_name = OmegaConf.select(cfg, "pipeline")
    if pipeline_name:
        _run_pipeline_mode(str(pipeline_name), cfg)
        return
    stage_cls = load_stage(cfg.stage.name)
    stage_cls(cfg).run()


if __name__ == "__main__":
    main()

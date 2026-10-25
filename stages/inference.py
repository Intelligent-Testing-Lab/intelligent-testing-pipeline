import json
import sys
from pathlib import Path

from omegaconf import DictConfig

from stage import Stage


class InferenceStage(Stage):
    """Guided (control-conditioned) generation with a RESIDENT model.

    The cosmos model is loaded once (``Control2WorldInference.__init__``) and
    then ``.generate`` is called per batch item into its own output directory,
    so the model is kept in memory instead of being reloaded for every item.

    This stage runs in an interpreter that has both the framework deps
    (hydra/omegaconf) and the cosmos dependencies (torch + `cosmos_transfer2`
    importable from ``cosmos_repo``).
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, stage="inference")

    def _batch(self) -> list[str]:
        """Resolve the batch of output dirs from config (manifest / dirs / glob).

        Prefers ``manifest`` (the runner's per-shard file), then ``dirs`` (a list or
        comma-separated string), then ``glob``. Falls back to the single
        ``run.output_dir`` when none is set (single-stage mode).
        """
        cfg = self.config
        batch = cfg.get("batch")
        if batch is not None:
            manifest = batch.get("manifest")
            if manifest is not None:
                dirs = [
                    ln.strip()
                    for ln in Path(str(manifest)).read_text().splitlines()
                    if ln.strip()
                ]
                if dirs:
                    return dirs
            dirs_val = batch.get("dirs")
            if dirs_val is not None:
                if isinstance(dirs_val, str):
                    parts = [d.strip() for d in dirs_val.split(",") if d.strip()]
                else:
                    parts = [str(x) for x in dirs_val]
                if parts:
                    return parts
            glob_pat = batch.get("glob")
            if glob_pat is not None:
                import glob as globlib

                found = sorted(globlib.glob(str(glob_pat)))
                if found:
                    return found
        return [str(cfg.run.output_dir)]

    def _generation_config(self, outdir: str) -> dict:
        """Build one item's guided-generation config (the cosmos input format)."""
        cfg = self.config.stage
        return {
            "name": "guided_generation",
            "prompt_path": f"{outdir}/prompt.txt",
            "video_path": f"{outdir}/sensor_recordings/front_rgb/front_rgb.mp4",
            "guidance": int(cfg.guidance),
            "edge": {
                "control_path": f"{outdir}/edge.mp4",
                "control_weight": float(cfg.edge.control_weight),
            },
            "seg": {
                "control_path": (
                    f"{outdir}/sensor_recordings/front_semantic_cityscapes/front_semantic_cityscapes.mp4"
                ),
                "mask_path": f"{outdir}/seg_mask.mp4",
                "control_weight": float(cfg.seg.control_weight),
            },
            "depth": {
                "control_path": f"{outdir}/sensor_recordings/front_depth_log/front_depth_log.mp4",
                "control_weight": float(cfg.depth.control_weight),
            },
        }

    def _run(self) -> None:
        cfg = self.config.stage
        items = self._batch()

        # 1. Assemble each item's generation_config.json.
        cfg_paths = []
        for outdir in items:
            cp = Path(outdir) / "generation_config.json"
            cp.write_text(json.dumps(self._generation_config(outdir), indent=2))
            cfg_paths.append(cp)
        self.logger.info("Assembled %d generation config(s): %s", len(items), ", ".join(items))

        # 2. Make the cosmos repo importable, then import the cosmos API.
        cosmos_repo = str(cfg.cosmos_repo)
        if cosmos_repo not in sys.path:
            sys.path.insert(0, cosmos_repo)
        from cosmos_transfer2.config import InferenceArguments, SetupArguments
        from cosmos_transfer2.inference import Control2WorldInference

        # 3. Derive the (identical) control hint keys from a single item and load
        #    the model ONCE. We deliberately do NOT load the whole batch at once:
        #    every item is named "guided_generation", and cosmos' from_files()
        #    enforces unique sample names (it sys.exit()s on a duplicate). Each
        #    item is therefore loaded individually in the loop below, so that check
        #    never sees two at once, while the resident model is still reused.
        _samples0, batch_hint_keys = InferenceArguments.from_files([cfg_paths[0]])
        setup = SetupArguments(
            output_dir=Path(items[0]),
            model=cfg.model,
            disable_guardrails=bool(cfg.disable_guardrails),
            context_parallel_size=cfg.context_parallel_size,
        )
        self.logger.info(
            "Loading cosmos model (hint keys: %s); %d sample(s)", batch_hint_keys, len(items)
        )
        inference = Control2WorldInference(setup, batch_hint_keys)

        # 4. Generate each item into its own output dir, reusing the resident model.
        for cp, outdir in zip(cfg_paths, items):
            samples, _ = InferenceArguments.from_files([cp])
            self.logger.info("Generating sample '%s' -> %s", samples[0].name, outdir)
            inference.generate(samples, output_dir=Path(outdir))

    def verify_completion(self) -> bool:
        return all((Path(d) / "guided_generation.mp4").exists() for d in self._batch())

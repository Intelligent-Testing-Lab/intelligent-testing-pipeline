from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from stage import Stage
from utils import save_mp4


def _fill_mask(mask: np.ndarray, truth_label: tuple, false_label: tuple) -> np.ndarray:
    """Convert a boolean ``(H, W)`` mask into a uint8 ``(H, W, 3)`` RGB frame."""
    frame = np.zeros((*mask.shape, 3), dtype=np.uint8)
    frame[mask] = truth_label
    frame[~mask] = false_label
    return frame


class SegMaskStage(Stage):
    """Build the segmentation control mask and the edge-detection ROI mask.

    From the raw CARLA semantic IDs (``raw_frames.npy``):
      * **seg mask** is white where the pixel is NOT a scenario/object class
        (i.e. background), and
      * **edge ROI** is white where the pixel is one of the edge classes.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, stage="seg_mask")

    def _run(self) -> None:
        cfg = self.config.stage
        run = self.config.run

        raw_data = np.load(str(run.raw_semantic))  # (N, H, W) semantic class IDs
        object_ids = list(cfg.object_class_ids)
        edge_ids = list(cfg.edge_class_ids)
        truth = tuple(cfg.truth_label)
        false = tuple(cfg.false_label)

        seg_frames = []
        edge_frames = []
        for frame in raw_data:
            # Seg mask: white where the pixel is NOT a scenario/object class.
            if object_ids:
                object_masks = np.stack([frame == c for c in object_ids])
                seg_mask = ~np.any(object_masks, axis=0)
            else:
                seg_mask = np.zeros(frame.shape, dtype=bool)
            seg_frames.append(_fill_mask(seg_mask, truth, false))

            # Edge ROI: white where the pixel IS one of the edge classes.
            if edge_ids:
                edge_masks = np.stack([frame == c for c in edge_ids])
                roi_mask = np.any(edge_masks, axis=0)
            else:
                roi_mask = np.zeros(frame.shape, dtype=bool)
            edge_frames.append(_fill_mask(roi_mask, truth, false))

        save_mp4(np.asarray(seg_frames, dtype=np.uint8), str(run.seg_mask_output), fps=cfg.fps)
        save_mp4(np.asarray(edge_frames, dtype=np.uint8), str(run.edge_mask_output), fps=cfg.fps)
        self.logger.info("Wrote seg mask -> %s and edge ROI mask -> %s", run.seg_mask_output, run.edge_mask_output)

    def verify_completion(self) -> bool:
        return (
            Path(self.config.run.seg_mask_output).exists()
            and Path(self.config.run.edge_mask_output).exists()
        )

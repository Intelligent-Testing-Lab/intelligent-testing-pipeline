from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from stage import Stage
from utils import get_frame_buffer, save_mp4


def _canny_edge(frame: np.ndarray, p: dict) -> np.ndarray:
    """Canny edge of a single BGR frame -> grayscale (H, W) of 0/255."""
    if p.get("adjust_image_features", True):
        frame = cv2.convertScaleAbs(frame, alpha=p["contrast"], beta=p["brightness"])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if p.get("blur", True):
        k = int(p["gaussian_kernel"])
        gray = cv2.GaussianBlur(gray, (k, k), float(p["gaussian_sigma"]))
    return cv2.Canny(
        gray,
        int(p["min_threshold"]),
        int(p["max_threshold"]),
        apertureSize=int(p["kernel_size"]),
        L2gradient=bool(p["gradient_func"]),
    )


def _semantic_edges(labels: np.ndarray) -> np.ndarray:
    """Boolean edge mask from label transitions.

    One pixel owns each transition: vertical transition ``(y, x)/(y+1, x)`` is
    assigned to the upper pixel; horizontal ``(y, x)/(y, x+1)`` to the left pixel.
    """
    vertical_change = labels[:, :-1, :] != labels[:, 1:, :]
    horizontal_change = labels[:, :, :-1] != labels[:, :, 1:]
    edge_mask = np.zeros_like(labels, dtype=bool)
    edge_mask[:, :-1, :] |= vertical_change
    edge_mask[:, :, :-1] |= horizontal_change
    return edge_mask


def _is_white(frame: np.ndarray, tolerance: int) -> np.ndarray:
    """Boolean mask of the white pixels of an (H, W, 3) (or N, H, W, 3) frame."""
    return np.all(np.abs(frame - (255, 255, 255)) < tolerance, axis=-1)


class EdgeDetection(Stage):
    """Combine ground-truth semantic edges with canny edges (gated by the ROI).

    Final edge = (canny edges AND inside the semantic ROI) OR the ground-truth
    semantic edges. Written as a white-on-black edge video.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, stage="edge")

    def _run(self) -> None:
        cfg = self.config.stage
        run = self.config.run
        canny_cfg = dict(cfg.canny)
        tolerance = int(cfg.roi_tolerance)

        labels = np.load(str(run.raw_semantic))  # (N, H, W)
        gt_edge = _semantic_edges(labels)

        rgb = get_frame_buffer(str(run.rgb_input))  # (N, H, W, 3)
        canny_rgb = np.stack(
            [cv2.cvtColor(_canny_edge(f, canny_cfg), cv2.COLOR_GRAY2BGR) for f in rgb],
            axis=0,
        )
        canny_mask = _is_white(canny_rgb, tolerance)

        roi = get_frame_buffer(str(run.edge_mask_output))  # (N, H, W, 3)
        roi_mask = _is_white(roi, tolerance)

        detected = (canny_mask & roi_mask) | gt_edge

        final_output = np.zeros((*detected.shape, 3), dtype=np.uint8)
        final_output[detected] = (255, 255, 255)
        final_output[~detected] = (0, 0, 0)
        save_mp4(final_output, str(run.edge_output), fps=cfg.fps)
        self.logger.info("Wrote edge video -> %s", run.edge_output)

    def verify_completion(self) -> bool:
        return Path(self.config.run.edge_output).exists()

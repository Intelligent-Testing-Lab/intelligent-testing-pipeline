"""Shared helpers + base class for the evaluation-metric stages.

All heavy dependencies (torch / pyiqa / skimage / ultralytics) are imported
lazily inside the concrete stages, so this module (and the framework) stay
importable in the lightweight framework venv. Each metric stage appends its
value to the *shared* results file so multiple stages can write to one file.
"""
import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig  # noqa: F401  (kept for typing parity with other stages)

from stage import Stage

# Cityscapes train-ID classes 0-18 (shared by the IoU metrics).
CITYSCAPES_CLASSES = [
    {"id": 0, "name": "road", "color": (128, 64, 128)},
    {"id": 1, "name": "sidewalk", "color": (244, 35, 232)},
    {"id": 2, "name": "building", "color": (70, 70, 70)},
    {"id": 3, "name": "wall", "color": (102, 102, 156)},
    {"id": 4, "name": "fence", "color": (190, 153, 153)},
    {"id": 5, "name": "pole", "color": (153, 153, 153)},
    {"id": 6, "name": "traffic light", "color": (250, 170, 30)},
    {"id": 7, "name": "traffic sign", "color": (220, 220, 0)},
    {"id": 8, "name": "vegetation", "color": (107, 142, 35)},
    {"id": 9, "name": "terrain", "color": (152, 251, 152)},
    {"id": 10, "name": "sky", "color": (70, 130, 180)},
    {"id": 11, "name": "person", "color": (220, 20, 60)},
    {"id": 12, "name": "rider", "color": (255, 0, 0)},
    {"id": 13, "name": "car", "color": (0, 0, 142)},
    {"id": 14, "name": "truck", "color": (0, 0, 70)},
    {"id": 15, "name": "bus", "color": (0, 60, 100)},
    {"id": 16, "name": "train", "color": (0, 80, 100)},
    {"id": 17, "name": "motorcycle", "color": (0, 0, 230)},
    {"id": 18, "name": "bicycle", "color": (119, 11, 32)},
]

IGNORE_LABEL = 255


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save_json(path: str, content: dict) -> None:
    Path(path).write_text(json.dumps(content, indent=4))


def calculate_outliers(metric_buffer: np.ndarray, multiplier: float = 1.5) -> dict:
    q1, q3 = np.percentile(metric_buffer, [25, 75])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = (metric_buffer < lower) | (metric_buffer > upper)
    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "outliers": [float(x) for x in metric_buffer[mask]],
    }


def calculate_metric_stats(metric_buffer: np.ndarray) -> dict:
    metric_buffer = np.asarray(metric_buffer, dtype=np.float64)
    return {
        "mean": float(np.mean(metric_buffer)),
        "median": float(np.median(metric_buffer)),
        "std": float(np.std(metric_buffer)),
        "min": float(np.min(metric_buffer)),
        "max": float(np.max(metric_buffer)),
        "outliers": calculate_outliers(metric_buffer),
        "values": [float(v) for v in metric_buffer.flatten()],
    }


def compute_iou_per_class(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    num_classes: int,
    ignore_label: int = IGNORE_LABEL,
    both_valid: bool = False,
) -> np.ndarray:
    """Per-class IoU between GT and prediction.

    By default a pixel is valid when ``gt_mask != ignore_label`` (matches the
    original single-frame IoU). With ``both_valid=True`` a pixel is valid only
    when BOTH masks are non-ignored (matches the original temporal-IoU validity).
    """
    ious = np.zeros(num_classes, dtype=np.float32)
    if both_valid:
        valid = (gt_mask != ignore_label) & (pred_mask != ignore_label)
    else:
        valid = gt_mask != ignore_label
    for cid in range(num_classes):
        gt = (gt_mask == cid) & valid
        pred = (pred_mask == cid) & valid
        intersection = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        ious[cid] = np.nan if union == 0 else intersection / union
    return ious


def colorize_mask(pred_mask: np.ndarray) -> np.ndarray:
    """Convert an (H, W) class-id mask to an (H, W, 3) BGR image."""
    h, w = pred_mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in CITYSCAPES_CLASSES:
        r, g, b = cls["color"]
        color_img[pred_mask == cls["id"]] = (b, g, r)  # RGB -> BGR
    return color_img


class EvalStage(Stage):
    """Base for an evaluation-metric stage.

    Subclasses set ``metric_name`` and implement ``_metric_value`` (the
    JSON-serializable value stored under that key). ``run`` loads the shared
    results file, adds the metric, and writes it back -- so multiple metric
    stages can append to the same file.
    """

    metric_name: str = "metric"

    def _metric_value(self):
        raise NotImplementedError

    def _run(self) -> None:
        value = self._metric_value()
        results_path = str(self.config.run.results_file)
        results = load_json(results_path)
        results[self.metric_name] = value
        save_json(results_path, results)
        self.logger.info("Wrote metric '%s' to %s", self.metric_name, results_path)

    def verify_completion(self) -> bool:
        return self.metric_name in load_json(str(self.config.run.results_file))

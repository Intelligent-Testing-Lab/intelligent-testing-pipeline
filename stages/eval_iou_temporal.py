from collections import defaultdict
from pathlib import Path  # noqa: F401  (kept for parity with eval_iou)

import cv2
import numpy as np

from stages.eval_base import (
    CITYSCAPES_CLASSES,
    IGNORE_LABEL,
    EvalStage,
    colorize_mask,
    compute_iou_per_class,
)
from utils import get_frame_buffer


def warp_mask_with_flow(mask: np.ndarray, flow: np.ndarray, ignore_label: int = IGNORE_LABEL) -> np.ndarray:
    """Warp a class-ID mask forward using dense optical flow."""
    grid_x, grid_y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        mask.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(ignore_label),
    ).astype(np.uint8)


class IoUTemporalMetric(EvalStage):
    """Temporal (optical-flow) per-class IoU of a video.

    Segments every frame, then measures per-class IoU between the optical-flow
    warped mask at ``t`` and the mask at ``t+1`` (temporal consistency).
    """

    metric_name = "iou_optical_flow"

    def _metric_value(self):
        from ultralytics import YOLO

        run = self.config.run
        model = YOLO(str(self.config.stage.yolo_weights))
        cosmos_rgb = get_frame_buffer(str(run.iou_temporal_rgb))
        num_frames, h, w, _ = cosmos_rgb.shape

        cap = cv2.VideoCapture(str(run.iou_temporal_rgb))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Semantic-segment every frame (and write the frames to the main output).
        out_main = cv2.VideoWriter(str(run.iou_temporal_flow_video), fourcc, fps, (w, h))
        sem_masks = []
        for t in range(num_frames):
            sem_masks.append(model.predict(cosmos_rgb[t], verbose=False)[0][0].semantic_mask.data.cpu().numpy())
            out_main.write(cosmos_rgb[t])
        out_main.release()

        # Optical-flow IoU between consecutive frames.
        flow_overlay = str(run.iou_temporal_flow_video).replace(".mp4", "_flow_overlay.mp4")
        out_flow = cv2.VideoWriter(flow_overlay, fourcc, fps, (w, h))
        per_class = defaultdict(list)
        for t in range(num_frames - 1):
            prev_gray = cv2.cvtColor(cosmos_rgb[t], cv2.COLOR_BGR2GRAY)
            next_gray = cv2.cvtColor(cosmos_rgb[t + 1], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                next_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            warped = warp_mask_with_flow(sem_masks[t], flow)
            ious_t = compute_iou_per_class(
                warped, sem_masks[t + 1], len(CITYSCAPES_CLASSES), IGNORE_LABEL, both_valid=True
            )
            for cid, iou_c in enumerate(ious_t):
                per_class[cid].append(iou_c)

            overlay = cv2.addWeighted(cosmos_rgb[t + 1], 0.5, colorize_mask(sem_masks[t + 1]), 0.3, 0)
            overlay = cv2.addWeighted(overlay, 0.8, colorize_mask(warped), 0.2, 0)
            out_flow.write(overlay)
        out_flow.release()

        result_dict = {
            "per_class_iou": [],
            "npy": str(run.iou_temporal_flow_raw),
            "target": {"input": str(run.iou_temporal_rgb), "output": str(run.iou_temporal_flow_video)},
        }
        for cls in CITYSCAPES_CLASSES:
            vals = np.array(per_class[cls["id"]], dtype=np.float32)
            valid = vals[~np.isnan(vals)]
            if valid.size == 0:
                continue
            vals[np.isnan(vals)] = -1
            result_dict["per_class_iou"].append({
                "class_id": cls["id"],
                "name": cls["name"],
                "mean_iou": float(valid.mean()),
                "std_iou": float(valid.std()),
                "min_iou": float(valid.min()),
                "max_iou": float(valid.max()),
                "frames": int(valid.size),
                "values": [float(v) for v in vals],
            })

        Path(str(run.iou_temporal_flow_raw)).parent.mkdir(parents=True, exist_ok=True)
        np.save(str(run.iou_temporal_flow_raw), {cid: np.array(vals) for cid, vals in per_class.items()})
        return result_dict

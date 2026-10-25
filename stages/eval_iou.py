from collections import defaultdict
from pathlib import Path

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


class IoUMetric(EvalStage):
    """Per-class semantic IoU between the generated and the original video.

    Runs YOLO26 semantic segmentation on both videos and compares the class
    maps (GT = original, prediction = generated), aggregated over frames.
    """

    metric_name = "iou"

    def _metric_value(self):
        from ultralytics import YOLO

        run = self.config.run
        model = YOLO(str(self.config.stage.yolo_weights))

        cosmos_rgb = get_frame_buffer(str(run.iou_cosmos))
        raw_rgb = get_frame_buffer(str(run.iou_raw))
        nr, hr, wr, _ = cosmos_rgb.shape
        ng, hg, wg, _ = raw_rgb.shape
        num_frames = min(nr, ng)
        if (hr, wr) != (hg, wg):
            raise ValueError("Cosmos RGB and raw RGB resolutions do not match")

        cap = cv2.VideoCapture(str(run.iou_cosmos))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_cosmos = cv2.VideoWriter(str(run.iou_cosmos_seg), fourcc, fps, (wr, hr))
        out_raw = cv2.VideoWriter(str(run.iou_raw_seg), fourcc, fps, (wr, hr))

        per_class = defaultdict(list)
        for t in range(num_frames):
            frame_cosmos = cosmos_rgb[t]
            frame_raw = raw_rgb[t]
            res = [model.predict(frame_cosmos, verbose=False), model.predict(frame_raw, verbose=False)]
            pred_mask = res[0][0].semantic_mask.data.cpu().numpy()
            gt_mask = res[1][0].semantic_mask.data.cpu().numpy()

            ious_t = compute_iou_per_class(gt_mask, pred_mask, len(CITYSCAPES_CLASSES), IGNORE_LABEL)
            for cid, iou_c in enumerate(ious_t):
                per_class[cid].append(iou_c)

            out_cosmos.write(cv2.addWeighted(frame_cosmos, 0.5, colorize_mask(pred_mask), 0.5, 0))
            out_raw.write(cv2.addWeighted(frame_raw, 0.5, colorize_mask(gt_mask), 0.5, 0))
        out_raw.release()
        out_cosmos.release()

        result_dict = {
            "per_class_iou": [],
            "npy": str(run.iou_raw_values),
            "target": {"input": str(run.iou_raw), "output": str(run.iou_raw_seg)},
            "prediction": {"input": str(run.iou_cosmos), "output": str(run.iou_cosmos_seg)},
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

        Path(str(run.iou_raw_values)).parent.mkdir(parents=True, exist_ok=True)
        np.save(str(run.iou_raw_values), {cid: np.array(vals) for cid, vals in per_class.items()})
        return result_dict

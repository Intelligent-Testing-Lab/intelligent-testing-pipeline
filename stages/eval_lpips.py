import numpy as np

from stages.eval_base import EvalStage, calculate_metric_stats
from utils import get_frame_buffer


class LPIPSMetric(EvalStage):
    """Learned Perceptual Image Patch Similarity between two videos (per frame)."""

    metric_name = "lpips"

    def _metric_value(self):
        import torch
        import pyiqa

        run = self.config.run
        chunk = int(self.config.stage.chunk_size)
        pred = get_frame_buffer(str(run.lpips_pred))
        true_ = get_frame_buffer(str(run.lpips_true))
        if pred.shape[0] != true_.shape[0]:
            raise ValueError("pred and true videos must have the same number of frames")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        metric = pyiqa.create_metric("lpips", device=device, as_loss=False)

        def to_tensor(frames: np.ndarray) -> "torch.Tensor":
            x = (frames / 255.0).astype("float32")
            return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(device)

        out = []
        for i in range(0, pred.shape[0], chunk):
            out.append(
                metric(to_tensor(true_[i:i + chunk]), to_tensor(pred[i:i + chunk])).cpu().numpy()
            )
        return calculate_metric_stats(np.concatenate(out))

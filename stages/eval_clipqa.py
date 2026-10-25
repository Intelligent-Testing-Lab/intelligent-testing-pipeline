import numpy as np

from stages.eval_base import EvalStage, calculate_metric_stats
from utils import get_frame_buffer


class CLIPIQAMetric(EvalStage):
    """CLIP-IQA+ no-reference image quality of a single video (per frame).

    The input video is whatever ``run.clipqa_video`` points at -- the user
    decides whether that is the generated video or the original RGB.
    """

    metric_name = "clipqa"

    def _metric_value(self):
        import torch
        import pyiqa

        run = self.config.run
        chunk = int(self.config.stage.chunk_size)
        video = get_frame_buffer(str(run.clipqa_video))

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        metric = pyiqa.create_metric("clipiqa+", device=device, as_loss=False)

        def to_tensor(frames: np.ndarray) -> "torch.Tensor":
            x = (frames / 255.0).astype("float32")
            return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(device)

        out = []
        for i in range(0, video.shape[0], chunk):
            out.append(metric(to_tensor(video[i:i + chunk])).cpu().numpy())
        return calculate_metric_stats(np.concatenate(out))

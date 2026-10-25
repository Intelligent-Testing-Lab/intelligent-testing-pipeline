import numpy as np

from stages.eval_base import EvalStage, calculate_metric_stats
from utils import get_frame_buffer


class SSIMMetric(EvalStage):
    """Structural Similarity Index between two videos (per frame)."""

    metric_name = "ssim"

    def _metric_value(self):
        from concurrent.futures import ThreadPoolExecutor

        from skimage.metrics import structural_similarity as ssim

        run = self.config.run
        num_threads = int(self.config.stage.num_threads)
        pred = get_frame_buffer(str(run.ssim_pred))
        true_ = get_frame_buffer(str(run.ssim_true))
        if pred.shape[0] != true_.shape[0]:
            raise ValueError("pred and true videos must have the same number of frames")

        def chunk_ssim(true_chunk: np.ndarray, pred_chunk: np.ndarray) -> np.ndarray:
            return np.asarray([
                ssim(t, p, data_range=255, channel_axis=-1) for t, p in zip(true_chunk, pred_chunk)
            ])

        pred_chunks = np.array_split(pred, num_threads, axis=0)
        true_chunks = np.array_split(true_, num_threads, axis=0)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(chunk_ssim, true_chunks[i], pred_chunks[i])
                for i in range(num_threads)
            ]
            ssim_buffer = np.concatenate([f.result() for f in futures])
        return calculate_metric_stats(ssim_buffer)

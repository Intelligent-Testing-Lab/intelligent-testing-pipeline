import cv2
import numpy as np

def get_frame_buffer(target: str, frames: int = -1) -> np.ndarray:
    """Read a video into a ``(N, H, W, 3)`` uint8 BGR frame buffer.

    ``frames`` selects how many frames to read:
        -1   -> all frames (default)
        > 0  -> the first ``frames`` frames (clamped to the video's frame count)
        < -1 -> ``frame_count + frames`` frames (i.e. all but the last ``abs(frames)``)
        0    -> raises ``ValueError``

    Raises:
        ValueError: if the video cannot be opened or ``frames`` is invalid.
    """

    cap = cv2.VideoCapture(target)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: is the path correct? {target!r}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frames == -1:
            n = frame_count
        elif frames < -1:
            n = max(1, frame_count + frames)
        elif frames == 0:
            raise ValueError("Cannot read zero frames from a video")
        else:
            n = min(frames, frame_count)

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        frame_buffer = np.empty((n, h, w, 3), dtype=np.uint8)

        index = 0
        while index < n:
            ret, frame = cap.read()
            if not ret:
                break
            frame_buffer[index] = frame
            index += 1

        return frame_buffer[:index]
    finally:
        cap.release()


def save_mp4(
    frames: np.ndarray,
    path: str,
    fps: float = 20.0,
) -> None:
    """Save uint8 BGR frames of shape (N, H, W, 3) as an MP4."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (N, H, W, 3)")

    frames = frames.astype(np.uint8, copy=False)
    _, height, width, _ = frames.shape

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),  # OpenCV expects (W, H), not (H, W)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")

    for frame in frames:
        writer.write(frame)

    writer.release()

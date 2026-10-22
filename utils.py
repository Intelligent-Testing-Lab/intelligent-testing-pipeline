import numpy as np
import cv2
import os
from pathlib import Path

def get_frame_buffer(target: str, frames: int = -1) -> np.ndarray:
    """
    Returns a frames length buffer. If set to -1 its the total length of the video.
    If frames is more than total framecount, return framecount
    """

    cap = cv2.VideoCapture(target)

    if not cap.isOpened():
        print("Failed to open video: is the path correct?")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames > frame_count or frames == -1:
        frames = frame_count
    elif frames < -1:
        frames = frame_count + frames
    elif frames == 0:
        print("Cannot get zero frames from video")

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_buffer = np.ndarray((frames, h,w, 3), dtype=np.uint8)

    frame_index = 0
    while True:
        ret, frame = cap.read()

        frame_buffer[frame_index] = frame

        # exit loop if video finished or target frames reached
        if not ret or frame_index == (frame_count - 1):
            break
        else:
            frame_index += 1

    cap.release()
    return frame_buffer


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

def validate_path(path: str, is_file: bool = True, create_dir: bool = False) -> Path:
    """
    Takes in a path string and validates if it exists and is the correct type (directory or file - symlinks not supported).
    If the supplied path is a directory that doesn't exist, it will be automatically created.

    Returns pathlib Path object
    """

    path_obj = Path(path)

    if not path_obj.exists():
        if is_file:
            raise FileNotFoundError(f"File {path_obj} doesn't exist")
        elif create_dir and not is_file:
            path_obj.mkdir(parents=True, exist_ok=True)

    return path_obj

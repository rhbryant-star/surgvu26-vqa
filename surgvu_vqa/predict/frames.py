"""Evenly-spaced frame sampling from a video clip (cv2 -> PIL RGB)."""
from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image

# 8 frames balances temporal coverage of a 30-s clip against T4 VRAM/time
# (each frame costs 256-512 visual tokens; see predict/model.py pixel budget).
DEFAULT_NUM_FRAMES = 8


def sample_frames(video_path: Path, num_frames: int = DEFAULT_NUM_FRAMES) -> list[Image.Image]:
    """Return up to `num_frames` RGB frames evenly spaced across the clip.

    Shorter clips return every frame. Raises ValueError when the file cannot
    be opened or no frame decodes.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"video reports no frames: {video_path}")
        count = min(num_frames, total)
        if count == 1:
            indices = [0]
        else:
            indices = [round(i * (total - 1) / (count - 1)) for i in range(count)]
        frames: list[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if not frames:
            raise ValueError(f"could not decode any frames: {video_path}")
        return frames
    finally:
        cap.release()

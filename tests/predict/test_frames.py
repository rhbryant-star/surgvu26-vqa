from __future__ import annotations

import cv2
import numpy as np
import pytest

from surgvu_vqa.predict.frames import sample_frames

# Distinct, codec-survivable solid colors (BGR), widely separated values.
_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]


def _write_video(path, n_frames, size=(64, 64)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    if not writer.isOpened():  # codec fallback (see plan risk 5)
        path = path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), 10.0, size)
        assert writer.isOpened(), "no usable cv2 video codec on this machine"
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), _COLORS[i % len(_COLORS)], dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_returns_requested_count(tmp_path):
    video = _write_video(tmp_path / "clip.mp4", n_frames=30)
    frames = sample_frames(video, num_frames=8)
    assert len(frames) == 8
    assert frames[0].mode == "RGB"


def test_spans_whole_clip(tmp_path):
    video = _write_video(tmp_path / "clip.mp4", n_frames=30)
    frames = sample_frames(video, num_frames=8)
    # First sampled frame ≈ color 0 (BGR 255,0,0 → RGB 0,0,255-ish), last ≈ color of frame 29 (29%5=4 → BGR 0,255,255 → RGB ~255,255,0)
    first_px = frames[0].getpixel((32, 32))
    last_px = frames[-1].getpixel((32, 32))
    assert first_px[2] > 200 and first_px[0] < 60      # blue-dominant start
    assert last_px[0] > 200 and last_px[2] < 60        # red+green (yellow) end
    # All frames are not identical (sampling actually moved through the clip)
    assert len({f.tobytes() for f in frames}) > 1


def test_short_video_returns_all_frames(tmp_path):
    video = _write_video(tmp_path / "short.mp4", n_frames=3)
    frames = sample_frames(video, num_frames=8)
    assert len(frames) == 3


def test_unreadable_video_raises(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"this is not an mp4")
    with pytest.raises(ValueError):
        sample_frames(bogus, num_frames=8)

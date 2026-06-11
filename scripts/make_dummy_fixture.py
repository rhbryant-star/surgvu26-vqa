#!/usr/bin/env python3
"""Generate a tiny SYNTHETIC clip as the committed container test fixture.

The official template's fixture mp4 is a 30-byte ASCII placeholder, not a
video — cv2 cannot decode it. A real (but synthetic, non-challenge) clip lets
the fake-model smoke tests exercise actual frame decoding inside the image.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT = (
    Path(__file__).resolve().parents[1]
    / "container" / "test" / "input" / "interf0"
    / "endoscopic-robotic-surgery-video.mp4"
)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    assert writer.isOpened(), "no mp4v codec available"
    for i in range(30):
        writer.write(np.full((64, 64, 3), COLORS[i % len(COLORS)], dtype=np.uint8))
    writer.release()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

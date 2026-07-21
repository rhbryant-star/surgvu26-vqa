from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

from scripts.build_case_dataset import build_case

TOOLS_CSV = """index,install_case_part,install_case_time,uninstall_case_part,uninstall_case_time,arm,commercial_toolname,groundtruth_toolname
0,1.0,00:00:00.000000,1.0,00:02:00.000000,USM1,Large Needle Driver,needle driver
"""
TASKS_CSV = """index,start_part,start_time,stop_part,stop_time,groundtruth_taskname
0,1,0.0,1,120.0,Suturing
"""


def _make_video(path: Path, seconds: int = 120, fps: int = 10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    assert w.isOpened()
    for i in range(seconds * fps):
        w.write(np.full((64, 64, 3), (i % 200, 50, 100), dtype=np.uint8))
    w.release()


def test_build_case_end_to_end(tmp_path):
    # Fake the remote zip with a local one (same inner layout).
    vid = tmp_path / "v.mp4"
    _make_video(vid)
    zpath = tmp_path / "videos.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.write(vid, "surgvu24/case_001/case_001_video_part_001.mp4")
    case_labels = tmp_path / "labels" / "case_001"
    case_labels.mkdir(parents=True)
    (case_labels / "tools.csv").write_text(TOOLS_CSV)
    (case_labels / "tasks.csv").write_text(TASKS_CSV)
    out = tmp_path / "out"

    n_clips = build_case(
        case_id="case_001",
        labels_dir=case_labels,
        out_dir=out,
        zip_source=str(zpath),       # local path -> zipfile; URL -> remotezip
        max_clips=3,
        seed=5,
    )

    assert n_clips >= 1
    qa = [json.loads(l) for l in (out / "case_001_qa.jsonl").read_text().splitlines()]
    assert all(r["images"] and len(r["images"]) == 8 for r in qa)
    # Fixed tautological assertion (revisions item 9):
    assert all(r["question"] and r["answer"] and r["answer"].endswith(".") for r in qa)
    with tarfile.open(out / "case_001_frames.tar") as t:
        names = t.getnames()
    # every referenced frame exists in the tar
    for r in qa:
        for img in r["images"]:
            assert img in names

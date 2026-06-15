#!/usr/bin/env python3
"""Build one case's VQA training shard: frames tar + qa.jsonl.

Streams ONLY this case's needed video parts out of the 344 GB GCS zip via
HTTP Range (remotezip); never materializes the whole archive. Designed to run
inside a CHTC job (videos land on quota-free scratch and are deleted), but
works locally too. zip_source may be an http(s) URL (remotezip) or a local
path (zipfile) — tests use the local path.

POST-CRITIQUE REVISIONS applied:
  Verified fact: video fps is read via cap.get(CAP_PROP_FPS) — never hardcoded.
  Rev 1 (clamping): open-ended tool intervals (end_s=None) are clamped to the
      part's actual video duration (frame_count/fps) before synthesis so the
      dataset never encodes infinite intervals.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import cv2
from PIL import Image

from surgvu_vqa.data.clip_plan import plan_clips
from surgvu_vqa.data.labels import CaseLabels, ToolInterval
from surgvu_vqa.data.qa_synthesis import synthesize_qa

JPEG_QUALITY = 90
FRAMES_PER_CLIP = 8


def _open_zip(zip_source: str):
    """Return an open zip-file-like object for the given source.

    Accepts an http(s) URL (uses remotezip for range-streaming) or a local
    filesystem path (uses the stdlib zipfile). Tests always pass a local path.
    """
    if zip_source.startswith(("http://", "https://")):
        from remotezip import RemoteZip  # noqa: PLC0415
        return RemoteZip(zip_source)
    return zipfile.ZipFile(zip_source)


def _clamp_labels_to_part_duration(
    labels: CaseLabels, part: int, duration_s: float
) -> CaseLabels:
    """Return a new CaseLabels with open-ended tool intervals clamped.

    Tools installed in `part` with end_s=None (spanning parts) are given an
    explicit end_s = duration_s so synthesis never operates on +inf intervals.
    Tools in other parts are unchanged.
    """
    clamped_tools: list[ToolInterval] = []
    for t in labels.tool_intervals:
        if t.part == part and t.end_s is None:
            clamped_tools.append(
                ToolInterval(
                    part=t.part,
                    start_s=t.start_s,
                    end_s=duration_s,
                    groundtruth=t.groundtruth,
                    commercial=t.commercial,
                )
            )
        else:
            clamped_tools.append(t)
    return CaseLabels(
        case_id=labels.case_id,
        tool_intervals=clamped_tools,
        task_intervals=labels.task_intervals,
    )


def _get_video_duration_s(cap: cv2.VideoCapture) -> tuple[float, float]:
    """Return (fps, duration_s) for an already-opened VideoCapture."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0.0
    return fps, duration_s


def _extract_frames(
    video_path: Path,
    start_s: float,
    end_s: float,
) -> list[Image.Image]:
    """Extract FRAMES_PER_CLIP evenly-spaced frames from [start_s, end_s)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video_path}")
    try:
        # Use real fps from the video file — never hardcode.
        fps, _ = _get_video_duration_s(cap)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        first = min(int(start_s * fps), max(total - 1, 0))
        last = min(int(end_s * fps) - 1, total - 1)
        if last <= first:
            return []
        idxs = [
            round(first + i * (last - first) / (FRAMES_PER_CLIP - 1))
            for i in range(FRAMES_PER_CLIP)
        ]
        frames: list[Image.Image] = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        return frames
    finally:
        cap.release()


def build_case(
    case_id: str,
    labels_dir: Path,
    out_dir: Path,
    zip_source: str,
    max_clips: int = 25,
    seed: int = 0,
) -> int:
    """Build one case shard: extract frames, synthesize QA, pack into tar+jsonl.

    Returns the number of usable clips written (0 if no clips were found).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)

    labels = CaseLabels.load(Path(labels_dir))
    clips = plan_clips(labels, max_clips=max_clips, seed=seed)
    if not clips:
        print(f"{case_id}: no usable clips")
        return 0

    needed_parts = sorted({c.part for c in clips})

    # Fetch only the needed video parts via range-streaming (or local zip).
    part_paths: dict[int, Path] = {}
    with _open_zip(zip_source) as z:
        for part in needed_parts:
            member = f"surgvu24/{case_id}/{case_id}_video_part_{part:03d}.mp4"
            print(f"{case_id}: fetching {member}")
            z.extract(member, path=work)
            part_paths[part] = work / member

    # Per-part video duration for clamping open-ended tool intervals (revision 1).
    part_durations: dict[int, float] = {}
    for part, vpath in part_paths.items():
        cap = cv2.VideoCapture(str(vpath))
        if cap.isOpened():
            _, dur = _get_video_duration_s(cap)
            part_durations[part] = dur
            cap.release()

    records: list[dict] = []
    tar_path = out_dir / f"{case_id}_frames.tar"
    with tarfile.open(tar_path, "w") as tar:
        for ci, clip in enumerate(clips):
            vpath = part_paths[clip.part]
            frames = _extract_frames(vpath, clip.start_s, clip.end_s)
            if len(frames) < FRAMES_PER_CLIP:
                continue

            img_names: list[str] = []
            tmp = work / "frame.jpg"
            for fi, img in enumerate(frames):
                name = f"frames/{case_id}/clip{ci:04d}_f{fi}.jpg"
                img.save(str(tmp), "JPEG", quality=JPEG_QUALITY)
                tar.add(str(tmp), arcname=name)
                img_names.append(name)

            # Clamp open-ended tools for this part before synthesis (revision 1).
            dur = part_durations.get(clip.part, float("inf"))
            clamped = _clamp_labels_to_part_duration(labels, clip.part, dur)

            for pair in synthesize_qa(clamped, clip, seed=seed + ci):
                records.append({
                    "case": case_id,
                    "kind": pair.kind,
                    "question": pair.question,
                    "answer": pair.answer,
                    "images": img_names,
                })

    with open(out_dir / f"{case_id}_qa.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Clean the (potentially GB-sized) videos from scratch.
    shutil.rmtree(work, ignore_errors=True)
    print(f"{case_id}: {len(clips)} clips, {len(records)} QA pairs")
    return len(clips)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build one SurgVU case's VQA shard (frames tar + qa.jsonl)."
    )
    p.add_argument("--case-id", required=True, help="e.g. case_001")
    p.add_argument("--labels-dir", type=Path, required=True,
                   help="Path to the case's label directory (contains tools.csv, tasks.csv)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for the shard files")
    p.add_argument(
        "--zip-source",
        default="https://storage.googleapis.com/isi-surgvu/surgvu24_videos_only.zip",
        help="HTTP(S) URL or local path to the videos zip",
    )
    p.add_argument("--max-clips", type=int, default=25,
                   help="Maximum clips to extract per case")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for deterministic clip selection")
    a = p.parse_args()
    build_case(
        case_id=a.case_id,
        labels_dir=a.labels_dir,
        out_dir=a.out_dir,
        zip_source=a.zip_source,
        max_clips=a.max_clips,
        seed=a.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

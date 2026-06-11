#!/usr/bin/env python3
"""Batch predictions over the public sample clips (runs INSIDE the SIF).

Writes predictions.json in the eval-harness format: {clip_id: answer}.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from surgvu_vqa.predict.answer import shape_answer
from surgvu_vqa.predict.frames import sample_frames
from surgvu_vqa.predict.model import QwenVqa

CASE_DIR_PATTERN = re.compile(r"^case\d+$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("predictions.json"))
    args = parser.parse_args()

    cases = sorted(d for d in args.samples_dir.iterdir() if d.is_dir() and CASE_DIR_PATTERN.match(d.name))
    print(f"Found {len(cases)} cases")
    t0 = time.time()
    model = QwenVqa()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    predictions = {}
    raw_outputs = {}
    for case in cases:
        question = json.loads((case / f"{case.name}_question.json").read_text())
        frames = sample_frames(case / f"{case.name}.mp4")
        t1 = time.time()
        raw = model.answer(frames, question)
        answer = shape_answer(raw)
        predictions[case.name] = answer
        raw_outputs[case.name] = raw
        print(f"{case.name} ({time.time() - t1:.1f}s) Q: {question!r} -> A: {answer!r}")

    args.out.write_text(json.dumps(predictions, indent=2))
    # Raw (pre-shaping) outputs let shaping changes be re-scored offline
    # without burning GPU time.
    args.out.with_name(args.out.stem + "_raw.json").write_text(json.dumps(raw_outputs, indent=2))
    print(f"Wrote {len(predictions)} predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

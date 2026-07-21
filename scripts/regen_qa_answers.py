#!/usr/bin/env python3
"""Regenerate per-case qa.jsonl ANSWERS with the echo-free templates, reusing
the existing frame tars (no video, no re-streaming).

plan_clips + synthesize_qa are deterministic given the seed, so the
(kind, question, images) for every surviving clip are identical to the original
build — only the `answer` text changes (M2 v2: drop the temporal clip-echo that
diluted BLEU). The clip index is recovered from each record's image name
(clip{ci:04d}_f{fi}.jpg). For every clip we re-run synthesize_qa and ASSERT the
regenerated (kind, question) sequence matches the staged one before swapping the
answer — so a silent drift aborts instead of corrupting the dataset.

Note: the original build clamped open-ended tool intervals to the part duration;
we skip that here because for any clip that lies within its video part the clamp
is a no-op (clamped_end = part_dur >= clip.end_s gives the same window overlap).
The per-clip question assertion is the backstop that proves equivalence.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

from surgvu_vqa.data.clip_plan import plan_clips
from surgvu_vqa.data.labels import CaseLabels
from surgvu_vqa.data.qa_synthesis import synthesize_qa

_CI = re.compile(r"clip(\d+)_f\d+")


def regen_case(case_id: str, labels_root: Path, staged_qa: Path, out_qa: Path) -> int:
    labels = CaseLabels.load(labels_root / case_id)
    clips = plan_clips(labels, max_clips=25, seed=0)

    existing = [json.loads(line) for line in staged_qa.read_text().splitlines() if line.strip()]
    by_ci: "OrderedDict[int, list[dict]]" = OrderedDict()
    for r in existing:
        m = _CI.search(r["images"][0])
        if not m:
            raise ValueError(f"{case_id}: cannot parse clip index from {r['images'][0]}")
        by_ci.setdefault(int(m.group(1)), []).append(r)

    out: list[dict] = []
    for ci, recs in by_ci.items():
        clip = clips[ci]
        newpairs = list(synthesize_qa(labels, clip, seed=0 + ci))
        if len(newpairs) != len(recs):
            raise AssertionError(
                f"{case_id} clip{ci}: regenerated {len(newpairs)} pairs vs staged {len(recs)}"
            )
        for r, pair in zip(recs, newpairs):
            if pair.question != r["question"] or pair.kind != r["kind"]:
                raise AssertionError(
                    f"{case_id} clip{ci}: drift\n  staged Q: {r['question']!r}\n  regen  Q: {pair.question!r}"
                )
            out.append({**r, "answer": pair.answer})

    out_qa.write_text("".join(json.dumps(r) + "\n" for r in out))
    return len(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-root", type=Path, required=True)
    p.add_argument("--staged-dir", type=Path, required=True, help="dir with {case}_qa.jsonl")
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    cases = sorted(f.name[: -len("_qa.jsonl")] for f in a.staged_dir.glob("*_qa.jsonl"))
    for case_id in cases:
        n = regen_case(case_id, a.labels_root, a.staged_dir / f"{case_id}_qa.jsonl", a.out_dir / f"{case_id}_qa.jsonl")
        total += n
        print(f"{case_id}: {n} QA pairs")
    print(f"TOTAL: {total} QA pairs across {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

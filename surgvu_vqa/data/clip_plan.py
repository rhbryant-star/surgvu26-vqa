"""Deterministic, task-stratified 30-s clip selection for one case.

Clips never cross video-part boundaries (each part is its own mp4). Sampling
is seeded (case-stable) so re-running dataset builds is reproducible.

Only segments whose task is in STEP_CLASSES are eligible; unknown/empty task
names are silently dropped so no clip ever carries task=''.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from surgvu_vqa.data.labels import CaseLabels

# The 8 surgical step classes used by SurgVU 2026, in sentence-case
# (per POST-CRITIQUE REVISIONS item 4 — overrides any Title-Case spelling
# in the plan body).
STEP_CLASSES: frozenset[str] = frozenset([
    "Suturing",
    "Uterine horn",
    "Suspensory ligaments",
    "Rectal artery/vein",
    "Skills application",
    "Range of motion",
    "Retraction and collision avoidance",
    "Other",
])


@dataclass(frozen=True)
class ClipSpec:
    part: int
    start_s: float
    end_s: float
    task: str


def plan_clips(
    labels: CaseLabels,
    clip_len_s: float = 30.0,
    max_clips: int = 25,
    seed: int = 0,
) -> list[ClipSpec]:
    """Round-robin over task segments, sampling non-overlapping windows.

    Only task intervals whose task name is in STEP_CLASSES and whose duration
    is at least clip_len_s are eligible. Returns a deterministically ordered
    list of at most max_clips ClipSpec objects, sorted by (part, start_s).
    """
    rng = random.Random(seed)

    # Filter: known step class AND long enough to hold at least one clip.
    segments = [
        a
        for a in labels.task_intervals
        if a.task in STEP_CLASSES and a.end_s - a.start_s >= clip_len_s
    ]
    if not segments:
        return []

    # Build a pool of candidate clips per segment (aligned grid, shuffled).
    per_segment: list[list[ClipSpec]] = []
    for a in segments:
        starts: list[float] = []
        s = a.start_s
        while s + clip_len_s <= a.end_s:
            starts.append(s)
            s += clip_len_s
        rng.shuffle(starts)
        per_segment.append(
            [ClipSpec(a.part, st, st + clip_len_s, a.task) for st in starts]
        )

    # Round-robin pick across segments until max_clips or all pools exhausted.
    clips: list[ClipSpec] = []
    i = 0
    while len(clips) < max_clips and any(per_segment):
        bucket = per_segment[i % len(per_segment)]
        if bucket:
            clips.append(bucket.pop())
        if not any(per_segment):
            break
        i += 1

    return sorted(clips, key=lambda c: (c.part, c.start_s))

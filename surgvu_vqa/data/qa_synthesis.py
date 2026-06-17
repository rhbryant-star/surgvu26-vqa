"""Synthesize grounded VQA pairs for one clip from weak labels.

Answers use the v3 BLEU-winning declarative style (docs/BASELINES.md):
Yes/No-led restatements for closed questions, subject-restating sentences for
identification. Question templates rotate deterministically per (clip, seed)
and alternate between the groundtruth and commercial tool vocabularies.

POST-CRITIQUE REVISIONS applied:
  Rev 2: Filter present tools to the 12 challenge classes before any sampling
          (drops nan(camera in), blanks, out-of-vocab names).
  Rev 3: Dedup present tools by groundtruth name before sampling positives so no
          tool can appear as two positive pairs (handles case_122 duplicate rows).
  Rev 4: Negative task pool uses real sentence-case task names from STEP_CLASSES
          (NOT Title Case from the plan body).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from surgvu_vqa.data.clip_plan import ClipSpec, STEP_CLASSES
from surgvu_vqa.data.labels import CaseLabels, ToolInterval

# The 12 challenge tool classes (groundtruth_toolname vocabulary).
# These are the ONLY valid classes; anything else (nan(camera in), blanks,
# suction irrigator, etc.) is filtered out before forming any QA pair.
ALL_GROUNDTRUTH_TOOLS: list[str] = [
    "needle driver",
    "cadiere forceps",
    "prograsp forceps",
    "monopolar curved scissors",
    "bipolar forceps",
    "stapler",
    "force bipolar",
    "vessel sealer",
    "permanent cautery hook/spatula",
    "clip applier",
    "tip-up fenestrated grasper",
    "grasping retractor",
]

# Lowercase set for O(1) membership check (case-insensitive, stripped).
_VALID_GROUNDTRUTH: frozenset[str] = frozenset(t.lower() for t in ALL_GROUNDTRUTH_TOOLS)

# Task negative pool: real sentence-case names from STEP_CLASSES (revision 4).
# "Other" is excluded from the negative pool so wrong-task questions use the
# 7 named classes only (matching actual label vocabulary).
_TASK_NEGATIVE_POOL: list[str] = sorted(
    t for t in STEP_CLASSES if t != "Other"
)

_TOOL_Q = [
    "Was a {tool} used during this clip?",
    "Is a {tool} being used here?",
    "Did the surgeon use a {tool} in this clip?",
    "Was a {tool} used during the surgery?",
]
# Answers use SHORT canonical restatements with NO temporal echo ("during this
# clip" / "in this clip" / "here"). The public refs drop the temporal clause
# even when the question carries it ("Is tissue being cut during this clip?" ->
# "Yes, tissue is being cut."), and echoing it back dilutes the BLEU n-gram
# overlap (empirically: case131 1.0000 -> 0.3457 with the echo). Questions keep
# their variety — only the answer is scored. (M2 v2 fix, 2026-06-17.)
_TOOL_YES = [
    "Yes, a {tool} was used.",
    "Yes, a {tool} is being used.",
    "Yes, the surgeon used a {tool}.",
    "Yes, a {tool} was utilized.",
]
_TOOL_NO = [
    "No, a {tool} was not used.",
    "No, a {tool} is not being used.",
    "No, the surgeon did not use a {tool}.",
    "No, a {tool} was not utilized.",
]
_TASK_WHAT_Q = [
    "What surgical task is being performed in this clip?",
    "Which surgical task is shown here?",
    "What task is the surgeon performing?",
]
_TASK_WHAT_A = [
    "The surgical task being performed is {task}.",
    "The task shown is {task}.",
    "The surgeon is performing {task}.",
]
_TASK_YESNO_Q = ["Is {task} being performed in this clip?"]
_TASK_YES = ["Yes, {task} is being performed."]
_TASK_NO = ["No, {task} is not being performed."]
_WHICH_Q = ["Which tools are being used in this clip?"]
_WHICH_A = ["The tools being used are {tools}."]


@dataclass(frozen=True)
class QAPair:
    kind: str       # tool_yes | tool_no | task_what | task_yesno | which_tools
    question: str
    answer: str


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _filter_valid_tools(tools: list[ToolInterval]) -> list[ToolInterval]:
    """Drop tools whose groundtruth name is not in the 12 challenge classes.

    Filters out nan(camera in), blank names, and any out-of-vocabulary entries
    (e.g. suction irrigator, synchroseal, bipolar dissector, etc.).
    Comparison is case-insensitive and strips leading/trailing whitespace.
    """
    return [
        t for t in tools
        if t.groundtruth.strip().lower() in _VALID_GROUNDTRUTH
    ]


def _dedup_by_groundtruth(tools: list[ToolInterval]) -> list[ToolInterval]:
    """Return one representative ToolInterval per unique groundtruth name.

    Preserves the first occurrence. This prevents duplicate rows (e.g.
    case_122's exact-duplicate tool rows) from producing two positive pairs
    for the same tool.
    """
    seen: set[str] = set()
    out: list[ToolInterval] = []
    for t in tools:
        key = t.groundtruth.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def synthesize_qa(
    labels: CaseLabels,
    clip: ClipSpec,
    seed: int = 0,
    max_pairs: int = 6,
) -> list[QAPair]:
    """Synthesize ≤max_pairs grounded VQA pairs for the given clip.

    Revisions applied:
    - Present tools are filtered to the 12 challenge classes and deduped by
      groundtruth name before any sampling (revisions 2+3).
    - Task negative pool uses real sentence-case names (revision 4).
    """
    rng = random.Random((seed, clip.part, int(clip.start_s)).__hash__())

    # --- Tool presence: filter to vocab + dedup ---
    raw_present = labels.tools_in_window(clip.part, clip.start_s, clip.end_s)
    valid_present = _filter_valid_tools(raw_present)     # drop nan/blank/OOV (rev 2)
    unique_present = _dedup_by_groundtruth(valid_present)  # one row per tool (rev 3)

    present_gt_names: set[str] = {t.groundtruth.strip().lower() for t in unique_present}

    pairs: list[QAPair] = []

    # Positives: up to 2 present tools, vocab alternating per pick.
    picks = rng.sample(unique_present, min(2, len(unique_present))) if unique_present else []
    for i, t in enumerate(picks):
        # Alternate: even index → groundtruth name, odd → lowercased commercial name
        name = t.groundtruth if i % 2 == 0 else t.commercial.lower()
        j = rng.randrange(len(_TOOL_Q))
        pairs.append(QAPair(
            "tool_yes",
            _TOOL_Q[j].format(tool=name),
            _TOOL_YES[j].format(tool=name),
        ))

    # Sorted list of present names for which-tools and negative exclusion.
    present_names_sorted = sorted(present_gt_names)

    # Negatives: 2 tools NOT present (groundtruth vocabulary).
    absent = [t for t in ALL_GROUNDTRUTH_TOOLS if t.lower() not in present_gt_names]
    for name in rng.sample(absent, min(2, len(absent))):
        j = rng.randrange(len(_TOOL_Q))
        pairs.append(QAPair(
            "tool_no",
            _TOOL_Q[j].format(tool=name),
            _TOOL_NO[j].format(tool=name),
        ))

    # Task identification (what task?).
    task = clip.task
    j = rng.randrange(len(_TASK_WHAT_Q))
    pairs.append(QAPair("task_what", _TASK_WHAT_Q[j], _TASK_WHAT_A[j].format(task=task)))

    # Task yes/no: 50% positive (the real task), 50% negative (wrong task).
    # Negative pool uses real sentence-case from STEP_CLASSES (revision 4).
    if rng.random() < 0.5:
        pairs.append(QAPair(
            "task_yesno",
            _TASK_YESNO_Q[0].format(task=task),
            _TASK_YES[0].format(task=task),
        ))
    else:
        wrong_pool = [t for t in _TASK_NEGATIVE_POOL if t.lower() != task.lower()]
        wrong = rng.choice(wrong_pool) if wrong_pool else "Other"
        pairs.append(QAPair(
            "task_yesno",
            _TASK_YESNO_Q[0].format(task=wrong),
            _TASK_NO[0].format(task=wrong),
        ))

    # Which-tools (only when ≥1 valid present tool and we have budget).
    if present_names_sorted and len(pairs) < max_pairs:
        pairs.append(QAPair(
            "which_tools",
            _WHICH_Q[0],
            _WHICH_A[0].format(tools=_join(present_names_sorted)),
        ))

    return pairs[:max_pairs]

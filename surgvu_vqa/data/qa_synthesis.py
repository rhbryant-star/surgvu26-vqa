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

# --- v3 (BERTScore) revisions -------------------------------------------------
# The ranked metric is BERTScore-F1, which scores EMBEDDING overlap against the
# reference wording. Two observed mismatches on the public set cost real score:
#
#  1. Raw label strings were injected verbatim mid-sentence, producing
#     "The organ being manipulated is Uterine horn." while the reference reads
#     "...is the uterine horn."  Naturalizing the noun phrase (lowercase + article)
#     matches the reference frame instead of the CSV's label casing.
#  2. Anatomy/organ questions ("What organ is being manipulated?") were never
#     synthesized at all — the step label is a TASK name, not an organ, so the
#     model answered with a task name. STEP_ANATOMY maps the anatomical steps to
#     the structure actually being worked on.
#
# Every SurgVU clip is da Vinci robot-assisted endoscopic/laparoscopic surgery on
# a porcine model, so the procedure-type answer is a dataset-wide constant, not a
# per-clip inference.
PROCEDURE_TYPE = "endoscopic or laparoscopic surgery"

# Anatomical structure worked on during each anatomy-bearing step. Steps that are
# skills-exercises rather than anatomy (Skills application, Range of motion,
# Retraction and collision avoidance, Suturing, Other) are deliberately absent —
# no organ question is generated for them.
STEP_ANATOMY: dict[str, str] = {
    "Uterine horn": "the uterine horn",
    "Suspensory ligaments": "the suspensory ligaments",
    "Rectal artery/vein": "the rectal artery and vein",
}


def naturalize_label(label: str) -> str:
    """Lowercase a CSV label for mid-sentence use as a TASK name.

    "Uterine horn" -> "uterine horn", so answers read "...is performing uterine
    horn." instead of echoing the CSV's sentence-case mid-sentence. Deliberately
    does NOT add an article: the label names a task here, and "performing the
    uterine horn" is ungrammatical. The anatomical noun phrase (with article)
    lives in STEP_ANATOMY and is used only for organ questions.
    """
    return label.lower()

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

# --- v3 question types (coverage gaps found on the public set) ---------------
# Organ/anatomy: public case127 asks "What organ is being manipulated?" and the
# reference is the anatomical structure, not the task name.
_ORGAN_Q = [
    "What organ is being manipulated?",
    "Which anatomical structure is being manipulated?",
    "What tissue is the surgeon working on?",
]
_ORGAN_A = [
    "The organ being manipulated is {organ}.",
    "The structure being manipulated is {organ}.",
    "The surgeon is working on {organ}.",
]
# Procedure type: public case129 asks what procedure the clip shows; the answer
# is a dataset-wide constant (all SurgVU footage is robotic endoscopic surgery).
_PROC_Q = [
    "What procedure is this summary describing?",
    "What type of procedure is being performed?",
    "What kind of surgery is shown in this clip?",
]
_PROC_A = [
    "The summary is describing {proc}.",
    "The procedure being performed is {proc}.",
    "This is {proc}.",
]
# Tool purpose: public case130 asks what a tool is FOR. Purposes are properties
# of the instrument class, not of the clip.
TOOL_PURPOSE: dict[str, str] = {
    "needle driver": "to grasp and drive the needle while suturing",
    "cadiere forceps": "to grasp and hold tissues or objects",
    "prograsp forceps": "to grasp and retract tissue",
    "bipolar forceps": "to grasp tissue and coagulate vessels",
    "force bipolar": "to grasp tissue and coagulate vessels",
    "monopolar curved scissors": "to cut and dissect tissue",
    "vessel sealer": "to seal and divide vessels",
    "stapler": "to staple and divide tissue",
    "clip applier": "to apply clips to vessels",
    "permanent cautery hook/spatula": "to dissect and cauterize tissue",
    "tip-up fenestrated grasper": "to grasp and hold tissue",
    "grasping retractor": "to grasp and retract tissue",
}
_PURPOSE_Q = [
    "What is the purpose of using {tool} in this procedure?",
    "Why is {tool} being used?",
    "What is {tool} used for in this procedure?",
]
_PURPOSE_A = [
    "The purpose of using {tool} in this procedure is {purpose}.",
    "{Tool} is being used {purpose}.",
    "{Tool} is used {purpose}.",
]


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
    max_pairs: int = 9,
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

    # Task identification (what task?). The answer naturalizes the CSV label so we
    # emit "...is the uterine horn." not "...is Uterine horn." (v3/BERTScore fix).
    task = clip.task
    j = rng.randrange(len(_TASK_WHAT_Q))
    pairs.append(QAPair("task_what", _TASK_WHAT_Q[j],
                        _TASK_WHAT_A[j].format(task=naturalize_label(task))))


    # Task yes/no: 50% positive (the real task), 50% negative (wrong task).
    # Negative pool uses real sentence-case from STEP_CLASSES (revision 4).
    if rng.random() < 0.5:
        pairs.append(QAPair(
            "task_yesno",
            _TASK_YESNO_Q[0].format(task=task),
            _TASK_YES[0].format(task=naturalize_label(task)),
        ))
    else:
        wrong_pool = [t for t in _TASK_NEGATIVE_POOL if t.lower() != task.lower()]
        wrong = rng.choice(wrong_pool) if wrong_pool else "Other"
        pairs.append(QAPair(
            "task_yesno",
            _TASK_YESNO_Q[0].format(task=wrong),
            _TASK_NO[0].format(task=naturalize_label(wrong)),
        ))

    # Which-tools (only when ≥1 valid present tool and we have budget).
    if present_names_sorted and len(pairs) < max_pairs:
        pairs.append(QAPair(
            "which_tools",
            _WHICH_Q[0],
            _WHICH_A[0].format(tools=_join(present_names_sorted)),
        ))

    # --- v3 question types (added AFTER the core types so they never starve the
    # tool/task pairs out of the max_pairs budget) ---------------------------
    # Organ/anatomy — public case127 asks this; we used to answer a task name.
    organ = STEP_ANATOMY.get(task)
    if organ and len(pairs) < max_pairs:
        j = rng.randrange(len(_ORGAN_Q))
        pairs.append(QAPair("organ_what", _ORGAN_Q[j], _ORGAN_A[j].format(organ=organ)))

    # Tool purpose — public case130 asks what an instrument is for.
    if unique_present and len(pairs) < max_pairs:
        t = rng.choice(unique_present)
        gt = t.groundtruth.strip().lower()
        purpose = TOOL_PURPOSE.get(gt)
        if purpose:
            j = rng.randrange(len(_PURPOSE_Q))
            phrase = f"a {gt}"
            pairs.append(QAPair(
                "tool_purpose",
                _PURPOSE_Q[j].format(tool=phrase),
                _PURPOSE_A[j].format(tool=phrase, Tool=f"A {gt}", purpose=purpose),
            ))

    # Procedure type — dataset-wide constant; public case129 gap. Sampled at 50%
    # so it does not dominate the corpus with one constant answer.
    if len(pairs) < max_pairs and rng.random() < 0.5:
        j = rng.randrange(len(_PROC_Q))
        pairs.append(QAPair("proc_type", _PROC_Q[j], _PROC_A[j].format(proc=PROCEDURE_TYPE)))

    return pairs[:max_pairs]

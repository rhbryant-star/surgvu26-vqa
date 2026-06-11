"""Bind the public Cat-2 sample set into our truth.json schema.

The real sample set layout (verified against data/raw/cat2_samples/ downloaded by
data/download_samples.sh) is ONE DIRECTORY PER CLIP:

    cat2_samples/
        caseNNN/
            caseNNN_question.json   — bare JSON string, e.g. "Are forceps visible?"
            caseNNN.json            — bare JSON array of 5 paraphrase strings

The clip_id is the directory stem (e.g. "case122").

If the on-disk format drifts, adjust the constants / _read_case() below.
The public interface (build_truth_json) and the truth schema are stable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ------------------------------------------------------------------
# Format constants — edit here if the download format changes.
# ------------------------------------------------------------------
CASE_DIR_PATTERN = re.compile(r"^case\d+$")
QUESTION_SUFFIX = "_question.json"
ANSWERS_SUFFIX = ".json"


def _read_case(case_dir: Path) -> dict:
    """Parse one caseNNN directory into {question, references}."""
    stem = case_dir.name
    question_path = case_dir / f"{stem}{QUESTION_SUFFIX}"
    answers_path = case_dir / f"{stem}{ANSWERS_SUFFIX}"

    question: str = json.loads(question_path.read_text(encoding="utf-8"))
    references: list = json.loads(answers_path.read_text(encoding="utf-8"))

    return {"question": question, "references": references}


def build_truth_json(annotations_path: Path) -> dict:
    """Map the public Cat-2 annotations directory → truth schema.

    Args:
        annotations_path: Path to the root directory containing caseNNN/
                          subdirectories (e.g. data/raw/cat2_samples/).

    Returns:
        dict mapping clip_id (str) → {"question": str, "references": [str, ...]}.
        Empty dict if no caseNNN directories are found.
    """
    truth: dict = {}
    for child in sorted(annotations_path.iterdir()):
        if child.is_dir() and CASE_DIR_PATTERN.match(child.name):
            truth[child.name] = _read_case(child)
    return truth

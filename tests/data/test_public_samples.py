"""Tests for surgvu_vqa.data.public_samples — per-clip file format.

Fixtures are shaped exactly like the real cat2_samples download:
  tmp_path/caseNNN/caseNNN_question.json  — bare JSON string
  tmp_path/caseNNN/caseNNN.json           — bare JSON array of 5 strings
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from surgvu_vqa.data.public_samples import build_truth_json


def _make_case(
    root: Path,
    case_id: str,
    question: str,
    answers: list,
) -> None:
    """Write the two annotation files for one case into root/case_id/."""
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / f"{case_id}_question.json").write_text(json.dumps(question))
    (case_dir / f"{case_id}.json").write_text(json.dumps(answers))


def test_build_truth_json_basic(tmp_path):
    """Two cases with 5 reference answers each — canonical happy path."""
    _make_case(
        tmp_path, "case122",
        question="Are there forceps being used here?",
        answers=[
            "No",
            "No, forceps are not mentioned.",
            "No forceps are being used.",
            "No, there's no indication of forceps.",
            "No forceps are listed.",
        ],
    )
    _make_case(
        tmp_path, "case124",
        question="What type of forceps is mentioned?",
        answers=[
            "Cadiere Forceps",
            "The type of forceps mentioned is Cadiere Forceps.",
            "Cadiere Forceps are the type mentioned.",
            "The forceps type is Cadiere Forceps.",
            "Cadiere Forceps is the specific type referenced.",
        ],
    )

    truth = build_truth_json(tmp_path)

    assert set(truth) == {"case122", "case124"}
    assert truth["case122"]["question"] == "Are there forceps being used here?"
    assert truth["case122"]["references"] == [
        "No",
        "No, forceps are not mentioned.",
        "No forceps are being used.",
        "No, there's no indication of forceps.",
        "No forceps are listed.",
    ]
    assert truth["case124"]["question"] == "What type of forceps is mentioned?"
    assert truth["case124"]["references"][0] == "Cadiere Forceps"
    assert len(truth["case124"]["references"]) == 5


def test_build_truth_json_answer_count_is_five(tmp_path):
    """Each entry must carry exactly 5 reference strings (per challenge spec)."""
    _make_case(
        tmp_path, "case130",
        question="Is a clip applier visible?",
        answers=[
            "Yes",
            "Yes, a clip applier is visible.",
            "A clip applier can be seen.",
            "Yes, the clip applier is present.",
            "The clip applier is visible in the frame.",
        ],
    )

    truth = build_truth_json(tmp_path)

    assert "case130" in truth
    assert len(truth["case130"]["references"]) == 5
    assert truth["case130"]["references"][0] == "Yes"


def test_build_truth_json_clip_id_is_directory_stem(tmp_path):
    """clip_id must be the bare directory stem string, not a numeric coercion."""
    _make_case(
        tmp_path, "case128",
        question="What instrument is used for grasping?",
        answers=["Grasper"] * 5,
    )

    truth = build_truth_json(tmp_path)

    key = list(truth.keys())[0]
    assert isinstance(key, str)
    assert key == "case128"


def test_build_truth_json_empty_directory(tmp_path):
    """An annotations root with no caseNNN subdirectories returns an empty dict."""
    truth = build_truth_json(tmp_path)
    assert truth == {}

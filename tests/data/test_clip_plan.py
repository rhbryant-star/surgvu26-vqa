from __future__ import annotations

from surgvu_vqa.data.clip_plan import STEP_CLASSES, plan_clips
from surgvu_vqa.data.labels import CaseLabels, TaskInterval, ToolInterval


def _labels(task_intervals):
    return CaseLabels(case_id="case_001", tool_intervals=[], task_intervals=task_intervals)


def test_clips_inside_task_segments_and_deterministic():
    labels = _labels([TaskInterval(part=1, start_s=0.0, end_s=300.0, task="Suturing")])
    a = plan_clips(labels, clip_len_s=30.0, max_clips=5, seed=7)
    b = plan_clips(labels, clip_len_s=30.0, max_clips=5, seed=7)
    assert a == b                                  # deterministic
    assert 1 <= len(a) <= 5
    for c in a:
        assert c.part == 1
        assert c.start_s >= 0.0 and c.start_s + 30.0 <= 300.0
        assert c.task == "Suturing"


def test_short_segments_skipped():
    labels = _labels([TaskInterval(part=1, start_s=0.0, end_s=20.0, task="Other")])
    assert plan_clips(labels, clip_len_s=30.0, max_clips=5, seed=7) == []


def test_clips_spread_across_tasks():
    labels = _labels([
        TaskInterval(part=1, start_s=0.0, end_s=120.0, task="Suturing"),
        TaskInterval(part=2, start_s=0.0, end_s=120.0, task="Range of motion"),
    ])
    clips = plan_clips(labels, clip_len_s=30.0, max_clips=6, seed=1)
    tasks = {c.task for c in clips}
    assert tasks == {"Suturing", "Range of motion"}     # stratified
    assert len([c for c in clips if c.part == 2]) >= 1


def test_unknown_task_dropped():
    """Task names not in STEP_CLASSES must never appear in plan output."""
    labels = _labels([
        TaskInterval(part=1, start_s=0.0, end_s=300.0, task="UnknownSurgery"),
        TaskInterval(part=1, start_s=300.0, end_s=600.0, task="Suturing"),
    ])
    clips = plan_clips(labels, clip_len_s=30.0, max_clips=10, seed=0)
    tasks = {c.task for c in clips}
    assert "UnknownSurgery" not in tasks
    assert "Suturing" in tasks


def test_no_empty_task_clips():
    """Empty task names must never reach clip output (defense-in-depth)."""
    labels = _labels([
        TaskInterval(part=1, start_s=0.0, end_s=300.0, task=""),
        TaskInterval(part=1, start_s=300.0, end_s=600.0, task="Suturing"),
    ])
    clips = plan_clips(labels, clip_len_s=30.0, max_clips=10, seed=0)
    assert all(c.task != "" for c in clips)
    assert any(c.task == "Suturing" for c in clips)


def test_step_classes_are_sentence_case():
    """STEP_CLASSES must use sentence-case per revisions item 4."""
    assert "Suturing" in STEP_CLASSES
    assert "Range of motion" in STEP_CLASSES
    assert "Retraction and collision avoidance" in STEP_CLASSES
    # Title-case variants must NOT be present
    assert "Range Of Motion" not in STEP_CLASSES
    assert "Retraction And Collision Avoidance" not in STEP_CLASSES


def test_real_case_122_produces_clips():
    """Sanity check: real case_122 labels yield at least one clip."""
    from pathlib import Path
    labels_dir = Path("data/raw/labels_root/labels/case_122")
    if not labels_dir.exists():
        import pytest
        pytest.skip("real labels not available")
    from surgvu_vqa.data.labels import CaseLabels as CL
    labels = CL.load(labels_dir)
    clips = plan_clips(labels, clip_len_s=30.0, max_clips=25, seed=0)
    assert len(clips) >= 1
    # All clips must have a valid step class task
    for c in clips:
        assert c.task in STEP_CLASSES
        assert c.task != ""

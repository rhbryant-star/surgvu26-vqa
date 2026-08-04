from __future__ import annotations

from surgvu_vqa.data.clip_plan import ClipSpec
from surgvu_vqa.data.labels import CaseLabels, TaskInterval, ToolInterval
from surgvu_vqa.data.qa_synthesis import ALL_GROUNDTRUTH_TOOLS, synthesize_qa


def _labels():
    return CaseLabels(
        case_id="case_001",
        tool_intervals=[
            ToolInterval(part=1, start_s=0.0, end_s=300.0, groundtruth="needle driver", commercial="Large Needle Driver"),
            ToolInterval(part=1, start_s=0.0, end_s=300.0, groundtruth="cadiere forceps", commercial="Cadiere Forceps"),
        ],
        task_intervals=[TaskInterval(part=1, start_s=0.0, end_s=300.0, task="Suturing")],
    )


CLIP = ClipSpec(part=1, start_s=30.0, end_s=60.0, task="Suturing")


def test_generates_positive_negative_step_and_which():
    qa = synthesize_qa(_labels(), CLIP, seed=3)
    kinds = {p.kind for p in qa}
    assert {"tool_yes", "tool_no", "task_what", "task_yesno"} <= kinds
    # v3 raised the budget to 9 so the added organ/purpose/procedure-type
    # questions cannot starve the core tool/task pairs out of the cut.
    assert len(qa) <= 9


def test_positive_answer_is_declarative_yes():
    qa = [p for p in synthesize_qa(_labels(), CLIP, seed=3) if p.kind == "tool_yes"]
    assert qa, "expected a positive tool question"
    p = qa[0]
    assert p.answer.startswith("Yes, ")
    assert p.answer.endswith(".")
    assert len(p.answer.split()) >= 5            # full declarative, never terse
    # the tool named in the question is the one in the answer
    assert any(tok in p.answer.lower() for tok in p.question.lower().split() if len(tok) > 4)


def test_negative_tool_not_in_clip():
    qa = [p for p in synthesize_qa(_labels(), CLIP, seed=3) if p.kind == "tool_no"]
    assert qa
    present = {"needle driver", "cadiere forceps"}
    for p in qa:
        assert not any(t in p.question.lower() for t in present)
        assert p.answer.startswith("No, ") and p.answer.endswith(".")


def test_task_answers_in_v3_style():
    qa = {p.kind: p for p in synthesize_qa(_labels(), CLIP, seed=3)}
    assert "suturing" in qa["task_what"].answer.lower()
    assert qa["task_what"].answer.endswith(".")
    assert qa["task_yesno"].answer.startswith(("Yes, ", "No, "))


def test_deterministic():
    a = synthesize_qa(_labels(), CLIP, seed=11)
    b = synthesize_qa(_labels(), CLIP, seed=11)
    assert [(p.question, p.answer) for p in a] == [(p.question, p.answer) for p in b]


def test_negative_pool_excludes_present_tools():
    absent = [t for t in ALL_GROUNDTRUTH_TOOLS if t not in {"needle driver", "cadiere forceps"}]
    assert "stapler" in absent and "needle driver" not in absent


# --- Revision 2+3+4 required tests ---

def test_junk_toolnames_excluded_from_all_qa_pairs():
    """nan(camera in) and blank groundtruth names must not appear in any QA pair."""
    labels = CaseLabels(
        case_id="case_junk",
        tool_intervals=[
            ToolInterval(part=1, start_s=0.0, end_s=300.0, groundtruth="nan(camera in)", commercial="Camera"),
            ToolInterval(part=1, start_s=0.0, end_s=300.0, groundtruth="", commercial="Unknown"),
            ToolInterval(part=1, start_s=0.0, end_s=300.0, groundtruth="needle driver", commercial="Large Needle Driver"),
        ],
        task_intervals=[TaskInterval(part=1, start_s=0.0, end_s=300.0, task="Suturing")],
    )
    clip = ClipSpec(part=1, start_s=30.0, end_s=60.0, task="Suturing")
    qa = synthesize_qa(labels, clip, seed=7)
    for pair in qa:
        assert "nan(camera in)" not in pair.question.lower()
        assert "nan(camera in)" not in pair.answer.lower()
        # blank tool should not appear (empty string won't produce valid question anyway)
        # but ensure no empty-name tool ends up as a positive
        assert pair.kind != "tool_yes" or pair.answer.strip() != "Yes, a  was used during this clip."


def test_duplicate_tool_rows_no_duplicate_positives():
    """Duplicate rows for the same groundtruth tool must never produce two positive pairs."""
    labels = CaseLabels(
        case_id="case_dup",
        tool_intervals=[
            # Two rows with same groundtruth — simulates case_122 exact duplicate rows
            ToolInterval(part=1, start_s=0.0, end_s=300.0, groundtruth="needle driver", commercial="Large Needle Driver"),
            ToolInterval(part=1, start_s=0.0, end_s=300.0, groundtruth="needle driver", commercial="Large Needle Driver"),
        ],
        task_intervals=[TaskInterval(part=1, start_s=0.0, end_s=300.0, task="Suturing")],
    )
    clip = ClipSpec(part=1, start_s=30.0, end_s=60.0, task="Suturing")
    qa = synthesize_qa(labels, clip, seed=5)
    positive_pairs = [p for p in qa if p.kind == "tool_yes"]
    # needle driver should appear at most once as a positive
    nd_positives = [p for p in positive_pairs if "needle driver" in p.question.lower() or "needle driver" in p.answer.lower()]
    assert len(nd_positives) <= 1


def test_task_yesno_negative_uses_sentence_case():
    """Negative task_yesno questions must use real sentence-case task names."""
    # Use a clip with "Suturing" so the yesno can pick a different task as wrong answer.
    # Run many seeds to catch a task_yesno negative case.
    labels = _labels()
    clip = ClipSpec(part=1, start_s=30.0, end_s=60.0, task="Suturing")
    found_negative = False
    for seed in range(30):
        qa = synthesize_qa(labels, clip, seed=seed)
        for pair in qa:
            if pair.kind == "task_yesno" and pair.answer.startswith("No, "):
                found_negative = True
                # Must NOT use Title Case variants
                assert "Range Of Motion" not in pair.question
                assert "Retraction And Collision Avoidance" not in pair.question
                assert "Skills Application" not in pair.question
                assert "Uterine Horn" not in pair.question
                assert "Suspensory Ligaments" not in pair.question
                assert "Rectal Artery/Vein" not in pair.question
    assert found_negative, "expected at least one negative task_yesno pair across seeds"


def test_out_of_vocab_tools_not_in_positives():
    """Tools not in ALL_GROUNDTRUTH_TOOLS (e.g. 'suction irrigator') must not appear as positives."""
    labels = CaseLabels(
        case_id="case_oov",
        tool_intervals=[
            ToolInterval(part=1, start_s=0.0, end_s=300.0, groundtruth="suction irrigator", commercial="Suction Irrigator"),
            ToolInterval(part=1, start_s=0.0, end_s=300.0, groundtruth="needle driver", commercial="Large Needle Driver"),
        ],
        task_intervals=[TaskInterval(part=1, start_s=0.0, end_s=300.0, task="Suturing")],
    )
    clip = ClipSpec(part=1, start_s=30.0, end_s=60.0, task="Suturing")
    qa = synthesize_qa(labels, clip, seed=3)
    positive_pairs = [p for p in qa if p.kind == "tool_yes"]
    for p in positive_pairs:
        assert "suction irrigator" not in p.question.lower()
        assert "suction irrigator" not in p.answer.lower()


# --- v3 (BERTScore) behavior -------------------------------------------------

def test_task_labels_are_naturalized_not_csv_case():
    """Answers must read '...the uterine horn.', never the CSV's 'Uterine horn'."""
    from surgvu_vqa.data.qa_synthesis import naturalize_label
    # Task context: lowercase only, no article ("performing uterine horn").
    assert naturalize_label("Uterine horn") == "uterine horn"
    assert naturalize_label("Skills application") == "skills application"
    # Organ context: the anatomical noun phrase carries the article.
    from surgvu_vqa.data.qa_synthesis import STEP_ANATOMY
    assert STEP_ANATOMY["Uterine horn"] == "the uterine horn"
    assert STEP_ANATOMY["Rectal artery/vein"] == "the rectal artery and vein"

    clip = ClipSpec(part=1, start_s=30.0, end_s=60.0, task="Uterine horn")
    labels = CaseLabels(
        case_id="case_v3",
        tool_intervals=[ToolInterval(part=1, start_s=0.0, end_s=300.0,
                                     groundtruth="needle driver",
                                     commercial="Large Needle Driver")],
        task_intervals=[TaskInterval(part=1, start_s=0.0, end_s=300.0, task="Uterine horn")],
    )
    for seed in range(8):
        for p in synthesize_qa(labels, clip, seed=seed):
            assert "Uterine horn" not in p.answer, p.answer


def test_v3_adds_organ_purpose_and_procedure_questions():
    """The three question types the public set exposed as coverage gaps."""
    clip = ClipSpec(part=1, start_s=30.0, end_s=60.0, task="Uterine horn")
    labels = CaseLabels(
        case_id="case_v3",
        tool_intervals=[ToolInterval(part=1, start_s=0.0, end_s=300.0,
                                     groundtruth="needle driver",
                                     commercial="Large Needle Driver")],
        task_intervals=[TaskInterval(part=1, start_s=0.0, end_s=300.0, task="Uterine horn")],
    )
    kinds = set()
    for seed in range(20):
        kinds |= {p.kind for p in synthesize_qa(labels, clip, seed=seed)}
    assert "organ_what" in kinds
    assert "tool_purpose" in kinds
    assert "proc_type" in kinds


def test_core_types_never_starved_by_v3_types():
    """Core tool/task pairs must survive the max_pairs cut in every sample."""
    clip = ClipSpec(part=1, start_s=30.0, end_s=60.0, task="Uterine horn")
    labels = CaseLabels(
        case_id="case_v3",
        tool_intervals=[ToolInterval(part=1, start_s=0.0, end_s=300.0,
                                     groundtruth="needle driver",
                                     commercial="Large Needle Driver")],
        task_intervals=[TaskInterval(part=1, start_s=0.0, end_s=300.0, task="Uterine horn")],
    )
    for seed in range(20):
        kinds = {p.kind for p in synthesize_qa(labels, clip, seed=seed)}
        assert {"tool_yes", "tool_no", "task_what", "task_yesno"} <= kinds, (seed, kinds)

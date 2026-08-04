from surgvu_vqa.predict.answer import (
    FALLBACK_ANSWER,
    GENERIC_INSTRUCTION,
    IDENTIFY_INSTRUCTION,
    SYSTEM_PROMPT,
    YESNO_INSTRUCTION,
    build_user_text,
    shape_answer,
    style_instruction,
)


def test_prompt_constants_are_nonempty():
    assert SYSTEM_PROMPT.strip()
    assert YESNO_INSTRUCTION.strip()
    assert IDENTIFY_INSTRUCTION.strip()
    assert GENERIC_INSTRUCTION.strip()
    assert FALLBACK_ANSWER.strip().endswith(".")


def test_router_yesno_questions():
    assert style_instruction("Was a stapler used?") == YESNO_INSTRUCTION
    assert style_instruction("Are there forceps being used here?") == YESNO_INSTRUCTION
    assert style_instruction("  is tissue being cut during this clip?") == YESNO_INSTRUCTION


def test_router_identification_questions():
    assert style_instruction("What type of forceps is mentioned?") == IDENTIFY_INSTRUCTION
    assert style_instruction("Which tools are being used?") == IDENTIFY_INSTRUCTION
    assert style_instruction("How many instruments are visible?") == IDENTIFY_INSTRUCTION


def test_router_falls_back_to_generic():
    assert style_instruction("Describe the activity in the clip.") == GENERIC_INSTRUCTION


def test_identify_instruction_has_no_clinical_example():
    # v1's concrete example ("The forceps type is ...") leaked verbatim into
    # answers (BASELINES.md case129/130) — identification questions must not
    # carry a copyable example sentence.
    assert "Forceps" not in IDENTIFY_INSTRUCTION
    assert "forceps" not in IDENTIFY_INSTRUCTION
    assert "stapler" not in IDENTIFY_INSTRUCTION.lower()


def test_build_user_text_contains_question_and_routed_style():
    text = build_user_text("  Was a stapler used?  ")
    assert text.startswith("Was a stapler used?")
    assert YESNO_INSTRUCTION in text


def test_shape_strips_quotes_and_whitespace():
    assert shape_answer('  "A stapler was not used."  ') == "A stapler was not used."


def test_shape_collapses_newlines_keeps_first_sentence():
    raw = "A stapler was not used. The clip shows suturing.\nExtra commentary."
    assert shape_answer(raw) == "A stapler was not used."


def test_shape_appends_terminal_period():
    assert shape_answer("The forceps type is Cadiere Forceps") == "The forceps type is Cadiere Forceps."


def test_shape_single_word_loses_period():
    # "No." scores exactly 0.0 against the bare "No" reference (tokenizer
    # glues punctuation); the bare word scores > 0. See BASELINES.md.
    assert shape_answer("No.") == "No"
    assert shape_answer("  Yes!  ") == "Yes"
    assert shape_answer("No") == "No"


def test_shape_empty_returns_fallback():
    assert shape_answer("   ") == FALLBACK_ANSWER


# --- BERTScore: naturalize CSV step labels echoed mid-sentence ---------------

def test_naturalize_step_labels_midsentence():
    from surgvu_vqa.predict.answer import shape_answer
    # measured on the public set: 0.7577 -> 1.0000 on case127
    assert shape_answer("The organ being manipulated is Uterine horn.") == \
        "The organ being manipulated is the uterine horn."
    assert shape_answer("This summary is describing Rectal artery/vein.") == \
        "This summary is describing the rectal artery and vein."
    assert shape_answer("The surgical task being performed is Suturing.") == \
        "The surgical task being performed is suturing."


def test_naturalize_does_not_double_article_or_touch_leading_label():
    from surgvu_vqa.predict.answer import shape_answer
    # model already emitted "the" -> must not become "the the uterine horn"
    assert shape_answer("The structure is the Uterine horn.") == \
        "The structure is the uterine horn."
    # a label that legitimately starts the sentence keeps its capital
    assert shape_answer("Uterine horn is being performed.") == \
        "Uterine horn is being performed."


def test_naturalize_leaves_ordinary_answers_untouched():
    from surgvu_vqa.predict.answer import shape_answer
    for a in ("Yes, a needle driver is involved.",
              "No, a large needle driver was not used.",
              "The type of forceps mentioned is Cadiere Forceps."):
        assert shape_answer(a) == a

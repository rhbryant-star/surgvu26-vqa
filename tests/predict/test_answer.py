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

from surgvu_vqa.predict.answer import (
    ANSWER_STYLE_INSTRUCTION,
    FALLBACK_ANSWER,
    SYSTEM_PROMPT,
    build_user_text,
    shape_answer,
)


def test_prompt_constants_are_nonempty():
    assert SYSTEM_PROMPT.strip()
    assert ANSWER_STYLE_INSTRUCTION.strip()
    assert FALLBACK_ANSWER.strip().endswith(".")


def test_build_user_text_contains_question_and_style():
    text = build_user_text("  Was a stapler used?  ")
    assert text.startswith("Was a stapler used?")
    assert ANSWER_STYLE_INSTRUCTION in text


def test_shape_strips_quotes_and_whitespace():
    assert shape_answer('  "A stapler was not used."  ') == "A stapler was not used."


def test_shape_collapses_newlines_keeps_first_sentence():
    raw = "A stapler was not used. The clip shows suturing.\nExtra commentary."
    assert shape_answer(raw) == "A stapler was not used."


def test_shape_appends_terminal_period():
    assert shape_answer("The forceps type is Cadiere Forceps") == "The forceps type is Cadiere Forceps."


def test_shape_empty_returns_fallback():
    assert shape_answer("   ") == FALLBACK_ANSWER

# surgvu_vqa/predict/answer.py
"""Prompt text + BLEU-aware answer shaping (v1 baseline).

The challenge scores answers with BLEU against 5 reference phrasings whose
canonical style is a short declarative sentence ("A stapler was not used.").
Full sentences dominate 1-token answers under BLEU-4 (a 1-token exact match
caps at ~0.18) — so both the prompt and the shaper push toward that form.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert surgical assistant analyzing a short clip from a "
    "robot-assisted surgery training session recorded on a da Vinci system. "
    "Answer questions about the clip accurately and concisely."
)

ANSWER_STYLE_INSTRUCTION = (
    "Answer with exactly one short declarative sentence that restates the "
    'answer, e.g. "A stapler was not used." or "The forceps type is Cadiere '
    'Forceps." Do not add explanations.'
)

FALLBACK_ANSWER = "The answer is not visible in the clip."

_SENTENCE_ENDS = (". ", "! ", "? ")


def build_user_text(question: str) -> str:
    """Text part of the user turn; frames are attached separately."""
    return f"{question.strip()}\n\n{ANSWER_STYLE_INSTRUCTION}"


def shape_answer(raw: str) -> str:
    """Normalize a model response into one clean declarative sentence."""
    text = raw.strip().strip('"').strip("'").strip()
    text = " ".join(text.split())
    if not text:
        return FALLBACK_ANSWER
    for sep in _SENTENCE_ENDS:
        cut = text.find(sep)
        if cut != -1:
            text = text[: cut + 1].rstrip()
            break
    if text[-1] not in ".!?":
        text += "."
    return text

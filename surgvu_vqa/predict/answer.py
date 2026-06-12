# surgvu_vqa/predict/answer.py
"""Prompt construction and BLEU-aware answer shaping (v2).

The challenge scores answers with BLEU against 5 reference phrasings whose
canonical style is a short declarative sentence ("A stapler was not used.").
v1 eval findings (docs/BASELINES.md): single-word answers score ~zero (the
tokenizer glues terminal punctuation: "no." != "no"), and a concrete clinical
example in the instruction leaked verbatim into answers to unrelated
questions. v2 therefore (a) routes the style instruction by question type,
(b) keeps examples away from identification questions, and (c) strips the
period from single-word answers as a deterministic backstop.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert surgical assistant analyzing a short clip from a "
    "robot-assisted surgery training session recorded on a da Vinci system. "
    "Answer questions about the clip accurately and concisely."
)

# Yes/no questions: references look like "No" / "No, X was not used." /
# "X was not used." — a Yes/No-led full restatement matches the most n-grams.
# The example instrument (scalpel) is deliberately NOT one of the 12 challenge
# tool classes, so even verbatim leakage cannot name a wrong challenge tool.
YESNO_INSTRUCTION = (
    "Answer with one short declarative sentence that starts with Yes or No "
    "and restates the question as a statement, never as a question. For "
    "example, if asked whether a scalpel was used, answer "
    '"No, a scalpel was not used." or "Yes, a scalpel was used." '
    "Do not add explanations."
)

# Identification/open questions: v2 showed bare noun-phrase answers ("The
# stomach.") score near zero, while a subject-restating sentence pattern
# scores high. The example object (marking pen / blue) is deliberately not a
# challenge tool class, so structural leakage cannot name a wrong tool.
IDENTIFY_INSTRUCTION = (
    "Answer with one short declarative sentence that restates the subject of "
    "the question and then names the answer. For example, if asked "
    '"What color is the marking pen?", answer "The marking pen color is '
    'blue." Never answer with a single word or a bare name. '
    "Do not add explanations."
)

GENERIC_INSTRUCTION = (
    "Answer with exactly one short declarative sentence that restates the "
    "subject of the question and states the answer. Never answer with a "
    "single word. Do not add explanations."
)

FALLBACK_ANSWER = "The answer is not visible in the clip."

_SENTENCE_ENDS = (". ", "! ", "? ")

_YESNO_STARTERS = (
    "was ", "were ", "is ", "are ", "did ", "does ", "do ",
    "has ", "have ", "had ", "can ", "could ", "will ", "would ",
)
_IDENTIFY_STARTERS = ("what", "which", "who", "where", "when", "how")


def style_instruction(question: str) -> str:
    """Pick the style instruction for a question (keyword router, spec §4)."""
    q = question.strip().lower()
    if q.startswith(_YESNO_STARTERS):
        return YESNO_INSTRUCTION
    if q.startswith(_IDENTIFY_STARTERS):
        return IDENTIFY_INSTRUCTION
    return GENERIC_INSTRUCTION


def build_user_text(question: str) -> str:
    """Text part of the user turn; frames are attached separately."""
    return f"{question.strip()}\n\n{style_instruction(question)}"


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
    words = text.split()
    if len(words) == 1:
        # BLEU-4 scores "No." as 0.0 against the bare "No" reference (the
        # tokenizer keeps punctuation glued to the word); bare form scores >0.
        return words[0].rstrip(".!?")
    if text[-1] not in ".!?":
        text += "."
    return text

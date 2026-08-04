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


# CSV step labels the model may echo verbatim from its training data. The
# references phrase these as ordinary noun phrases, and BERTScore-F1 (the ranked
# metric) compares embeddings against that phrasing, so emitting the CSV's
# sentence-case mid-sentence costs real score: measured on the public set,
# "The organ being manipulated is Uterine horn." -> 0.7577 while
# "...is the uterine horn." -> 1.0000. Mean over the 11 clips: 0.7764 -> 0.8043.
# Applied post-generation so it fixes the SHIPPED model without a retrain.
_STEP_LABEL_PHRASES: dict[str, str] = {
    "Uterine horn": "the uterine horn",
    "Suspensory ligaments": "the suspensory ligaments",
    "Rectal artery/vein": "the rectal artery and vein",
    "Range of motion": "range of motion",
    "Skills application": "skills application",
    "Retraction and collision avoidance": "retraction and collision avoidance",
    "Suturing": "suturing",
}


def naturalize_step_labels(text: str) -> str:
    """Rewrite a CSV step label echoed mid-sentence as a natural noun phrase.

    Only rewrites a label that appears AFTER the first character, so an answer
    that legitimately begins with the label keeps its leading capital.
    """
    for label, phrase in _STEP_LABEL_PHRASES.items():
        idx = text.find(label)
        if idx > 0:
            # "... is the uterine horn" reads wrong as "... is the the uterine
            # horn"; drop a preceding article the model already emitted.
            prefix = text[:idx]
            for article in ("the ", "a ", "an "):
                if phrase.startswith("the ") and prefix.lower().endswith(article):
                    prefix = prefix[: -len(article)]
                    break
            text = prefix + phrase + text[idx + len(label):]
    return text


def shape_answer(raw: str) -> str:
    """Normalize a model response into one clean declarative sentence."""
    text = raw.strip().strip('"').strip("'").strip()
    text = naturalize_step_labels(text)
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

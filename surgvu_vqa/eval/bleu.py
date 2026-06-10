"""SurgVU Category-2 BLEU metric (local re-implementation).

The challenge scores each answer with BLEU against five reference answers,
takes the maximum, then averages over all questions. Uniform 1-4 gram
weights (0.25 each) with NLTK smoothing (spec sections 1 and 11).

NOTE: the official evaluator's exact tokenizer/smoothing method are not
published. This module is used for RELATIVE ranking of our own iterations;
reconcile the constants below with the official evaluation container once it
is available.
"""
from __future__ import annotations

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
_SMOOTHING = SmoothingFunction().method1


def tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization (documented default)."""
    return text.lower().split()


def question_bleu(prediction: str, references: list[str]) -> float:
    """Maximum BLEU of `prediction` against each reference answer.

    Returns 0.0 when there are no references or the prediction is empty.
    """
    if not references or not prediction.strip():
        return 0.0
    hyp = tokenize(prediction)
    return max(
        sentence_bleu(
            [tokenize(ref)], hyp, weights=_WEIGHTS, smoothing_function=_SMOOTHING
        )
        for ref in references
    )

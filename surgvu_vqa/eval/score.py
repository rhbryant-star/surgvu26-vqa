"""Aggregate per-question BLEU into the challenge metric, plus a scoring CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from surgvu_vqa.eval.bleu import question_bleu


def mean_bleu(items: list[dict]) -> float:
    """Mean of per-question max-BLEU over a list of {prediction, references}."""
    if not items:
        return 0.0
    return sum(question_bleu(it["prediction"], it["references"]) for it in items) / len(items)


def score_run(truth: dict, predictions: dict) -> dict:
    """Score predictions against truth.

    truth:       {clip_id: {"question": str, "references": [str, ...]}}
    predictions: {clip_id: "answer string"}   (missing → scored 0.0)
    """
    per_question: dict[str, float] = {}
    for clip_id, entry in truth.items():
        per_question[clip_id] = question_bleu(predictions.get(clip_id, ""), entry["references"])
    mean = sum(per_question.values()) / len(per_question) if per_question else 0.0
    return {"mean_bleu": mean, "per_question": per_question}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score SurgVU VQA predictions (local BLEU).")
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    truth = json.loads(args.truth.read_text())
    predictions = json.loads(args.predictions.read_text())
    result = score_run(truth, predictions)
    print(f"mean_bleu: {result['mean_bleu']:.4f}")
    for clip_id, s in result["per_question"].items():
        print(f"  {clip_id}: {s:.4f}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

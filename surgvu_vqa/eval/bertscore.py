"""BERTScore-F1 scorer — the CHALLENGE'S ACTUAL RANKING METRIC (not BLEU).

Verified against the organizers' own evaluation container
(github.com/isi-challenges/surgvu26-category-2-eval-public, evaluation/evaluate.py):

    bert_scorer = BERTScorer(model_type="roberta-large", rescale_with_baseline=True, ...)
    _P, _R, F1 = bert_scorer.score(cands_expanded, refs)
    bertscore_f1s.append(F1.max().item())        # max over the multiple refs
    ... mean(r["bertscore_f1"] for r in results) # mean over cases

The eval container also reports NLI entailment, nli*bertscore, BLEU and ROUGE,
but ONLY `bertscore_f1` ranks the leaderboard.

We reimplement BERTScore's greedy-matching directly on transformers rather than
importing the `bert_score` package, because that package pulls matplotlib/pandas
at import time and their compiled extensions cannot load from group staging.
The numeric definition here matches bert_score's defaults for roberta-large:
  * representation layer 17 (bert_score.utils.model2layers["roberta-large"])
  * no IDF weighting (BERTScorer default idf=False)
  * baseline rescaling (F - b) / (1 - b) with b = 0.83122575, read from
    bert_score/rescale_baseline/en/roberta-large.tsv row LAYER=17
  * special tokens (<s>, </s>) excluded from the greedy match, as bert_score does
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_NAME = "roberta-large"
LAYER = 17                      # bert_score default for roberta-large
BASELINE_F = 0.83122575         # rescale_baseline/en/roberta-large.tsv, LAYER=17


class _Embedder:
    """roberta-large hidden states at LAYER, L2-normalized, minus special tokens."""

    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
        self.model.eval().to(self.device)

    def encode(self, texts: list[str], batch_size: int = 64):
        """-> list of (n_tokens, hidden) L2-normalized tensors, specials removed."""
        torch = self._torch
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                           max_length=512).to(self.device)
            with torch.no_grad():
                hs = self.model(**enc).hidden_states[LAYER]
            hs = torch.nn.functional.normalize(hs, dim=-1)
            for j in range(len(chunk)):
                mask = enc["attention_mask"][j].bool()
                ids = enc["input_ids"][j]
                # drop <s> / </s> / <pad>, matching bert_score's special handling
                special = torch.zeros_like(mask)
                for sid in (self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id):
                    if sid is not None:
                        special |= ids == sid
                keep = mask & ~special
                out.append(hs[j][keep].cpu())
        return out


def _greedy_f1(cand_emb, ref_emb) -> float:
    """BERTScore greedy matching: P over cand tokens, R over ref tokens, F1."""
    if cand_emb.shape[0] == 0 or ref_emb.shape[0] == 0:
        return 0.0
    sim = cand_emb @ ref_emb.T          # cosine (already normalized)
    p = sim.max(dim=1).values.mean().item()
    r = sim.max(dim=0).values.mean().item()
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def score_run(truth: dict, predictions: dict, embedder: "_Embedder | None" = None) -> dict:
    """Max-over-references BERTScore-F1 per clip, then mean — the challenge metric."""
    emb = embedder or _Embedder()

    texts: list[str] = []
    index: dict[str, tuple[int, list[int]]] = {}
    for clip_id, entry in truth.items():
        pred = predictions.get(clip_id, "") or ""
        cand_i = len(texts)
        texts.append(pred)
        ref_is = []
        for ref in entry["references"]:
            ref_is.append(len(texts))
            texts.append(ref)
        index[clip_id] = (cand_i, ref_is)

    embs = emb.encode(texts)

    per_question: dict[str, float] = {}
    for clip_id, (cand_i, ref_is) in index.items():
        raw = max(_greedy_f1(embs[cand_i], embs[r]) for r in ref_is)
        # baseline rescaling, exactly as rescale_with_baseline=True does
        per_question[clip_id] = (raw - BASELINE_F) / (1 - BASELINE_F)

    mean = sum(per_question.values()) / len(per_question) if per_question else 0.0
    return {"mean_bertscore_f1": mean, "per_question": per_question}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score predictions with the challenge's BERTScore-F1.")
    p.add_argument("--truth", required=True, type=Path)
    p.add_argument("--predictions", required=True, type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    truth = json.loads(a.truth.read_text())
    embedder = _Embedder()          # load roberta-large once for all files
    results = {}
    for pred_path in a.predictions:
        res = score_run(truth, json.loads(pred_path.read_text()), embedder)
        results[pred_path.name] = res
        print(f"\n=== {pred_path.name} ===", flush=True)
        print(f"mean_bertscore_f1: {res['mean_bertscore_f1']:.4f}", flush=True)
        for clip_id, s in sorted(res["per_question"].items()):
            print(f"  {clip_id}: {s:.4f}", flush=True)

    if len(results) > 1:
        print("\n=== RANKING (mean BERTScore-F1) ===", flush=True)
        for name, res in sorted(results.items(), key=lambda kv: -kv[1]["mean_bertscore_f1"]):
            print(f"  {res['mean_bertscore_f1']:.4f}  {name}", flush=True)

    if a.out:
        a.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

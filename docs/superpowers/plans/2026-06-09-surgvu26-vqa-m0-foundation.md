# SurgVU26 VQA — M0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the repo scaffold and a trusted local re-implementation of the SurgVU Category-2 BLEU metric, with a scoring CLI, so every later approach can be ranked against the public sample set.

**Architecture:** A small Python package `surgvu_vqa/` with an `eval` submodule implementing the exact challenge metric (NLTK sentence BLEU, uniform 1–4-gram weights + smoothing, max over the 5 reference answers, mean over all questions), a JSON schema for predictions/ground-truth, and a CLI. A `data/` helper downloads the public sample set and binds it into the truth schema. TDD throughout, with concrete fixtures including the real public example.

**Tech Stack:** Python ≥3.9 (system interpreter on this box is 3.9.25 — the only one available; all modules use `from __future__ import annotations` so PEP-604/generic annotations stay 3.9-compatible), `nltk` (BLEU), `pytest`. (Heavy deps — torch/transformers/peft/opencv — arrive in later plans.)

This plan implements the **M0 slice only** of `docs/superpowers/specs/2026-06-09-surgvu26-vqa-design.md` (spec §1 metric, §5 repo scaffold, §11 local BLEU harness). Training, container, and synthesis come in later plans.

---

## File structure (M0)

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, pytest config |
| `.gitignore` | Ignore caches, venv, downloaded data |
| `README.md` | One-paragraph orientation + pointer to the spec |
| `surgvu_vqa/__init__.py` | Package marker |
| `surgvu_vqa/eval/__init__.py` | Subpackage marker |
| `surgvu_vqa/eval/bleu.py` | `tokenize`, `question_bleu` — the per-question metric |
| `surgvu_vqa/eval/score.py` | `mean_bleu`, `score_run`, CLI `main` |
| `surgvu_vqa/data/__init__.py` | Subpackage marker |
| `surgvu_vqa/data/public_samples.py` | `build_truth_json` — bind sample set → truth schema |
| `data/download_samples.sh` | Fetch + unzip the public Cat-2 sample set |
| `tests/eval/test_bleu.py` | Tests for the metric |
| `tests/eval/test_score.py` | Tests for aggregation + CLI |
| `tests/data/test_public_samples.py` | Tests for the sample-set loader |

**Data schemas (used throughout):**
- `truth.json`: `{ "<clip_id>": {"question": str, "references": [str, ...]} }`
- `predictions.json`: `{ "<clip_id>": "<answer string>" }` (missing entries score 0.0)

---

## Task 1: Repo scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`
- Create: `surgvu_vqa/__init__.py`, `surgvu_vqa/eval/__init__.py`, `surgvu_vqa/data/__init__.py`
- Create: `tests/__init__.py`, `tests/eval/__init__.py`, `tests/data/__init__.py`, `tests/test_smoke.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "surgvu-vqa"
version = "0.1.0"
description = "SurgVU 2026 Category 2 (Surgical VQA) challenge solution"
requires-python = ">=3.9"
dependencies = ["nltk>=3.8"]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["surgvu_vqa*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
.pytest_cache/
.venv/
*.egg-info/
data/raw/
*.zip
```

- [ ] **Step 3: Create `README.md`**

```markdown
# surgvu-vqa

Solution for SurgVU 2026 Category 2 (Surgical Visual Question Answering).
Design spec: `docs/superpowers/specs/2026-06-09-surgvu26-vqa-design.md`.

## Dev setup
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest
```

- [ ] **Step 4: Create the package and test marker files**

Create each of these as an **empty** file:
`surgvu_vqa/__init__.py`, `surgvu_vqa/eval/__init__.py`, `surgvu_vqa/data/__init__.py`,
`tests/__init__.py`, `tests/eval/__init__.py`, `tests/data/__init__.py`

- [ ] **Step 5: Create `tests/test_smoke.py`**

```python
def test_package_imports():
    import surgvu_vqa  # noqa: F401
```

- [ ] **Step 6: Install and run the smoke test**

Run:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_smoke.py -v
```
Expected: PASS (1 passed).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore README.md surgvu_vqa tests
git commit -m "chore: scaffold surgvu-vqa package + pytest"
```

---

## Task 2: Per-question BLEU metric

**Files:**
- Create: `surgvu_vqa/eval/bleu.py`
- Test: `tests/eval/test_bleu.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_bleu.py
from surgvu_vqa.eval.bleu import question_bleu, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("A Large  Needle") == ["a", "large", "needle"]


def test_exact_match_scores_one():
    score = question_bleu(
        "A large needle driver was not used.",
        ["A large needle driver was not used."],
    )
    assert score == 1.0


def test_max_is_taken_over_references():
    refs = [
        "No",
        "No, a large needle driver was not used",
        "A large needle driver was not used.",
    ]
    score = question_bleu("No", refs)
    assert 0.0 < score <= 1.0
    # Choosing the best reference must beat scoring against only the long one.
    long_only = question_bleu("No", ["A large needle driver was not used."])
    assert score >= long_only


def test_empty_prediction_scores_zero():
    assert question_bleu("", ["No"]) == 0.0


def test_no_references_scores_zero():
    assert question_bleu("No", []) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_bleu.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'surgvu_vqa.eval.bleu'`).

- [ ] **Step 3: Write minimal implementation**

```python
# surgvu_vqa/eval/bleu.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_bleu.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add surgvu_vqa/eval/bleu.py tests/eval/test_bleu.py
git commit -m "feat: per-question SurgVU BLEU metric"
```

---

## Task 3: Dataset aggregation + scoring

**Files:**
- Create: `surgvu_vqa/eval/score.py`
- Test: `tests/eval/test_score.py` (aggregation cases only; CLI added in Task 4)

- [ ] **Step 1: Write the failing test**

> **Fixture note (discovered in Task 2 review):** BLEU-4 with method1 smoothing on a **1-token** exact match (e.g. `"No"` vs `"No"`) scores ≈0.178, NOT 1.0 — higher-order n-grams cannot fire. All fixtures that assert a score of exactly 1.0 must therefore use **4+-token sentences**.

```python
# tests/eval/test_score.py
from surgvu_vqa.eval.score import mean_bleu, score_run

_A = "A stapler was not used."
_B = "The suturing step is shown."


def test_mean_bleu_averages_questions():
    items = [
        {"prediction": _A, "references": [_A]},
        {"prediction": _B, "references": [_B]},
    ]
    assert mean_bleu(items) == 1.0


def test_mean_bleu_empty_is_zero():
    assert mean_bleu([]) == 0.0


def test_score_run_handles_missing_prediction():
    truth = {
        "clip_0": {"question": "Q?", "references": [_A]},
        "clip_1": {"question": "Q?", "references": [_B]},
    }
    predictions = {"clip_0": _A}  # clip_1 deliberately missing
    result = score_run(truth, predictions)
    assert result["per_question"]["clip_0"] == 1.0
    assert result["per_question"]["clip_1"] == 0.0
    assert result["mean_bleu"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_score.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'surgvu_vqa.eval.score'`).

- [ ] **Step 3: Write minimal implementation**

```python
# surgvu_vqa/eval/score.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_score.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add surgvu_vqa/eval/score.py tests/eval/test_score.py
git commit -m "feat: dataset BLEU aggregation + score_run"
```

---

## Task 4: Scoring CLI

**Files:**
- Modify: `surgvu_vqa/eval/score.py` (already has `main`; this task tests it end-to-end)
- Test: `tests/eval/test_score.py` (append the CLI test)

- [ ] **Step 1: Write the failing test (append to `tests/eval/test_score.py`)**

```python
import json

from surgvu_vqa.eval.score import main


def test_cli_prints_mean(tmp_path, capsys):
    # 4+-token answer: 1-token fixtures cannot reach BLEU 1.0 (see fixture note above).
    answer = "A stapler was not used."
    truth = {"clip_0": {"question": "Q?", "references": [answer]}}
    predictions = {"clip_0": answer}
    t = tmp_path / "truth.json"
    p = tmp_path / "pred.json"
    t.write_text(json.dumps(truth))
    p.write_text(json.dumps(predictions))

    rc = main(["--truth", str(t), "--predictions", str(p)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "mean_bleu: 1.0000" in out
    assert "clip_0: 1.0000" in out
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `pytest tests/eval/test_score.py::test_cli_prints_mean -v`
Expected: PASS (the `main` from Task 3 already satisfies it). If it FAILS, fix `main` to match the printed format `mean_bleu: <4dp>` and `  <clip_id>: <4dp>` before continuing.

- [ ] **Step 3: Verify the CLI by hand**

Run:
```bash
python -m surgvu_vqa.eval.score --truth /tmp/t.json --predictions /tmp/p.json 2>/dev/null || true
```
(Skip if no scratch files; the pytest case is the source of truth.)

- [ ] **Step 4: Commit**

```bash
git add tests/eval/test_score.py
git commit -m "test: CLI scoring end-to-end"
```

---

## Task 5: Public sample-set download script

**Files:**
- Create: `data/download_samples.sh`

This is a network utility (no unit test). The URL is the public GCS bucket — no registration required.

- [ ] **Step 1: Create `data/download_samples.sh`**

```bash
#!/usr/bin/env bash
# Download the public SurgVU Category-2 VQA sample set (10 clips, 5 references each).
# Public GCS bucket — no registration required.
set -euo pipefail

DEST="$(cd "$(dirname "$0")" && pwd)/raw"
URL="https://storage.googleapis.com/isi-surgvu/SURGVU25_cat_2_sample_set_public.zip"

mkdir -p "$DEST"
echo "Downloading public Cat-2 sample set..."
curl -fL -o "$DEST/cat2_samples.zip" "$URL"
echo "Unzipping..."
unzip -o "$DEST/cat2_samples.zip" -d "$DEST/cat2_samples"
echo "Done. Top-level contents:"
find "$DEST/cat2_samples" -maxdepth 2 | head -50
```

- [ ] **Step 2: Make it executable and run it**

Run:
```bash
chmod +x data/download_samples.sh
./data/download_samples.sh
```
Expected: a `data/raw/cat2_samples/` directory containing 10 video clips and an annotations file (CSV or JSON) listing each clip's question and its reference answers. **Record the exact annotation filename and field structure** printed by the `find` — Task 6 binds to it.

- [ ] **Step 3: Commit (script only; `data/raw/` is gitignored)**

```bash
git add data/download_samples.sh
git commit -m "feat: public Cat-2 sample-set download script"
```

---

## Task 6: Bind the public samples into the truth schema

**Files:**
- Create: `surgvu_vqa/data/public_samples.py`
- Test: `tests/data/test_public_samples.py`

> **Inspect-then-bind.** The exact field names in the public annotations file must be confirmed against the real download from Task 5. The loader centralizes field names as constants at the top of the module — if the real file uses different keys, change only those constants. The unit test pins the *shape* via a synthetic fixture so the parser logic stays verified regardless.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_public_samples.py
import json

from surgvu_vqa.data.public_samples import build_truth_json


def test_build_truth_json_from_list(tmp_path):
    annotations = [
        {"clip_id": "0", "question": "Was a stapler used?",
         "answers": ["No", "No, a stapler was not used", "A stapler was not used."]},
        {"clip_id": "1", "question": "What step is shown?",
         "answers": ["Suturing", "The suturing step.", "Suturing is being performed."]},
    ]
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotations))

    truth = build_truth_json(path)

    assert set(truth) == {"0", "1"}
    assert truth["0"]["question"] == "Was a stapler used?"
    assert truth["0"]["references"] == ["No", "No, a stapler was not used", "A stapler was not used."]
    assert truth["1"]["references"][0] == "Suturing"


def test_build_truth_json_wrapped_in_object(tmp_path):
    annotations = {"annotations": [
        {"clip_id": 5, "question": "Q?", "answers": ["Yes"]},
    ]}
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotations))

    truth = build_truth_json(path)

    assert truth["5"] == {"question": "Q?", "references": ["Yes"]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_public_samples.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'surgvu_vqa.data.public_samples'`).

- [ ] **Step 3: Write minimal implementation**

```python
# surgvu_vqa/data/public_samples.py
"""Bind the public Cat-2 sample set into our truth.json schema.

Verify these field-name constants against the real annotations file produced
by `data/download_samples.sh` (Task 5). If the real file uses different keys
(or is a CSV rather than JSON), adjust ONLY the constants / the `_records`
reader below — the public interface and tests stay the same.
"""
from __future__ import annotations

import json
from pathlib import Path

CLIP_ID_FIELD = "clip_id"
QUESTION_FIELD = "question"
ANSWERS_FIELD = "answers"


def _records(raw) -> list[dict]:
    """Accept either a top-level list or an object wrapping `annotations`."""
    if isinstance(raw, list):
        return raw
    return raw.get("annotations", [])


def build_truth_json(annotations_path: Path) -> dict:
    """Map the public annotations file → {clip_id: {question, references}}."""
    raw = json.loads(Path(annotations_path).read_text())
    truth: dict[str, dict] = {}
    for rec in _records(raw):
        truth[str(rec[CLIP_ID_FIELD])] = {
            "question": rec[QUESTION_FIELD],
            "references": list(rec[ANSWERS_FIELD]),
        }
    return truth
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_public_samples.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Verify against the REAL download**

Run (after Task 5's download):
```bash
python -c "from pathlib import Path; from surgvu_vqa.data.public_samples import build_truth_json; \
import json; \
t = build_truth_json(Path('data/raw/cat2_samples/<ANNOTATIONS_FILE>')); \
print('clips:', len(t)); print(json.dumps(next(iter(t.values())), indent=2))"
```
Replace `<ANNOTATIONS_FILE>` with the real filename from Task 5. Expected: `clips: 10`, and each entry showing a question + a list of reference answers. **If the keys differ, update the constants in `public_samples.py` and re-run.**

- [ ] **Step 6: Commit**

```bash
git add surgvu_vqa/data/public_samples.py tests/data/test_public_samples.py
git commit -m "feat: bind public sample set into truth schema"
```

---

## Task 7: Full-suite green + smoke-score the real samples

**Files:** none (verification task)

- [ ] **Step 1: Run the whole test suite**

Run: `pytest -v`
Expected: all tests PASS.

- [ ] **Step 2: Produce a truth.json + a dummy prediction and score it**

Run:
```bash
python - <<'PY'
import json
from pathlib import Path
from surgvu_vqa.data.public_samples import build_truth_json
ann = next(Path("data/raw/cat2_samples").rglob("*.json"))  # adjust if CSV
truth = build_truth_json(ann)
Path("data/raw/truth.json").write_text(json.dumps(truth, indent=2))
# Dummy baseline: answer "No" to everything, to confirm the harness runs end-to-end.
preds = {k: "No" for k in truth}
Path("data/raw/preds_dummy.json").write_text(json.dumps(preds))
print("wrote", len(truth), "clips")
PY
python -m surgvu_vqa.eval.score --truth data/raw/truth.json --predictions data/raw/preds_dummy.json
```
Expected: prints a `mean_bleu:` line and a per-clip score for all 10 clips. (The absolute number is meaningless for a dummy baseline — this only confirms the harness runs against real data.)

- [ ] **Step 3: Commit any constant adjustments made during verification**

```bash
git add -A
git commit -m "chore: verify BLEU harness against public samples" || echo "nothing to commit"
```

---

## Self-Review

**1. Spec coverage (M0 slice):** repo scaffold (spec §5) → Task 1 ✓; BLEU metric exactly as spec §1/§11 (uniform 0.25×4 weights, smoothing, max-over-5-refs, mean-over-questions) → Tasks 2–3 ✓; local harness usable on the public samples (spec §11) → Tasks 5–7 ✓. Training/container/synthesis are explicitly out of scope for M0 and deferred to later plans.

**2. Placeholder scan:** no TBD/TODO; every code step has complete code. The one observe-and-bind point (Task 6 field-name constants) is a concrete, tested parser with an explicit verification step against real data — not a placeholder.

**3. Type consistency:** `tokenize` / `question_bleu` (bleu.py) → consumed by `mean_bleu` / `score_run` / `main` (score.py) → consistent signatures. Truth schema `{clip_id: {question, references}}` and predictions `{clip_id: str}` are identical in `score_run`, the CLI, `build_truth_json`, and Task 7. `references` (not `answers`) is the internal field everywhere after binding; `answers` is only the raw public-file key, mapped once in `build_truth_json`.

**Known limitation (carried to a later plan):** the official evaluator's exact tokenizer/smoothing method are unpublished, so absolute BLEU may differ from the leaderboard; the harness is for relative ranking. Reconcile with the official evaluation container when SurgVU 2026 publishes it.

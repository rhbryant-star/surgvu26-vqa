# surgvu-vqa

Solution for SurgVU 2026 Category 2 (Surgical Visual Question Answering).
Design spec: `docs/superpowers/specs/2026-06-09-surgvu26-vqa-design.md`.

## Dev setup
    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest

## Local BLEU harness
    ./data/download_samples.sh   # fetch the public Cat-2 sample set (11 clips, 5 refs each)
    python -m surgvu_vqa.eval.score --truth data/raw/truth.json --predictions <preds.json>

`truth.json` is built from the download via `surgvu_vqa.data.public_samples.build_truth_json`;
predictions are `{"<clip_id>": "<answer string>"}`.

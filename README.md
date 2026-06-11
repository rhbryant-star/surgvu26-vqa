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

## Container (Grand Challenge submission)
    # build: GitHub Actions → ghcr.io/rhbryant-star/surgvu26-vqa-cat2 (no local docker needed)
    gh workflow run build-container --ref <branch>   # build + push image + SIF artifact
    ./hpc/fetch_sif.sh <ci-run-id>                   # CI SIF artifact → group staging
    ./hpc/fetch_weights.sh                           # one-time: AWQ weights (lm_head-patched) + model tarball + sample mirror
    cd hpc && condor_submit rehearse_container.sub   # GC-equivalent rehearsal (offline, VRAM check)
    cd hpc && condor_submit eval_samples.sub         # 11-clip predictions → score locally

GC upload artifacts: trigger `build-container` with `save_tarball=true` → image
tar.gz artifact; model tarball at
`/staging/groups/bhaskar_opscribe/surgvu26/tarballs/qwen25vl7b-awq-model.tar.gz`.
Scores: see `docs/BASELINES.md`.

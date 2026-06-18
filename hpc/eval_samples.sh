#!/usr/bin/env bash
# Run scripts/predict_samples.py inside the SIF over the staged sample clips.
# Binds the repo's surgvu_vqa over the baked copy so prompt iterations don't
# need an image rebuild (transfer_input_files brings the current repo copy).
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26
# Which AWQ tarball to evaluate. Default = the shipped v3 baseline; override with
# MODEL_TARBALL (e.g. the v2 self-quantized model) to eval a fine-tuned variant.
MODEL_TARBALL="${MODEL_TARBALL:-$STG/tarballs/qwen25vl7b-awq-model.tar.gz}"
OUT_JSON="${OUT_JSON:-/work/predictions.json}"
echo "eval tarball: $MODEL_TARBALL"

mkdir -p model
tar -xzf "$MODEL_TARBALL" -C model

apptainer exec --nv --containall \
  --bind "$STG/cat2_samples:/samples:ro,$(pwd)/model:/opt/ml/model:ro,$(pwd)/surgvu_vqa:/opt/app/surgvu_vqa:ro,$(pwd):/work" \
  --pwd /work \
  "$STG/surgvu26-vqa-cat2-current.sif" \
  python /work/predict_samples.py --samples-dir /samples --out "$OUT_JSON"

rm -rf model
echo "===== $OUT_JSON ====="
cat "$OUT_JSON"

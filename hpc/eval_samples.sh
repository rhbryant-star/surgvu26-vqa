#!/usr/bin/env bash
# Run scripts/predict_samples.py inside the SIF over the staged sample clips.
# Binds the repo's surgvu_vqa over the baked copy so prompt iterations don't
# need an image rebuild (transfer_input_files brings the current repo copy).
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26

mkdir -p model
tar -xzf "$STG/tarballs/qwen25vl7b-awq-model.tar.gz" -C model

apptainer exec --nv --containall \
  --bind "$STG/cat2_samples:/samples:ro,$(pwd)/model:/opt/ml/model:ro,$(pwd)/surgvu_vqa:/opt/app/surgvu_vqa:ro,$(pwd):/work" \
  --pwd /work \
  "$STG/surgvu26-vqa-cat2-current.sif" \
  python /work/predict_samples.py --samples-dir /samples --out /work/predictions.json

rm -rf model
echo "===== predictions.json ====="
cat predictions.json

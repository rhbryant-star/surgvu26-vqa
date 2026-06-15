#!/usr/bin/env bash
# Build one case's VQA training shard on a CHTC execute node.
# Runs build_case_dataset.py INSIDE the inference SIF (has cv2/pillow/numpy),
# binding the CURRENT surgvu_vqa over the baked (stale M1) copy, and
# pip-installing pure-python remotezip into scratch (noexec-safe: no .so).
# The job streams only this case's video parts over HTTPS (needs internet).
set -euo pipefail
CASE="$1"
STG=/staging/groups/bhaskar_opscribe/surgvu26
SIF="$STG/surgvu26-vqa-cat2-current.sif"

# Extract just this case's label CSVs from the staging zip.
mkdir -p labels_root
unzip -oq "$STG/labels/surgvu24_labels_updated_v2.zip" "labels/$CASE/*" -d labels_root

apptainer exec --containall --env CASE="$CASE" \
  --bind "$(pwd):/work,$(pwd)/surgvu_vqa:/opt/app/surgvu_vqa:ro" \
  --pwd /work \
  "$SIF" bash -lc '
    set -euo pipefail
    pip install --quiet --target=/work/pp remotezip 2>&1 | tail -1 || true
    export PYTHONPATH=/work/pp:/opt/app:/work
    python /work/build_case_dataset.py \
      --case-id "$CASE" \
      --labels-dir /work/labels_root/labels/"$CASE" \
      --out-dir /work/out \
      --seed 42
  '

mkdir -p "$STG/datasets/surgvu_vqa_v1"
cp "out/${CASE}_frames.tar" "out/${CASE}_qa.jsonl" "$STG/datasets/surgvu_vqa_v1/"
echo "done ${CASE}: $(wc -l < "out/${CASE}_qa.jsonl") QA pairs, frames tar $(du -h "out/${CASE}_frames.tar" | cut -f1)"

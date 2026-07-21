#!/usr/bin/env bash
# Outer wrapper: apptainer-exec the dedicated training SIF with GPUs, binding
# group staging (base model + data + adapter output) and job scratch as /work.
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26
GROUP=/staging/groups/bhaskar_opscribe
SIF="$STG/surgvu26-train.sif"

apptainer exec --nv \
  --bind "$GROUP:$GROUP,$(pwd):/work" \
  --pwd /work \
  "$SIF" bash /work/finetune_inner.sh

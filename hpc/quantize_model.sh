#!/usr/bin/env bash
# Merge the fine-tuned LoRA into the bf16 Qwen2.5-VL-7B base, AutoAWQ-quantize the
# result to 4-bit, patch lm_head, and tar it to the SAME artifact shape the
# inference container loads (top-level dir = qwen2.5-vl-7b-awq, so the container's
# TARBALL_MODEL_DIR works unchanged against v0).
#
# Runs inside surgvu26-quant.sif (the DEDICATED quant stack: torch 2.5.1 +
# transformers 4.51.3 + autoawq 0.2.9 + autoawq-kernels 0.0.9 + peft 0.15.2 —
# the proven prod combo; transformers >=4.52 is FORBIDDEN). NOT the train SIF:
# that one has transformers 4.50.3 + LLaMA-Factory and lacks the AWQ kernels.
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26
GROUP=/staging/groups/bhaskar_opscribe
SIF="$STG/surgvu26-quant.sif"

# Resolve the cached base-model snapshot the same way the train/eval jobs do —
# the snapshot dir name is a content hash, so glob for it rather than hard-code.
BASE=$(find "$GROUP/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots" \
  -maxdepth 1 -mindepth 1 -type d | head -1)
# Defaults target the v2 (echo-free) adapter + v2 calib corpus; override via env
# (ADAPTER/CALIB/MODEL_TAG) to quantize a different adapter, e.g. surgvu_vqa_v1.
ADAPTER="${ADAPTER:-$GROUP/adapters/surgvu_vqa_v2}"          # PEFT LoRA to merge
CALIB="${CALIB:-$STG/datasets/surgvu_vqa_v1/lf_v2/train.jsonl}"  # text-only calib (match adapter's style!)
MODEL_TAG="${MODEL_TAG:-surgvu-v2}"
# AWQ model dir -> JOB SCRATCH (/work), NOT staging: group staging sits near its
# 10k-file ceph quota and only the single tarball needs to persist (the ~15-file
# model dir would otherwise blow the file quota — observed: save_quantized mkdir
# failed with Errno 122). The host sees this same dir at $(pwd)/awq_out (bind),
# which the post-exec tar reads.
OUT="/work/awq_out"
OUT_HOST="$(pwd)/awq_out"
TARBALL="$STG/tarballs/qwen25vl7b-${MODEL_TAG}-awq-model.tar.gz"

# Merge to JOB SCRATCH bound at /work (NOT host $(pwd): this script is the condor
# executable and runs on the bare host BEFORE apptainer exec, so $(pwd) would be a
# host path the container can't see). /work is the exec-dir scratch bound below;
# the ~16GB merged bf16 is throwaway intermediate.
MERGED="/work/merged_bf16"

echo "base snapshot : $BASE"
echo "adapter       : $ADAPTER"
echo "calib jsonl   : $CALIB"
echo "merged (scratch): $MERGED"
echo "out (scratch) : $OUT  (host: $OUT_HOST)"
echo "tarball (staging): $TARBALL"

mkdir -p "$STG/tarballs"
# Clear any prior scratch output (fresh exec dir, but be explicit).
rm -rf "$OUT_HOST"

# apptainer exec the quant SIF with GPUs:
#   --nv             : bind the host CUDA driver libs (H200)
#   --bind GROUP     : base weights + adapter + hf_cache + staging out all live here
#   --bind cwd:/work : the transferred merge_and_quantize.py + scratch merged dir
#   --env offline    : HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE so
#                      no from_pretrained or load_dataset call ever reaches the hub
#   PYTHONPATH=/work : let the script import any sibling helpers if added later
#   expandable_segments: tame fragmentation during the bf16 merge -> fp16 quant
apptainer exec --nv \
  --bind "$GROUP:$GROUP,$(pwd):/work" \
  --pwd /work \
  --env "HF_HOME=$GROUP/hf_cache,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_DATASETS_OFFLINE=1,PYTHONPATH=/work,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  "$SIF" python /work/merge_and_quantize.py \
    --base "$BASE" \
    --adapter "$ADAPTER" \
    --merged-dir "$MERGED" \
    --out-dir "$OUT" \
    --calib-jsonl "$CALIB" \
    --max-calib-samples 128 \
    --max-calib-seq-len 512

# Build the deployable tarball. -C "$(dirname "$OUT")" + a renamed transform so
# the archive's single top-level dir is 'qwen2.5-vl-7b-awq' (matching v0), even
# though the on-disk dir is qwen25vl7b-surgvu-v1-awq. --transform rewrites the
# leading path component during archiving.
# tar runs on the HOST after apptainer exec, so read the scratch dir at its HOST
# path ($(pwd)/awq_out). --transform rewrites the archive root to qwen2.5-vl-7b-awq
# (matching v0 so the container's TARBALL_MODEL_DIR works unchanged).
echo "Building model tarball (top-level dir = qwen2.5-vl-7b-awq)..."
tar -czf "$TARBALL" \
  -C "$(pwd)" \
  --transform "s,^awq_out,qwen2.5-vl-7b-awq," \
  awq_out

echo "=== tarball ==="
ls -lh "$TARBALL"
# Verify the archive root is exactly qwen2.5-vl-7b-awq/ (catch a bad --transform).
echo "=== archive top-level entries ==="
tar -tzf "$TARBALL" | awk -F/ '{print $1}' | sort -u
echo "Done."

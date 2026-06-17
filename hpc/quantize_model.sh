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
OUT="$STG/models/qwen25vl7b-${MODEL_TAG}-awq"               # AWQ model lands here on staging
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
echo "out (staging) : $OUT"

mkdir -p "$STG/models" "$STG/tarballs"
# Clear any prior AWQ output: save_quantized overwrites model.safetensors but NOT
# orphaned shards from a run with a different shard count — those would bloat the
# tarball and confuse the index. Start clean.
rm -rf "$OUT"

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
echo "Building model tarball (top-level dir = qwen2.5-vl-7b-awq)..."
tar -czf "$TARBALL" \
  -C "$(dirname "$OUT")" \
  --transform "s,^$(basename "$OUT"),qwen2.5-vl-7b-awq," \
  "$(basename "$OUT")"

echo "=== tarball ==="
ls -lh "$TARBALL"
# Verify the archive root is exactly qwen2.5-vl-7b-awq/ (catch a bad --transform).
echo "=== archive top-level entries ==="
tar -tzf "$TARBALL" | awk -F/ '{print $1}' | sort -u
echo "Done."

#!/usr/bin/env bash
# bf16 preview eval (the plan's "bridge path A"): run predict_samples.py over
# the 11 public clips with the FINE-TUNED LoRA adapter on the bf16 base, and
# again with the bf16 base alone as a clean control. Runs inside the training
# SIF (has transformers 4.50.3 + peft 0.15.2 + the base weights). Predictions
# are scored offline on the submit node against data/raw/truth.json.
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26
GROUP=/staging/groups/bhaskar_opscribe
SIF="$STG/surgvu26-train.sif"
BASE=$(find "$GROUP/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)
ADAPTER="$GROUP/adapters/surgvu_vqa_v2"
echo "base snapshot: $BASE"
echo "adapter:       $ADAPTER"

run_cfg () {  # $1=output name  $2=adapter dir ("" for base control)
  local name="$1" adapter="$2"
  echo "===== eval config: ${name} (adapter='${adapter:-<none>}') ====="
  apptainer exec --nv \
    --bind "${GROUP}:${GROUP},${STG}/cat2_samples:/samples:ro,$(pwd):/work" \
    --pwd /work \
    --env "SURGVU_MODEL_DIR=${BASE},SURGVU_ADAPTER_DIR=${adapter},SURGVU_DTYPE=bfloat16,HF_HOME=${GROUP}/hf_cache,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,PYTHONPATH=/work,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    "${SIF}" python /work/predict_samples.py --samples-dir /samples --out "/work/predictions_${name}.json"
}

run_cfg base_bf16 ""
run_cfg ft_v2_bf16 "${ADAPTER}"

echo "===== predictions_base_bf16.json ====="; cat predictions_base_bf16.json
echo "===== predictions_ft_v2_bf16.json ====="; cat predictions_ft_v2_bf16.json

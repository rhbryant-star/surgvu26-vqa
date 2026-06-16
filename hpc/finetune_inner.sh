#!/usr/bin/env bash
# Runs INSIDE surgvu26-train.sif (has the pinned LLaMA-Factory stack).
# Untars frame shards to scratch, rewrites image paths, registers the dataset,
# resolves the cached base-model snapshot, and launches LoRA SFT.
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26
GROUP=/staging/groups/bhaskar_opscribe
export HF_HOME="$GROUP/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /work

echo "=== untar frame shards -> scratch ==="
mkdir -p frames_root
for t in "$STG"/datasets/surgvu_vqa_v1/*_frames.tar; do tar -xf "$t" -C frames_root; done
echo "frames extracted: $(find frames_root -name '*.jpg' | wc -l)"

echo "=== rewrite LF jsonl image paths to scratch ==="
mkdir -p data
python3 - "$STG" "$(pwd)/frames_root" <<'PY'
import json, os, sys
stg, root = sys.argv[1], sys.argv[2]
for split in ("train", "val"):
    with open(f"data/{split}.jsonl", "w") as out:
        for line in open(f"{stg}/datasets/surgvu_vqa_v1/lf/{split}.jsonl"):
            r = json.loads(line)
            r["images"] = [os.path.join(root, p) for p in r["images"]]
            out.write(json.dumps(r) + "\n")
print("prepped train/val jsonl")
PY

cat > data/dataset_info.json <<EOF
{
  "surgvu_vqa_train": {"file_name": "$(pwd)/data/train.jsonl", "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}},
  "surgvu_vqa_val": {"file_name": "$(pwd)/data/val.jsonl", "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}}
}
EOF

SNAP=$(find "$GROUP/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)
echo "base model snapshot: $SNAP"
sed -i "s|^model_name_or_path:.*|model_name_or_path: ${SNAP}|" finetune_surgvu_vqa.yaml
echo "dataset_dir: $(pwd)/data" >> finetune_surgvu_vqa.yaml

# nvidia-smi isn't on PATH inside the SIF (--nv binds driver libs, not the binary).
# Count visible GPUs via torch, falling back to CUDA_VISIBLE_DEVICES.
NGPU=$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || true)
if [ -z "${NGPU}" ] || [ "${NGPU}" = "0" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
  else
    NGPU=1
  fi
fi
echo "=== launching LoRA SFT on ${NGPU} GPU(s) ==="
mkdir -p "$GROUP/adapters/surgvu_vqa_v1"
torchrun --nproc_per_node="${NGPU}" --master_port=29501 -m llamafactory.launcher finetune_surgvu_vqa.yaml
echo "=== done; adapter contents ==="
ls -la "$GROUP/adapters/surgvu_vqa_v1" | tail -8

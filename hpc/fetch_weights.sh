#!/usr/bin/env bash
# Download Qwen2.5-VL-7B-Instruct-AWQ to group staging and build the GC model
# tarball (extracts to /opt/ml/model/qwen2.5-vl-7b-awq at runtime).
# Run directly on the CHTC submit node. Also mirrors the public sample clips
# to staging for the eval job.
set -euo pipefail

export HF_HOME=/staging/groups/bhaskar_opscribe/hf_cache
STG=/staging/groups/bhaskar_opscribe/surgvu26
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$STG/models" "$STG/tarballs"

python3 -m pip install --user --quiet "huggingface_hub[cli]"
python3 -m huggingface_hub.commands.huggingface_cli download \
  Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
  --local-dir "$STG/models/qwen2.5-vl-7b-awq"

echo "Building model tarball (top-level dir = qwen2.5-vl-7b-awq)..."
tar -czf "$STG/tarballs/qwen25vl7b-awq-model.tar.gz" -C "$STG/models" qwen2.5-vl-7b-awq
ls -lh "$STG/tarballs/qwen25vl7b-awq-model.tar.gz"

echo "Mirroring public sample clips to staging for the eval job..."
rsync -a "$REPO/data/raw/cat2_samples/" "$STG/cat2_samples/" --exclude "__MACOSX"
echo "Done."
du -sh "$STG/models/qwen2.5-vl-7b-awq" "$STG/cat2_samples"

#!/usr/bin/env bash
# Drive the v3 build: wait for the dataset fan-out, merge to lf_v3, launch the
# LoRA retrain. Quantize + eval are submitted separately once training lands
# (they need the finished adapter).
#
# Usage: run_v3_pipeline.sh <fanout_cluster_id>
set -euo pipefail
CLUSTER="${1:?usage: run_v3_pipeline.sh <fanout_cluster_id>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STG=/staging/groups/bhaskar_opscribe/surgvu26
DS="$STG/datasets/surgvu_vqa_v1"

echo "=== waiting for fan-out cluster $CLUSTER ==="
while true; do
  n=$(condor_q "$CLUSTER" -af:j ClusterId 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && break
  sleep 120
done
echo "fan-out done"

# Every shard must carry the v3 question types, else the merge would silently
# build a v2-style corpus and we'd burn 13 h of GPU on the wrong data.
echo "=== verifying shards carry v3 question types ==="
python3 - "$DS" <<'PY'
import json, sys, glob, collections
ds = sys.argv[1]
kinds = collections.Counter()
files = sorted(glob.glob(f"{ds}/*_qa.jsonl"))
for f in files:
    for line in open(f):
        line = line.strip()
        if line:
            kinds[json.loads(line)["kind"]] += 1
print(f"{len(files)} shards, {sum(kinds.values())} QA pairs")
for k, v in kinds.most_common():
    print(f"  {k:14} {v}")
missing = {"organ_what", "tool_purpose", "proc_type"} - set(kinds)
if missing:
    sys.exit(f"ERROR: shards are missing v3 question types {missing} — "
             "the fan-out ran stale code; do NOT train on this.")
print("v3 question types present OK")
PY

echo "=== merging -> lf_v3 ==="
mkdir -p "$DS/lf_v3"
cd "$REPO"
PYTHONPATH="$REPO" python3 scripts/merge_qa_dataset.py \
  --shards-dir "$DS" --out-dir "$DS/lf_v3" \
  --train-cases hpc/cases_train.txt --val-cases hpc/cases_val.txt
wc -l "$DS/lf_v3/train.jsonl" "$DS/lf_v3/val.jsonl"

echo "=== submitting LoRA retrain (lf_v3 -> surgvu_vqa_v3) ==="
cd "$REPO/hpc"
condor_submit finetune_vqa.sub

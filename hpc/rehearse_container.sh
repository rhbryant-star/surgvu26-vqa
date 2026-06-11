#!/usr/bin/env bash
# GC-equivalent rehearsal: real weights, one real case, offline, /input + /output
# + /opt/ml/model mounted exactly as Grand Challenge does.
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26
CASE="${1:-case122}"

mkdir -p input output model
echo "Extracting model tarball (as GC does at runtime)..."
tar -xzf "$STG/tarballs/qwen25vl7b-awq-model.tar.gz" -C model

cp "$STG/cat2_samples/$CASE/$CASE.mp4" input/endoscopic-robotic-surgery-video.mp4
cp "$STG/cat2_samples/$CASE/${CASE}_question.json" input/visual-context-question.json
cat > input/inputs.json <<'EOF'
[
  {"interface": {"slug": "endoscopic-robotic-surgery-video", "kind": "MP4 file",
   "super_kind": "File", "relative_path": "endoscopic-robotic-surgery-video.mp4"}},
  {"interface": {"slug": "visual-context-question", "kind": "String",
   "super_kind": "Value", "relative_path": "visual-context-question.json"}}
]
EOF

NETFLAGS=""
if apptainer exec --net --network none "$STG/surgvu26-vqa-cat2-current.sif" true 2>/dev/null; then
  NETFLAGS="--net --network none"
  echo "Network isolation: ENABLED (--network none)"
else
  echo "Network isolation: unavailable unprivileged; relying on TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE (baked into image)"
fi

# AWQ kernel check: without awq_ext, autoawq silently uses a pure-torch
# dequant path that is several-fold slower — tolerable on H200, fatal on T4.
if apptainer exec --nv "$STG/surgvu26-vqa-cat2-current.sif" python -c "import awq_ext" 2>/dev/null; then
  echo "awq kernels OK"
else
  echo "WARNING: awq_ext missing — pure-torch fallback (slow); investigate before T4 submission"
fi

# During iteration the repo's surgvu_vqa (transferred by the .sub) is bound
# over the baked copy so code fixes don't wait on an image rebuild. For the
# FINAL pre-submission rehearsal, comment the surgvu_vqa bind out to test the
# pure image exactly as Grand Challenge will run it.
CODE_BIND=""
[ -d surgvu_vqa ] && CODE_BIND=",$(pwd)/surgvu_vqa:/opt/app/surgvu_vqa:ro"

# (/usr/bin/time is absent on the EL9 nodes; inference.py prints its own
# load/generate timings and CUDA peak, which is everything Step 3 consumes.)
apptainer run --nv --containall $NETFLAGS \
  --bind "$(pwd)/input:/input:ro,$(pwd)/output:/output,$(pwd)/model:/opt/ml/model:ro$CODE_BIND" \
  "$STG/surgvu26-vqa-cat2-current.sif" 2>&1 | tee rehearsal_log.txt

echo "===== RESULT ====="
cat output/visual-context-response.json
echo
python3 - <<'PY'
import json, re
out = json.load(open("output/visual-context-response.json"))
assert isinstance(out, str) and out.strip(), f"bad output: {out!r}"
log = open("rehearsal_log.txt").read()
m = re.search(r"CUDA peak memory: ([0-9.]+) GiB", log)
peak = float(m.group(1)) if m else -1
print(f"answer={out!r}  cuda_peak={peak} GiB")
assert peak > 0, "no CUDA peak reported - model did not run on GPU"
assert peak < 14.0, f"T4 BUDGET EXCEEDED: peak {peak} GiB >= 14 GiB"
print("T4 VRAM budget check: OK")
PY
rm -rf model  # don't transfer 7GB back

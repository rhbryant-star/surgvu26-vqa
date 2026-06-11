#!/usr/bin/env bash
# Stage a REAL-input fixture dir (case122) with canonical GC filenames for
# in-container smoke runs. Entirely gitignored: challenge data never committed.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/data/raw/cat2_samples/case122"
DEST="$REPO/container/test/input/interf_real"
mkdir -p "$DEST"
cp "$SRC/case122.mp4" "$DEST/endoscopic-robotic-surgery-video.mp4"
cp "$SRC/case122_question.json" "$DEST/visual-context-question.json"
cp "$REPO/container/test/input/interf0/inputs.json" "$DEST/inputs.json"
echo "Real fixture staged at $DEST (gitignored)."

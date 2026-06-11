#!/usr/bin/env bash
# Download the public SurgVU Category-2 VQA sample set (10 clips, 5 references each).
# Public GCS bucket — no registration required.
set -euo pipefail

DEST="$(cd "$(dirname "$0")" && pwd)/raw"
URL="https://storage.googleapis.com/isi-surgvu/SURGVU25_cat_2_sample_set_public.zip"

mkdir -p "$DEST"
echo "Downloading public Cat-2 sample set..."
curl -fL -o "$DEST/cat2_samples.zip" "$URL"
echo "Unzipping..."
unzip -o "$DEST/cat2_samples.zip" -d "$DEST/cat2_samples"
echo "Done. Top-level contents:"
find "$DEST/cat2_samples" -maxdepth 2 | head -50

#!/usr/bin/env bash
# Fetch the public label CSVs for local dev/tests (data/raw is gitignored).
set -euo pipefail
DEST="$(cd "$(dirname "$0")" && pwd)/raw"
mkdir -p "$DEST"
curl -fsSL -o "$DEST/labels.zip" "https://storage.googleapis.com/isi-surgvu/surgvu24_labels_updated_v2.zip"
unzip -oq "$DEST/labels.zip" -d "$DEST/labels_root"
echo "cases: $(ls "$DEST/labels_root/labels" | wc -l)"

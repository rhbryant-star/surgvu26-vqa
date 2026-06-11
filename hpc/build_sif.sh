#!/usr/bin/env bash
# Convert the CI-built GHCR image into a SIF in group staging (submit node).
# Scratch AND layer cache are forced to staging: /tmp has a 256 MB quota and
# $HOME has <3.5 GB headroom — the default cache (~/.apptainer*) would blow
# the home quota on a ~10 GB image pull.
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26
TAG="${1:-latest}"
mkdir -p "$STG/tmp" "$STG/apptainer_cache"
export APPTAINER_TMPDIR="$STG/tmp"
export APPTAINER_CACHEDIR="$STG/apptainer_cache"

apptainer build --force \
  "$STG/surgvu26-vqa-cat2-${TAG}.sif" \
  "docker://ghcr.io/rhbryant-star/surgvu26-vqa-cat2:${TAG}"
ln -sf "$STG/surgvu26-vqa-cat2-${TAG}.sif" "$STG/surgvu26-vqa-cat2-current.sif"
ls -lh "$STG"/surgvu26-vqa-cat2-*.sif

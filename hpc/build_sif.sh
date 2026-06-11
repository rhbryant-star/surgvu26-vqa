#!/usr/bin/env bash
# Convert the CI-built GHCR image into a SIF in group staging (submit node).
# Scratch AND layer cache are forced to PERSONAL staging: /tmp has a 256 MB
# quota, $HOME has <3.5 GB headroom, and the GROUP dir runs close to its
# 10,000-FILE CephFS quota (check: getfattr -n ceph.dir.rfiles on it), so the
# multi-file OCI layer cache must not land there. Only the single SIF file
# is written to the group dir.
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26
TAG="${1:-latest}"
mkdir -p /staging/r/rhbryant/apptainer_tmp /staging/r/rhbryant/apptainer_cache
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/staging/r/rhbryant/apptainer_tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/staging/r/rhbryant/apptainer_cache}"

apptainer build --force \
  "$STG/surgvu26-vqa-cat2-${TAG}.sif" \
  "docker://ghcr.io/rhbryant-star/surgvu26-vqa-cat2:${TAG}"
ln -sf "$STG/surgvu26-vqa-cat2-${TAG}.sif" "$STG/surgvu26-vqa-cat2-current.sif"
ls -lh "$STG"/surgvu26-vqa-cat2-*.sif

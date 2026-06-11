#!/usr/bin/env bash
# Convert the CI-built GHCR image into a SIF in group staging.
#
# MUST run as an HTCondor job (condor_submit build_sif.sub): unpacking the
# ~15 GB rootfs creates ~100k temp files, which exceeds EVERY quota reachable
# from the submit node (/tmp 256 MB; $HOME <3.5 GB headroom; personal staging
# 2,000 files; group staging near its 10,000-file ceph quota). Job scratch
# disk is local and quota-free. Only the single SIF file lands in staging.
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26
TAG="${1:-latest}"

SCRATCH="${_CONDOR_SCRATCH_DIR:?run via condor_submit build_sif.sub — see header}"
export APPTAINER_TMPDIR="$SCRATCH/aptmp"
export APPTAINER_CACHEDIR="$SCRATCH/apcache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# `pull`, not `build`: build's unprivileged path needs proot for root
# emulation, and proot is broken on these execute nodes (no user namespaces,
# seccomp/loader failures). pull converts OCI->SIF as the calling user with
# no proot involvement — sufficient since there is no %post to execute.
apptainer pull --force \
  "surgvu26-vqa-cat2-${TAG}.sif" \
  "docker://ghcr.io/rhbryant-star/surgvu26-vqa-cat2:${TAG}"

cp "surgvu26-vqa-cat2-${TAG}.sif" "$STG/"
ln -sf "$STG/surgvu26-vqa-cat2-${TAG}.sif" "$STG/surgvu26-vqa-cat2-current.sif"
ls -lh "$STG"/surgvu26-vqa-cat2-*.sif
rm -f "surgvu26-vqa-cat2-${TAG}.sif"  # don't transfer the SIF back home

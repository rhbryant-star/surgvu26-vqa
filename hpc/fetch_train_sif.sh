#!/usr/bin/env bash
# Download the CI-built training SIF artifact into group staging (submit node).
# Built AS ROOT in GitHub Actions because CHTC nodes can't do unprivileged
# OCI->SIF conversion (proot broken there).
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26
RUN_ID="${1:?usage: fetch_train_sif.sh <ci-run-id>}"
TMP="$STG/sif_download_train"
rm -rf "$TMP"; mkdir -p "$TMP" /staging/r/rhbryant/ghtmp
# gh buffers the artifact zip in $TMPDIR; /tmp has a 256 MB quota on the submit node.
TMPDIR=/staging/r/rhbryant/ghtmp gh run download "$RUN_ID" -R rhbryant-star/surgvu26-vqa \
  -n surgvu26-train-sif --dir "$TMP"
rm -rf /staging/r/rhbryant/ghtmp
mv -f "$TMP/surgvu26-train-latest.sif" "$STG/surgvu26-train.sif"
rm -rf "$TMP"
ls -lh "$STG/surgvu26-train.sif"

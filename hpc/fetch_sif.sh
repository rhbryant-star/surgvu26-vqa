#!/usr/bin/env bash
# Download the CI-built SIF artifact into group staging (submit node).
# The SIF is built AS ROOT in GitHub Actions because CHTC nodes cannot do the
# unprivileged OCI->SIF conversion (proot broken: seccomp/loader errors).
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26
RUN_ID="${1:?usage: fetch_sif.sh <ci-run-id>}"
TMP="$STG/sif_download"
rm -rf "$TMP"; mkdir -p "$TMP"
gh run download "$RUN_ID" -R rhbryant-star/surgvu26-vqa -n surgvu26-vqa-cat2-sif --dir "$TMP"
mv -f "$TMP/surgvu26-vqa-cat2-latest.sif" "$STG/surgvu26-vqa-cat2-latest.sif"
rm -rf "$TMP"
ln -sf "$STG/surgvu26-vqa-cat2-latest.sif" "$STG/surgvu26-vqa-cat2-current.sif"
ls -lh "$STG"/surgvu26-vqa-cat2-*.sif

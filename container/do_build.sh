#!/usr/bin/env bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-surgvu26-vqa-cat2}"

docker build \
  --platform=linux/amd64 \
  --file "$SCRIPT_DIR/Dockerfile" \
  --tag "$DOCKER_IMAGE_TAG" \
  "$REPO_ROOT" 2>&1

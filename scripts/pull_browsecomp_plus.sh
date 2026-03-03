#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/texttron/BrowseComp-Plus.git"
TARGET_DIR="${1:-third_party/BrowseComp-Plus}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -d "${TARGET_DIR}/.git" ]]; then
  echo "Updating ${TARGET_DIR} ..."
  git -C "${TARGET_DIR}" pull --ff-only
else
  echo "Cloning ${REPO_URL} into ${TARGET_DIR} ..."
  git clone --depth 1 "${REPO_URL}" "${TARGET_DIR}"
fi

mkdir -p "${PROJECT_ROOT}/benchmark/browsecomp/topics-qrels"
cp "${TARGET_DIR}/topics-qrels/qrel_evidence.txt" "${PROJECT_ROOT}/benchmark/browsecomp/topics-qrels/"
cp "${TARGET_DIR}/topics-qrels/qrel_golds.txt" "${PROJECT_ROOT}/benchmark/browsecomp/topics-qrels/"

echo "Synced qrels to ${PROJECT_ROOT}/benchmark/browsecomp/topics-qrels"

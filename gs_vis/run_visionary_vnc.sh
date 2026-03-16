#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_SRC="${MODEL_SRC:-${ROOT_DIR}/../da3/output_gs/gs_ply/0000.ply}"
GPU_DEVICE="${GPU_DEVICE:-0}"

if [[ ! -f "${MODEL_SRC}" ]]; then
  echo "Missing source GS model: ${MODEL_SRC}" >&2
  exit 1
fi

export MODEL_SRC
export GPU_DEVICE

cd "${ROOT_DIR}"
exec docker compose up --build

#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISIONARY_DIR="${ROOT_DIR}/visionary"
MODEL_SRC="${MODEL_SRC:-${ROOT_DIR}/../da3/output_gs/gs_ply/0000.ply}"
MODEL_DST="${VISIONARY_DIR}/models/current_output.ply"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-3000}"

if [[ ! -f "${MODEL_SRC}" ]]; then
  echo "Missing source GS model: ${MODEL_SRC}" >&2
  exit 1
fi

if [[ ! -x "/home/axgu/.bun/bin/bun" ]]; then
  echo "bun is not available at /home/axgu/.bun/bin/bun" >&2
  exit 1
fi

if [[ ! -d "${VISIONARY_DIR}/node_modules" ]]; then
  echo "Installing Visionary dependencies with bun..."
  (cd "${VISIONARY_DIR}" && /home/axgu/.bun/bin/bun install)
fi

mkdir -p "${VISIONARY_DIR}/models"
cp -f "${MODEL_SRC}" "${MODEL_DST}"

echo "Synced GS model:"
echo "  source: ${MODEL_SRC}"
echo "  target: ${MODEL_DST}"
echo
echo "Open after startup:"
echo "  http://localhost:${PORT}/demo/simple/current_output.html"
echo

cd "${VISIONARY_DIR}"
exec /home/axgu/.bun/bin/bun run dev --host "${HOST}" --port "${PORT}"

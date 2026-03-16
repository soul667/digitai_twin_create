#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISIONARY_DIR="${ROOT_DIR}/visionary"
MODEL_SRC="${MODEL_SRC:-${ROOT_DIR}/../da3/output_gs/gs_ply/0000.ply}"
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
  (cd "${VISIONARY_DIR}" && /home/axgu/.bun/bin/bun install --no-save)
fi

echo "3DGS model:"
echo "  ${MODEL_SRC}"
echo
echo "Open after startup:"
echo "  http://localhost:${PORT}/demo/simple/index.html"
echo
echo "Then click '选择文件' or drag the model into the page."
echo

cd "${VISIONARY_DIR}"
exec /home/axgu/.bun/bin/bun run dev --host "${HOST}" --port "${PORT}"

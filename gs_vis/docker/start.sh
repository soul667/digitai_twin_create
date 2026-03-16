#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VISIONARY_DIR="${VISIONARY_DIR:-${WORKSPACE}/gs_vis/visionary}"
MODEL_SRC="${MODEL_SRC:-${WORKSPACE}/da3/output_gs/gs_ply/0000.ply}"
VISIONARY_PORT="${VISIONARY_PORT:-3000}"
LOG_DIR="${HOME:-/home/ubuntu}/.cache/visionary"

mkdir -p "${LOG_DIR}"

if [[ ! -d "${VISIONARY_DIR}" ]]; then
  echo "Missing Visionary workspace: ${VISIONARY_DIR}" >> "${LOG_DIR}/autostart.log"
  exit 1
fi

if [[ ! -f "${MODEL_SRC}" ]]; then
  echo "Missing 3DGS PLY: ${MODEL_SRC}" >> "${LOG_DIR}/autostart.log"
  exit 1
fi

if ! command -v bun >/dev/null 2>&1; then
  export PATH="/root/.bun/bin:${PATH}"
fi

if [[ ! -d "${VISIONARY_DIR}/node_modules" ]]; then
  echo "[visionary] Installing frontend dependencies..." >> "${LOG_DIR}/autostart.log"
  (cd "${VISIONARY_DIR}" && bun install --no-save) >> "${LOG_DIR}/bun-install.log" 2>&1
fi

if ! pgrep -f "vite --host 0.0.0.0 --port ${VISIONARY_PORT}" >/dev/null 2>&1; then
  (
    cd "${VISIONARY_DIR}"
    bun run dev --host 0.0.0.0 --port "${VISIONARY_PORT}"
  ) >> "${LOG_DIR}/vite.log" 2>&1 &
fi

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${VISIONARY_PORT}/" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

google-chrome-stable \
  --no-sandbox \
  --user-data-dir="${LOG_DIR}/chrome-profile" \
  --window-size=1920,1080 \
  ${CHROME_FLAGS:-} \
  "http://127.0.0.1:${VISIONARY_PORT}/demo/simple/index.html" \
  >> "${LOG_DIR}/chrome.log" 2>&1 &

{
  echo "[visionary] App:   http://127.0.0.1:${VISIONARY_PORT}/demo/simple/index.html"
  echo "[visionary] Model: ${MODEL_SRC}"
  echo "[visionary] Load the model from the browser via file picker or drag-and-drop."
} >> "${LOG_DIR}/autostart.log"

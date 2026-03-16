#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VISIONARY_DIR="${VISIONARY_DIR:-${WORKSPACE}/gs_vis/visionary}"
MODEL_SRC="${MODEL_SRC:-${WORKSPACE}/da3/output_gs/gs_ply/0000.ply}"
DISPLAY="${DISPLAY:-:1}"
VNC_PORT="${VNC_PORT:-5901}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
VISIONARY_PORT="${VISIONARY_PORT:-3000}"

mkdir -p /tmp/.X11-unix /var/log/visionary
rm -f /tmp/.X1-lock

if [[ ! -d "${VISIONARY_DIR}" ]]; then
  echo "Missing Visionary workspace: ${VISIONARY_DIR}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_SRC}" ]]; then
  echo "Missing 3DGS PLY: ${MODEL_SRC}" >&2
  exit 1
fi

if [[ ! -d "${VISIONARY_DIR}/node_modules" ]]; then
  echo "[visionary-vnc] Installing frontend dependencies..."
  (cd "${VISIONARY_DIR}" && bun install --no-save)
fi

Xorg "${DISPLAY}" -config /etc/X11/xorg.conf -noreset +extension GLX +extension RANDR +extension RENDER > /var/log/visionary/xorg.log 2>&1 &
sleep 2

export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p "${XDG_RUNTIME_DIR}"
fluxbox > /var/log/visionary/fluxbox.log 2>&1 &
x11vnc -display "${DISPLAY}" -forever -shared -nopw -rfbport "${VNC_PORT}" > /var/log/visionary/x11vnc.log 2>&1 &
websockify --web=/usr/share/novnc/ "${NOVNC_PORT}" "localhost:${VNC_PORT}" > /var/log/visionary/novnc.log 2>&1 &

(cd "${VISIONARY_DIR}" && bun run dev --host 0.0.0.0 --port "${VISIONARY_PORT}") > /var/log/visionary/vite.log 2>&1 &

sleep 5

google-chrome-stable \
  --user-data-dir=/tmp/chrome-profile \
  --window-size=1920,1080 \
  ${CHROME_FLAGS} \
  "http://127.0.0.1:${VISIONARY_PORT}/demo/simple/index.html" \
  > /var/log/visionary/chrome.log 2>&1 &

echo "[visionary-vnc] VNC:   ${VNC_PORT}"
echo "[visionary-vnc] noVNC: ${NOVNC_PORT}"
echo "[visionary-vnc] App:   ${VISIONARY_PORT}"
echo "[visionary-vnc] Model: ${MODEL_SRC}"
echo "[visionary-vnc] Load the model from the browser via file picker or drag-and-drop."

tail -f /var/log/visionary/*.log

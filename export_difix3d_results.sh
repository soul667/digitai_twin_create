#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data2/axgu/micromamba/envs/da3/bin/python}"
SCENE_ID="${SCENE_ID:-video_20260316_194727}"
STEP="${STEP:-1999}"
RENDER_TAG="${RENDER_TAG:-novel}"
RENDER_STAGE="${RENDER_STAGE:-Fixed}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/da3/difix3d_runs/${SCENE_ID}}"
CKPT_PATH="${CKPT_PATH:-${RUN_DIR}/train/ckpts/ckpt_${STEP}_rank0.pt}"
PLY_OUT="${PLY_OUT:-${RUN_DIR}/exports/ply/ckpt_${STEP}_rank0.ply}"
FRAMES_DIR="${FRAMES_DIR:-${RUN_DIR}/train/renders/${RENDER_TAG}/${STEP}/${RENDER_STAGE}}"
VIDEO_OUT="${VIDEO_OUT:-${RUN_DIR}/exports/video/${RENDER_TAG}_${RENDER_STAGE}_${STEP}.mp4}"
FPS="${FPS:-24}"

echo "[1/2] Exporting checkpoint to PLY"
"${PYTHON_BIN}" "${ROOT_DIR}/da3/export_difix3d_ckpt_to_ply.py" \
  --ckpt "${CKPT_PATH}" \
  --out "${PLY_OUT}"

echo "[2/2] Packing rendered frames to video"
"${PYTHON_BIN}" "${ROOT_DIR}/da3/make_video_from_frames.py" \
  --frames-dir "${FRAMES_DIR}" \
  --out "${VIDEO_OUT}" \
  --fps "${FPS}"

echo
echo "[OK] PLY:   ${PLY_OUT}"
echo "[OK] Video: ${VIDEO_OUT}"

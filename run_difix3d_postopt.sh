#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Multi-GPU presets:
# 1 GPU:
export CUDA_VISIBLE_DEVICES=1
STEPS_SCALER=1.0
# 2 GPUs:
# export CUDA_VISIBLE_DEVICES=1,2
# STEPS_SCALER=0.5
# 3 GPUs:
# export CUDA_VISIBLE_DEVICES=1,2,3
# STEPS_SCALER=0.333

# Training presets:
# smoke: verify the full pipeline quickly
TRAINER_ARGS=(
  --max-steps 2000
  --eval-steps 1000 2000
  --save-steps 1000 2000
  --fix-steps 1000 2000
)

# medium: longer run for a first useful result
# TRAINER_ARGS=(
#   --max-steps 10000
#   --eval-steps 2000 4000 6000 8000 10000
#   --save-steps 2000 4000 6000 8000 10000
#   --fix-steps 3000 6000 9000
# )

# full: follow Difix3D default long training profile
# TRAINER_ARGS=()

CMD=(
  python ./da3/run_difix3d_postopt.py
  --frames-dir ./da3/frames/video_20260316_194727
  --scene-id video_20260316_194727
  --da3-export-dir ./da3/output_gs
  --output-root ./da3/difix3d_runs
  --gpus "${CUDA_VISIBLE_DEVICES}"
  --da3-max-images 48
  --process-res 504
  --steps-scaler "${STEPS_SCALER}"
  --data-factor 4
  --trainer-args
)

for arg in "${TRAINER_ARGS[@]}"; do
  CMD+=("$arg")
done

exec "${CMD[@]}"

# Set GPU device and memory optimization
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_ALLOC_CONF=expandable_segments:True

python ./da3/run_da3_export.py \
  --frames-dir ./da3/frames \
  --gpu 2 \
  --max-images 128 \
  --infer-gs \
  --export-dir ./da3/output_gs \
  --export-format npz-glb-gs_ply-gs_video \
  --process-res 756

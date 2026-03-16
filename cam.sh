rm -rf ./da3/frames/*
python ./da3/video_to_frames.py \
  --input ./da3/video_20260316_194727.mp4 \
  --output ./da3/frames \
  --count 48 \
  --write-manifest
```shell
conda activate da3
CUDA_VISIBLE_DEVICES=0 python ./da3/test.py
```

```shell
# 安装清晰帧提取器（二选一）
python3.12 -m pip install sharp-frame-extractor
# 或
# uv tool install sharp-frame-extractor
# 或（推荐，不依赖当前conda环境Python版本）
# uvx --python 3.12 sharp-frame-extractor --help

# 清晰抽帧：每0.3秒选1张最清晰帧（推荐）
python ./da3/video_to_frames.py \
  --input ./da3/video_20260316_194727.mp4 \
  --output ./da3/frames \
  --every 0.3 \
  --write-manifest

# 也支持 .MOV
python ./da3/video_to_frames.py \
  --input ./da3/IMG_9018.MOV \
  --output ./da3/frames \
  --count 120

# 目录批处理（递归）
python ./da3/video_to_frames.py \
  --input ./da3 \
  --output ./da3/frames \
  --every 0.25 \
  --recursive \
  --overwrite
```

```shell
# DA3导出（含高斯）
python ./da3/run_da3_export.py \
  --frames-dir ./da3/frames \
  --gpu 3 \
  --max-images 48 \
  --infer-gs \
  --export-dir ./da3/output_gs \
  --export-format npz-glb-gs_ply-gs_video
```

```shell
# 只做 DA3 -> Difix3D 数据桥接
/data2/axgu/micromamba/envs/da3/bin/python ./da3/build_difix3d_dataset.py \
  --da3-export-dir ./da3/output_gs \
  --frames-dir ./da3/frames/IMG_9018 \
  --scene-id img9018 \
  --out-dir ./da3/difix3d_runs/img9018/data
```

```shell
# 把 DA3 的 gs_ply 转成 Difix3D 初始化 checkpoint
/data2/axgu/micromamba/envs/da3/bin/python ./da3/convert_da3_gs_to_difix_ckpt.py \
  --ply ./da3/output_gs/gs_ply/0000.ply \
  --out ./da3/difix3d_runs/img9018/init_ckpt/ckpt_0000_rank0.pt
```

```shell
# 一键跑 DA3 + Difix3D 后优化
python ./da3/run_difix3d_postopt.py \
  --frames-dir ./da3/frames/IMG_9018 \
  --scene-id img9018 \
  --da3-export-dir ./da3/output_gs \
  --gpu 0
```

```shell
# 只跑后优化，不重跑 DA3；额外覆盖 Difix3D trainer 参数
python ./da3/run_difix3d_postopt.py \
  --frames-dir ./da3/frames/IMG_9018 \
  --scene-id img9018 \
  --da3-export-dir ./da3/output_gs \
  --trainer-arg=--max_steps \
  --trainer-arg=2000
```

## 处理后的高斯后优化
- 当前仓库不会改 `/data2/axgu/code/Difix3D`。
- 训练阶段通过 `PYTHONPATH` 注入最小兼容层：
  - `da3/pycolmap.py`: 兼容 Difix3D 期望的 `SceneManager`
  - `da3/nerfview.py` / `da3/viser.py`: 在 `disable_viewer=True` 时绕开 viewer import
  - `da3/fused_ssim.py`: 用纯 PyTorch 版本兼容 `fused_ssim`
- 结果默认落到 `da3/difix3d_runs/{scene_id}/`，其中包含：
  - `data/{scene_id}`: Difix3D 数据集
  - `init_ckpt/`: 从 DA3 高斯转换的初始 checkpoint
  - `train/`: Difix3D 训练输出和 renders

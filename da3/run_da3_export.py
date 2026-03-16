#!/usr/bin/env python3
"""
Run Depth Anything 3 inference on sampled frames and export results.

Example:
  python da3/run_da3_export.py \
    --frames-dir da3/frames \
    --gpu 3 \
    --max-images 48 \
    --infer-gs \
    --export-dir da3/output_gs \
    --export-format npz-glb-gs_ply-gs_video
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    frames_dir: Path
    pattern: str
    max_images: int | None
    stride: int
    gpu: int | None
    model_id: str
    export_dir: Path | None
    export_format: str
    infer_gs: bool
    process_res: int
    process_res_method: str
    ref_view_strategy: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DA3 on extracted frames and export depth/mesh/gaussian results."
    )
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory containing frames.")
    parser.add_argument(
        "--pattern",
        default="**/*",
        help="Glob pattern under frames-dir (default: **/*).",
    )
    parser.add_argument("--max-images", type=int, default=48, help="Max number of frames to run.")
    parser.add_argument("--stride", type=int, default=1, help="Use every N-th image from sorted list.")
    parser.add_argument("--gpu", type=int, default=None, help="Physical GPU id (e.g. 3).")
    parser.add_argument(
        "--model-id",
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="HF model id.",
    )
    parser.add_argument("--export-dir", type=Path, default=Path("da3/output"), help="Output directory.")
    parser.add_argument(
        "--export-format",
        default="npz-glb-gs_ply-gs_video",
        help="Dash-joined formats, e.g. mini_npz-glb or npz-glb-gs_ply-gs_video.",
    )
    parser.add_argument("--infer-gs", action="store_true", help="Enable gaussian branch for gs exports.")
    parser.add_argument("--process-res", type=int, default=504, help="Inference process resolution.")
    parser.add_argument(
        "--process-res-method",
        default="upper_bound_resize",
        choices=[
            "upper_bound_resize",
            "upper_bound_crop",
            "lower_bound_resize",
            "lower_bound_crop",
        ],
        help="Input resize/crop method.",
    )
    parser.add_argument(
        "--ref-view-strategy",
        default="saddle_balanced",
        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
        help="Reference view strategy for multi-view inference.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    if not args.frames_dir.exists():
        raise ValueError(f"frames dir does not exist: {args.frames_dir}")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be > 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.gpu is not None and args.gpu < 0:
        raise ValueError("--gpu must be >= 0")
    return RunConfig(
        frames_dir=args.frames_dir,
        pattern=args.pattern,
        max_images=args.max_images,
        stride=args.stride,
        gpu=args.gpu,
        model_id=args.model_id,
        export_dir=args.export_dir,
        export_format=args.export_format,
        infer_gs=args.infer_gs,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        ref_view_strategy=args.ref_view_strategy,
    )


def collect_images(cfg: RunConfig) -> list[str]:
    allowed_exts = {".jpg", ".jpeg", ".png"}
    pattern = str(cfg.frames_dir / cfg.pattern)
    images = sorted(glob.glob(pattern, recursive=True))
    images = [
        p
        for p in images
        if os.path.isfile(p) and Path(p).suffix.lower() in allowed_exts
    ]
    if cfg.stride > 1:
        images = images[:: cfg.stride]
    if cfg.max_images is not None:
        images = images[: cfg.max_images]
    if not images:
        raise RuntimeError(
            f"No images found. frames_dir={cfg.frames_dir}, pattern={cfg.pattern}. "
            "Expected files like da3/frames/<video_name>/*.(png|jpg|jpeg)"
        )
    return images


def main() -> int:
    try:
        cfg = build_config(parse_args())
    except Exception as exc:
        print(f"[ERROR] bad arguments: {exc}", file=sys.stderr)
        return 2

    if cfg.gpu is not None:
        # Must be set before torch initializes CUDA.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    try:
        import torch
        from depth_anything_3.api import DepthAnything3
    except Exception as exc:
        print(f"[ERROR] import failed: {exc}", file=sys.stderr)
        return 3

    try:
        images = collect_images(cfg)
    except Exception as exc:
        print(f"[ERROR] input failed: {exc}", file=sys.stderr)
        return 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}, visible_gpu={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    if cfg.gpu is not None and device.type != "cuda":
        print(
            "[WARN] --gpu was set but CUDA is unavailable in this process. "
            "If your machine has N GPUs, valid physical ids are 0..N-1 "
            "(e.g., 4 GPUs => ids 0,1,2,3).",
            file=sys.stderr,
        )
    print(f"[INFO] images={len(images)} from {cfg.frames_dir}")
    print(f"[INFO] infer_gs={cfg.infer_gs}, export_format={cfg.export_format}")

    model = DepthAnything3.from_pretrained(cfg.model_id).to(device)
    model.eval()

    export_dir = str(cfg.export_dir) if cfg.export_dir is not None else None
    if export_dir is not None:
        cfg.export_dir.mkdir(parents=True, exist_ok=True)

    prediction = model.inference(
        image=images,
        process_res=cfg.process_res,
        process_res_method=cfg.process_res_method,
        ref_view_strategy=cfg.ref_view_strategy,
        infer_gs=cfg.infer_gs,
        export_dir=export_dir,
        export_format=cfg.export_format,
    )

    print(f"[OK] processed_images={prediction.processed_images.shape}")
    print(f"[OK] depth={prediction.depth.shape}, conf={prediction.conf.shape}")
    print(f"[OK] extrinsics={prediction.extrinsics.shape}, intrinsics={prediction.intrinsics.shape}")
    if export_dir is not None:
        print(f"[OK] exported to {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

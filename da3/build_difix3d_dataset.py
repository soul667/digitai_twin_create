#!/usr/bin/env python3
"""
Bridge DA3 exports into a Difix3D gsplat dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from difix3d_bridge import (
    build_sparse_points,
    build_split_entries,
    choose_pose_semantics,
    ensure_dir,
    load_da3_results,
    save_image_pyramid,
    write_colmap_text_model,
    write_dataset_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Difix3D dataset from DA3 exports.")
    parser.add_argument("--da3-export-dir", type=Path, required=True, help="DA3 export root that contains exports/npz/results.npz.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Original frames directory used for DA3.")
    parser.add_argument("--scene-id", required=True, help="Scene identifier under the output root.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output root directory.")
    parser.add_argument("--eval-every", type=int, default=8, help="Mark every Nth image as eval.")
    parser.add_argument("--sample-stride", type=int, default=16, help="Pixel stride when sampling sparse depth points.")
    parser.add_argument("--min-conf-percentile", type=float, default=85.0, help="Keep sampled pixels whose confidence is above this percentile.")
    parser.add_argument("--max-points-per-image", type=int, default=2048, help="Maximum sparse points to emit per image.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    npz_path = args.da3_export_dir / "exports" / "npz" / "results.npz"
    if not npz_path.exists():
        print(f"[ERROR] missing DA3 npz export: {npz_path}", file=sys.stderr)
        return 2

    scene_dir = ensure_dir(args.out_dir / args.scene_id)
    data = load_da3_results(npz_path)
    entries = build_split_entries(len(data["image"]), args.eval_every)
    pose_semantics, camtoworlds, pose_scores = choose_pose_semantics(
        data["depth"], data["intrinsics"], data["extrinsics"]
    )

    save_image_pyramid(data["image"], entries, scene_dir)
    sparse_points, by_image = build_sparse_points(
        data["image"],
        data["depth"],
        data["conf"],
        data["intrinsics"],
        camtoworlds,
        entries,
        sample_stride=args.sample_stride,
        min_conf_percentile=args.min_conf_percentile,
        max_points_per_image=args.max_points_per_image,
    )
    if not sparse_points:
        print("[ERROR] sparse point generation produced zero points", file=sys.stderr)
        return 3

    sparse_dir = scene_dir / "colmap" / "sparse" / "0"
    write_colmap_text_model(
        sparse_dir,
        data["image"],
        data["intrinsics"],
        camtoworlds,
        entries,
        sparse_points,
        by_image,
    )
    write_dataset_manifest(
        scene_dir / "dataset_manifest.json",
        scene_id=args.scene_id,
        npz_path=npz_path,
        frames_dir=args.frames_dir,
        entries=entries,
        pose_semantics=pose_semantics,
        pose_scores=pose_scores,
        sample_stride=args.sample_stride,
        min_conf_percentile=args.min_conf_percentile,
        sparse_points=sparse_points,
    )

    train_count = sum(entry.split == "train" for entry in entries)
    eval_count = sum(entry.split == "eval" for entry in entries)
    print(f"[OK] scene_dir={scene_dir}")
    print(f"[OK] pose_semantics={pose_semantics}, scores={pose_scores}")
    print(f"[OK] train={train_count}, eval={eval_count}, sparse_points={len(sparse_points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

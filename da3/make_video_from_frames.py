#!/usr/bin/env python3
"""
Create an MP4 video from a directory of rendered PNG frames.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make MP4 video from rendered frames.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory containing .png frames.")
    parser.add_argument("--out", type=Path, required=True, help="Output .mp4 path.")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second.")
    parser.add_argument(
        "--macro-block-size",
        type=int,
        default=2,
        help="FFmpeg macro block size. Default 2 keeps MP4 compatible while minimizing padding.",
    )
    return parser.parse_args()


def pad_frame(frame: np.ndarray, macro_block_size: int) -> np.ndarray:
    if macro_block_size <= 1:
        return frame
    height, width = frame.shape[:2]
    padded_height = ((height + macro_block_size - 1) // macro_block_size) * macro_block_size
    padded_width = ((width + macro_block_size - 1) // macro_block_size) * macro_block_size
    if padded_height == height and padded_width == width:
        return frame
    pad_h = padded_height - height
    pad_w = padded_width - width
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def main() -> int:
    args = parse_args()
    if not args.frames_dir.exists():
        print(f"[ERROR] missing frames dir: {args.frames_dir}", file=sys.stderr)
        return 2

    frames = sorted(args.frames_dir.glob("*.png"))
    if not frames:
        print(f"[ERROR] no PNG frames found under {args.frames_dir}", file=sys.stderr)
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    first = imageio.imread(frames[0])
    height, width = first.shape[:2]
    first_padded = pad_frame(first, args.macro_block_size)
    out_height, out_width = first_padded.shape[:2]
    with imageio.get_writer(
        str(args.out),
        fps=args.fps,
        macro_block_size=args.macro_block_size,
    ) as writer:
        writer.append_data(first_padded)
        for frame in frames[1:]:
            writer.append_data(pad_frame(imageio.imread(frame), args.macro_block_size))

    print(
        f"[OK] wrote {len(frames)} frames at {args.fps} fps, "
        f"input={width}x{height}, output={out_width}x{out_height} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

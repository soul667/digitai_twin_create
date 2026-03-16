#!/usr/bin/env python3
"""
Extract sharp frames from videos via sharp-frame-extractor.

Examples:
  python da3/video_to_frames.py --input da3/VID20260316191402.mp4 --output da3/frames --every 0.3
  python da3/video_to_frames.py --input da3/VID20260316191402.MOV --output da3/frames --count 120
  python da3/video_to_frames.py --input da3 --output da3/frames --every 0.25 --recursive --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov"}


@dataclass(frozen=True)
class SampleConfig:
    count: Optional[int]
    every: Optional[float]
    jobs: Optional[int]
    workers: Optional[int]
    memory_limit_mb: Optional[int]
    overwrite: bool
    write_manifest: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sharp frames from MP4/MOV videos using sharp-frame-extractor."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input video file (.mp4/.mov) or directory that contains videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for extracted frames.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Target number of sharp frames to extract per video.",
    )
    parser.add_argument(
        "--every",
        type=float,
        default=None,
        help="Extract one sharp frame every N seconds (supports decimals).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Max number of videos processed in parallel by extractor (forwarded as -j).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Total analysis workers used by extractor (forwarded as -w).",
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=None,
        help="Global frame-buffer memory limit in MB (forwarded as -m).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search .mp4/.mov files when --input is a directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directories for each video.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write metadata manifest.json next to each video's extracted frames.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> SampleConfig:
    if (args.count is None) == (args.every is None):
        raise ValueError("Specify exactly one of --count or --every.")
    if args.count is not None and args.count <= 0:
        raise ValueError("--count must be > 0.")
    if args.every is not None and args.every <= 0:
        raise ValueError("--every must be > 0.")
    if args.jobs is not None and args.jobs <= 0:
        raise ValueError("--jobs must be > 0.")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be > 0.")
    if args.memory_limit is not None and args.memory_limit <= 0:
        raise ValueError("--memory-limit must be > 0.")
    return SampleConfig(
        count=args.count,
        every=args.every,
        jobs=args.jobs,
        workers=args.workers,
        memory_limit_mb=args.memory_limit,
        overwrite=args.overwrite,
        write_manifest=args.write_manifest,
    )


def find_videos(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
            supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTS))
            raise ValueError(f"Unsupported video extension: {input_path}. Supported: {supported}")
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    patterns = (
        ("**/*.mp4", "**/*.mov", "**/*.MP4", "**/*.MOV")
        if recursive
        else ("*.mp4", "*.mov", "*.MP4", "*.MOV")
    )
    files_set = set()
    for pattern in patterns:
        files_set.update(p for p in input_path.glob(pattern) if p.is_file())
    files = sorted(files_set)
    if not files:
        raise ValueError(f"No .mp4/.mov files found under: {input_path}")
    return files


def resolve_extractor_cmd() -> list[str]:
    cli = shutil.which("sharp-frame-extractor")
    if cli:
        return [cli]

    uvx = shutil.which("uvx")
    if uvx:
        # sharp-frame-extractor v2 requires Python >= 3.12.
        return [uvx, "--python", "3.12", "sharp-frame-extractor"]

    # Fallback to module execution in current Python environment.
    # This only works if the package is installed and compatible with current Python.
    return [sys.executable, "-m", "sharp_frame_extractor"]


def read_video_meta(video_path: Path) -> dict[str, object]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if source_fps <= 0:
        return {}
    return {
        "source_fps": source_fps,
        "source_total_frames": total_frames,
        "source_resolution": {"width": width, "height": height},
    }


def build_extractor_command(
    base_cmd: list[str],
    video_path: Path,
    output_root: Path,
    cfg: SampleConfig,
) -> list[str]:
    cmd = [*base_cmd, str(video_path), "-o", str(output_root)]
    if cfg.count is not None:
        cmd.extend(["--count", str(cfg.count)])
    else:
        cmd.extend(["--every", str(cfg.every)])
    if cfg.jobs is not None:
        cmd.extend(["-j", str(cfg.jobs)])
    if cfg.workers is not None:
        cmd.extend(["-w", str(cfg.workers)])
    if cfg.memory_limit_mb is not None:
        cmd.extend(["-m", str(cfg.memory_limit_mb)])
    return cmd


def extract_one_video(
    video_path: Path,
    output_root: Path,
    cfg: SampleConfig,
    extractor_cmd: list[str],
) -> int:
    video_stem = video_path.stem
    out_dir = output_root / video_stem

    if out_dir.exists():
        if not cfg.overwrite:
            raise FileExistsError(
                f"Output exists for '{video_stem}'. Use --overwrite or change --output."
            )
        shutil.rmtree(out_dir)

    cmd = build_extractor_command(extractor_cmd, video_path, output_root, cfg)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        if "No module named sharp_frame_extractor" in detail:
            detail = (
                "sharp-frame-extractor is unavailable in current environment. "
                "Install uv and run via uvx (Python 3.12): "
                "`uvx --python 3.12 sharp-frame-extractor ...`"
            )
        raise RuntimeError(f"sharp-frame-extractor failed: {detail}")

    images = sorted(p for p in out_dir.glob("*.png") if p.is_file())
    return len(images)


def write_manifest_file(
    video_path: Path,
    output_root: Path,
    cfg: SampleConfig,
    written_frames: int,
    extractor_cmd: list[str],
) -> None:
    out_dir = output_root / video_path.stem
    manifest = {
        "video_path": str(video_path.resolve()),
        "output_dir": str(out_dir.resolve()),
        "extractor": {
            "command": extractor_cmd,
            "mode": "count" if cfg.count is not None else "every",
            "count": cfg.count,
            "every": cfg.every,
            "jobs": cfg.jobs,
            "workers": cfg.workers,
            "memory_limit_mb": cfg.memory_limit_mb,
        },
        "written_frames": written_frames,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest.update(read_video_meta(video_path))
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    try:
        cfg = validate_args(args)
        videos = find_videos(args.input, args.recursive)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    extractor_cmd = resolve_extractor_cmd()
    total_written = 0
    failed = 0

    for video in videos:
        try:
            written = extract_one_video(
                video_path=video,
                output_root=args.output,
                cfg=cfg,
                extractor_cmd=extractor_cmd,
            )
            if cfg.write_manifest:
                write_manifest_file(video, args.output, cfg, written, extractor_cmd)
            total_written += written
            print(f"[OK] {video} -> {written} frame(s)")
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {video}: {exc}", file=sys.stderr)

    if failed == len(videos):
        print("[DONE] no video succeeded", file=sys.stderr)
        return 1

    print(f"[DONE] extracted {total_written} sharp frame(s) from {len(videos) - failed} video(s)")
    if failed > 0:
        print(f"[WARN] failed videos: {failed}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

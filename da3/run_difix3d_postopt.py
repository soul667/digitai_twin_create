#!/usr/bin/env python3
"""
Run DA3 export, bridge data for Difix3D, and launch gsplat post-optimization.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DA3_PYTHON = Path("/data2/axgu/micromamba/envs/da3/bin/python")
DIFIX3D_PYTHON = Path("/data2/axgu/micromamba/envs/difix3d/bin/python")
DIFIX3D_ROOT = Path("/data2/axgu/code/Difix3D")
DIFIX3D_TRAINER = DIFIX3D_ROOT / "examples" / "gsplat" / "simple_trainer_difix3d.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click DA3 -> Difix3D gsplat post-optimization.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Frames directory used as DA3 input.")
    parser.add_argument("--scene-id", required=True, help="Scene identifier under da3/difix3d_runs.")
    parser.add_argument("--da3-export-dir", type=Path, default=Path("da3/output_gs"), help="DA3 export directory.")
    parser.add_argument("--output-root", type=Path, default=Path("da3/difix3d_runs"), help="Output root for bridged data and training outputs.")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated physical GPU ids. Example: 0 or 0,1,2,3",
    )
    parser.add_argument("--run-da3", action="store_true", help="Force rerun DA3 export before post-optimization.")
    parser.add_argument("--da3-max-images", type=int, default=48, help="Max images for the DA3 export step.")
    parser.add_argument("--da3-export-format", default="npz-glb-gs_ply-gs_video", help="DA3 export formats.")
    parser.add_argument("--process-res", type=int, default=504, help="DA3 process resolution.")
    parser.add_argument("--steps-scaler", type=float, default=1.0, help="Difix3D trainer steps scaler.")
    parser.add_argument("--data-factor", type=int, default=4, help="Difix3D image pyramid factor.")
    parser.add_argument(
        "--trainer-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Pass all remaining arguments directly to the Difix3D trainer.",
    )
    parser.add_argument(
        "--trainer-arg",
        action="append",
        default=[],
        help="Extra raw arguments forwarded to the Difix3D trainer. Repeat this flag for multiple args.",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("[CMD]", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd is not None else None)


def main() -> int:
    args = parse_args()
    frames_dir = args.frames_dir.resolve()
    da3_export_dir = args.da3_export_dir.resolve()
    output_root = args.output_root.resolve()
    gpu_list = [part.strip() for part in args.gpus.split(",")] if args.gpus else []
    da3_gpu = gpu_list[0] if gpu_list else None

    npz_path = da3_export_dir / "exports" / "npz" / "results.npz"
    ply_path = da3_export_dir / "gs_ply" / "0000.ply"
    if args.run_da3 or not npz_path.exists() or not ply_path.exists():
        da3_cmd = [
            str(DA3_PYTHON),
            str(REPO_ROOT / "da3" / "run_da3_export.py"),
            "--frames-dir",
            str(frames_dir),
            "--max-images",
            str(args.da3_max_images),
            "--infer-gs",
            "--export-dir",
            str(da3_export_dir),
            "--export-format",
            args.da3_export_format,
            "--process-res",
            str(args.process_res),
        ]
        if da3_gpu is not None:
            da3_cmd.extend(["--gpu", da3_gpu])
        run_cmd(da3_cmd, cwd=REPO_ROOT)

    scene_root = output_root / args.scene_id
    data_root = scene_root / "data"
    init_root = scene_root / "init_ckpt"
    train_root = scene_root / "train"

    build_cmd = [
        str(DA3_PYTHON),
        str(REPO_ROOT / "da3" / "build_difix3d_dataset.py"),
        "--da3-export-dir",
        str(da3_export_dir),
        "--frames-dir",
        str(frames_dir),
        "--scene-id",
        args.scene_id,
        "--out-dir",
        str(data_root),
    ]
    run_cmd(build_cmd, cwd=REPO_ROOT)

    ckpt_path = init_root / "ckpt_0000_rank0.pt"
    convert_cmd = [
        str(DA3_PYTHON),
        str(REPO_ROOT / "da3" / "convert_da3_gs_to_difix_ckpt.py"),
        "--ply",
        str(ply_path),
        "--out",
        str(ckpt_path),
    ]
    run_cmd(convert_cmd, cwd=REPO_ROOT)

    trainer_env = os.environ.copy()
    pythonpath_parts = [str(REPO_ROOT / "da3"), str(DIFIX3D_ROOT)]
    if trainer_env.get("PYTHONPATH"):
        pythonpath_parts.append(trainer_env["PYTHONPATH"])
    trainer_env["PYTHONPATH"] = ":".join(pythonpath_parts)
    trainer_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    trainer_env.setdefault("HF_HOME", "/tmp/huggingface")
    if args.gpus is not None:
        trainer_env["CUDA_VISIBLE_DEVICES"] = args.gpus

    train_cmd = [
        str(DIFIX3D_PYTHON),
        str(DIFIX3D_TRAINER),
        "default",
        "--data_dir",
        str(data_root / args.scene_id),
        "--data_factor",
        str(args.data_factor),
        "--result_dir",
        str(train_root),
        "--test_every",
        "1",
        "--no-normalize-world-space",
        "--steps_scaler",
        str(args.steps_scaler),
    ]
    train_cmd.extend(args.trainer_arg)
    train_cmd.extend(args.trainer_args)
    train_cmd.extend(
        [
            "--ckpt",
            str(ckpt_path),
        ]
    )
    run_cmd(train_cmd, env=trainer_env, cwd=DIFIX3D_ROOT / "examples" / "gsplat")

    print(f"[OK] outputs under {scene_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)

#!/usr/bin/env python3
"""
Export a Difix3D gsplat checkpoint to a standard 3DGS PLY file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Difix3D checkpoint to 3DGS PLY.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to Difix3D ckpt_*.pt")
    parser.add_argument("--out", type=Path, required=True, help="Output PLY path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.ckpt.exists():
        print(f"[ERROR] missing checkpoint: {args.ckpt}", file=sys.stderr)
        return 2

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "splats" not in ckpt:
        print("[ERROR] unsupported checkpoint: missing 'splats'", file=sys.stderr)
        return 3

    splats = ckpt["splats"]
    required = {"means", "quats", "scales", "opacities", "sh0", "shN"}
    missing = required - set(splats.keys())
    if missing:
        print(f"[ERROR] unsupported checkpoint: missing keys {sorted(missing)}", file=sys.stderr)
        return 4

    means = splats["means"].detach().cpu().numpy().astype(np.float32)
    quats = splats["quats"].detach().cpu().numpy().astype(np.float32)
    scales = splats["scales"].detach().cpu().numpy().astype(np.float32)
    opacities = splats["opacities"].detach().cpu().numpy().astype(np.float32)
    sh0 = splats["sh0"].detach().cpu().numpy().astype(np.float32)
    shN = splats["shN"].detach().cpu().numpy().astype(np.float32)

    if means.ndim != 2 or means.shape[1] != 3:
        print(f"[ERROR] invalid means shape: {means.shape}", file=sys.stderr)
        return 5
    if quats.ndim != 2 or quats.shape[1] != 4:
        print(f"[ERROR] invalid quats shape: {quats.shape}", file=sys.stderr)
        return 6
    if scales.ndim != 2 or scales.shape[1] != 3:
        print(f"[ERROR] invalid scales shape: {scales.shape}", file=sys.stderr)
        return 7
    if sh0.ndim != 3 or sh0.shape[1:] != (1, 3):
        print(f"[ERROR] invalid sh0 shape: {sh0.shape}", file=sys.stderr)
        return 8
    if shN.ndim != 3 or shN.shape[2] != 3:
        print(f"[ERROR] invalid shN shape: {shN.shape}", file=sys.stderr)
        return 9

    sh0_flat = sh0[:, 0, :]
    shN_flat = shN.reshape(shN.shape[0], -1)

    dtype_fields: list[tuple[str, str]] = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]
    dtype_fields.extend((f"f_rest_{idx}", "f4") for idx in range(shN_flat.shape[1]))
    dtype_fields.extend(
        [
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]
    )

    vertex = np.empty(means.shape[0], dtype=np.dtype(dtype_fields))
    vertex["x"] = means[:, 0]
    vertex["y"] = means[:, 1]
    vertex["z"] = means[:, 2]
    vertex["f_dc_0"] = sh0_flat[:, 0]
    vertex["f_dc_1"] = sh0_flat[:, 1]
    vertex["f_dc_2"] = sh0_flat[:, 2]
    for idx in range(shN_flat.shape[1]):
        vertex[f"f_rest_{idx}"] = shN_flat[:, idx]
    vertex["opacity"] = opacities
    vertex["scale_0"] = scales[:, 0]
    vertex["scale_1"] = scales[:, 1]
    vertex["scale_2"] = scales[:, 2]
    vertex["rot_0"] = quats[:, 0]
    vertex["rot_1"] = quats[:, 1]
    vertex["rot_2"] = quats[:, 2]
    vertex["rot_3"] = quats[:, 3]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(args.out))
    print(f"[OK] exported {means.shape[0]} gaussians -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

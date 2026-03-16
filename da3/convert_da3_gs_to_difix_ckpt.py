#!/usr/bin/env python3
"""
Convert a DA3 Gaussian PLY to a Difix3D gsplat checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


EXPECTED_FIELDS = {
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DA3 Gaussian PLY to a Difix3D checkpoint.")
    parser.add_argument("--ply", type=Path, required=True, help="Path to DA3 gs_ply/0000.ply")
    parser.add_argument("--out", type=Path, required=True, help="Output checkpoint path.")
    parser.add_argument("--sh-degree", type=int, default=3, help="Target SH degree for Difix3D.")
    return parser.parse_args()


def to_tensor(vertex_data: np.ndarray, names: tuple[str, ...]) -> torch.Tensor:
    return torch.from_numpy(np.stack([vertex_data[name].astype(np.float32) for name in names], axis=1))


def main() -> int:
    args = parse_args()
    if not args.ply.exists():
        print(f"[ERROR] missing PLY: {args.ply}", file=sys.stderr)
        return 2

    ply = PlyData.read(str(args.ply))
    if "vertex" not in ply:
        print("[ERROR] PLY does not contain a vertex element", file=sys.stderr)
        return 3
    vertex = ply["vertex"].data
    fields = set(vertex.dtype.names or [])
    missing = EXPECTED_FIELDS - fields
    if missing:
        print(f"[ERROR] unsupported PLY schema, missing fields: {sorted(missing)}", file=sys.stderr)
        return 4

    sh_channels = (args.sh_degree + 1) ** 2 - 1
    sh0 = to_tensor(vertex, ("f_dc_0", "f_dc_1", "f_dc_2")).unsqueeze(1)
    shN = torch.zeros((len(vertex), sh_channels, 3), dtype=torch.float32)
    splats = {
        "means": to_tensor(vertex, ("x", "y", "z")),
        "quats": to_tensor(vertex, ("rot_0", "rot_1", "rot_2", "rot_3")),
        "scales": to_tensor(vertex, ("scale_0", "scale_1", "scale_2")),
        "opacities": torch.from_numpy(vertex["opacity"].astype(np.float32)),
        "sh0": sh0,
        "shN": shN,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 0, "splats": splats}, args.out)
    print(f"[OK] converted {len(vertex)} gaussians -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

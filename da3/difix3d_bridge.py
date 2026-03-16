#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SplitEntry:
    index: int
    split: str
    image_name: str
    camera_id: int


@dataclass(frozen=True)
class SparsePoint:
    point_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_id: int
    xy: np.ndarray


def load_da3_results(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    required = {"image", "depth", "conf", "extrinsics", "intrinsics"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Missing arrays in {npz_path}: {sorted(missing)}")
    return {key: data[key] for key in required}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    mats = np.repeat(np.eye(4, dtype=np.float32)[None, ...], extrinsics.shape[0], axis=0)
    mats[:, :3, :4] = extrinsics
    return mats


def invert_homogeneous(mats: np.ndarray) -> np.ndarray:
    return np.linalg.inv(mats).astype(np.float32)


def unproject_depth(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    camtoworld: np.ndarray,
    pixels_xy: np.ndarray,
) -> np.ndarray:
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    x = pixels_xy[:, 0]
    y = pixels_xy[:, 1]
    z = depth[y.astype(np.int32), x.astype(np.int32)].astype(np.float32)
    cam_xyz = np.stack(
        [
            (x - cx) / fx * z,
            (y - cy) / fy * z,
            z,
        ],
        axis=1,
    )
    world_xyz = cam_xyz @ camtoworld[:3, :3].T + camtoworld[:3, 3]
    return world_xyz.astype(np.float32)


def project_points(
    world_xyz: np.ndarray,
    intrinsic: np.ndarray,
    camtoworld: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_to_cam = np.linalg.inv(camtoworld)
    cam_xyz = world_xyz @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    z = cam_xyz[:, 2]
    uv = cam_xyz @ intrinsic.T
    uv = uv[:, :2] / np.clip(z[:, None], 1e-8, None)
    return uv.astype(np.float32), z.astype(np.float32)


def sample_grid(height: int, width: int, stride: int) -> np.ndarray:
    ys = np.arange(stride // 2, height, stride, dtype=np.int32)
    xs = np.arange(stride // 2, width, stride, dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)


def score_pose_interpretation(
    depths: np.ndarray,
    intrinsics: np.ndarray,
    candidate_c2w: np.ndarray,
    sample_stride: int = 24,
    max_views: int = 8,
) -> float:
    num_views, height, width = depths.shape
    view_ids = np.linspace(0, num_views - 1, min(num_views, max_views), dtype=int)
    total_score = 0.0
    total_weight = 0.0

    for i in view_ids:
        pixels = sample_grid(height, width, sample_stride)
        depth = depths[i]
        valid = depth[pixels[:, 1], pixels[:, 0]] > 1e-5
        pixels = pixels[valid]
        if len(pixels) == 0:
            continue
        world_xyz = unproject_depth(depth, intrinsics[i], candidate_c2w[i], pixels)
        neighbor_ids = [j for j in (i - 1, i + 1) if 0 <= j < num_views]
        if not neighbor_ids:
            continue
        for j in neighbor_ids:
            uv, z = project_points(world_xyz, intrinsics[j], candidate_c2w[j])
            in_front = z > 1e-5
            in_bounds = (
                (uv[:, 0] >= 0)
                & (uv[:, 0] < width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < height)
            )
            total_score += float(np.mean(in_front & in_bounds))
            total_weight += 1.0

    if total_weight == 0:
        return -math.inf
    return total_score / total_weight


def choose_pose_semantics(
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> tuple[str, np.ndarray, dict[str, float]]:
    extr_h = to_homogeneous(extrinsics)
    candidates = {
        "c2w": extr_h,
        "w2c": invert_homogeneous(extr_h),
    }
    scores = {
        name: score_pose_interpretation(depths, intrinsics, mats)
        for name, mats in candidates.items()
    }
    semantics = max(scores, key=scores.get)
    return semantics, candidates[semantics], scores


def split_frames(num_images: int, eval_every: int) -> list[str]:
    if num_images < 2:
        return ["train"] * num_images
    splits = ["eval" if (i % eval_every == 0) else "train" for i in range(num_images)]
    if "train" not in splits:
        splits[0] = "train"
    if "eval" not in splits:
        splits[-1] = "eval"
    return splits


def build_split_entries(num_images: int, eval_every: int) -> list[SplitEntry]:
    splits = split_frames(num_images, eval_every)
    counters = {"train": 0, "eval": 0}
    entries: list[SplitEntry] = []
    for index, split in enumerate(splits):
        counters[split] += 1
        image_name = f"image_{split}_{counters[split]:06d}.png"
        entries.append(SplitEntry(index=index, split=split, image_name=image_name, camera_id=index + 1))
    return entries


def save_image_pyramid(images: np.ndarray, entries: Iterable[SplitEntry], scene_dir: Path) -> None:
    image_dirs = {
        1: ensure_dir(scene_dir / "images"),
        2: ensure_dir(scene_dir / "images_2"),
        4: ensure_dir(scene_dir / "images_4"),
        8: ensure_dir(scene_dir / "images_8"),
    }
    for entry in entries:
        image = Image.fromarray(images[entry.index])
        width, height = image.size
        for factor, image_dir in image_dirs.items():
            if factor == 1:
                resized = image
            else:
                resized = image.resize((max(1, width // factor), max(1, height // factor)), Image.LANCZOS)
            resized.save(image_dir / entry.image_name)


def rotation_matrix_to_qvec(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    return qvec / np.linalg.norm(qvec)


def build_sparse_points(
    images: np.ndarray,
    depths: np.ndarray,
    confs: np.ndarray,
    intrinsics: np.ndarray,
    camtoworlds: np.ndarray,
    entries: list[SplitEntry],
    sample_stride: int,
    min_conf_percentile: float,
    max_points_per_image: int,
) -> tuple[list[SparsePoint], dict[int, list[SparsePoint]]]:
    sparse_points: list[SparsePoint] = []
    by_image: dict[int, list[SparsePoint]] = {entry.camera_id: [] for entry in entries}
    point_id = 1

    for entry in entries:
        conf = confs[entry.index]
        threshold = float(np.percentile(conf, min_conf_percentile))
        pixels = sample_grid(conf.shape[0], conf.shape[1], sample_stride)
        valid = (depths[entry.index][pixels[:, 1], pixels[:, 0]] > 1e-5) & (
            conf[pixels[:, 1], pixels[:, 0]] >= threshold
        )
        pixels = pixels[valid]
        if len(pixels) == 0:
            continue
        if len(pixels) > max_points_per_image:
            scores = conf[pixels[:, 1], pixels[:, 0]]
            keep = np.argsort(scores)[-max_points_per_image:]
            pixels = pixels[keep]

        world_xyz = unproject_depth(depths[entry.index], intrinsics[entry.index], camtoworlds[entry.index], pixels)
        rgbs = images[entry.index][pixels[:, 1], pixels[:, 0]]
        errors = 1.0 / np.clip(conf[pixels[:, 1], pixels[:, 0]], 1e-6, None)
        for xyz, rgb, error, xy in zip(world_xyz, rgbs, errors, pixels):
            point = SparsePoint(
                point_id=point_id,
                xyz=xyz.astype(np.float64),
                rgb=rgb.astype(np.uint8),
                error=float(error),
                image_id=entry.camera_id,
                xy=xy.astype(np.float64),
            )
            sparse_points.append(point)
            by_image[entry.camera_id].append(point)
            point_id += 1
    return sparse_points, by_image


def write_colmap_text_model(
    sparse_dir: Path,
    images: np.ndarray,
    intrinsics: np.ndarray,
    camtoworlds: np.ndarray,
    entries: list[SplitEntry],
    sparse_points: list[SparsePoint],
    by_image: dict[int, list[SparsePoint]],
) -> None:
    ensure_dir(sparse_dir)
    height, width = images.shape[1], images.shape[2]

    with (sparse_dir / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for entry in entries:
            intrinsic = intrinsics[entry.index]
            fx = float(intrinsic[0, 0])
            fy = float(intrinsic[1, 1])
            cx = float(intrinsic[0, 2])
            cy = float(intrinsic[1, 2])
            f.write(f"{entry.camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    with (sparse_dir / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for entry in entries:
            world_to_cam = np.linalg.inv(camtoworlds[entry.index])
            qvec = rotation_matrix_to_qvec(world_to_cam[:3, :3])
            tvec = world_to_cam[:3, 3]
            f.write(
                f"{entry.camera_id} "
                f"{qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                f"{tvec[0]} {tvec[1]} {tvec[2]} "
                f"{entry.camera_id} {entry.image_name}\n"
            )
            obs = by_image.get(entry.camera_id, [])
            if obs:
                triples: list[str] = []
                for point in obs:
                    triples.extend([f"{point.xy[0]:.6f}", f"{point.xy[1]:.6f}", str(point.point_id)])
                f.write(" ".join(triples) + "\n")
            else:
                f.write("\n")

    with (sparse_dir / "points3D.txt").open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        point2d_index = {point.point_id: idx for obs in by_image.values() for idx, point in enumerate(obs)}
        for point in sparse_points:
            rgb = point.rgb.astype(int)
            f.write(
                f"{point.point_id} "
                f"{point.xyz[0]} {point.xyz[1]} {point.xyz[2]} "
                f"{rgb[0]} {rgb[1]} {rgb[2]} "
                f"{point.error} "
                f"{point.image_id} {point2d_index[point.point_id]}\n"
            )


def write_dataset_manifest(
    manifest_path: Path,
    *,
    scene_id: str,
    npz_path: Path,
    frames_dir: Path,
    entries: list[SplitEntry],
    pose_semantics: str,
    pose_scores: dict[str, float],
    sample_stride: int,
    min_conf_percentile: float,
    sparse_points: list[SparsePoint],
) -> None:
    payload = {
        "scene_id": scene_id,
        "npz_path": str(npz_path.resolve()),
        "frames_dir": str(frames_dir.resolve()),
        "pose_semantics": pose_semantics,
        "pose_scores": pose_scores,
        "sample_stride": sample_stride,
        "min_conf_percentile": min_conf_percentile,
        "num_sparse_points": len(sparse_points),
        "splits": [
            {
                "source_index": entry.index,
                "split": entry.split,
                "image_name": entry.image_name,
                "camera_id": entry.camera_id,
            }
            for entry in entries
        ],
        "image_pyramid": ["images", "images_2", "images_4", "images_8"],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

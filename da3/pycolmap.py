from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


@dataclass
class Camera:
    camera_id: int
    camera_type: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class ImageRecord:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str

    def R(self) -> np.ndarray:
        return qvec_to_rotmat(self.qvec)


class SceneManager:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.cameras: dict[int, Camera] = {}
        self.images: dict[int, ImageRecord] = {}
        self.points3D = np.zeros((0, 3), dtype=np.float32)
        self.point3D_errors = np.zeros((0,), dtype=np.float32)
        self.point3D_colors = np.zeros((0, 3), dtype=np.uint8)
        self.point3D_id_to_images: dict[int, list[tuple[int, int]]] = {}
        self.point3D_id_to_point3D_idx: dict[int, int] = {}
        self.name_to_image_id: dict[str, int] = {}

    def _iter_data_lines(self, file_path: Path):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    yield line

    def load_cameras(self) -> None:
        for line in self._iter_data_lines(self.path / "cameras.txt"):
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            fx, fy, cx, cy = map(float, parts[4:8])
            self.cameras[camera_id] = Camera(camera_id, model, width, height, fx, fy, cx, cy)

    def load_images(self) -> None:
        lines = []
        with (self.path / "images.txt").open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.rstrip("\n")
                if stripped.startswith("#"):
                    continue
                lines.append(stripped)
        if len(lines) % 2 != 0:
            raise ValueError("images.txt must contain pairs of lines")
        for idx in range(0, len(lines), 2):
            header = lines[idx].split()
            image_id = int(header[0])
            qvec = np.array(list(map(float, header[1:5])), dtype=np.float64)
            tvec = np.array(list(map(float, header[5:8])), dtype=np.float64)
            camera_id = int(header[8])
            name = header[9]
            self.images[image_id] = ImageRecord(image_id, qvec, tvec, camera_id, name)
            self.name_to_image_id[name] = image_id

    def load_points3D(self) -> None:
        xyzs = []
        errs = []
        rgbs = []
        for point_index, line in enumerate(self._iter_data_lines(self.path / "points3D.txt")):
            parts = line.split()
            point_id = int(parts[0])
            xyzs.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rgbs.append([int(parts[4]), int(parts[5]), int(parts[6])])
            errs.append(float(parts[7]))
            track = parts[8:]
            obs = []
            for i in range(0, len(track), 2):
                obs.append((int(track[i]), int(track[i + 1])))
            self.point3D_id_to_images[point_id] = obs
            self.point3D_id_to_point3D_idx[point_id] = point_index
        self.points3D = np.asarray(xyzs, dtype=np.float32)
        self.point3D_errors = np.asarray(errs, dtype=np.float32)
        self.point3D_colors = np.asarray(rgbs, dtype=np.uint8)

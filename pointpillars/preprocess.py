#!/usr/bin/env python3
"""PointPillars KITTI point-cloud preprocessing and BEV scatter."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

POINT_CLOUD_RANGE = np.array([0.0, -39.68, -3.0, 69.12, 39.68, 1.0], np.float32)
VOXEL_SIZE = np.array([0.16, 0.16, 4.0], np.float32)
GRID_X, GRID_Y = 432, 496
MAX_POINTS = 32
MAX_PILLARS = 40000
PFN_CHANNELS = 10

def load_points(path: Path) -> np.ndarray:
    flat = np.fromfile(path, dtype=np.float32)
    if flat.size % 4:
        raise ValueError(f"{path} does not contain x,y,z,intensity float32 points")
    return flat.reshape(-1, 4)


def voxelize_and_decorate(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Match the checkpoint's legacy 10-channel PillarFeatureNet input."""
    lower = POINT_CLOUD_RANGE[:3]
    upper = POINT_CLOUD_RANGE[3:]
    keep = np.all((points[:, :3] >= lower) & (points[:, :3] < upper), axis=1)
    points = points[keep]

    xyz_index = np.floor((points[:, :3] - lower) / VOXEL_SIZE).astype(np.int32)
    pillars: OrderedDict[tuple[int, int, int], list[np.ndarray]] = OrderedDict()
    for point, (x_idx, y_idx, z_idx) in zip(points, xyz_index):
        key = (int(z_idx), int(y_idx), int(x_idx))
        bucket = pillars.get(key)
        if bucket is None:
            if len(pillars) >= MAX_PILLARS:
                continue
            bucket = []
            pillars[key] = bucket
        if len(bucket) < MAX_POINTS:
            bucket.append(point.copy())

    count = len(pillars)
    decorated = np.zeros((MAX_PILLARS, MAX_POINTS, PFN_CHANNELS), np.float32)
    coordinates = np.zeros((count, 3), np.int32)  # z, y, x

    for pillar_index, ((z_idx, y_idx, x_idx), bucket) in enumerate(pillars.items()):
        raw = np.asarray(bucket, dtype=np.float32)
        num_points = raw.shape[0]
        mean_xyz = raw[:, :3].mean(axis=0, keepdims=True)
        cluster = raw[:, :3] - mean_xyz
        center = np.array([
            x_idx * VOXEL_SIZE[0] + VOXEL_SIZE[0] / 2 + POINT_CLOUD_RANGE[0],
            y_idx * VOXEL_SIZE[1] + VOXEL_SIZE[1] / 2 + POINT_CLOUD_RANGE[1],
            z_idx * VOXEL_SIZE[2] + VOXEL_SIZE[2] / 2 + POINT_CLOUD_RANGE[2],
        ], np.float32)
        center_offset = raw[:, :3] - center

        # The training code used legacy=True and a 3D center slice.  That
        # slice aliases raw xyz, so the first xyz and final xyz are identical.
        features = np.concatenate([
            center_offset,
            raw[:, 3:4],
            cluster,
            center_offset,
        ], axis=1)
        decorated[pillar_index, :num_points] = features
        coordinates[pillar_index] = (z_idx, y_idx, x_idx)

    return decorated, coordinates, int(points.shape[0])



def scatter(pillar_features: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    spatial = np.zeros((1, 64, GRID_Y, GRID_X), np.float32)
    count = coordinates.shape[0]
    y = coordinates[:, 1]
    x = coordinates[:, 2]
    # NumPy advanced indexing returns [pillar_count, channels] here.
    spatial[0, :, y, x] = pillar_features[:count]
    return spatial

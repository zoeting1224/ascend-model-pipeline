#!/usr/bin/env python3
"""PointPillars KITTI anchor decoding and rotated NMS."""

from __future__ import annotations

import math

import cv2
import numpy as np

CLASSES = ["Pedestrian", "Cyclist", "Car"]
FEATURE_H, FEATURE_W = 248, 216
SCORE_THRESHOLD = 0.1
NMS_THRESHOLD = 0.01
NMS_PRE = 4096
MAX_DETECTIONS = 100
DIR_OFFSET = -math.pi / 2
DIR_LIMIT_OFFSET = 0.0

ANCHOR_RANGES = np.array([
    [0.0, -39.68, -0.60, 69.12, 39.68, -0.60],
    [0.0, -39.68, -0.60, 69.12, 39.68, -0.60],
    [0.0, -39.68, -1.78, 69.12, 39.68, -1.78],
], np.float32)
ANCHOR_SIZES = np.array([
    [0.80, 0.60, 1.73],
    [1.76, 0.60, 1.73],
    [3.90, 1.60, 1.56],
], np.float32)
ANCHOR_ROTATIONS = np.array([0.0, 1.57], np.float32)

def sigmoid(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


def generate_anchors() -> np.ndarray:
    x_step = (ANCHOR_RANGES[0, 3] - ANCHOR_RANGES[0, 0]) / FEATURE_W
    y_step = (ANCHOR_RANGES[0, 4] - ANCHOR_RANGES[0, 1]) / FEATURE_H
    xs = ANCHOR_RANGES[0, 0] + (np.arange(FEATURE_W, dtype=np.float32) + 0.5) * x_step
    ys = ANCHOR_RANGES[0, 1] + (np.arange(FEATURE_H, dtype=np.float32) + 0.5) * y_step
    anchors = np.empty((FEATURE_H, FEATURE_W, 3, 2, 7), np.float32)
    for cls_index in range(3):
        for rot_index, rotation in enumerate(ANCHOR_ROTATIONS):
            anchors[:, :, cls_index, rot_index, 0] = xs[None, :]
            anchors[:, :, cls_index, rot_index, 1] = ys[:, None]
            anchors[:, :, cls_index, rot_index, 2] = ANCHOR_RANGES[cls_index, 2]
            anchors[:, :, cls_index, rot_index, 3:6] = ANCHOR_SIZES[cls_index]
            anchors[:, :, cls_index, rot_index, 6] = rotation
    return anchors.reshape(-1, 7)


def decode_boxes(anchors: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    diagonal = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 3] ** 2)
    anchor_center_z = anchors[:, 2] + anchors[:, 5] / 2
    boxes = np.empty_like(deltas, dtype=np.float32)
    boxes[:, 0] = deltas[:, 0] * diagonal + anchors[:, 0]
    boxes[:, 1] = deltas[:, 1] * diagonal + anchors[:, 1]
    gravity_z = deltas[:, 2] * anchors[:, 5] + anchor_center_z
    boxes[:, 3] = np.exp(deltas[:, 3]) * anchors[:, 3]
    boxes[:, 4] = np.exp(deltas[:, 4]) * anchors[:, 4]
    boxes[:, 5] = np.exp(deltas[:, 5]) * anchors[:, 5]
    boxes[:, 2] = gravity_z - boxes[:, 5] / 2
    boxes[:, 6] = deltas[:, 6] + anchors[:, 6]
    return boxes


def rotated_nms(boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    rotated_rects = []
    for box in boxes:
        # OpenCV uses degrees. Negating all angles preserves pairwise IoU and
        # accounts for its image-coordinate clockwise convention.
        rotated_rects.append((
            (float(box[0]), float(box[1])),
            (float(box[3]), float(box[4])),
            float(-box[6] * 180.0 / math.pi),
        ))
    indices = cv2.dnn.NMSBoxesRotated(
        rotated_rects,
        scores.astype(float).tolist(),
        SCORE_THRESHOLD,
        NMS_THRESHOLD,
    )
    if indices is None or len(indices) == 0:
        return np.empty((0,), np.int64)
    return np.asarray(indices, dtype=np.int64).reshape(-1)


def limit_period(value: np.ndarray, offset: float, period: float) -> np.ndarray:
    return value - np.floor(value / period + offset) * period


def postprocess(cls_output: np.ndarray, box_output: np.ndarray,
                dir_output: np.ndarray) -> list[dict]:
    scores = sigmoid(cls_output[0].transpose(1, 2, 0).reshape(-1, 3))
    deltas = box_output[0].transpose(1, 2, 0).reshape(-1, 7)
    directions = dir_output[0].transpose(1, 2, 0).reshape(-1, 2).argmax(axis=1)
    anchors = generate_anchors()

    max_scores = scores.max(axis=1)
    if max_scores.size > NMS_PRE:
        top = np.argpartition(max_scores, -NMS_PRE)[-NMS_PRE:]
        top = top[np.argsort(max_scores[top])[::-1]]
        scores, deltas, directions, anchors = (
            scores[top], deltas[top], directions[top], anchors[top]
        )

    decoded = decode_boxes(anchors, deltas)
    collected = []
    for class_index, class_name in enumerate(CLASSES):
        mask = scores[:, class_index] > SCORE_THRESHOLD
        if not np.any(mask):
            continue
        class_boxes = decoded[mask].copy()
        class_scores = scores[mask, class_index]
        class_directions = directions[mask]
        selected = rotated_nms(class_boxes, class_scores)
        for local_index in selected:
            box = class_boxes[local_index]
            direction = int(class_directions[local_index])
            yaw = limit_period(
                np.asarray([box[6] - DIR_OFFSET], np.float32),
                DIR_LIMIT_OFFSET,
                math.pi,
            )[0]
            box[6] = yaw + DIR_OFFSET + math.pi * direction
            collected.append({
                "class_id": class_index,
                "class_name": class_name,
                "score": float(class_scores[local_index]),
                "box_lidar_bottom_center": {
                    "x": float(box[0]), "y": float(box[1]), "z": float(box[2]),
                    "width_x": float(box[3]), "length_y": float(box[4]),
                    "height_z": float(box[5]), "yaw": float(box[6]),
                },
            })
    collected.sort(key=lambda item: item["score"], reverse=True)
    return collected[:MAX_DETECTIONS]



#!/usr/bin/env python3
"""End-to-end PointPillars inference using reusable AscendCL helpers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.acl_runner import ACLModel, ACLRuntime
from pointpillars.postprocess import CLASSES, postprocess
from pointpillars.preprocess import (
    MAX_PILLARS,
    load_points,
    scatter,
    voxelize_and_decorate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    total_start = time.perf_counter()
    points = load_points(args.points)
    preprocess_start = time.perf_counter()
    decorated, coordinates, in_range_points = voxelize_and_decorate(points)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000
    print(f"points: total={points.shape[0]}, in_range={in_range_points}")
    print(f"pillars: {coordinates.shape[0]} / {MAX_PILLARS}")

    with ACLRuntime(args.device):
        with ACLModel(args.model_dir / "pillar_encoder.om") as encoder:
            encoder_outputs, enc_h2d, enc_exec, enc_d2h = encoder.execute([decorated])
        scatter_start = time.perf_counter()
        spatial = scatter(encoder_outputs[0], coordinates)
        scatter_ms = (time.perf_counter() - scatter_start) * 1000
        with ACLModel(args.model_dir / "backbone_head.om") as backbone:
            head_outputs, head_h2d, head_exec, head_d2h = backbone.execute([spatial])

    post_start = time.perf_counter()
    detections = postprocess(*head_outputs)
    post_ms = (time.perf_counter() - post_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    report = {
        "input": str(args.points),
        "point_count": int(points.shape[0]),
        "in_range_point_count": in_range_points,
        "pillar_count": int(coordinates.shape[0]),
        "classes": CLASSES,
        "detections": detections,
        "timing_ms": {
            "cpu_preprocess": preprocess_ms,
            "encoder_h2d": enc_h2d,
            "encoder_execute": enc_exec,
            "encoder_d2h": enc_d2h,
            "cpu_scatter": scatter_ms,
            "head_h2d": head_h2d,
            "head_execute": head_exec,
            "head_d2h": head_d2h,
            "cpu_postprocess": post_ms,
            "total_python_wall": total_ms,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"detections: {len(detections)}")
    for detection in detections[:20]:
        box = detection["box_lidar_bottom_center"]
        print(
            f"{detection['class_name']:10s} score={detection['score']:.4f} "
            f"xyz=({box['x']:.2f},{box['y']:.2f},{box['z']:.2f}) "
            f"wlh=({box['width_x']:.2f},{box['length_y']:.2f},{box['height_z']:.2f}) "
            f"yaw={box['yaw']:.3f}"
        )
    print("timing_ms:", json.dumps(report["timing_ms"], indent=2))
    print("saved:", args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Execute arbitrary static-shape OM models once with zero inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from common.acl_runner import ACLModel, ACLRuntime
except ModuleNotFoundError:  # Supports execution by absolute script path.
    from acl_runner import ACLModel, ACLRuntime


def summarize(index: int, value: np.ndarray) -> None:
    finite = bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else True
    message = f"OUT {index}: shape={list(value.shape)} dtype={value.dtype} finite={finite}"
    if value.size and np.issubdtype(value.dtype, np.number):
        message += (
            f" min={float(value.min()):.6g} max={float(value.max()):.6g}"
            f" mean={float(value.mean()):.6g}"
        )
    print(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+", help="one or more OM files")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    with ACLRuntime(args.device):
        for path in args.models:
            with ACLModel(path) as model:
                outputs, h2d_ms, execute_ms, d2h_ms = model.execute(model.zero_inputs())
                print(f"\nPASS: {path}")
                print(
                    f"H2D={h2d_ms:.3f} ms execute={execute_ms:.3f} ms "
                    f"D2H={d2h_ms:.3f} ms (first run may include warm-up)"
                )
                for index, output in enumerate(outputs):
                    summarize(index, output)


if __name__ == "__main__":
    main()

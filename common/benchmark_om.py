#!/usr/bin/env python3
"""Benchmark one or more OM models with AISBench."""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path, help="OM model files")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-count", type=int, default=10)
    parser.add_argument("--loop", type=int, default=100)
    parser.add_argument(
        "--pure-data-type",
        choices=("zero", "random"),
        default="zero",
        help="generated input data when no real input is supplied",
    )
    parser.add_argument(
        "--extra-ais-arg",
        action="append",
        default=[],
        help="additional complete AISBench argument; may be repeated",
    )
    args = parser.parse_args()

    if importlib.util.find_spec("ais_bench") is None:
        raise RuntimeError("ais_bench is not installed in the current Python environment")
    if args.batch_size < 1 or args.warmup_count < 0 or args.loop < 1:
        parser.error("batch size and loop must be positive; warmup count cannot be negative")

    missing = [str(model) for model in args.models if not model.is_file()]
    if missing:
        raise FileNotFoundError("missing OM model(s): " + ", ".join(missing))

    for index, model in enumerate(args.models, start=1):
        command = [
            sys.executable,
            "-m",
            "ais_bench",
            "--model",
            str(model),
            "--device",
            str(args.device),
            "--batchsize",
            str(args.batch_size),
            "--warmup_count",
            str(args.warmup_count),
            "--loop",
            str(args.loop),
            "--pure_data_type",
            args.pure_data_type,
            "--display_all_summary",
            "true",
            *args.extra_ais_arg,
        ]
        print(f"\n=== [{index}/{len(args.models)}] {model} ===", flush=True)
        print("RUN:", " ".join(shlex.quote(item) for item in command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

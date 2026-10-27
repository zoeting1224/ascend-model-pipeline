#!/usr/bin/env python3
"""Convert one ONNX model to OM through ATC with reusable CLI arguments."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output prefix without .om")
    parser.add_argument("--soc-version", default="Ascend310P1")
    parser.add_argument("--input-format", help="for example NCHW or ND")
    parser.add_argument("--input-shape", help="for example images:1,3,640,640")
    parser.add_argument("--precision-mode", default="allow_fp32_to_fp16")
    parser.add_argument("--log", default="info")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument(
        "--extra-atc-arg", action="append", default=[],
        help="additional complete ATC argument; may be repeated",
    )
    args = parser.parse_args()

    atc = shutil.which("atc")
    if atc is None:
        raise RuntimeError("atc is not in PATH; source the CANN set_env.sh first")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        atc,
        "--framework=5",
        f"--model={args.model}",
        f"--output={args.output}",
        f"--soc_version={args.soc_version}",
        f"--precision_mode={args.precision_mode}",
        f"--log={args.log}",
    ]
    if args.input_format:
        command.append(f"--input_format={args.input_format}")
    if args.input_shape:
        command.append(f"--input_shape={args.input_shape}")
    command.extend(args.extra_atc_arg)

    print("RUN:", " ".join(shlex.quote(item) for item in command))
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        with args.log_file.open("w", encoding="utf-8") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        print("log:", args.log_file)
    else:
        subprocess.run(command, check=True)
    output_file = args.output.with_suffix(".om")
    if not output_file.is_file():
        raise RuntimeError(f"ATC returned successfully but {output_file} was not created")
    print(f"OK: {output_file} ({output_file.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()

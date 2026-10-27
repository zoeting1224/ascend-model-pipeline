#!/usr/bin/env python3
"""Print input and output metadata for arbitrary OM files."""

from __future__ import annotations

import argparse
from pathlib import Path

import acl


def check(ret: int, operation: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{operation} failed, ret={ret}")


def inspect_model(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    desc = acl.mdl.create_desc()
    if desc is None:
        raise RuntimeError("acl.mdl.create_desc returned None")
    try:
        check(acl.mdl.get_desc_from_file(desc, str(path)), f"read {path}")
        print(f"\nMODEL: {path}")
        for index in range(acl.mdl.get_num_inputs(desc)):
            dims, ret = acl.mdl.get_input_dims(desc, index)
            check(ret, f"get input {index} dims")
            print(
                "IN ", index, acl.mdl.get_input_name_by_index(desc, index), dims,
                "dtype=", acl.mdl.get_input_data_type(desc, index),
                "bytes=", acl.mdl.get_input_size_by_index(desc, index),
            )
        for index in range(acl.mdl.get_num_outputs(desc)):
            dims, ret = acl.mdl.get_output_dims(desc, index)
            check(ret, f"get output {index} dims")
            print(
                "OUT", index, acl.mdl.get_output_name_by_index(desc, index), dims,
                "dtype=", acl.mdl.get_output_data_type(desc, index),
                "bytes=", acl.mdl.get_output_size_by_index(desc, index),
            )
    finally:
        acl.mdl.destroy_desc(desc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+", help="one or more OM files")
    args = parser.parse_args()
    check(acl.init(), "acl.init")
    try:
        for model in args.models:
            inspect_model(model)
    finally:
        check(acl.finalize(), "acl.finalize")


if __name__ == "__main__":
    main()

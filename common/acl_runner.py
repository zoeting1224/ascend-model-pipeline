#!/usr/bin/env python3
"""Reusable AscendCL runtime for static-shape OM models."""

from __future__ import annotations

import time
from pathlib import Path

import acl
import numpy as np


ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

# aclDataType values used by CANN. bfloat16 is represented as raw uint16
# because NumPy support varies between installations.
ACL_DTYPE_TO_NUMPY = {
    0: np.dtype(np.float32),
    1: np.dtype(np.float16),
    2: np.dtype(np.int8),
    3: np.dtype(np.int32),
    4: np.dtype(np.uint8),
    6: np.dtype(np.int16),
    7: np.dtype(np.uint16),
    8: np.dtype(np.uint32),
    9: np.dtype(np.int64),
    10: np.dtype(np.uint64),
    11: np.dtype(np.float64),
    12: np.dtype(np.bool_),
    16: np.dtype(np.uint16),
}


def check(ret: int, operation: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{operation} failed, ret={ret}")


def numpy_dtype(acl_dtype: int) -> np.dtype:
    try:
        return ACL_DTYPE_TO_NUMPY[acl_dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported ACL data type: {acl_dtype}") from exc


def tensor_dims(desc, index: int, is_input: bool) -> list[int]:
    getter = acl.mdl.get_input_dims if is_input else acl.mdl.get_output_dims
    info, ret = getter(desc, index)
    check(ret, "get tensor dims")
    dims = list(info["dims"])
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(
            f"dynamic or invalid tensor shape {dims}; set dynamic dimensions "
            "before using the static runner"
        )
    return dims


class ACLRuntime:
    """Initialize one Ascend device for use in a with statement."""

    def __init__(self, device: int = 0) -> None:
        self.device = device
        self.initialized = False
        self.device_set = False

    def __enter__(self) -> "ACLRuntime":
        check(acl.init(), "acl.init")
        self.initialized = True
        try:
            check(acl.rt.set_device(self.device), f"set device {self.device}")
            self.device_set = True
        except Exception:
            acl.finalize()
            self.initialized = False
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        errors = []
        if self.device_set:
            ret = acl.rt.reset_device(self.device)
            if ret != 0:
                errors.append(f"reset device failed, ret={ret}")
            self.device_set = False
        if self.initialized:
            ret = acl.finalize()
            if ret != 0:
                errors.append(f"acl.finalize failed, ret={ret}")
            self.initialized = False
        if errors and exc_type is None:
            raise RuntimeError("; ".join(errors))


class ACLModel:
    """Load and execute one static-shape OM model."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.model_id, ret = acl.mdl.load_from_file(str(self.path))
        check(ret, f"load {self.path}")
        self.desc = acl.mdl.create_desc()
        check(acl.mdl.get_desc(self.desc, self.model_id), "get model desc")

    def close(self) -> None:
        if self.desc is not None:
            acl.mdl.destroy_desc(self.desc)
            self.desc = None
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
            self.model_id = None

    def __enter__(self) -> "ACLModel":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def input_spec(self, index: int) -> tuple[list[int], np.dtype, int]:
        dims = tensor_dims(self.desc, index, True)
        dtype = numpy_dtype(acl.mdl.get_input_data_type(self.desc, index))
        size = acl.mdl.get_input_size_by_index(self.desc, index)
        return dims, dtype, size

    def output_spec(self, index: int) -> tuple[list[int], np.dtype, int]:
        dims = tensor_dims(self.desc, index, False)
        dtype = numpy_dtype(acl.mdl.get_output_data_type(self.desc, index))
        size = acl.mdl.get_output_size_by_index(self.desc, index)
        return dims, dtype, size

    def zero_inputs(self) -> list[np.ndarray]:
        return [
            np.zeros(dims, dtype=dtype)
            for dims, dtype, _ in (
                self.input_spec(i)
                for i in range(acl.mdl.get_num_inputs(self.desc))
            )
        ]

    def execute(
        self, inputs: list[np.ndarray]
    ) -> tuple[list[np.ndarray], float, float, float]:
        input_count = acl.mdl.get_num_inputs(self.desc)
        if len(inputs) != input_count:
            raise ValueError(f"received {len(inputs)} inputs, OM expects {input_count}")

        input_dataset = acl.mdl.create_dataset()
        output_dataset = acl.mdl.create_dataset()
        input_resources = []
        output_resources = []

        try:
            h2d_start = time.perf_counter()
            for index, value in enumerate(inputs):
                dims, dtype, expected = self.input_spec(index)
                host = np.ascontiguousarray(value, dtype=dtype)
                if list(host.shape) != dims:
                    raise ValueError(
                        f"input {index} shape {list(host.shape)} != OM shape {dims}"
                    )
                if host.nbytes != expected:
                    raise ValueError(
                        f"input {index} bytes {host.nbytes} != OM bytes {expected}"
                    )
                device_ptr, ret = acl.rt.malloc(expected, ACL_MEM_MALLOC_HUGE_FIRST)
                check(ret, "malloc input")
                check(
                    acl.rt.memcpy(
                        device_ptr, expected, acl.util.numpy_to_ptr(host), expected,
                        ACL_MEMCPY_HOST_TO_DEVICE,
                    ),
                    "input H2D",
                )
                data_buffer = acl.create_data_buffer(device_ptr, expected)
                input_dataset, ret = acl.mdl.add_dataset_buffer(input_dataset, data_buffer)
                check(ret, "add input buffer")
                input_resources.append((device_ptr, data_buffer, host))
            h2d_ms = (time.perf_counter() - h2d_start) * 1000

            outputs = []
            for index in range(acl.mdl.get_num_outputs(self.desc)):
                dims, dtype, expected = self.output_spec(index)
                host = np.empty(dims, dtype=dtype)
                if host.nbytes != expected:
                    raise ValueError(
                        f"output {index} bytes {host.nbytes} != OM bytes {expected}"
                    )
                device_ptr, ret = acl.rt.malloc(expected, ACL_MEM_MALLOC_HUGE_FIRST)
                check(ret, "malloc output")
                data_buffer = acl.create_data_buffer(device_ptr, expected)
                output_dataset, ret = acl.mdl.add_dataset_buffer(output_dataset, data_buffer)
                check(ret, "add output buffer")
                output_resources.append((device_ptr, data_buffer, host))
                outputs.append(host)

            execute_start = time.perf_counter()
            check(acl.mdl.execute(self.model_id, input_dataset, output_dataset), "execute model")
            execute_ms = (time.perf_counter() - execute_start) * 1000

            d2h_start = time.perf_counter()
            for device_ptr, _, host in output_resources:
                check(
                    acl.rt.memcpy(
                        acl.util.numpy_to_ptr(host), host.nbytes, device_ptr, host.nbytes,
                        ACL_MEMCPY_DEVICE_TO_HOST,
                    ),
                    "output D2H",
                )
            d2h_ms = (time.perf_counter() - d2h_start) * 1000
            return outputs, h2d_ms, execute_ms, d2h_ms
        finally:
            for device_ptr, data_buffer, _ in output_resources:
                if data_buffer is not None:
                    acl.destroy_data_buffer(data_buffer)
                if device_ptr:
                    acl.rt.free(device_ptr)
            if output_dataset is not None:
                acl.mdl.destroy_dataset(output_dataset)
            for device_ptr, data_buffer, _ in input_resources:
                if data_buffer is not None:
                    acl.destroy_data_buffer(data_buffer)
                if device_ptr:
                    acl.rt.free(device_ptr)
            if input_dataset is not None:
                acl.mdl.destroy_dataset(input_dataset)

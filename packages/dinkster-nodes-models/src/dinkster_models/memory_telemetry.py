"""Measured memory from the tensor runtime owned by the models worker."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any, cast

from dinkster_api.v1 import MeasuredMemory


def _optional_counter(read: Callable[[], object]) -> int | None:
    try:
        value = read()
        return (
            int(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
            else None
        )
    except Exception:
        return None


@cache
def _nvml_device(uuid: str) -> tuple[Any, Any]:
    nvml = importlib.import_module("pynvml")
    nvml.nvmlInit()
    return nvml, nvml.nvmlDeviceGetHandleByUUID(uuid)


def _nvml_telemetry(properties: Any) -> dict[str, int]:
    # Physical UUIDs remain stable when CUDA_VISIBLE_DEVICES remaps torch indices.
    try:
        uuid = str(properties.uuid)
        if not uuid.startswith("GPU-"):
            uuid = f"GPU-{uuid}"
        nvml, handle = _nvml_device(uuid)
    except Exception:
        return {}
    readings = {
        "driver_free_bytes": lambda: nvml.nvmlDeviceGetMemoryInfo(handle).free,
        "gpu_utilization_percent": lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
        "gpu_temperature_celsius": lambda: nvml.nvmlDeviceGetTemperature(
            handle, nvml.NVML_TEMPERATURE_GPU
        ),
        "gpu_power_milliwatts": lambda: nvml.nvmlDeviceGetPowerUsage(handle),
        "gpu_power_limit_milliwatts": lambda: nvml.nvmlDeviceGetPowerManagementLimit(handle),
    }
    return {
        name: value
        for name, read in readings.items()
        if (value := _optional_counter(read)) is not None
    }


def _device_memory(torch: Any, family: str, index: int) -> MeasuredMemory | None:
    namespace = getattr(torch, family)
    try:
        device = torch.device(f"{family}:{index}")
        free, total = namespace.mem_get_info(device)
        hardware: dict[str, int] = {}
        name: str | None = None
        try:
            properties = namespace.get_device_properties(device)
            name = str(properties.name)
            if family == "cuda" and not getattr(torch.version, "hip", None):
                hardware = _nvml_telemetry(properties)
        except Exception:
            pass
        manager = sys.modules.get("dinkster_inference.model_management")
        return MeasuredMemory(
            free_bytes=int(free),
            total_bytes=int(total),
            torch_allocated_bytes=_optional_counter(lambda: namespace.memory_allocated(device)),
            torch_reserved_bytes=_optional_counter(lambda: namespace.memory_reserved(device)),
            process_id=os.getpid(),
            process_count=1,
            process_rss_bytes=_optional_counter(
                lambda: importlib.import_module("psutil").Process().memory_info().rss
            ),
            pinned_host_bytes=_optional_counter(lambda: cast("Any", manager).TOTAL_PINNED_MEMORY),
            gpu_name=name,
            **hardware,
        )
    except Exception:
        # An unavailable device must not hide measurements from its siblings.
        return None


def vram_telemetry_snapshot() -> Mapping[str, MeasuredMemory]:
    """Report visible CUDA/ROCm and XPU devices; omit unavailable measurements.

    ROCm uses torch's CUDA namespace. CPU-only and MPS runtimes have no
    separately measurable accelerator capacity and return no device rows.
    """
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return {}
    snapshot: dict[str, MeasuredMemory] = {}
    for family in ("cuda", "xpu"):
        try:
            namespace = getattr(torch, family, None)
            if namespace is None or not namespace.is_available():
                continue
            count = int(namespace.device_count())
        except Exception:
            continue
        for index in range(count):
            measured = _device_memory(torch, family, index)
            if measured is not None:
                snapshot[f"vram:{family}:{index}"] = measured
    return snapshot

"""The official models manifest supplies measured worker memory reports."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from dinkster_workers import load_manifest
from dinkster_workers.host import load_telemetry


def test_models_report_each_visible_card_and_omit_cpu_only_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_manifest("packages/dinkster-nodes-models/dinkster-pack.toml")
    probe = load_telemetry(manifest)
    assert probe is not None
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        mem_get_info=lambda device: {"cuda:0": (3_000, 8_000), "cuda:1": (7_000, 24_000)}[device],
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(device=lambda value: value, cuda=cuda)
    )
    measured = probe()
    assert {device: (row.free_bytes, row.total_bytes) for device, row in measured.items()} == {
        "vram:cuda:0": (3_000, 8_000),
        "vram:cuda:1": (7_000, 24_000),
    }
    assert all(row.torch_allocated_bytes is None for row in measured.values())
    cuda.is_available = lambda: False
    assert dict(probe()) == {}


def test_unavailable_device_and_counters_do_not_hide_other_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_manifest("packages/dinkster-nodes-models/dinkster-pack.toml")
    probe = load_telemetry(manifest)
    assert probe is not None

    def memory(device: str) -> tuple[int, int]:
        if device == "cuda:0":
            raise RuntimeError("device unavailable")
        return (3_000, 8_000) if device == "cuda:1" else (9_000, 12_000)

    def allocated(device: str) -> int:
        if device == "cuda:1":
            raise RuntimeError("counter unavailable")
        return 120

    def namespace(count: int) -> SimpleNamespace:
        return SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: count,
            mem_get_info=memory,
            memory_allocated=allocated,
            memory_reserved=lambda device: 0 if device == "cuda:1" else 300,
        )

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(device=lambda value: value, cuda=namespace(2), xpu=namespace(1)),
    )
    measured = probe()
    assert set(measured) == {"vram:cuda:1", "vram:xpu:0"}
    assert measured["vram:cuda:1"].free_bytes == 3_000
    assert measured["vram:cuda:1"].torch_allocated_bytes is None
    assert measured["vram:cuda:1"].torch_reserved_bytes == 0
    assert measured["vram:xpu:0"].total_bytes == 12_000
    assert measured["vram:xpu:0"].torch_allocated_bytes == 120
    assert measured["vram:xpu:0"].torch_reserved_bytes == 300

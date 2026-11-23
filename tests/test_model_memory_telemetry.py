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

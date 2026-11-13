"""SeedVR2 temporal chunking and merging."""

import math

import dinkster_seedvr.seedvr_temporal as seedvr_temporal
import pytest
import torch
from dinkster_seedvr.seedvr_temporal import SeedVR2TemporalChunk, SeedVR2TemporalMerge


def _latent(frames: int = 7) -> dict[str, object]:
    return {
        "samples": torch.arange(16 * frames * 2 * 2, dtype=torch.float64).reshape(
            1, 16, frames, 2, 2
        ),
        "metadata": {"source": "seedvr"},
        "noise_mask": torch.zeros(1),
    }


def test_manual_chunks_and_merges_with_runt_final_chunk(monkeypatch) -> None:
    monkeypatch.setattr(seedvr_temporal, "initialize_inference_process", lambda: None)
    source = _latent()
    chunks = SeedVR2TemporalChunk.execute(
        latent=source,
        temporal_overlap=2,
        chunking_mode={"chunking_mode": "manual", "frames_per_chunk": 13},
    )
    assert [chunk["samples"].shape[2] for chunk in chunks["latents"]] == [4, 4, 3]
    assert chunks["temporal_overlap"] == 2
    merged = SeedVR2TemporalMerge.execute(
        latents=chunks["latents"], temporal_overlap=chunks["temporal_overlap"]
    )["latent"]
    assert merged["samples"].shape == source["samples"].shape
    assert merged["samples"].dtype == source["samples"].dtype
    assert merged["metadata"] == source["metadata"]
    assert "noise_mask" not in merged


def test_hann_crossfade_weights_overlap(monkeypatch) -> None:
    monkeypatch.setattr(seedvr_temporal, "initialize_inference_process", lambda: None)
    overlap = 6
    first = {"samples": torch.zeros((1, 16, 7, 1, 1), dtype=torch.float64)}
    second = {"samples": torch.ones((1, 16, 7, 1, 1), dtype=torch.float64)}
    merged = SeedVR2TemporalMerge.execute(latents=[first, second], temporal_overlap=overlap)[
        "latent"
    ]["samples"]
    assert torch.equal(merged[:, :, 0], first["samples"][:, :, 0])
    assert torch.equal(merged[:, :, -1], second["samples"][:, :, -1])
    expected = []
    for index in range(overlap):
        ramp = min(1.0, max(0.0, (index / (overlap - 1) - 1.0 / 3.0) / (1.0 / 3.0)))
        previous = 0.5 + 0.5 * math.cos(math.pi * ramp)
        expected.append(1.0 - previous)
    actual = merged[0, 0, 1 : 1 + overlap, 0, 0].tolist()
    assert actual == pytest.approx(expected)
    assert actual[3] != pytest.approx(3 / (overlap - 1))

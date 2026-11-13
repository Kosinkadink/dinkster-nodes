"""CPU transport checks for structural hook strengths."""

from __future__ import annotations

import torch
from dinkster_api.v1 import (
    FLOATS_TYPE,
    materialize_latent,
)
from dinkster_inference_wire import register_inference_types
from dinkster_values import TypeRegistry


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def test_floats_tensor_codec_preserves_dtype_and_bytes() -> None:
    source = torch.tensor([0.5, -3.25], dtype=torch.bfloat16)
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(FLOATS_TYPE)
    decoded = spec.decode(spec.encode(source))
    restored = materialize_latent(decoded)

    assert restored.dtype == source.dtype
    assert restored.shape == source.shape
    assert _raw_bytes(restored) == _raw_bytes(source)

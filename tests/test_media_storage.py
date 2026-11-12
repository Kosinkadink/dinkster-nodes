"""Pack registration preserves audio storage and compositor memory accounting."""

from multiprocessing.shared_memory import SharedMemory
from typing import Any, cast

import numpy as np
import pytest
from dinkster_api.v1 import AUDIO_TYPE
from dinkster_values import COST_META_KEY


@pytest.mark.parametrize("host_registration", [False, True])
@pytest.mark.parametrize("dtype", ["int16", "float32", "float16", "float64", "int32"])
@pytest.mark.parametrize("shm", [False, True])
def test_audio_storage_facts_match_canonical_payload_across_boundary(
    host_registration: bool, dtype: str, shm: bool
) -> None:
    from dinkster_audio import register_types as register_audio_types
    from dinkster_inference_wire import register_inference_types
    from dinkster_values import TypeRegistry
    from dinkster_workers import ValueCodec
    from dinkster_workers.boundary import release_segment

    registry = TypeRegistry()
    if host_registration:
        register_inference_types(registry)
    else:
        register_audio_types(registry)
    original = np.arange(24, dtype=dtype).reshape(1, 2, 12)
    value = registry.wrap(AUDIO_TYPE, {"waveform": original, "sample_rate": 48000})
    resolved = cast(dict[str, Any], value.resolve())
    stored = cast(dict[str, Any], resolved["source"])["pcm"]
    expected_dtype = "int16" if dtype == "int16" else "float32"
    assert stored.dtype == expected_dtype
    assert original.dtype == dtype
    assert (stored is original) == (dtype in ("int16", "float32"))
    assert value.meta.get("storage_dtype") == ("int16" if dtype == "int16" else "fp32")
    assert value.meta.get(COST_META_KEY) == {"ram": stored.nbytes}
    codec = ValueCodec(registry, shm_threshold=1 if shm else 1000000)
    blobs: list[bytes] = []
    segments: list[SharedMemory] = []
    wire, sent = codec.encode(value, blobs, segments)
    try:
        received, _ = codec.decode(wire, blobs, [])
        received_value = cast(dict[str, Any], received.resolve())
        loaded = cast(dict[str, Any], received_value["source"])["pcm"]
        assert loaded.dtype == expected_dtype
        assert received.meta.get("storage_dtype") == value.meta.get("storage_dtype")
        assert received.meta.get(COST_META_KEY) == {"ram": sent.size_bytes}
        assert received.fingerprint == value.fingerprint
        np.testing.assert_array_equal(loaded, stored)
    finally:
        for segment in segments:
            release_segment(segment)


def test_inline_layer_rasters_share_array_accounting_without_duplicate_layers() -> None:
    from dinkster_image.compositor_types import (
        CompositorSourceLayer,
        LayerStack,
        layer_stack_meta,
    )

    layer = CompositorSourceLayer(
        image=np.zeros((1, 4, 4, 3), dtype=np.float32),
        mask=np.ones((1, 4, 4), dtype=np.float32),
        name="pixels",
        x=0,
        y=0,
        width=4,
        height=4,
    )
    meta = layer_stack_meta(LayerStack((layer, layer)))
    assert meta["layers"] == 2
    assert meta["storage_dtype"] == ["fp32"]
    assert layer.mask is not None
    expected = layer.image.nbytes + layer.mask.nbytes
    assert meta[COST_META_KEY] == {"ram": expected}

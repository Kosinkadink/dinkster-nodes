"""Pack media registrars preserve inline storage and mask contracts."""

from typing import cast

import numpy as np
import pytest
from dinkster_values import TypeRegistry, annotate_mask, encode_image_array


@pytest.mark.parametrize("image_first", [False, True])
def test_pack_registration_order_preserves_mask_contract(image_first: bool) -> None:
    from dinkster_image import register_types as register_image_types
    from dinkster_video import register_types as register_video_types

    registry = TypeRegistry()
    registrations = [register_image_types, register_video_types]
    for register in registrations if image_first else reversed(registrations):
        register(registry)
    mask = annotate_mask(np.zeros((1, 1, 1), np.float32), polarity="transparency", semantic="alpha")
    value = registry.wrap("MASK", mask)
    assert value.meta.get("polarity") == "transparency"
    assert value.meta.get("semantic") == "alpha"
    assert "channels" not in value.meta.entries
    for type_id in ("MASK", "IMAGE"):
        assert registry.spec(type_id).validate_encoded is not None
        assert registry.spec(type_id).validate_encoded_buffer is not None


@pytest.mark.parametrize(
    ("registration", "type_id"),
    [
        (registration, type_id)
        for registration in (
            "image",
            "3d",
            "3d_models",
            "minimax",
            "upscale",
            "video",
            "vision_background_removal",
            "vision_depth_anything_3",
            "vision_marigold",
            "vision_mediapipe",
            "vision_moge",
            "vision_rtdetr",
            "vision_sam3",
            "vision_sam3d_body",
            "vision_sdpose",
            "void",
        )
        for type_id in ("IMAGE", "MASK")
    ],
)
def test_inline_media_has_no_external_asset_dependencies(type_id, registration) -> None:
    import importlib

    from dinkster_assets import AssetRef, digest_bytes
    from dinkster_assets.value import value_asset_references
    from dinkster_values import default_encode
    from dinkster_workers.boundary import EncodedPayloadValidationError, ValueCodec

    registry = TypeRegistry()
    importlib.import_module(f"dinkster_{registration}").register_types(registry)
    array = np.arange(6, dtype=np.float32).reshape((1, 2, 3) if type_id == "MASK" else (1, 1, 2, 3))
    wrapped = registry.wrap(type_id, array)
    assert value_asset_references({"media": wrapped}, complete_registry=registry) == {}
    codec = ValueCodec(registry, shm_threshold=1 << 20)
    blobs, segments = [], []
    wire, _ = codec.encode(wrapped, blobs, segments)
    assert segments == []
    restored, _ = codec.decode(wire, blobs, [])
    assert value_asset_references({"media": restored}, complete_registry=registry) == {}
    np.testing.assert_array_equal(restored.resolve(), array)
    validator = registry.spec(type_id).validate_encoded_buffer
    assert validator is not None
    invented_ref = AssetRef(digest_bytes(b"outside"), "outside.bin", 7).to_wire()
    with memoryview(encode_image_array(array)) as data:
        validator(data, {**wrapped.meta.entries, "asset_refs": []})
        for references in ([invented_ref], None):
            with pytest.raises(ValueError, match="asset_refs"):
                validator(data, {**wrapped.meta.entries, "asset_refs": references})
    blobs[cast(int, wire["metaBlob"])] = default_encode(
        {**wrapped.meta.entries, "asset_refs": [invented_ref]}
    )
    with pytest.raises(EncodedPayloadValidationError, match="asset_refs"):
        codec.decode(wire, blobs, [])

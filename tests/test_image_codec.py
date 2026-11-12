"""The image pack registers canonical image bytes, renditions and asset providers."""

import struct

import dinkster_values.image_codec as image_codec_module
import numpy as np
import pytest
from dinkster_api.v1 import IMAGE_TYPE as COMFY_IMAGE_TYPE
from dinkster_api.v1 import MASK_TYPE as COMFY_MASK_TYPE
from dinkster_image import register_types as register_image_types
from dinkster_values import (
    PNG_CONTAINER_VERSION,
    EncodedPayload,
    TypeRegistry,
    Value,
    ValueMeta,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    prepare_image_array_encoding,
    render_mask_png,
)


def png_header(data: bytes) -> tuple[int, int, int]:
    """Read the image dimensions and color type from its IHDR chunk."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    width, height, depth, color_type = struct.unpack(">IIBB", data[16:26])
    assert depth == 8
    return width, height, color_type


def make_host_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_image_types(registry)
    return registry


def test_comfy_host_types_declare_png_rendition() -> None:
    registry = make_host_registry()
    assert COMFY_IMAGE_TYPE in registry
    assert registry.spec(COMFY_IMAGE_TYPE).prepare_buffer_encoding is prepare_image_array_encoding
    specs = registry.renditions_of(COMFY_IMAGE_TYPE)
    assert [(s.kind, s.mime, s.default) for s in specs] == [("png", "image/png", True)]
    assert specs[0].version == PNG_CONTAINER_VERSION


def test_comfy_host_mask_shares_image_bytes_with_mask_rendition() -> None:
    """MASK gets IMAGE's byte contract and renders through the mask PNG."""
    registry = make_host_registry()
    assert COMFY_MASK_TYPE in registry
    assert registry.spec(COMFY_MASK_TYPE).prepare_buffer_encoding is prepare_image_array_encoding
    specs = registry.renditions_of(COMFY_MASK_TYPE)
    assert [(s.kind, s.mime, s.default) for s in specs] == [("png", "image/png", True)]
    mask = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 24
    rendition = registry.render(registry.wrap(COMFY_MASK_TYPE, mask), "png")
    assert rendition.mime == "image/png"
    assert rendition.data == render_mask_png(mask)


def test_comfy_image_direct_shm_matches_npy_bytes() -> None:
    from dinkster_workers import ValueCodec
    from dinkster_workers.boundary import release_segment

    registry = make_host_registry()
    array = np.arange(60, dtype=np.float32).reshape(1, 4, 5, 3)
    value = registry.wrap(COMFY_IMAGE_TYPE, array)
    blobs: list[bytes] = []
    segments = []
    wire, stat = ValueCodec(registry, shm_threshold=1).encode(value, blobs, segments)
    try:
        segment = segments[0]
        assert segment.buf is not None
        assert bytes(segment.buf[: stat.size_bytes]) == encode_image_array(array)
        assert wire["fingerprint"] == value.fingerprint
        assert stat.transport == "shm"
        assert stat.reused is False
        assert len(blobs) == 1
    finally:
        for segment in segments:
            release_segment(segment)


def test_large_comfy_image_shm_does_not_materialize_encoded_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    import dinkster_workers.boundary as boundary_module
    from dinkster_workers import ValueCodec
    from dinkster_workers.boundary import release_segment

    registry = make_host_registry()
    array = np.zeros((1, 1280, 1280, 3), dtype=np.float32)
    value = registry.wrap(COMFY_IMAGE_TYPE, array)

    def forbid_bytes_io(*_args: object, **_kwargs: object) -> object:
        pytest.fail("full npy BytesIO encoding was used")

    monkeypatch.setattr(image_codec_module, "io", types.SimpleNamespace(BytesIO=forbid_bytes_io))
    monkeypatch.setattr(
        boundary_module,
        "_create_segment",
        lambda _data: pytest.fail("copy-based shared-memory creation was used"),
    )
    blobs: list[bytes] = []
    segments = []
    wire, stat = ValueCodec(registry, shm_threshold=1).encode(value, blobs, segments)
    try:
        payload = wire["payload"]
        assert isinstance(payload, dict)
        assert payload["transport"] == "shm"
        assert stat.size_bytes > array.nbytes
        assert len(blobs) == 1
        assert segments[0].buf is not None
        assert bytes(segments[0].buf[:6]) == b"\x93NUMPY"
    finally:
        for segment in segments:
            release_segment(segment)


def test_comfy_host_types_registration_is_idempotent() -> None:
    registry = make_host_registry()
    register_image_types(registry)  # second spec / reload: no raise
    assert "ASSET" in registry
    assert len(registry.renditions_of(COMFY_IMAGE_TYPE)) == 1


def test_comfy_host_base_asset_registration_is_independent_of_image() -> None:
    registry = TypeRegistry()
    registry.register(COMFY_IMAGE_TYPE)
    register_image_types(registry)
    register_image_types(registry)
    assert "ASSET" in registry


def test_comfy_host_types_wrap_typed_image_asset_without_manual_base_registration() -> None:
    from dinkster_assets import AssetRef

    registry = make_host_registry()
    ref = AssetRef(digest="blake3:" + "a" * 64, name="input.png", size=3)
    wrapped = registry.wrap("asset<IMAGE>", ref)
    assert wrapped.type_id == "asset<IMAGE>"
    assert wrapped.fingerprint == ref.digest
    resolved = wrapped.resolve()
    assert isinstance(resolved, AssetRef)
    assert resolved.resolver is None


def test_encoded_comfy_image_renders_without_torch() -> None:
    """The /api/values path in miniature: a worker-encoded IMAGE
    arrives as npy bytes (EncodedPayload, exactly what the boundary builds)
    and the host renders PNG from numpy alone."""
    registry = make_host_registry()
    array = np.zeros((1, 2, 3, 3), dtype=np.float32)
    array[0, 0, 0] = [0.0, 1.0, 0.0]
    spec = registry.spec(COMFY_IMAGE_TYPE)
    value = Value(
        type_id=COMFY_IMAGE_TYPE,
        fingerprint=image_array_fingerprint(COMFY_IMAGE_TYPE)(array),
        meta=ValueMeta(dict(image_array_meta(array))),
        payload=EncodedPayload(COMFY_IMAGE_TYPE, spec.encode(array), spec.decode),
    )
    rendition = registry.render(value, "png")
    assert (rendition.kind, rendition.mime) == ("png", "image/png")
    assert png_header(rendition.data) == (3, 2, 2)


def test_comfy_host_types_declare_asset_providers() -> None:
    """Typed assets: the host registers the asset<IMAGE> decoder and
    the IMAGE batch merge under their canonical provider ids."""
    from dinkster_values import (
        IMAGE_BATCH_MERGER_ID,
        IMAGE_FILE_DECODER_ID,
    )

    registry = make_host_registry()
    decoder = registry.asset_decoder_for(COMFY_IMAGE_TYPE)
    assert IMAGE_FILE_DECODER_ID == "dinkster.image-file@3"
    assert decoder is not None and decoder.provider_id == IMAGE_FILE_DECODER_ID
    merger = registry.batch_merge_for(COMFY_IMAGE_TYPE)
    assert merger is not None and merger.provider_id == IMAGE_BATCH_MERGER_ID

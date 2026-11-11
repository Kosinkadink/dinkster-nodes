from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest
from dinkster_image.compositor import CreateLayeredImage
from dinkster_image.types import register_image_types
from dinkster_image_document.document import (
    ImageDocument,
    append_raster,
    apply_commands,
    empty_document,
    flatten,
    raster_png,
)
from dinkster_image_document.format import InvalidDocument, canonical_json
from dinkster_values import TypeRegistry
from dinkster_workers.boundary import ValueCodec


def document() -> ImageDocument:
    return append_raster(
        empty_document(2, 1),
        raster_png(np.array([[[1, 0, 0, 1], [0, 1, 0, 0.5]]], dtype=np.float32)),
    )


def test_legacy_layer_admission_does_not_publish_migrated_rasters(monkeypatch) -> None:
    from dinkster_assets import AssetError, AssetVault
    from dinkster_assets.value import value_asset_references
    from dinkster_image.compositor_types import (
        CompositorSourceLayer,
        LayerStack,
        encode_layer_stack,
    )
    from dinkster_workers.boundary import EncodedPayloadValidationError

    pixels = np.random.default_rng(41).random((1, 64, 64, 3), dtype=np.float32)
    legacy = encode_layer_stack(
        LayerStack(
            (
                CompositorSourceLayer(
                    pixels,
                    None,
                    "Noise",
                    0,
                    0,
                    64,
                    64,
                ),
            )
        )
    )
    registry = TypeRegistry()
    register_image_types(registry)
    codec = ValueCodec(registry)
    blobs, segments = [], []
    wire, _ = codec.encode(registry.wrap("dinkster.layers", document()), blobs, segments)
    payload = cast(dict[str, object], wire["payload"])
    blobs[cast(int, payload["blob"])] = legacy

    def forbid_publication(*args, **kwargs):
        pytest.fail("legacy validation must not publish migrated raster resources")

    monkeypatch.setattr(AssetVault, "writer", forbid_publication)
    received, _ = codec.decode(wire, blobs, [])
    with pytest.raises(AssetError, match="incomplete"):
        value_asset_references({"legacy": received}, complete_registry=registry)
    blobs[cast(int, payload["blob"])] = legacy + b"trailing"
    with pytest.raises(EncodedPayloadValidationError):
        codec.decode(wire, blobs, [])


def test_delta_codecs_bind_edits_and_reject_stale_documents() -> None:
    from dinkster_assets import digest_bytes
    from dinkster_image.layer_recipe import replay_compositor

    value = document()
    identifier = value.to_record()["rootLayerIds"][0]
    delta = {
        "version": 2,
        "documentDigest": digest_bytes(value.data),
        "commands": [{"op": "layer", "id": identifier, "changes": {"visible": False}}],
    }
    registry = TypeRegistry()
    register_image_types(registry)
    for type_id in ("COMPOSITOR", "comfy.COMPOSITOR"):
        spec = registry.spec(type_id)
        decoded = spec.decode(spec.encode(delta))
        assert decoded == delta
        edited, stale = replay_compositor(value, decoded, value)
        assert not stale
        assert np.all(flatten(edited)[1] == 1)
        unchanged, stale = replay_compositor(edited, decoded, edited)
        assert stale and unchanged.data == edited.data


@pytest.mark.parametrize("type_id", ["COMPOSITOR", "comfy.COMPOSITOR"])
def test_compositor_boundary_retains_auxiliary_references_without_retrieving_document(
    type_id, monkeypatch
) -> None:
    from dinkster_assets import AssetRef, AssetVault, digest_bytes
    from dinkster_assets.value import value_asset_references
    from dinkster_image.layer_recipe import replay_compositor
    from dinkster_values import default_encode

    value = document()
    identifier = value.to_record()["rootLayerIds"][0]
    delta = {
        "version": 2,
        "documentDigest": digest_bytes(value.data),
        "commands": [{"op": "layer", "id": identifier, "changes": {"visible": False}}],
    }
    registry = TypeRegistry()
    register_image_types(registry)
    codec = ValueCodec(registry)
    wrapped = registry.wrap(type_id, delta)
    blobs, segments = [], []
    wire, _ = codec.encode(wrapped, blobs, segments)
    expected = canonical_json(delta).encode()
    assert blobs[cast(int, cast(dict[str, object], wire["payload"])["blob"])] == expected
    auxiliary = AssetRef(digest_bytes(b"auxiliary"), "source.json", 9)
    blobs[cast(int, wire["metaBlob"])] = default_encode(
        {**wrapped.meta.entries, "asset_refs": [auxiliary.to_wire()]}
    )

    def forbid_io(*args, **kwargs):
        pytest.fail("compositor admission must not retrieve or publish a document")

    monkeypatch.setattr(ImageDocument, "read_resource", forbid_io)
    monkeypatch.setattr(AssetVault, "writer", forbid_io)
    received, _ = codec.decode(wire, blobs, [])
    assert received.fingerprint == wrapped.fingerprint
    assert received.resolve() == delta
    assert value_asset_references({"delta": received}, complete_registry=registry) == {
        auxiliary.digest: auxiliary
    }
    edited, stale = replay_compositor(value, received.resolve(), value)
    assert not stale
    assert edited.to_record()["layers"][identifier]["visible"] is False
    unchanged, stale = replay_compositor(edited, received.resolve(), edited)
    assert stale and unchanged.data == edited.data
    payload = cast(dict[str, object], wire["payload"])
    blobs[cast(int, payload["blob"])] = json.dumps(delta, indent=2).encode()
    noncanonical, _ = codec.decode(wire, blobs, [])
    assert noncanonical.resolve() == delta
    first = TypeRegistry()
    first_spec = first.register(type_id)
    register_image_types(first)
    assert first.spec(type_id) is first_spec
    assert not first_spec.asset_references_complete


@pytest.mark.parametrize("type_id", ["COMPOSITOR", "comfy.COMPOSITOR"])
@pytest.mark.parametrize("data", [b"not json", b"[]", b'{"unknown":true}'])
def test_compositor_boundary_refuses_invalid_encoding_before_exposure(type_id, data) -> None:
    from dinkster_workers.boundary import EncodedPayloadValidationError

    registry = TypeRegistry()
    register_image_types(registry)
    codec = ValueCodec(registry)
    blobs, segments = [], []
    wire, _ = codec.encode(registry.wrap(type_id, {}), blobs, segments)
    payload = cast(dict[str, object], wire["payload"])
    blobs[cast(int, payload["blob"])] = data
    with pytest.raises(EncodedPayloadValidationError):
        codec.decode(wire, blobs, [])


@pytest.mark.parametrize("type_id", ["COMPOSITOR", "comfy.COMPOSITOR"])
@pytest.mark.parametrize(
    "recipe",
    [
        {},
        {"version": 1, "canvas": {"w": 3, "h": 2}, "inputs": [], "layers": []},
        {"version": 2, "documentDigest": None, "commands": [{"op": "unknown"}]},
    ],
)
def test_compositor_boundary_preserves_existing_legacy_and_deferred_command_forms(
    type_id, recipe
) -> None:
    from dinkster_assets.value import value_asset_references
    from dinkster_image.layer_recipe import replay_compositor

    registry = TypeRegistry()
    register_image_types(registry)
    codec = ValueCodec(registry)
    expected = (
        {"version": 2, "documentDigest": None, "commands": []}
        if type_id == "COMPOSITOR" and recipe == {}
        else recipe
    )
    wrapped = registry.wrap(type_id, recipe)
    blobs, segments = [], []
    wire, _ = codec.encode(wrapped, blobs, segments)
    assert blobs[cast(int, cast(dict[str, object], wire["payload"])["blob"])] == (
        canonical_json(expected).encode()
    )
    received, _ = codec.decode(wire, blobs, [])
    assert received.resolve() == expected
    assert received.fingerprint == wrapped.fingerprint
    assert value_asset_references({"recipe": received}, complete_registry=registry) == {}
    if recipe.get("commands"):
        value = document()
        with pytest.raises(InvalidDocument, match="unknown document command"):
            replay_compositor(value, received.resolve(), value)


@pytest.mark.parametrize("color_space", ["perceptual", "linear"])
def test_compositor_preview_digest_binds_repeated_apply_to_input(monkeypatch, color_space) -> None:
    from dinkster_assets import digest_bytes
    from dinkster_image import compositor as module

    value = document()
    original = value.data
    identifier = value.to_record()["rootLayerIds"][0]
    events = []
    monkeypatch.setattr(module, "report_event", lambda name, data: events.append((name, data)))
    delta = {"version": 2, "documentDigest": None, "commands": []}
    for opacity in (65535, 32768, 16384):
        delta["commands"] = [{"op": "layer", "id": identifier, "changes": {"opacity": opacity}}]
        CreateLayeredImage.execute(layers=value, compositor=delta, color_space=color_space)
        name, state = events[-1]
        assert name == "dinkster.compositor.state"
        assert state["stale"] is False
        assert state["documentDigest"] == digest_bytes(original)
        assert state["document"]["layers"][identifier]["opacity"] == opacity
        assert state["document"]["canvas"]["compositing"] == (
            "linear-premultiplied-alpha" if color_space == "linear" else "premultiplied-alpha"
        )
        delta["documentDigest"] = state["documentDigest"]
    assert value.data == original

    changed = apply_commands(value, [{"op": "layer", "id": identifier, "changes": {"name": "New"}}])
    CreateLayeredImage.execute(layers=changed, compositor=delta, color_space=color_space)
    state = events[-1][1]
    assert state["stale"] is True
    assert state["documentDigest"] == digest_bytes(changed.data)
    assert state["document"]["layers"][identifier]["opacity"] == 65535

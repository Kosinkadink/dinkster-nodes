"""Image registration preserves the shared box codec at the worker boundary."""

from __future__ import annotations

import itertools
import pickle
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    COMFY_ARRAY_TYPE,
    TypeRegistry,
    register_inference_types,
)
from dinkster_assets.value import value_asset_references
from dinkster_image import register_types
from dinkster_workers.boundary import EncodedPayloadValidationError, ValueCodec

BOX = {"x": -3, "y": 2.5, "width": 0, "height": 7.0}


def _reconstruct(marker: str) -> list[object]:
    Path(marker).write_text("reconstructed", encoding="ascii")
    return []


class OpaqueBoxes:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _reconstruct, (str(self.marker),)


@pytest.mark.parametrize("order", tuple(itertools.permutations(("native", "image"))))
@pytest.mark.parametrize("type_id", (BOUNDING_BOX_TYPE, COMFY_ARRAY_TYPE))
def test_all_registration_orders_refuse_pickle_at_real_boundary_admission(
    tmp_path: Path, order: tuple[str, ...], type_id: str
) -> None:
    registry = TypeRegistry()
    registrations = {
        "native": register_inference_types,
        "image": register_types,
    }
    first = None
    for name in order:
        registrations[name](registry)
        spec = registry.spec(type_id)
        if first is None:
            first = spec
        assert spec is first and spec.declared_codec and spec.validate_encoded is not None
    spec = registry.spec(type_id)
    codec = ValueCodec(registry, use_shm=False, accept_shm=False)
    blobs: list[bytes] = []
    wire, _ = codec.encode(registry.wrap(type_id, [BOX]), blobs, [])
    restored, _ = codec.decode(wire, blobs, [])
    assert restored.resolve() == [BOX]
    assert value_asset_references({"boxes": restored}, complete_registry=registry) == {}
    payload = wire["payload"]
    assert isinstance(payload, dict)
    marker = tmp_path / "marker.txt"
    marker.write_text("unchanged", encoding="ascii")
    malicious = b"pickle:" + pickle.dumps(OpaqueBoxes(marker), protocol=5)
    blobs[payload["blob"]] = malicious
    with pytest.raises(EncodedPayloadValidationError, match="JSON framing"):
        codec.decode(wire, blobs, [])
    assert marker.read_text() == "unchanged"
    assert spec.validate_encoded is not None
    with pytest.raises(ValueError, match="JSON framing"):
        spec.validate_encoded(malicious, {})
    with pytest.raises(ValueError, match="JSON framing"):
        spec.decode(malicious)
    assert marker.read_text() == "unchanged"

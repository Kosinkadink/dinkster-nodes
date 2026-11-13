"""Geometry pack schemas stay cold and model registrations retain their authority."""

from pathlib import Path

import pytest
from dinkster_api.v1 import TypeRegistry
from dinkster_workers import load_manifest
from dinkster_workers.manifest import resolve_entry


@pytest.mark.parametrize(
    "type_id,pack",
    (
        ("DA3_MODEL", "depth-anything-3"),
        ("FACE_DETECTION_MODEL", "mediapipe"),
        ("SAM3D_BODY_MODEL", "sam3d-body"),
    ),
)
def test_family_models_retain_their_complete_holding_process_object(
    type_id: str, pack: str
) -> None:
    manifest = load_manifest(Path(f"packages/dinkster-nodes-vision-{pack}/dinkster-pack.toml"))
    assert manifest.types_entry is not None
    register_types = resolve_entry(manifest.types_entry)
    assert callable(register_types)
    registry = TypeRegistry()
    with registry.registering_pack(manifest.name, manifest.type_namespaces, allow_bare=True):
        register_types(registry)
    spec = registry.spec(type_id)
    wrapper = {"model": object(), "patchers": {"short": object(), "full": object()}}
    assert spec.decode(spec.encode(wrapper)) is wrapper
    with registry.registering_pack(manifest.name, manifest.type_namespaces, allow_bare=True):
        register_types(registry)
    assert registry.spec(type_id) is spec

    authority = TypeRegistry()
    with authority.registering_pack(manifest.name, manifest.type_namespaces, allow_bare=True):
        previous = authority.register(type_id, encode=lambda _: b"first", decode=lambda _: wrapper)
        register_types(authority)
    assert authority.spec(type_id) is previous
    assert authority.spec(type_id).encode(wrapper) == b"first"

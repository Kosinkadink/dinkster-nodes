"""Geometry pack schemas stay cold and model registrations retain their authority."""

import subprocess
import sys
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


def test_geometry_pack_schemas_are_cold_and_match_their_manifests() -> None:
    script = """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_3d import NODES as geometry_nodes
from dinkster_hunyuan import NODES as hunyuan_nodes
from dinkster_workers import load_manifest
runtimes = {node.schema().node_type: node.schema() for node in (*geometry_nodes, *hunyuan_nodes)}
names = {'dinkster.3d.rotate_mesh', 'dinkster.3d.get_mesh_info',
         'dinkster.hunyuan.vae_decode_hunyuan3_d',
         'dinkster.3d.voxel_to_mesh_basic', 'dinkster.3d.voxel_to_mesh'}
assert names <= runtimes.keys()
for pack, nodes in (('3d', geometry_nodes), ('hunyuan', hunyuan_nodes)):
    manifest = load_manifest(f'packages/dinkster-nodes-{pack}/dinkster-pack.toml')
    assert all(any(node.schema().node_type.startswith(ns + '.') for ns in manifest.namespaces)
               for node in nodes)
assert runtimes['dinkster.3d.voxel_to_mesh_basic'].deprecation is not None
assert runtimes['dinkster.3d.voxel_to_mesh_basic'].search_visibility == 'deprecated'
assert sys.modules['torch'] is None
assert sys.modules['dinkster_inference'] is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr

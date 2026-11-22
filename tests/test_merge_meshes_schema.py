"""Mesh merge declarations remain available without tensor runtimes."""

import subprocess
import sys


def test_merge_meshes_cold_schema_registration_and_family_contract() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import MESH_TYPE, schema_from_wire, schema_to_wire
from dinkster_3d import NODES
from dinkster_3d.mesh import MergeMeshes
from dinkster_workers import load_manifest
from tests.body_rows import groups
schema = schema_from_wire(schema_to_wire(MergeMeshes.schema()))
assert schema.node_type == 'dinkster.3d.merge_meshes'
assert schema.display_name == 'Merge Meshes' and schema.category == '3d/mesh'
family, = groups(schema)
assert family.id == 'meshes' and family.keys.prefix == 'mesh'
assert family.min_members == 2 and family.max_members == 50
assert family.optional_after_minimum
assert family.value.type.types == (MESH_TYPE,)
assert [o.id for o in schema.outputs] == ['mesh']
assert NODES.count(MergeMeshes) == 1
manifest = load_manifest('packages/dinkster-nodes-3d/dinkster-pack.toml')
assert 'dinkster.3d' in manifest.namespaces
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

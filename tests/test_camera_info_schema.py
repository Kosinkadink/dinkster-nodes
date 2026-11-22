"""Cold public contracts for world-space camera construction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dinkster_3d import NODES
from dinkster_3d.camera_info import CreateCameraInfo
from dinkster_api.v1 import (
    LOAD3D_CAMERA_TYPE,
    Output,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_workers import load_manifest

from tests.body_rows import advanced_ids, choices, scope


def test_camera_info_schema_and_production_claim_are_canonical() -> None:
    schema = CreateCameraInfo.schema()
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert schema.outputs[0].type.types == (LOAD3D_CAMERA_TYPE,)
    assert [
        getattr(row, "id", "") for row in scope(schema.body) if not isinstance(row, Output)
    ] == [
        "mode",
        "roll",
        "fov",
        "zoom",
        "camera_type",
        "target_x",
        "target_y",
        "target_z",
    ]
    assert advanced_ids(schema.body) == ("target_x", "target_y", "target_z")
    assert [option.key for option in choices(schema)[0].options] == [
        "orbit",
        "look_at",
        "quaternion",
    ]
    assert sum(node is CreateCameraInfo for node in NODES) == 1
    manifest = load_manifest(
        Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-3d/dinkster-pack.toml"
    )
    assert "dinkster.3d" in manifest.namespaces


def test_camera_info_discovery_does_not_import_inference_or_torch() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_3d.camera_info import CreateCameraInfo\n"
            "assert CreateCameraInfo.schema().node_type == 'dinkster.3d.create_camera_info'\n",
        ],
        check=True,
    )

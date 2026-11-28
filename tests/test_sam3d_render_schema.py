"""SAM3D renderer discovery preserves cold imports and ordered native claims."""

import json
import subprocess
import sys
from pathlib import Path

from dinkster_api.v1 import schema_to_wire
from dinkster_vision_sam3d_body.sam3d_render import SAM3DBody_Render

from tools.compare_nodes import compare


def test_render_schema_matches_committed_reference() -> None:
    reference = json.loads(Path("tools/data/comfy_core_reference.json").read_text())
    row = next(row for row in reference["nodes"] if row["class_name"] == "SAM3DBody_Render")
    assert (
        compare(
            {**row, "class_name": "dinkster.vision.sam3d.body.sam3_d_body_render"},
            schema_to_wire(SAM3DBody_Render.define_schema()),
        )
        == []
    )


def test_render_schema_is_cold_and_registered() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import IMAGE_TYPE, MHR_POSE_DATA_TYPE, KIMODO_POSE_DATA_TYPE, Output, TypeExpr
from dinkster_vision_sam3d_body import NODES
from dinkster_vision_sam3d_body.sam3d_render import SAM3DBody_Render
from dinkster_workers import load_manifest
from tests.body_rows import choices, option_rows, scope
schema = SAM3DBody_Render.define_schema()
assert SAM3DBody_Render in NODES
assert schema.node_type == 'dinkster.vision.sam3d.body.sam3_d_body_render'
manifest = load_manifest('packages/dinkster-nodes-vision-sam3d-body/dinkster-pack.toml')
assert manifest.namespaces == ('dinkster.vision.sam3d.body',)
assert [row.id for row in scope(schema.body) if not isinstance(row, Output)] == [
    'pose_data', 'background', 'width', 'height', 'camera_info', 'render_style']
assert schema.inputs[0].type == TypeExpr.union(MHR_POSE_DATA_TYPE, KIMODO_POSE_DATA_TYPE)
assert schema.outputs[0].type == TypeExpr.concrete(IMAGE_TYPE)
assert schema.outputs[0].id == 'image'
render_style, = choices(schema)
assert [option.key for option in render_style.options] == [
    'mesh', 'silhouette', 'openpose_2d', 'openpose_3d', 'scail']
shader = option_rows(render_style, render_style.options[0])[0]
assert [option.key for option in shader.options] == [
    'default', 'normals', 'rainbow', 'rainbow_face_normal', 'rainbow_face_semantic', 'depth']
assert 'dinkster_vision_sam3d_body.sam3d_rasterizer' not in sys.modules
assert 'dinkster_vision_sam3d_body.sam3d_openpose' not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

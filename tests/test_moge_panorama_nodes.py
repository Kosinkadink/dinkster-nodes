"""Panorama geometry declarations stay cold and registered once."""

import subprocess
import sys


def test_moge_panorama_cold_schema_and_registration() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_vision_moge import NODES
from dinkster_vision_moge.moge_panorama import MoGePanoramaInference
from dinkster_workers import load_manifest
from tests.body_rows import advanced_ids
schema = MoGePanoramaInference.define_schema()
manifest = load_manifest('packages/dinkster-nodes-vision-moge/dinkster-pack.toml')
assert NODES.count(MoGePanoramaInference) == 1
assert manifest.namespaces.count('dinkster.vision.moge') == 1
assert schema.node_type == 'dinkster.vision.moge.mo_ge_panorama_inference'
assert [i.id for i in schema.inputs] == [
    'moge_model', 'image', 'resolution_level', 'split_resolution',
    'merge_resolution', 'batch_size', 'refine_steps'
]
assert [i.default for i in schema.inputs[2:]] == [9, 512, 1920, 4, 3]
assert advanced_ids(schema.body) == (schema.inputs[-1].id,)
assert [o.id for o in schema.outputs] == ['moge_geometry']
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

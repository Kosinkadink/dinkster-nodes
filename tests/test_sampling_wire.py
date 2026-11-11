"""Sampling node schemas and resident codecs register without the inference runtime."""

import subprocess
import sys


def test_cold_registration_and_resident_inspection_need_no_runtime() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["torch"] = None
sys.modules["dinkster_inference"] = None
from dinkster_api.v1 import TypeRegistry, register_inference_types
from dinkster_sampling import NODES
registry = TypeRegistry()
register_inference_types(registry)
for node in NODES:
    node.schema()
for type_id in ("SAMPLER", "NOISE"):
    prefix = type_id + ".resident.v1"
    registry.spec(type_id).validate_encoded(
        prefix.encode() + b'\\x00{"residentId":"foreign"}', {"format": prefix}
    )
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

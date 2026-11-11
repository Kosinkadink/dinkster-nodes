"""Schema discovery and public codecs do not load the tensor runtime."""

import json
import subprocess
import sys

import pytest

NODE_MODULES = ("dinkster_ltx.ltx_duration", "dinkster_conditioning.controlnet")


@pytest.fixture(scope="module")
def cold_modules() -> set[str]:
    result = subprocess.run(
        [sys.executable, "-c", COLD_DISCOVERY, json.dumps(NODE_MODULES)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return set(json.loads(result.stdout))


@pytest.mark.parametrize("module", NODE_MODULES)
def test_node_module_discovery_is_torch_and_inference_free(
    module: str, cold_modules: set[str]
) -> None:
    assert module in cold_modules


COLD_DISCOVERY = """
import importlib
import json
import sys

from dinkster_schema import schema_to_wire
from dinkster_api.v1 import Node

modules = json.loads(sys.argv[1])
public_names = set()
for name in modules:
    module = importlib.import_module(name)
    for attr, node in vars(module).items():
        if (not attr.startswith('_') and isinstance(node, type)
                and issubclass(node, Node) and node.__module__ == name):
            assert schema_to_wire(node.schema()), (name, node)
            public_names.add(node.schema().node_type)
    forbidden = [name for name in sys.modules if name == 'torch' or name.startswith('torch.')
                 or name == 'dinkster_inference' or name.startswith('dinkster_inference.')]
    assert not forbidden, (module.__name__, forbidden)

assert {'dinkster.ltx.ltxv_duration_predictor',
        'dinkster.conditioning.set_union_control_net_type'} <= public_names
print(json.dumps(modules))
"""

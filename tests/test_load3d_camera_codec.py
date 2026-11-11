"""The 3D pack registers the portable camera types without loading inference."""

import subprocess
import sys


def test_load3d_camera_public_registration_is_cold_and_shared() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from dinkster_api.v1 import LOAD3D_CAMERA_TYPE, TypeRegistry\n"
            "from dinkster_3d import register_types\n"
            "from dinkster_values.builtin_types import LOAD3D_CAMERA_TYPE as canonical\n"
            "assert LOAD3D_CAMERA_TYPE is canonical\n"
            "registry = TypeRegistry()\n"
            "register_types(registry)\n"
            "register_types(registry)\n"
            "value = {'position': {'x': 0, 'y': 1, 'z': 2},\n"
            "         'target': {'x': 0, 'y': 0, 'z': 0},\n"
            "         'zoom': 1, 'cameraType': 'perspective'}\n"
            "spec = registry.spec(canonical)\n"
            "payload = spec.encode(value)\n"
            "spec.validate_encoded(payload, {})\n"
            "assert spec.decode(payload) == value\n"
            "from dinkster_api.v1 import LOAD3D_MODEL_INFO_TYPE\n"
            "from dinkster_values.builtin_types import LOAD3D_MODEL_INFO_TYPE as model_type\n"
            "assert LOAD3D_MODEL_INFO_TYPE is model_type\n"
            "models = registry.spec(model_type)\n"
            "assert models.decode(models.encode([])) == []\n"
            "assert not any(\n"
            "    name == 'torch' or name.startswith('torch.')\n"
            "    or name == 'dinkster_inference'\n"
            "    or name.startswith('dinkster_inference.')\n"
            "    for name in sys.modules\n"
            ")\n",
        ],
        check=True,
        timeout=30,
    )

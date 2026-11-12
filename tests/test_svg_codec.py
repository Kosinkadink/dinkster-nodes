"""Image registration preserves the portable SVG type and its first authority."""

import subprocess
import sys

from dinkster_api.v1 import SVG_TYPE, TypeRegistry
from dinkster_image import register_types
from dinkster_values import svg_meta


def test_svg_registration_preserves_the_first_authority() -> None:
    registry = TypeRegistry()
    register_types(registry)
    spec = registry.spec(SVG_TYPE)
    register_types(registry)
    assert registry.spec(SVG_TYPE) is spec
    value = (b"<svg/>", b"", b"<svg/>")
    encoded = spec.encode(value)
    assert spec.decode(encoded) == value
    assert spec.validate_encoded is not None
    spec.validate_encoded(encoded, svg_meta(value))


def test_svg_public_type_registers_in_two_fresh_python_processes() -> None:
    script = """
import sys
from dinkster_api.v1 import SVG_TYPE, TypeRegistry, materialize_svg
from dinkster_image import register_types
from dinkster_values.builtin_types import SVG_TYPE as canonical

assert SVG_TYPE is canonical
registry = TypeRegistry()
register_types(registry)
register_types(registry)
spec = registry.spec(SVG_TYPE)
value = (b'<svg/>', b'', b'\\xff')
encoded = spec.encode(value)
assert spec.decode(encoded) == value
assert spec.fingerprint(value) == spec.fingerprint(spec.decode(encoded))
assert tuple(stream.read() for stream in materialize_svg(spec.decode(encoded))) == value
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
"""
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr

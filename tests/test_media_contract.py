"""Pack media registrars preserve inline storage and mask contracts."""

import numpy as np
import pytest
from dinkster_values import TypeRegistry, annotate_mask


@pytest.mark.parametrize("image_first", [False, True])
def test_pack_registration_order_preserves_mask_contract(image_first: bool) -> None:
    from dinkster_image import register_types as register_image_types
    from dinkster_video import register_types as register_video_types

    registry = TypeRegistry()
    registrations = [register_image_types, register_video_types]
    for register in registrations if image_first else reversed(registrations):
        register(registry)
    mask = annotate_mask(np.zeros((1, 1, 1), np.float32), polarity="transparency", semantic="alpha")
    value = registry.wrap("MASK", mask)
    assert value.meta.get("polarity") == "transparency"
    assert value.meta.get("semantic") == "alpha"
    assert "channels" not in value.meta.entries
    for type_id in ("MASK", "IMAGE"):
        assert registry.spec(type_id).validate_encoded is not None
        assert registry.spec(type_id).validate_encoded_buffer is not None

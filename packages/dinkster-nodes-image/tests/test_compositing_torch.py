"""CPU tensor and hosted execution behavior for ComfyUI compositing nodes."""

from __future__ import annotations

import numpy as np
import torch
from dinkster_api.v1 import (
    CORE_FLOAT,
    IMAGE_TYPE,
    MASK_TYPE,
    TypeExpr,
)
from dinkster_image.ported.compositing import (
    JoinImageWithAlpha,
    PorterDuffImageComposite,
    SplitImageWithAlpha,
)

_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)


def test_compositing_materializes_decoded_masks() -> None:
    source = torch.full((1, 2, 3, 3), 0.8)
    destination = torch.full((1, 2, 3, 3), 0.2)
    source_alpha = np.full((1, 2, 3), 0.75, dtype=np.float32)
    destination_alpha = np.full((1, 2, 3), 0.25, dtype=np.float32)
    result = PorterDuffImageComposite.execute(
        source=source,
        source_alpha=source_alpha,
        destination=destination,
        destination_alpha=destination_alpha,
        mode="SRC_OVER",
    )
    torch.testing.assert_close(result["IMAGE"], torch.full((1, 2, 3, 3), 5.0 / 13.0))
    torch.testing.assert_close(result["MASK"], torch.full((1, 2, 3), 3.0 / 16.0))

    split = SplitImageWithAlpha.execute(
        image=torch.cat((source, torch.full((1, 2, 3, 1), 0.25)), -1)
    )
    joined = JoinImageWithAlpha.execute(image=split["IMAGE"], alpha=source_alpha)["IMAGE"]
    torch.testing.assert_close(joined[..., 3], torch.full((1, 2, 3), 0.25))

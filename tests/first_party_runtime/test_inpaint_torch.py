"""CPU tensor and ordinary-pack execution behavior for inpainting nodes."""

from __future__ import annotations

import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    IMAGE_TYPE,
    MASK_TYPE,
    VAE_TYPE,
    TypeExpr,
)
from dinkster_latent.inpaint import VAEEncodeForInpaint

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


class FakeVAE:
    def __init__(self) -> None:
        self.encoded: list[torch.Tensor] = []

    def spacial_compression_encode(self) -> int:
        return 4

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.encoded.append(pixels.clone())
        return pixels.movedim(-1, 1) + len(self.encoded)


def test_inpaint_values_use_center_crop_and_only_masked_pixels() -> None:
    pixels = torch.arange(7 * 11 * 3, dtype=torch.float32).reshape(1, 7, 11, 3) / 256
    mask = torch.zeros((1, 7, 11), dtype=torch.float32)
    mask[0, 2, 3] = 1
    result = VAEEncodeForInpaint.execute(
        vae=FakeVAE(), pixels=pixels.numpy(), mask=mask.numpy(), grow_mask_by=3
    )["LATENT"]
    # A 4x8 crop starts at (1,1); the mask lands at (1,2).
    expected_mask = torch.zeros((1, 1, 4, 8))
    expected_mask[0, 0, :3, 1:4] = 1
    assert torch.equal(result["noise_mask"], expected_mask)
    assert torch.equal(result["samples"][0, :, 1, 2], torch.full((3,), 1.5))
    assert torch.equal(
        result["samples"][0, :, 0, 0],
        torch.tensor([1 + 36 / 256, 1 + 37 / 256, 1 + 38 / 256]),
    )

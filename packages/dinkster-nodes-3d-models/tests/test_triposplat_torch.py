"""CPU TripoSplat foreground geometry and ordinary host transport."""

from __future__ import annotations

import pytest
import torch
from dinkster_3d_models.triposplat import (
    TripoSplatPreprocessImage,
)


@pytest.mark.parametrize("erode_radius", (0, 2))
@pytest.mark.parametrize("mask_size", ((64, 96), (32, 48)))
def test_primary_colors_repeat_resize_masks_and_preserve_inputs(
    erode_radius: int, mask_size: tuple[int, int]
) -> None:
    # Hand-derived: RGB values 2/-1 clamp to primary colors at the center;
    # a centered rectangular alpha region leaves black corners after square padding.
    # The declared stride rounds size 271 down to 256, independently of input size.
    colors = torch.eye(3)
    image = (colors * 3 - 1)[:, None, None, :].expand(3, 64, 96, 3).clone()
    mask = torch.zeros(1, *mask_size)
    height, width = mask_size
    mask[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1
    original_image, original_mask = image.clone(), mask.clone()
    result = TripoSplatPreprocessImage.execute(
        image=image.numpy(), mask=mask.numpy(), erode_radius=erode_radius, size=271
    )["image"]
    assert result.shape == (3, 256, 256, 3) and result.dtype == torch.float32
    torch.testing.assert_close(result[:, 128, 128], colors, rtol=0, atol=0)
    assert torch.count_nonzero(result[:, 0, 0]) == 0
    assert torch.all((result >= 0) & (result <= 1))
    torch.testing.assert_close(image, original_image, rtol=0, atol=0)
    torch.testing.assert_close(mask, original_mask, rtol=0, atol=0)


def test_empty_mask_reports_foreground_error() -> None:
    # Hand-derived: an all-zero mask has no foreground, regardless of RGB values.
    with pytest.raises(ValueError, match="mask is empty .*no foreground pixels"):
        TripoSplatPreprocessImage.execute(
            image=torch.ones(1, 48, 80, 3),
            mask=torch.zeros(1, 48, 80),
            erode_radius=1,
            size=256,
        )


def test_erosion_removes_a_foreground_strip_narrower_than_the_kernel() -> None:
    # Hand-derived: a 5x5 minimum filter removes a three-row strip. Input and
    # requested size are both 256, so the first resize cannot widen the strip.
    image = torch.ones(1, 256, 256, 3)
    mask = torch.zeros(1, 256, 256)
    mask[:, 100:103, 50:200] = 1
    un_eroded = TripoSplatPreprocessImage.execute(image=image, mask=mask, erode_radius=0, size=256)[
        "image"
    ]
    assert torch.count_nonzero(un_eroded) > 0
    with pytest.raises(ValueError, match="mask is empty .*no foreground pixels"):
        TripoSplatPreprocessImage.execute(image=image, mask=mask, erode_radius=2, size=256)

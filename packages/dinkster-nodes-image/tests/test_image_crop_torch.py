"""Public CPU behavior for native image crop primitives."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from dinkster_image.ported.image_crop import ImageCropToMask, ImageCropV2
from dinkster_schema import use_reporter
from PIL import Image


def _image() -> torch.Tensor:
    return torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5, 1) / 30


def _mask_crop_image() -> torch.Tensor:
    return torch.tensor([0.2, 0.4, 0.8, 0.1]).expand(2, 8, 12, 4).clone()


def test_mask_crop_composites_broadcast_resized_mask_without_mutating_rgba() -> None:
    image = _mask_crop_image()
    mask = torch.full((1, 4, 6), 0.5)
    original_image, original_mask = image.clone(), mask.clone()
    output = ImageCropToMask.execute(
        images=image.numpy(),
        masks=mask.numpy(),
        width=96,
        height=64,
        pad_factor=1.0,
        grow_mask=0,
        background="#204080",
    )["IMAGE"]
    # Half-opacity RGB over the requested background, then the Lanczos byte round trip.
    expected = (
        torch.floor((torch.tensor([0.2, 0.4, 0.8]) + torch.tensor([32, 64, 128]) / 255) * 127.5)
        / 255
    )
    assert output.shape == (2, 64, 96, 3)
    assert output.dtype == torch.float32 and output.device.type == "cpu"
    assert torch.equal(output, expected.expand_as(output))
    assert torch.equal(image, original_image)
    assert torch.equal(mask, original_mask)


def test_mask_crop_rejects_mismatched_batches() -> None:
    with pytest.raises(ValueError, match="Mask batch 3 does not match image batch 2"):
        ImageCropToMask.execute(
            images=_mask_crop_image(),
            masks=torch.ones(3, 8, 12),
            width=96,
            height=64,
            pad_factor=1.0,
            grow_mask=0,
            background="#000000",
        )


def test_mask_crop_uses_foreground_bounds_not_the_full_image() -> None:
    image = torch.arange(8 * 12 * 4, dtype=torch.float32).reshape(1, 8, 12, 4) / 384
    mask = torch.zeros(1, 8, 12)
    mask[:, 2:6, 5:11] = 1
    actual = ImageCropToMask.execute(
        images=image,
        masks=mask,
        width=96,
        height=64,
        pad_factor=1.0,
        grow_mask=0,
        background="#204080",
    )["IMAGE"]
    # A 6x4 foreground gives a 6x4 crop at (4, 1) after integer centering.
    crop = image[0, 1:5, 4:10, :3].clone()
    crop[0, :] = torch.tensor([32, 64, 128]) / 255
    crop[:, 0] = torch.tensor([32, 64, 128]) / 255
    pixels = Image.fromarray((crop.numpy() * 255).astype(np.uint8))
    expected = (
        torch.from_numpy(np.array(pixels.resize((96, 64), Image.Resampling.LANCZOS))).float() / 255
    )
    assert torch.equal(actual, expected.unsqueeze(0))


def test_image_crop_v2_uses_asymmetric_bounds_and_reports_the_cropped_preview() -> None:
    image = _image()
    reports: list[tuple[str, dict[str, object], bytes | None]] = []

    with use_reporter(lambda name, data, blob: reports.append((name, dict(data), blob))):
        result = ImageCropV2.execute(
            image=image, crop_region={"x": 3, "y": 1, "width": 2, "height": 1}
        )

    assert torch.equal(result["IMAGE"], image[:, 1:2, 3:5, :])
    assert len(reports) == 1
    name, data, blob = reports[0]
    assert (name, data) == ("preview", {"mime": "image/png", "width": 2, "height": 1})
    assert blob is not None and blob.startswith(b"\x89PNG\r\n\x1a\n")


def test_image_crop_v2_uses_region_defaults_and_clamps_only_the_origin() -> None:
    image = _image()

    defaults = ImageCropV2.execute(image=image, crop_region={})["IMAGE"]
    clamped = ImageCropV2.execute(
        image=image, crop_region={"x": 99, "y": 99, "width": 10, "height": 10}
    )["IMAGE"]

    assert torch.equal(defaults, image)
    assert torch.equal(clamped, image[:, 2:3, 4:5, :])

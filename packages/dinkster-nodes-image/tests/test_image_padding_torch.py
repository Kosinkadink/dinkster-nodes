"""Public tensor behavior of native image padding."""

from __future__ import annotations

import pytest
import torch
from dinkster_image.ported.image_transform import PadImageNode


def test_mirror_repeats_without_edge_duplication_and_preserves_batches() -> None:
    image = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3, 1)
    actual = PadImageNode.execute(
        image=image,
        padding_type={
            "padding_type": "pad dimensions",
            "left": 4,
            "top": 1,
            "right": 1,
            "bottom": 2,
        },
        padding_style={"padding_style": "mirror image"},
    )["image"]
    expected = image[:, [1, 0, 1, 0, 1], :, :][:, :, [0, 1, 2, 1, 0, 1, 2, 1], :]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "selected,values",
    [
        ("pad by multiplier", {"padding_type.multiplier": 1.5}),
        ("pad to multiple", {"padding_type.multiple": 4}),
    ],
)
def test_odd_padding_places_extra_pixel_right_and_bottom(selected: str, values: dict) -> None:
    image = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3) / 20
    actual = PadImageNode.execute(
        image=image,
        padding_type=selected,
        padding_style="color",
        **values,
        **{"padding_style.color": "#20408080", "padding_style.feathering": 0},
    )["image"]
    height = 3 if selected == "pad by multiplier" else 4
    expected = torch.tensor([32 / 255, 64 / 255, 128 / 255]).expand(1, height, 4, 3).clone()
    top = 0 if selected == "pad by multiplier" else 1
    expected[:, top : top + 2, :3] = image
    assert torch.equal(actual, expected)
    assert torch.equal(image, torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3) / 20)


def test_feathering_only_uses_padded_sides_and_ignores_color_alpha() -> None:
    image = torch.zeros((1, 3, 4, 4), dtype=torch.float16)
    actual = PadImageNode.execute(
        image=image,
        padding_type={
            "padding_type": "pad dimensions",
            "left": 1,
            "top": 0,
            "right": 0,
            "bottom": 0,
        },
        padding_style={"padding_style": "color", "color": "#80402000", "feathering": 2},
    )["image"]
    fill = torch.tensor([128 / 255, 64 / 255, 32 / 255, 1], dtype=torch.float16)
    expected = torch.zeros((1, 3, 5, 4), dtype=torch.float16)
    expected[:, :, 0] = fill
    expected[:, :, 1] = fill * torch.tensor(4 / 9, dtype=torch.float16)
    expected[:, :, 2] = fill * torch.tensor(1 / 9, dtype=torch.float16)
    assert torch.equal(actual, expected)
    assert torch.equal(image, torch.zeros_like(image))


def test_single_pixel_mirror_and_zero_padding_bypass_style() -> None:
    image = torch.tensor([[[[0.2, 0.4, 0.8]]]])
    actual = PadImageNode.execute(
        image=image,
        padding_type={
            "padding_type": "pad dimensions",
            "left": 3,
            "top": 2,
            "right": 4,
            "bottom": 1,
        },
        padding_style={"padding_style": "mirror image"},
    )["image"]
    assert torch.equal(actual, image.expand(1, 4, 8, 3))
    unchanged = PadImageNode.execute(
        image=image,
        padding_type={"padding_type": "pad to multiple", "multiple": 1},
        padding_style={"padding_style": "unused"},
    )["image"]
    assert unchanged is image


@pytest.mark.parametrize("color", ["oops", "#gg0000"])
def test_invalid_color_uses_reference_white_fallback(color: str) -> None:
    image = torch.zeros((1, 1, 2, 3))
    actual = PadImageNode.execute(
        image=image,
        padding_type={
            "padding_type": "pad dimensions",
            "left": 1,
            "top": 0,
            "right": 0,
            "bottom": 0,
        },
        padding_style={"padding_style": "color", "color": color, "feathering": 0},
    )["image"]
    assert torch.equal(actual[:, :, :1], torch.ones((1, 1, 1, 3)))
    assert torch.equal(actual[:, :, 1:], image)

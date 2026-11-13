"""Marigold prediction conversion and ordinary manifest execution."""

from __future__ import annotations

from typing import Any

import torch
from dinkster_vision_marigold.marigold import MarigoldV2PostProcess


def test_depth_normalizes_each_image_and_preserves_constant_images() -> None:
    values = torch.tensor([[[1.0, 2.0], [3.0, 5.0]], [[11.0, 13.0], [15.0, 19.0]]])
    source = values.unsqueeze(-1).repeat(1, 1, 1, 3)
    before = source.clone()
    result: Any = MarigoldV2PostProcess.execute(image=source, prediction="depth")["IMAGE"]
    expected = torch.tensor([[1.0, 0.75], [0.5, 0.0]]).expand(2, 2, 2)
    assert torch.equal(result, expected.unsqueeze(-1).repeat(1, 1, 1, 3))
    assert torch.equal(source, before)
    constant: Any = MarigoldV2PostProcess.execute(
        image=torch.full((2, 2, 3, 3), 0.7), prediction="depth"
    )["IMAGE"]
    assert torch.count_nonzero(constant).item() == 0


def test_normals_use_channel_axis_and_zero_vectors_remain_midpoint() -> None:
    source = torch.tensor([[[[0.65, 0.7, 0.5], [0.5, 0.5, 0.5]]]], dtype=torch.float64)
    result: Any = MarigoldV2PostProcess.execute(image=source, prediction="normals")["IMAGE"]
    assert torch.allclose(
        result, torch.tensor([[[[0.8, 0.9, 0.5], [0.5, 0.5, 0.5]]]], dtype=torch.float64)
    )
    assert result.dtype == source.dtype


def test_albedo_transfer_includes_negative_and_above_one_values() -> None:
    source = torch.tensor([[[[-0.1, 0.0, 0.0031308], [0.003131, 0.25, 1.2]]]])
    expected = torch.tensor(
        [
            [
                [
                    [-1.292, 0.0, 0.0031308 * 12.92],
                    [
                        1.055 * 0.003131 ** (1 / 2.4) - 0.055,
                        1.055 * 0.25 ** (1 / 2.4) - 0.055,
                        1.055 * 1.2 ** (1 / 2.4) - 0.055,
                    ],
                ]
            ]
        ]
    )
    result: Any = MarigoldV2PostProcess.execute(image=source, prediction="albedo")["IMAGE"]
    assert torch.allclose(result, expected)
    assert torch.isfinite(result).all()

"""Hunyuan vision conditioning keeps view indices and crosses ordinary hosts."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_hunyuan.hunyuan3d import (
    Hunyuan3Dv2Conditioning,
    Hunyuan3Dv2ConditioningMultiView,
)


def _features(index: int) -> torch.Tensor:
    return torch.arange(12, dtype=torch.float64).reshape(1, 3, 4) / 8 + index * 2


def _expected(indices: tuple[int, ...]) -> torch.Tensor:
    return torch.cat(
        [
            _features(index)
            + torch.cat(
                (
                    torch.sin(torch.tensor([index, index / 100], dtype=torch.float32)),
                    torch.cos(torch.tensor([index, index / 100], dtype=torch.float32)),
                )
            ).reshape(1, 1, 4)
            for index in indices
        ],
        dim=1,
    )


@pytest.mark.parametrize("indices", ((0,), (1,), (3,), (0, 2), (1, 3), (0, 1, 2, 3)))
def test_ordered_views_preserve_slot_positions_and_sources(indices: tuple[int, ...]) -> None:
    inputs = {
        name: {"last_hidden_state": _features(index)}
        for index, name in enumerate(("front", "left", "back", "right"))
        if index in indices
    }
    before = {name: value["last_hidden_state"].clone() for name, value in inputs.items()}
    result: Any = Hunyuan3Dv2ConditioningMultiView.execute(**inputs)
    assert torch.equal(result["positive"][0][0], _expected(indices))
    assert result["positive"][0][1] == result["negative"][0][1] == {}
    assert torch.equal(result["negative"][0][0], torch.zeros_like(_expected(indices)))
    for name, value in inputs.items():
        assert torch.equal(value["last_hidden_state"], before[name])


def test_single_view_and_empty_multiview() -> None:
    source = _features(2)
    result: Any = Hunyuan3Dv2Conditioning.execute(clip_vision_output={"last_hidden_state": source})
    assert torch.equal(result["positive"][0][0], source)
    assert torch.equal(result["negative"][0][0], torch.zeros_like(source))
    with pytest.raises(ValueError, match="non-empty"):
        Hunyuan3Dv2ConditioningMultiView.execute()

"""Gaussian counts and passthrough through native CPU execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from dinkster_3d.gaussian_splat import GetSplatCount


def _source(case: str) -> SimpleNamespace:
    size = 0 if case == "empty" else 3
    return SimpleNamespace(
        positions=torch.arange(18, dtype=torch.float64).reshape(2, 3, 3)[:, :size],
        scales=torch.ones(2, size, 3),
        rotations=torch.tensor([1.0, 0, 0, 0]).expand(2, size, 4),
        opacities=torch.ones(2, size, 1),
        sh=torch.zeros(2, size, 1, 3),
        counts=(
            torch.tensor([1, 3])
            if case == "nonuniform"
            else torch.zeros(2, dtype=torch.int32)
            if case == "zero-counts"
            else None
        ),
    )


@pytest.mark.parametrize(
    "case,expected", [("uniform", 6), ("nonuniform", 4), ("empty", 0), ("zero-counts", 0)]
)
def test_splat_count_preserves_input_and_counts_only_valid_rows(case: str, expected: int) -> None:
    source = _source(case)
    result = GetSplatCount.execute(splat=source)
    assert result["splat"] is source
    assert result["count"] == expected
    assert source.positions.flatten().tolist() == ([] if case == "empty" else list(range(18)))

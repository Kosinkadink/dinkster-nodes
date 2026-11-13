"""CPU merge geometry and execution through the ordinary native manifest."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from dinkster_3d.gaussian_splat import MergeSplat
from dinkster_api.v1 import (
    materialize_splat,
)


def _sources() -> tuple[SimpleNamespace, SimpleNamespace]:
    def value(offset: int, size: int, degree: int, counts: torch.Tensor | None) -> SimpleNamespace:
        return SimpleNamespace(
            positions=torch.arange(2 * size * 3, dtype=torch.float64).reshape(2, size, 3) + offset,
            scales=torch.full((2, size, 3), offset + 1, dtype=torch.float32),
            rotations=torch.tensor([1.0, 0, 0, 0] if offset == 10 else [0.0, 1, 0, 0])
            .expand(2, size, 4)
            .clone(),
            opacities=torch.full((2, size, 1), offset / 100, dtype=torch.float32),
            sh=torch.arange(2 * size * degree * 3, dtype=torch.float32).reshape(2, size, degree, 3)
            + offset,
            counts=counts,
        )

    return value(10, 3, 1, torch.tensor([1, 3], dtype=torch.int32)), value(40, 2, 4, None)


def _assert_merged(value: object) -> None:
    result = materialize_splat(value)
    assert result.positions.dtype == torch.float64
    assert result.scales.dtype == torch.float32
    assert result.counts.dtype == torch.int64
    assert result.counts.tolist() == [3, 5]
    assert result.positions.tolist() == [
        [[10, 11, 12], [40, 41, 42], [43, 44, 45], [0, 0, 0], [0, 0, 0]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27], [46, 47, 48], [49, 50, 51]],
    ]
    for batch, first_size, valid_size in ((0, 1, 3), (1, 3, 5)):
        assert result.scales[batch, :first_size].tolist() == [[11] * 3] * first_size
        assert result.scales[batch, first_size:valid_size].tolist() == [[41] * 3] * 2
        assert result.rotations[batch, :valid_size].tolist() == (
            [[1, 0, 0, 0]] * first_size + [[0, 1, 0, 0]] * 2
        )
        torch.testing.assert_close(
            result.opacities[batch, :valid_size, 0],
            torch.tensor([0.1] * first_size + [0.4] * 2),
            rtol=0,
            atol=0,
        )
        assert result.sh[batch, :first_size, 1:].count_nonzero().item() == 0
        expected_dc = [
            [10 + (batch * 3 + index) * 3 + axis for axis in range(3)]
            for index in range(first_size)
        ]
        assert result.sh[batch, :first_size, 0].tolist() == expected_dc
        expected_sh = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) + 40 + batch * 24
        torch.testing.assert_close(
            result.sh[batch, first_size:valid_size], expected_sh, rtol=0, atol=0
        )
        for name in ("positions", "scales", "rotations", "opacities", "sh"):
            assert getattr(result, name)[batch, valid_size:].count_nonzero().item() == 0


def test_merge_splat_truncates_real_counts_pads_sh_and_preserves_sources() -> None:
    first, second = _sources()
    snapshots = [
        {key: item.clone() if item is not None else None for key, item in vars(value).items()}
        for value in (first, second)
    ]
    # The body consumes mapping order, not lexical slot order.
    result = MergeSplat.execute(splats={"splat10": first, "splat2": second, "splat31": None})
    _assert_merged(result["splat"])
    for value, snapshot in zip((first, second), snapshots, strict=True):
        for name, before in snapshot.items():
            after = getattr(value, name)
            if before is None:
                assert after is None
            else:
                torch.testing.assert_close(after, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_merge_splat_preserves_each_float_dtype_and_cpu_device(dtype: torch.dtype) -> None:
    first, second = _sources()
    names = ("positions", "scales", "rotations", "opacities", "sh")
    for value in (first, second):
        for name in names:
            setattr(value, name, getattr(value, name).to(dtype=dtype, device="cpu"))
    result = materialize_splat(
        MergeSplat.execute(splats={"splat0": first, "splat1": second})["splat"]
    )
    for name in names:
        tensor = getattr(result, name)
        assert tensor.dtype == dtype and tensor.device == torch.device("cpu")
        assert tensor[0, 3:].count_nonzero().item() == 0
        torch.testing.assert_close(tensor[1, 3:], getattr(second, name)[1], rtol=0, atol=0)
    assert result.sh[0, 0, 1:].count_nonzero().item() == 0
    assert result.sh[1, :3, 1:].count_nonzero().item() == 0
    assert result.positions[0, :3, 0].tolist() == [10, 40, 43]
    assert result.counts.dtype == torch.int64
    assert result.counts.device == torch.device("cpu")
    assert result.counts.tolist() == [3, 5]


@pytest.mark.parametrize("case,size", [("uniform", 5), ("empty", 0), ("zero-counts", 0)])
def test_merge_splat_uniform_lengths_omit_counts(case: str, size: int) -> None:
    first, second = _sources()
    first.counts = None
    if case == "empty":
        for value in (first, second):
            for name in ("positions", "scales", "rotations", "opacities", "sh"):
                setattr(value, name, getattr(value, name)[:, :0])
    elif case == "zero-counts":
        first.counts = second.counts = torch.zeros(2, dtype=torch.int64)
    merged = MergeSplat.execute(splats={"splat0": first, "splat1": second})["splat"]
    result = materialize_splat(merged)
    assert result.counts is None
    assert result.positions.shape == (2, size, 3)
    assert result.sh.shape == (2, size, 4, 3)
    if case == "uniform":
        assert result.positions[0, :, 0].tolist() == [10, 13, 16, 40, 43]
        assert result.positions[1, :, 0].tolist() == [19, 22, 25, 46, 49]


def test_merge_splat_rejects_missing_inputs_and_batch_mismatch() -> None:
    with pytest.raises(ValueError, match="connect at least one splat"):
        MergeSplat.execute(splats={"splat0": None})
    first, second = _sources()
    for name in ("positions", "scales", "rotations", "opacities", "sh"):
        setattr(second, name, getattr(second, name)[:1])
    with pytest.raises(ValueError, match=r"batch size mismatch \(2 vs 1\)"):
        MergeSplat.execute(splats={"splat0": first, "splat1": second})

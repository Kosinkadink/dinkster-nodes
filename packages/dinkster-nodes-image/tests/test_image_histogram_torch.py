"""First-frame histogram contracts through direct and ordinary native hosts."""

from __future__ import annotations

import pytest
import torch
from dinkster_image.ported.image_histogram import ImageHistogram


def source_image() -> torch.Tensor:
    pixels = torch.tensor(
        [
            [-1.0, 0.0, 2.0, 0.125],
            [1.0, 0.0, 0.0, 0.25],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.75],
            [0.5, 0.5, 0.5, 1.0],
            [2.0, -1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.375],
        ]
    ).reshape(1, 1, 7, 4)
    return torch.cat((pixels, torch.ones_like(pixels)), dim=0)


def expected_histograms() -> dict[str, list[int]]:
    # Hand-derived from the seven first-frame pixels above. Luminance bins use
    # raw RGB before clipping; the RGB composite floors each bin independently.
    counts = {
        "rgb": {0: 3, 127: 1, 255: 2},
        "luminance": {0: 2, 18: 1, 54: 1, 127: 1, 182: 1, 255: 1},
        "red": {0: 3, 127: 1, 255: 3},
        "green": {0: 4, 127: 1, 255: 2},
        "blue": {0: 3, 127: 1, 255: 3},
    }
    return {name: [bins.get(index, 0) for index in range(256)] for name, bins in counts.items()}


@pytest.mark.parametrize("channels", (3, 4))
def test_histogram_uses_first_frame_and_raw_luminance_without_mutation(channels: int) -> None:
    source = source_image()[..., :channels]
    before = source.clone()
    result = ImageHistogram.execute(image=source)
    assert result == expected_histograms()
    assert tuple(result) == ("rgb", "luminance", "red", "green", "blue")
    torch.testing.assert_close(source, before, rtol=0, atol=0)

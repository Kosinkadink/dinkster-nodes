"""SeedVR2 padding values, boundaries, and ordinary pack execution."""

import pytest
import torch
from dinkster_seedvr.seedvr_preprocess import SeedVR2Preprocess


def frames(count: int, batch: bool = False) -> torch.Tensor:
    shape = (2, count, 3, 17, 4) if batch else (count, 3, 17, 4)
    size = 2 * count * 3 * 17 * 4 if batch else count * 3 * 17 * 4
    return torch.linspace(-0.3, 1.4, size, dtype=torch.float64).reshape(shape)


def expected(image: torch.Tensor) -> torch.Tensor:
    source = image.unsqueeze(0) if image.ndim == 4 else image
    count = source.shape[1]
    length = count if count == 1 else 1 + ((count + 2) // 4) * 4
    output = torch.zeros((source.shape[0], length, 16, 32, 3), dtype=image.dtype)
    for batch in range(source.shape[0]):
        for time in range(length):
            output[batch, time, :3, :17] = source[batch, min(time, count - 1), ..., :3].clamp(0, 1)
    return output


@pytest.mark.parametrize("count", (1, 2, 4, 5, 6, 9))
@pytest.mark.parametrize("batch", (False, True))
def test_preprocess_padding_and_source_preservation(count: int, batch: bool) -> None:
    image = frames(count, batch)
    before = image.clone()
    output = SeedVR2Preprocess.execute(resized_images=image)["images"]
    assert output.dtype == image.dtype and output.device == image.device
    assert output.is_contiguous()
    assert torch.equal(output, expected(image))
    assert torch.equal(image, before)


@pytest.mark.parametrize(
    "shape, message",
    (((2, 3), "4-D or 5-D"), ((1, 1, 3, 3), "shorter edge"), ((0, 3, 17, 4), "at least one frame")),
)
def test_preprocess_invalid_shape(shape: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SeedVR2Preprocess.execute(resized_images=torch.zeros(shape))

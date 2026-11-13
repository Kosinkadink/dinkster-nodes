"""WanDancer frame placement, audio slicing and ordinary-host transport."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    materialize_audio,
    materialize_image,
)
from dinkster_wan.wandancer_keyframes import (
    WanDancerPadKeyframes,
    WanDancerPadKeyframesList,
)


def images() -> torch.Tensor:
    return torch.arange(7 * 2 * 3 * 3, dtype=torch.float32).reshape(7, 2, 3, 3) / 128


def audio(length: int = 17) -> dict[str, Any]:
    return {
        "waveform": torch.arange(2 * length, dtype=torch.float32).reshape(1, 2, length) / 64,
        "sample_rate": 30,
    }


def check(result: dict[str, Any], index: int) -> None:
    frames = materialize_image(result["keyframes_sequence"])
    mask = materialize_image(result["keyframes_mask"])
    sound = materialize_audio(result["audio_segment"])
    expected = torch.zeros((5, 2, 3, 3))
    expected[0], expected[4] = images()[index], images()[index + 1]
    expected_mask = torch.zeros((5, 2, 3))
    expected_mask[0] = expected_mask[4] = 1
    torch.testing.assert_close(frames, expected, rtol=0, atol=0)
    torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)
    assert frames.dtype == mask.dtype == torch.float32
    assert frames.device.type == mask.device.type == "cpu"
    assert sound["sample_rate"] == 30
    torch.testing.assert_close(
        sound["waveform"],
        audio()["waveform"][:, :, index * 5 : min(index * 5 + 5, 17)],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("index", (0, 1, 2, 3))
def test_keyframe_positions_audio_slice_and_sources(index: int) -> None:
    pixels, sound = images(), audio()
    before_image, before_sound = pixels.clone(), sound["waveform"].clone()
    result = WanDancerPadKeyframes.execute(
        images=pixels, segment_length=5, segment_index=index, audio=sound
    )
    check(dict(result), index)
    assert torch.equal(pixels, before_image) and torch.equal(sound["waveform"], before_sound)


def test_list_outputs_and_audio_buffer_boundary() -> None:
    outputs = WanDancerPadKeyframesList.execute(
        images=images(), segment_length=5, num_segments=4, audio=audio()
    )
    assert all(len(value) == 4 for value in outputs.values())
    for index in range(4):
        check({key: value[index] for key, value in outputs.items()}, index)
    for count in (0, 7):
        result = WanDancerPadKeyframes.execute(
            images=images()[:count], segment_length=5, segment_index=0, audio=audio(6)
        )
        assert not result["keyframes_sequence"].count_nonzero()
        assert not result["keyframes_mask"].count_nonzero()
        assert result["audio_segment"]["waveform"].shape == (1, 2, 5)
    result = WanDancerPadKeyframes.execute(
        images=images()[:1], segment_length=5, segment_index=0, audio=audio(7)
    )
    assert torch.equal(result["keyframes_sequence"][4], images()[0])
    assert result["keyframes_mask"].sum() == 6

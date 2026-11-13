"""CPU behavior for native ComfyUI audio combine nodes."""

from __future__ import annotations

import pytest
import torch
from dinkster_api.v1 import (
    AUDIO_TYPE,
    TypeExpr,
)
from dinkster_audio.audio_combine import (
    AudioConcat,
    AudioMerge,
    JoinAudioChannels,
)

_AUDIO = TypeExpr.concrete(AUDIO_TYPE)


def _audio(values: list[list[list[float]]], rate: int = 8_000) -> dict[str, object]:
    return {"waveform": torch.tensor(values, dtype=torch.float32), "sample_rate": rate}


def test_null_audio_returns_the_original_object_without_materializing() -> None:
    opaque = object()
    assert JoinAudioChannels.execute(audio_left=opaque, audio_right=None)["audio"] is opaque
    assert AudioConcat.execute(audio1=None, audio2=opaque, direction="after")["AUDIO"] is opaque
    assert AudioMerge.execute(audio1=opaque, audio2=None, merge_method="add")["AUDIO"] is opaque


def test_join_rejects_non_mono_and_trims_with_left_right_order() -> None:
    left = _audio([[[1, 2, 3, 4]]])
    right = _audio([[[5, 6]]])
    joined = JoinAudioChannels.execute(audio_left=left, audio_right=right)["audio"]
    assert torch.equal(joined["waveform"], torch.tensor([[[1, 2], [5, 6]]], dtype=torch.float32))
    with pytest.raises(ValueError, match="Both input audios must be mono"):
        JoinAudioChannels.execute(audio_left=_audio([[[1], [2]]]), audio_right=right)


def test_concat_duplicates_mono_and_preserves_direction() -> None:
    first = _audio([[[1, 2]]])
    second = _audio([[[3]]])
    after = AudioConcat.execute(audio1=first, audio2=second, direction="after")["AUDIO"]
    before = AudioConcat.execute(audio1=first, audio2=second, direction="before")["AUDIO"]
    assert torch.equal(
        after["waveform"], torch.tensor([[[1, 2, 3], [1, 2, 3]]], dtype=torch.float32)
    )
    assert torch.equal(
        before["waveform"], torch.tensor([[[3, 1, 2], [3, 1, 2]]], dtype=torch.float32)
    )


@pytest.mark.parametrize(
    ("method", "expected"),
    (
        ("add", [0.75, 0.5, 0.5]),
        ("mean", [0.375, 0.25, 0.25]),
        ("subtract", [0.25, 0.5, 0.5]),
        ("multiply", [0.125, 0.0, 0.0]),
    ),
)
def test_merge_methods_pad_to_first_input_extent(method: str, expected: list[float]) -> None:
    first = _audio([[[0.5, 0.5, 0.5]]])
    second = _audio([[[0.25]]])
    result = AudioMerge.execute(audio1=first, audio2=second, merge_method=method)["AUDIO"]
    assert torch.equal(result["waveform"], torch.tensor([[expected]], dtype=torch.float32))


def test_merge_normalizes_only_above_one_and_retains_dtype_device() -> None:
    first = _audio([[[0.75, 0.75]]])
    result = AudioMerge.execute(audio1=first, audio2=first, merge_method="add")["AUDIO"]
    assert torch.equal(result["waveform"], torch.ones((1, 1, 2)))
    assert result["waveform"].dtype is torch.float32
    assert result["waveform"].device.type == "cpu"


@pytest.mark.parametrize("left_rate,right_rate", ((8_000, 16_000), (16_000, 8_000)))
def test_join_resamples_only_the_lower_rate_with_asymmetric_batches(
    left_rate: int, right_rate: int
) -> None:
    left = _audio([[[0.0, 1.0, -0.5]], [[0.5, -1.0, 0.25]]], left_rate)
    right = _audio([[[0.25, -0.75, 1.0, 0.5]], [[-0.5, 0.75, 0.0, -1.0]]], right_rate)
    result = JoinAudioChannels.execute(audio_left=left, audio_right=right)["audio"]
    import dinkster_inference.audio

    target_rate = max(left_rate, right_rate)
    expected_left = (
        left["waveform"]
        if left_rate == target_rate
        else dinkster_inference.audio.resample(left["waveform"], left_rate, target_rate)
    )
    expected_right = (
        right["waveform"]
        if right_rate == target_rate
        else dinkster_inference.audio.resample(right["waveform"], right_rate, target_rate)
    )
    length = min(expected_left.shape[-1], expected_right.shape[-1])
    assert set(result) == {"waveform", "sample_rate"}
    assert result["sample_rate"] == target_rate
    assert result["waveform"].dtype is torch.float32
    assert result["waveform"].device.type == "cpu"
    assert torch.equal(
        result["waveform"],
        torch.cat((expected_left[..., :length], expected_right[..., :length]), dim=1),
    )


def test_merge_trims_longer_second_and_returns_first_for_empty_audio() -> None:
    first = _audio([[[1.0, -0.5]]])
    longer = _audio([[[0.25, 0.5, 0.75]]])
    merged = AudioMerge.execute(audio1=first, audio2=longer, merge_method="subtract")["AUDIO"]
    assert torch.equal(merged["waveform"], torch.tensor([[[0.75, -1.0]]]))
    empty = _audio([[[]]])
    result = AudioMerge.execute(audio1=first, audio2=empty, merge_method="add")["AUDIO"]
    assert result["waveform"] is first["waveform"]


def test_combine_preserves_float64_inputs_and_metadata() -> None:
    first = {
        "waveform": torch.tensor([[[0.5, -0.25, 0.75]]], dtype=torch.float64),
        "sample_rate": 8_000,
        "metadata": object(),
    }
    second = {
        "waveform": torch.tensor([[[0.125, 0.5]]], dtype=torch.float64),
        "sample_rate": 8_000,
    }
    first_before, second_before = first["waveform"].clone(), second["waveform"].clone()
    results = (
        JoinAudioChannels.execute(audio_left=first, audio_right=second)["audio"],
        AudioConcat.execute(audio1=first, audio2=second, direction="before")["AUDIO"],
        AudioMerge.execute(audio1=first, audio2=second, merge_method="subtract")["AUDIO"],
    )
    expected = (
        [[[0.5, -0.25], [0.125, 0.5]]],
        [[[0.125, 0.5, 0.5, -0.25, 0.75], [0.125, 0.5, 0.5, -0.25, 0.75]]],
        [[[0.375, -0.75, 0.75]]],
    )
    for result, values in zip(results, expected, strict=True):
        assert set(result) == {"waveform", "sample_rate"}
        assert result["sample_rate"] == 8_000
        assert result["waveform"].dtype is torch.float64
        assert result["waveform"].device.type == "cpu"
        assert torch.equal(result["waveform"], torch.tensor(values, dtype=torch.float64))
    assert set(first) == {"waveform", "sample_rate", "metadata"}
    assert torch.equal(first["waveform"], first_before)
    assert torch.equal(second["waveform"], second_before)

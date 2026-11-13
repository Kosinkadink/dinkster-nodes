"""CPU behavior for native ComfyUI audio effect nodes."""

from __future__ import annotations

import pytest
import torch
from dinkster_api.v1 import (
    AUDIO_TYPE,
    TypeExpr,
)
from dinkster_audio.audio_effects import AudioEqualizer3Band, TrimAudioDuration

_AUDIO = TypeExpr.concrete(AUDIO_TYPE)


def _audio(values: list[list[list[float]]], rate: int = 8) -> dict[str, object]:
    return {
        "waveform": torch.tensor(values, dtype=torch.float64),
        "sample_rate": rate,
        "extra": object(),
    }


def _equalize(audio: object, **gains: object) -> object:
    return AudioEqualizer3Band.execute(
        audio=audio,
        low_gain_dB=gains.get("low_gain_dB", 0.0),
        low_freq=100,
        mid_gain_dB=gains.get("mid_gain_dB", 0.0),
        mid_freq=2,
        mid_q=0.9,
        high_gain_dB=gains.get("high_gain_dB", 0.0),
        high_freq=3,
    )["AUDIO"]


def test_trim_handles_null_rounding_negative_starts_clamps_empty_and_invalid_ranges() -> None:
    assert TrimAudioDuration.execute(audio=None, start_index=0.0, duration=1.0) == {"AUDIO": None}
    audio = _audio([[[0, 1, 2, 3, 4, 5]]])
    original = audio["waveform"].clone()
    positive = TrimAudioDuration.execute(audio=audio, start_index=0.1875, duration=0.3125)["AUDIO"]
    assert torch.equal(positive["waveform"], torch.tensor([[[2.0, 3.0]]], dtype=torch.float64))
    negative = TrimAudioDuration.execute(audio=audio, start_index=-0.375, duration=0.25)["AUDIO"]
    assert torch.equal(negative["waveform"], torch.tensor([[[3.0, 4.0]]], dtype=torch.float64))
    clamped = TrimAudioDuration.execute(audio=audio, start_index=-2.0, duration=2.0)["AUDIO"]
    assert torch.equal(clamped["waveform"], original)
    empty = _audio([[[]]])
    assert TrimAudioDuration.execute(audio=empty, start_index=0.0, duration=1.0)["AUDIO"] is empty
    with pytest.raises(ValueError, match="TrimAudioDuration: Start time must be less"):
        TrimAudioDuration.execute(audio=audio, start_index=1.0, duration=0.0)
    assert torch.equal(audio["waveform"], original)


def test_equalizer_uses_pinned_dsp_in_source_order_and_preserves_source() -> None:
    audio = _audio(
        [
            [[0.1, -0.2, 0.3, -0.4], [0.4, -0.3, 0.2, -0.1]],
            [[-0.2, 0.1, -0.4, 0.3], [0.3, -0.4, 0.1, -0.2]],
        ]
    )
    original = audio["waveform"].clone()
    assert AudioEqualizer3Band.execute(
        audio=None,
        low_gain_dB=1.0,
        low_freq=100,
        mid_gain_dB=0.0,
        mid_freq=2,
        mid_q=0.9,
        high_gain_dB=0.0,
        high_freq=3,
    ) == {"AUDIO": None}
    zero = _equalize(audio)
    assert zero["waveform"] is not audio["waveform"]
    assert torch.equal(zero["waveform"], original)
    import dinkster_inference.audio as filters

    expected = filters.bass_biquad(original.clone(), 8, gain=3.0, central_freq=100.0, Q=0.707)
    expected = filters.equalizer_biquad(expected, 8, center_freq=2.0, gain=-2.0, Q=0.9)
    expected = filters.treble_biquad(expected, 8, gain=1.5, central_freq=3.0, Q=0.707)
    actual = _equalize(audio, low_gain_dB=3.0, mid_gain_dB=-2.0, high_gain_dB=1.5)
    assert set(actual) == {"waveform", "sample_rate"}
    assert actual["sample_rate"] == 8
    assert actual["waveform"].dtype is torch.float64
    assert actual["waveform"].device.type == "cpu"
    assert torch.equal(actual["waveform"], expected)
    assert torch.equal(audio["waveform"], original)
    empty = _audio([[[]]])
    assert _equalize(empty) is empty

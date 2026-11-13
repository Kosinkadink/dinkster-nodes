"""CPU audio channel and volume behavior through native pack hosts."""

from __future__ import annotations

import pytest
import torch
from dinkster_audio.audio_transform import AudioAdjustVolume, SplitAudioChannels


def _audio() -> dict[str, object]:
    return {
        "waveform": torch.tensor(
            [
                [[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]],
                [[-9.0, 10.0, -11.0, 12.0], [13.0, -14.0, 15.0, -16.0]],
            ]
        ),
        "sample_rate": 44100,
        "extra": "not-output",
    }


def test_split_and_volume_preserve_audio_contracts() -> None:
    audio = _audio()
    waveform = audio["waveform"].to(torch.float64)
    audio["waveform"] = waveform
    original = waveform.clone()
    left, right = (SplitAudioChannels.execute(audio=audio)[name] for name in ("left", "right"))
    assert set(left) == set(right) == {"waveform", "sample_rate"}
    assert left["sample_rate"] == right["sample_rate"] == 44100
    assert torch.equal(left["waveform"], waveform[..., 0:1, :])
    assert torch.equal(right["waveform"], waveform[..., 1:2, :])
    assert left["waveform"].dtype == waveform.dtype
    assert left["waveform"].device == waveform.device
    assert SplitAudioChannels.execute(audio=None) == {"left": None, "right": None}
    for channels in (1, 3):
        invalid = {"waveform": torch.zeros((2, channels, 4)), "sample_rate": 44100}
        with pytest.raises(ValueError, match=rf"got {channels} channel\(s\)"):
            SplitAudioChannels.execute(audio=invalid)

    assert AudioAdjustVolume.execute(audio=None, volume=6) == {"AUDIO": None}
    assert AudioAdjustVolume.execute(audio=audio, volume=0)["AUDIO"] is audio
    opaque = object()
    assert AudioAdjustVolume.execute(audio=opaque, volume=0)["AUDIO"] is opaque
    for decibels in (6, -6):
        output = AudioAdjustVolume.execute(audio=audio, volume=decibels)["AUDIO"]
        assert set(output) == {"waveform", "sample_rate"}
        assert output["sample_rate"] == 44100
        assert output["waveform"].dtype == waveform.dtype
        assert output["waveform"].device == waveform.device
        assert torch.allclose(output["waveform"], waveform * 10 ** (decibels / 20))
    assert audio["waveform"] is waveform
    assert torch.equal(waveform, original)
    assert audio["extra"] == "not-output"

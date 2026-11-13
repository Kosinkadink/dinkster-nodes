"""Audio file admission preserves decoded channel order and PCM scaling."""

import wave
from pathlib import Path

import numpy as np
import pytest
from dinkster_api.v1 import (
    AssetIntegrityError,
    AssetRef,
    digest_file,
)
from dinkster_audio.audio_io import LoadAudio, PreviewAudio, RecordAudio


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        return self.path


@pytest.mark.parametrize("node", (LoadAudio, RecordAudio))
@pytest.mark.parametrize("bits", (16, 32))
def test_load_stereo_pcm_and_preview_without_writes(
    tmp_path: Path, node: type[LoadAudio], bits: int
) -> None:
    samples = (
        np.array([[-32768, 1234], [-1, 32767], [0, -16384], [8192, 11]], dtype=np.int16)
        if bits == 16
        else np.array([[-2147483648], [2147483647], [-245689], [902134]], dtype=np.int32)
    )
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(samples.shape[1])
        output.setsampwidth(bits // 8)
        output.setframerate(22050)
        output.writeframes(samples.tobytes())
    original = path.read_bytes()
    ref = AssetRef(digest_file(path), path.name, len(original), resolver=_Resolver(path))
    audio = node.execute(audio=ref)["AUDIO"]
    assert audio["sample_rate"] == 22050
    np.testing.assert_array_equal(
        audio["waveform"], samples.T[None].astype(np.float32) / (2 ** (bits - 1))
    )
    assert PreviewAudio.execute(audio=audio)["audio"] is audio
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    path.write_bytes(original + b"changed")
    with pytest.raises(AssetIntegrityError):
        node.execute(audio=ref)


@pytest.mark.parametrize("node", (LoadAudio, RecordAudio))
def test_file_input_rejects_host_path_and_preview_rejects_missing_audio(
    node: type[LoadAudio],
) -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        node.execute(audio="unbound.wav")
    with pytest.raises(ValueError, match="input audio is None"):
        PreviewAudio.execute(audio=None)

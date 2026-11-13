"""CPU contracts and ordinary-host transport for native WanDancer audio features."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    AUDIO_ENCODER_OUTPUT_TYPE,
    AUDIO_TYPE,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_audio,
    materialize_latent,
)
from dinkster_wan.wandancer_audio import WanDancerEncodeAudio


def _audio(waveform: torch.Tensor, sample_rate: int = 15360) -> dict[str, object]:
    return {"waveform": waveform, "sample_rate": sample_rate}


def _assert_output(
    result: dict[str, object], *, fps: float, scale: float, fps_string: str
) -> torch.Tensor:
    encoded = result["audio_encoder_output"]
    assert isinstance(encoded, dict)
    feature = encoded["audio_feature"]
    assert isinstance(feature, torch.Tensor)
    assert feature.ndim == 3 and feature.shape[0] == 1 and feature.shape[2] == 35
    assert feature.dtype == torch.float32 and feature.device.type == "cpu"
    assert encoded["fps"] == fps and encoded["audio_inject_scale"] == scale
    assert result["fps_string"] == fps_string
    return feature


def test_silence_features_have_only_the_constant_spectrum_mfcc_offset() -> None:
    silence = torch.zeros((1, 1, 15360), dtype=torch.float32)
    before = silence.clone()
    result = dict(
        WanDancerEncodeAudio.execute(audio=_audio(silence), video_frames=4, audio_inject_scale=1.0)
    )
    feature = _assert_output(result, fps=3.75, scale=1.0, fps_string=" \u5e27\u7387\u662f3.7500")
    assert feature.shape == (1, 31, 35)
    assert torch.equal(silence, before)
    torch.testing.assert_close(feature[:, :, 0], torch.zeros_like(feature[:, :, 0]), rtol=0, atol=0)
    torch.testing.assert_close(
        feature[:, :, 1], torch.full_like(feature[:, :, 1], -100 * np.sqrt(128)), rtol=0, atol=2e-4
    )
    torch.testing.assert_close(
        feature[:, :, 2:], torch.zeros_like(feature[:, :, 2:]), rtol=0, atol=0
    )


def test_real_mono_stereo_first_batch_features_and_input_immutability() -> None:
    samples = torch.arange(15360, dtype=torch.float32) / 15360
    first = torch.stack((samples, samples * 3)).unsqueeze(0)
    ignored = torch.full((1, 2, samples.numel()), 99.0)
    batched = torch.cat((first, ignored), dim=0)
    before = batched.clone()
    mono = first.mean(dim=1, keepdim=True)
    stereo = dict(
        WanDancerEncodeAudio.execute(audio=_audio(batched), video_frames=3, audio_inject_scale=2.5)
    )
    expected = dict(
        WanDancerEncodeAudio.execute(audio=_audio(mono), video_frames=3, audio_inject_scale=2.5)
    )
    stereo_feature = _assert_output(
        stereo, fps=3.0, scale=2.5, fps_string=" \u5e27\u7387\u662f3.0000"
    )
    expected_feature = _assert_output(
        expected, fps=3.0, scale=2.5, fps_string=" \u5e27\u7387\u662f3.0000"
    )
    torch.testing.assert_close(stereo_feature, expected_feature, rtol=0, atol=0)
    assert torch.isfinite(stereo_feature).all()
    assert torch.count_nonzero(stereo_feature[:, :, 21:33]) > 0
    assert torch.equal(batched, before)


@pytest.mark.parametrize(
    ("video_frames", "fps", "text"),
    (
        (1, 10.0, " \u5e27\u7387\u662f10.0000"),
        (2, 15.0, " \u5e27\u7387\u662f15.0000"),
        (3, 30.0, ", \u5e27\u7387\u662f30fps\u3002"),
    ),
)
def test_controlled_feature_rows_preserve_column_order_and_fps_branches(
    monkeypatch: pytest.MonkeyPatch, video_frames: int, fps: float, text: str
) -> None:
    feature_module = types.ModuleType("dinkster_wan._wandancer_audio_features")
    feature_module.quick_tempo_estimate = lambda *_args, **_kwargs: 120.0
    feature_module._compute_mel_spectrogram = lambda *_args, **_kwargs: np.ones(
        (128, 3), dtype=np.float32
    )
    feature_module.power_to_db = lambda values, **_kwargs: values
    feature_module.compute_onset_envelope = lambda *_args, **_kwargs: np.array(
        [1, 2, 3], dtype=np.float32
    )
    feature_module.compute_mfcc = lambda *_args, **_kwargs: (
        np.arange(60, dtype=np.float32).reshape(3, 20) + 10
    )
    feature_module.compute_chroma_cens = lambda **_kwargs: (
        np.arange(36, dtype=np.float32).reshape(12, 3) + 100
    )
    feature_module.detect_onset_peaks = lambda *_args, **_kwargs: np.array([1])
    feature_module.estimate_tempo_from_onset = lambda *_args, **_kwargs: 120.0
    feature_module.track_beats = lambda *_args, **_kwargs: np.array([2])
    audio_module = types.ModuleType("dinkster_inference.audio")
    audio_module.resample = lambda waveform, *_args: waveform
    management_module = types.ModuleType("dinkster_inference.model_management")
    management_module.intermediate_device = lambda: torch.device("cpu")
    monkeypatch.setitem(sys.modules, feature_module.__name__, feature_module)
    monkeypatch.setitem(sys.modules, audio_module.__name__, audio_module)
    monkeypatch.setitem(sys.modules, management_module.__name__, management_module)

    waveform = torch.ones((1, 1, 512), dtype=torch.float32)
    result = dict(
        WanDancerEncodeAudio.execute(
            audio=_audio(waveform), video_frames=video_frames, audio_inject_scale=0.25
        )
    )
    feature = _assert_output(result, fps=fps, scale=0.25, fps_string=text)
    envelope = np.array([1, 2, 3], dtype=np.float32)[:, None]
    mfcc = np.arange(60, dtype=np.float32).reshape(3, 20) + 10
    chroma = (np.arange(36, dtype=np.float32).reshape(12, 3) + 100).T
    expected = np.concatenate(
        (
            envelope,
            mfcc,
            chroma,
            np.array([[0], [1], [0]], dtype=np.float32),
            np.array([[0], [0], [1]], dtype=np.float32),
        ),
        axis=1,
    )
    torch.testing.assert_close(feature, torch.from_numpy(expected).unsqueeze(0), rtol=0, atol=0)


class AudioSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wandancer-audio-fixture.source",
            outputs=(OutputSpec("audio", TypeExpr.concrete(AUDIO_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        waveform = torch.stack((-torch.ones(15360), torch.ones(15360))).reshape(1, 2, 15360)
        return cls.outputs(audio=_audio(waveform))


class AudioConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wandancer-audio-fixture.consumer",
            inputs=(
                InputSpec("encoded", TypeExpr.concrete(AUDIO_ENCODER_OUTPUT_TYPE)),
                InputSpec("fps_string", TypeExpr.concrete(CORE_STRING)),
                InputSpec("audio", TypeExpr.concrete(AUDIO_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, encoded: object, fps_string: str, audio: object) -> dict[str, object]:
        source = materialize_audio(audio)
        waveform = source["waveform"]
        assert isinstance(waveform, torch.Tensor) and waveform.shape == (1, 2, 15360)
        assert torch.equal(waveform[0, 0], -torch.ones(15360))
        assert torch.equal(waveform[0, 1], torch.ones(15360))
        value = materialize_latent(encoded)
        feature = value["audio_feature"]
        assert isinstance(feature, torch.Tensor) and feature.shape == (1, 31, 35)
        assert feature.dtype is torch.float32 and feature.device.type == "cpu"
        assert value["fps"] == 3.0 and value["audio_inject_scale"] == 0.5
        assert fps_string == " \u5e27\u7387\u662f3.0000"
        expected = torch.zeros((1, 31, 35), dtype=torch.float32)
        expected[:, :, 1] = -100 * np.sqrt(128)
        torch.testing.assert_close(feature[:, :, 1], expected[:, :, 1], rtol=0, atol=2e-4)
        torch.testing.assert_close(feature[:, :, 0], expected[:, :, 0], rtol=0, atol=0)
        torch.testing.assert_close(feature[:, :, 2:], expected[:, :, 2:], rtol=0, atol=0)
        return cls.outputs(summary="wandancer-audio-host-values")


HOST_NODES = (AudioSource, AudioConsumer)

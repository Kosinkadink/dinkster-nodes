"""CPU contracts for portable Wan camera, track, and audio encoder values."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from dinkster_api.v1 import (
    AUDIO_ENCODER_OUTPUT_TYPE,
    TRACKS_TYPE,
    WAN_CAMERA_EMBEDDING_TYPE,
    TypeExpr,
    TypeRegistry,
    materialize_latent,
)
from dinkster_wan import register_types

_CAMERA = TypeExpr.concrete(WAN_CAMERA_EMBEDDING_TYPE)
_TRACKS = TypeExpr.concrete(TRACKS_TYPE)
_AUDIO_ENCODER_OUTPUT = TypeExpr.concrete(AUDIO_ENCODER_OUTPUT_TYPE)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return bytes(value.contiguous().reshape(-1).view(torch.uint8).tolist())


def _assert_tensor(actual: object, expected: torch.Tensor) -> None:
    assert type(actual) is torch.Tensor
    assert actual.dtype is expected.dtype
    assert actual.shape == expected.shape
    assert actual.device.type == "cpu"
    assert _tensor_bytes(actual) == _tensor_bytes(expected)


def _camera() -> torch.Tensor:
    return torch.arange(30, dtype=torch.float16).reshape(3, 2, 5).transpose(0, 2)


def _tracks(include_visibility: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "track_path": torch.arange(48, dtype=torch.bfloat16).reshape(2, 3, 4, 2).transpose(1, 2),
        "name": "non-square tracks",
        "frame_indices": (3, 7),
        "metadata": {"enabled": True, "none": None},
    }
    if include_visibility:
        result["track_visibility"] = torch.tensor(
            [[[True, False, True], [False, True, False]]], dtype=torch.bool
        ).transpose(1, 2)
    return result


def _audio(layers_as_tuple: bool) -> dict[str, object]:
    layers: list[torch.Tensor] = [
        torch.arange(30, dtype=torch.float32).reshape(2, 3, 5).transpose(0, 2),
        torch.empty((2, 0, 3), dtype=torch.float16),
    ]
    return {
        "encoded_audio": torch.arange(42, dtype=torch.float32).reshape(2, 3, 7).transpose(0, 2),
        "encoded_audio_all_layers": tuple(layers) if layers_as_tuple else layers,
        "audio_samples": 16001,
        "sample_rate": 16000,
        "metadata": {"source": "cpu", "normalized": False},
    }


def _assert_tracks(actual: object, expected: Mapping[str, object]) -> None:
    assert type(actual) is dict
    assert set(actual) == set(expected)
    _assert_tensor(actual["track_path"], expected["track_path"])
    assert actual["name"] == expected["name"]
    assert actual["frame_indices"] == expected["frame_indices"]
    assert actual["metadata"] == expected["metadata"]
    if "track_visibility" in expected:
        _assert_tensor(actual["track_visibility"], expected["track_visibility"])
        assert actual["track_visibility"].dtype is torch.bool


def _assert_audio(actual: object, expected: Mapping[str, object]) -> None:
    assert type(actual) is dict
    assert set(actual) == set(expected)
    _assert_tensor(actual["encoded_audio"], expected["encoded_audio"])
    actual_layers = actual["encoded_audio_all_layers"]
    expected_layers = expected["encoded_audio_all_layers"]
    assert type(actual_layers) is type(expected_layers)
    assert len(actual_layers) == len(expected_layers)
    for received, source in zip(actual_layers, expected_layers, strict=True):
        _assert_tensor(received, source)
    assert actual["audio_samples"] == expected["audio_samples"]
    assert type(actual["audio_samples"]) is int
    assert actual["sample_rate"] == expected["sample_rate"]
    assert actual["metadata"] == expected["metadata"]


@pytest.mark.parametrize("include_visibility", (False, True))
@pytest.mark.parametrize("layers_as_tuple", (False, True))
def test_wan_portable_value_codecs_preserve_cpu_tensors_and_source_storage(
    include_visibility: bool, layers_as_tuple: bool
) -> None:
    registry = TypeRegistry()
    registered = register_types(registry)
    assert registry.spec(WAN_CAMERA_EMBEDDING_TYPE) in registered
    assert registry.spec(TRACKS_TYPE) in registered
    assert registry.spec(AUDIO_ENCODER_OUTPUT_TYPE) not in registered
    assert registry.spec(AUDIO_ENCODER_OUTPUT_TYPE).encode is not None
    assert register_types(registry) == registered

    camera = _camera()
    tracks = _tracks(include_visibility)
    audio = _audio(layers_as_tuple)
    source_storages = (
        camera.untyped_storage().data_ptr(),
        tracks["track_path"].untyped_storage().data_ptr(),
        *((tracks["track_visibility"].untyped_storage().data_ptr(),) if include_visibility else ()),
        audio["encoded_audio"].untyped_storage().data_ptr(),
        *(layer.untyped_storage().data_ptr() for layer in audio["encoded_audio_all_layers"]),
    )

    camera_spec = registry.spec(WAN_CAMERA_EMBEDDING_TYPE)
    tracks_spec = registry.spec(TRACKS_TYPE)
    audio_spec = registry.spec(AUDIO_ENCODER_OUTPUT_TYPE)
    decoded_camera = camera_spec.decode(camera_spec.encode(camera))
    decoded_tracks = tracks_spec.decode(tracks_spec.encode(tracks))
    decoded_audio = audio_spec.decode(audio_spec.encode(audio))
    materialized_camera = materialize_latent(decoded_camera)
    materialized_tracks = materialize_latent(decoded_tracks)
    materialized_audio = materialize_latent(decoded_audio)

    _assert_tensor(materialized_camera, camera)
    _assert_tracks(materialized_tracks, tracks)
    _assert_audio(materialized_audio, audio)
    assert camera_spec.fingerprint is not None
    assert tracks_spec.fingerprint is not None
    assert audio_spec.fingerprint is not None
    assert camera_spec.fingerprint(camera) == camera_spec.fingerprint(decoded_camera)
    assert tracks_spec.fingerprint(tracks) == tracks_spec.fingerprint(decoded_tracks)
    assert audio_spec.fingerprint(audio) == audio_spec.fingerprint(decoded_audio)
    assert materialize_latent(camera) is camera
    assert materialize_latent(tracks) is tracks
    assert materialize_latent(audio) is audio
    assert source_storages == (
        camera.untyped_storage().data_ptr(),
        tracks["track_path"].untyped_storage().data_ptr(),
        *((tracks["track_visibility"].untyped_storage().data_ptr(),) if include_visibility else ()),
        audio["encoded_audio"].untyped_storage().data_ptr(),
        *(layer.untyped_storage().data_ptr() for layer in audio["encoded_audio_all_layers"]),
    )
    _assert_tensor(camera, _camera())
    _assert_tracks(tracks, _tracks(include_visibility))
    _assert_audio(audio, _audio(layers_as_tuple))

"""Uploaded audio keeps canonical asset identity through production loading."""

from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path
from typing import cast

import av
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetRef, AssetVault, LibraryStore
from dinkster_audio.audio_io import LoadAudio
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_server import ServerLibrary, create_app
from dinkster_values import (
    TypeRegistry,
    audio_meta,
    coerce_audio,
    register_core_types,
    render_audio_wav,
)
from dinkster_values.audio_codec import normalize_audio_waveform_request, render_audio_waveform
from dinkster_workers import InProcessWorker
from PIL import Image

from tests.async_scenarios import run_scenario


def writable_media_recorder_webm_audio_encoders() -> tuple[str, ...]:
    available: list[str] = []
    for codec in ("libopus", "libvorbis"):
        try:
            av.Codec(codec, "w")
        except Exception:  # noqa: BLE001 - PyAV uses several codec error classes
            continue
        available.append(codec)
    return tuple(available)


def _real_audio_webm() -> bytes:
    output = BytesIO()
    with av.open(output, "w", format="webm") as container:
        stream = container.add_stream("libopus", rate=48_000)
        stream.layout = "mono"
        stream.codec_context.thread_count = 1
        samples = np.arange(960, dtype=np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = 48_000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def _media_recorder_audio_webm(codec: str) -> bytes:
    output = BytesIO()
    with av.open(output, "w", format="webm", options={"live": "1"}) as container:
        stream = cast(av.AudioStream, container.add_stream(codec, rate=48_000))
        stream.layout = "stereo"
        stream.codec_context.thread_count = 1
        if codec == "vorbis":
            stream.codec_context.options = {"strict": "experimental"}
        for index in range(5):
            samples = np.arange(index * 1920, (index + 1) * 1920, dtype=np.int16).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="stereo")
            frame.sample_rate = 48_000
            frame.pts = index * 960
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def _real_audio_m4a() -> bytes:
    output = BytesIO()
    with av.open(output, "w", format="ipod") as container:
        stream = container.add_stream("aac", rate=48_000)
        stream.layout = "mono"
        stream.codec_context.thread_count = 1
        samples = np.arange(1024, dtype=np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = 48_000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


async def _media_client(tmp_path: Path) -> tuple[TestClient, ServerLibrary]:
    library = ServerLibrary(
        vault=AssetVault(tmp_path / "vault"),
        store=LibraryStore(tmp_path / "library.sqlite"),
        upload_limit=1024 * 1024,
        media_upload_limits=dict.fromkeys(
            ("media/image", "media/audio", "media/video", "media/model3d"), 1024 * 1024
        ),
    )

    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        return Engine(
            schemas={},
            registry=registry,
            worker=InProcessWorker({}, registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    client = TestClient(TestServer(create_app(factory, {}, library=library)))
    await client.start_server()
    return client, library


def _media_path(scope: str, kind: str, name: str) -> str:
    return f"/api/assets/media?scope={scope}&kind={kind}&name={name}"


def _media_headers(media_type: str) -> dict[str, str]:
    return {"Content-Type": media_type}


def test_audio_webm_upload_load_and_rendition_use_canonical_asset(tmp_path: Path) -> None:
    async def scenario() -> None:
        payload = _real_audio_webm()
        client, library = await _media_client(tmp_path)
        try:
            mislabeled = await client.post(
                _media_path("local", "media/audio", "recording.webm"),
                data=payload,
                headers=_media_headers("video/webm"),
            )
            assert mislabeled.status == 409
            assert (await mislabeled.json())["error"]["code"] == "asset.media.content_type_mismatch"
            response = await client.post(
                _media_path("local", "media/audio", "recording.webm"),
                data=payload,
                headers=_media_headers("audio/webm;codecs=opus"),
            )
            assert response.status == 201, await response.text()
            body = await response.json()
            assert body["kind"] == "media/audio"
            assert body["asset"]["mediaType"] == "audio/webm"
            asset = AssetRef.from_wire(body["asset"], resolver=library.vault)
            audio = coerce_audio(LoadAudio.execute(audio=asset)["AUDIO"])
            # The fixture has 960 samples; container duration includes Opus padding.
            assert audio_meta(audio)["frames"] == 960
            assert audio_meta(audio)["duration"] == 0.02
            assert audio_meta(audio)["shape"] == (1, 1, 960)
            rendition = render_audio_wav(audio)
            with wave.open(BytesIO(rendition), "rb") as decoded:
                assert decoded.getframerate() == 48_000
                assert decoded.getnchannels() == 1
                assert decoded.getnframes() == 960
        finally:
            await client.close()

    run_scenario(scenario())


async def _assert_media_recorder_audio_webm_upload(tmp_path: Path, codec: str) -> None:
    payload = _media_recorder_audio_webm(codec)
    with av.open(BytesIO(payload), mode="r") as container:
        stream = container.streams.audio[0]
        assert container.duration is None
        assert stream.duration is None
        expected_frames = sum(frame.samples for frame in container.decode(stream))

    client, library = await _media_client(tmp_path)
    try:
        response = await client.post(
            _media_path("local", "media/audio", "media-recorder.webm"),
            data=payload,
            headers=_media_headers("audio/webm"),
        )
        assert response.status == 201, await response.text()
        asset = AssetRef.from_wire((await response.json())["asset"], resolver=library.vault)
        audio = coerce_audio(LoadAudio.execute(audio=asset)["AUDIO"])
        metadata = audio_meta(audio)
        assert metadata["frames"] == expected_frames
        assert metadata["duration"] == expected_frames / 48_000
        assert metadata["shape"] == (1, 2, expected_frames)

        parameters = normalize_audio_waveform_request({"waveform": "32x16"}, metadata)
        with Image.open(BytesIO(render_audio_waveform(audio, parameters))) as image:
            assert image.format == "PNG"
            assert image.size == (32, 16)
    finally:
        await client.close()


@pytest.mark.parametrize("codec", writable_media_recorder_webm_audio_encoders())
def test_media_recorder_audio_webm_publishes_length_and_waveform(
    tmp_path: Path, codec: str
) -> None:
    run_scenario(_assert_media_recorder_audio_webm_upload(tmp_path, codec))


def test_media_recorder_webm_fixture_encoders_follow_pyav_availability() -> None:
    available = writable_media_recorder_webm_audio_encoders()
    expected: list[str] = []
    for codec in ("libopus", "libvorbis"):
        try:
            av.Codec(codec, "w")
        except Exception:  # noqa: BLE001 - PyAV uses several codec error classes
            continue
        expected.append(codec)
    assert available == tuple(expected)
    assert "libopus" in available


def test_vorbis_webm_upload_remains_admitted(tmp_path: Path) -> None:
    run_scenario(_assert_media_recorder_audio_webm_upload(tmp_path, "vorbis"))


def test_audio_m4a_upload_load_and_rendition_use_canonical_asset(tmp_path: Path) -> None:
    async def scenario() -> None:
        payload = _real_audio_m4a()
        client, library = await _media_client(tmp_path)
        try:
            mislabeled = await client.post(
                _media_path("local", "media/audio", "recording.m4a"),
                data=payload,
                headers=_media_headers("video/mp4"),
            )
            assert mislabeled.status == 409
            assert (await mislabeled.json())["error"]["code"] == "asset.media.content_type_mismatch"
            response = await client.post(
                _media_path("local", "media/audio", "recording.m4a"),
                data=payload,
                headers=_media_headers("audio/mp4"),
            )
            assert response.status == 201, await response.text()
            body = await response.json()
            assert body["kind"] == "media/audio"
            assert body["asset"]["mediaType"] == "audio/mp4"
            asset = AssetRef.from_wire(body["asset"], resolver=library.vault)
            audio = coerce_audio(LoadAudio.execute(audio=asset)["AUDIO"])
            rendition = render_audio_wav(audio)
            with wave.open(BytesIO(rendition), "rb") as decoded:
                assert decoded.getframerate() == 48_000
                assert decoded.getnchannels() == 1
                assert decoded.getnframes() > 0
        finally:
            await client.close()

    run_scenario(scenario())

"""CPU behavior for the native SaveVideo output node."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from dinkster_api.v1 import ExportSnapshot, Record
from dinkster_assets.identity import AssetError
from dinkster_video.video_components import CreateVideo
from dinkster_video.video_save import SaveVideo

pytestmark = pytest.mark.usefixtures("saved_artifact_context")


def _mounts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    default, alternate, readonly = (
        tmp_path / name for name in ("default", "alternate", "readonly")
    )
    for root in (default, alternate, readonly):
        root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "outputMount": "default",
                "mounts": [
                    {"id": "default", "root": str(default), "mode": "readwrite"},
                    {"id": "alternate", "root": str(alternate), "mode": "readwrite"},
                    {"id": "readonly", "root": str(readonly), "mode": "read"},
                ],
            }
        ),
        encoding="ascii",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return default, alternate, readonly


def _images() -> np.ndarray[Any, Any]:
    images = np.zeros((3, 16, 24, 3), dtype=np.float32)
    images[:, :, :12, 0] = 1
    images[:, :, 12:, 1] = 1
    return images


def _video() -> object:
    return CreateVideo.execute(images=_images(), fps=4.0)["video"]


def _format(kind: str, codec: str, crf: float | None = None) -> Record:
    """The format value execute receives: the container, its codec and the encoding."""
    if codec == "auto":
        return Record(kind, {"codec": Record("auto", {})})
    encoding = Record("auto", {}) if crf is None else Record("re-encode", {"crf": crf})
    return Record(kind, {"codec": Record(codec, {"encoding": encoding})})


def _tags(path: Path) -> tuple[dict[str, object], Any, int]:
    with av.open(path) as container:
        tags = {
            name: json.loads(container.metadata[name])
            for name in ("prompt", "workflow")
            if name in container.metadata
        }
        stream = container.streams.video[0]
        return tags, stream, len(list(container.decode(stream)))


def test_save_video_writes_selected_mount_metadata_and_original_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_video import video_save

    default, alternate, _ = _mounts(tmp_path, monkeypatch)
    video = _video()
    prompt = {"4": {"class_type": "SaveVideo", "inputs": {"format": "mp4"}}}
    workflow = {"nodes": [{"id": 4, "type": "SaveVideo"}]}
    monkeypatch.setattr(
        video_save,
        "current_export_snapshot",
        lambda: ExportSnapshot(prompt=prompt, extra_pnginfo={"workflow": workflow}),
    )

    assert (
        SaveVideo.execute(
            video=video,
            filename_prefix="video/%width%x%height%/clip",
            format=_format("mp4", "h264", 18.0),
            mount="alternate",
        )["video"]
        is video
    )
    assert not list(default.rglob("*"))
    files = list(alternate.glob("video/24x16/clip_*.mp4"))
    assert len(files) == 1
    tags, stream, frame_count = _tags(files[0])
    assert {name: tags[name] for name in ("prompt", "workflow")} == {
        "prompt": prompt,
        "workflow": workflow,
    }
    assert stream.codec_context.name == "h264"
    assert (stream.width, stream.height) == (24, 16)
    assert float(stream.average_rate) == 4.0
    assert frame_count == 3


@pytest.mark.parametrize(
    ("kind", "codec", "crf", "extension", "encoder"),
    (
        ("auto", "auto", None, "mp4", "h264"),
        ("auto", "h264", None, "mp4", "h264"),
        ("auto", "h264", 18.0, "mp4", "h264"),
        ("auto", "av1", None, "webm", "av1"),
        ("auto", "av1", 40.0, "webm", "av1"),
        ("mp4", "auto", None, "mp4", "h264"),
        ("mp4", "h264", 30.0, "mp4", "h264"),
        ("mp4", "av1", None, "mp4", "av1"),
        ("mkv", "auto", None, "mkv", "h264"),
        ("mkv", "h264", None, "mkv", "h264"),
        ("mkv", "av1", 40.0, "mkv", "av1"),
        ("webm", "auto", None, "webm", "av1"),
        ("webm", "av1", 40.0, "webm", "av1"),
    ),
)
def test_save_video_writes_each_container_codec_and_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    codec: str,
    crf: float | None,
    extension: str,
    encoder: str,
) -> None:
    default, _, _ = _mounts(tmp_path, monkeypatch)
    video = _video()
    result = SaveVideo.execute(
        video=video, filename_prefix="clip/out", format=_format(kind, codec, crf)
    )
    assert result["video"] is video
    path = next(default.glob(f"clip/out_*.{extension}"))
    _, stream, frame_count = _tags(path)
    assert encoder in stream.codec_context.name
    assert (stream.width, stream.height) == (24, 16)
    assert frame_count == 3


@pytest.mark.parametrize(
    ("format", "codec", "prefix"),
    (
        ("gif", "auto", "video/clip"),
        ("webm", "h264", "video/clip"),
        ("mp4", "vp9", "video/clip"),
        ("mp4", "auto", "../escape"),
    ),
)
def test_save_video_refuses_unsafe_values_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format: str, codec: str, prefix: str
) -> None:
    default, alternate, readonly = _mounts(tmp_path, monkeypatch)
    with pytest.raises((ValueError, AssetError)):
        SaveVideo.execute(video=_video(), filename_prefix=prefix, format=_format(format, codec))
    assert not list(default.rglob("*"))
    assert not list(alternate.rglob("*"))
    assert not list(readonly.rglob("*"))


def test_save_video_refuses_readonly_mount_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default, alternate, readonly = _mounts(tmp_path, monkeypatch)
    with pytest.raises(AssetError, match="mount"):
        SaveVideo.execute(
            video=_video(),
            filename_prefix="video/clip",
            format=_format("mp4", "auto"),
            mount="readonly",
        )
    assert not list(default.rglob("*"))
    assert not list(alternate.rglob("*"))
    assert not list(readonly.rglob("*"))

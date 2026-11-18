"""CPU behavior for the native SaveVideo output node."""

from __future__ import annotations

import json
from collections import UserDict
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from dinkster_api.v1 import (
    ExportSnapshot,
)
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


def _tags(path: Path) -> tuple[dict[str, object], Any, int]:
    with av.open(path) as container:
        raw = {name.lower(): value for name, value in container.metadata.items()}
        tags = {
            name: json.loads(raw[name])
            for name in (
                "prompt",
                "workflow",
                "dinkster.modelprovenance",
                "dinkster.modelprovenance.1",
            )
            if name in raw
        }
        stream = container.streams.video[0]
        return tags, stream, len(list(container.decode(stream)))


@pytest.mark.parametrize("format", ["mp4", "mkv"])
@pytest.mark.parametrize(
    ("disabled", "absent_prompt"), [(False, False), (False, True), (True, False)]
)
def test_save_video_writes_selected_mount_metadata_and_original_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format: str,
    disabled: bool,
    absent_prompt: bool,
) -> None:
    from dinkster_api.v1 import initialize_inference_process
    from dinkster_inference.cli_args import args
    from dinkster_video import video_save

    initialize_inference_process()
    monkeypatch.setattr(args, "disable_metadata", disabled)
    default, alternate, _ = _mounts(tmp_path, monkeypatch)
    video = _video()
    prompt = {
        "4": {"class_type": "SaveVideo", "inputs": {"format": format, "seed": 9007199254740993}}
    }
    workflow = {"nodes": [{"id": 4, "type": "SaveVideo"}]}
    extra = {
        "workflow": workflow,
        "dinkster.modelProvenance": {"caller": "keep"},
        "dinkster.modelProvenance.1": {"version": 1, "models": [{"name": "same.bin", "size": 23}]},
    }

    if absent_prompt:
        extra["prompt"] = {"caller": "keep"}

    def snapshot() -> ExportSnapshot:
        assert not disabled, "disabled metadata must not read the export snapshot"
        return ExportSnapshot(
            prompt=None if absent_prompt else UserDict(prompt), extra_pnginfo=extra
        )

    monkeypatch.setattr(video_save, "current_export_snapshot", snapshot)

    assert (
        SaveVideo.execute(
            video=video,
            filename_prefix="video/%width%x%height%/clip",
            format={
                "format": format,
                "codec": {"codec": "h264", "encoding": {"crf": 18.0}},
            },
            mount="alternate",
        )["video"]
        is video
    )
    assert not list(default.rglob("*"))
    files = list(alternate.glob(f"video/24x16/clip_*.{format}"))
    assert len(files) == 1
    tags, stream, frame_count = _tags(files[0])
    assert tags == (
        {}
        if disabled
        else {"prompt": prompt, **{key.lower(): value for key, value in extra.items()}}
    )
    assert stream.codec_context.name == "h264"
    assert (stream.width, stream.height) == (24, 16)
    assert float(stream.average_rate) == 4.0
    assert frame_count == 3


@pytest.mark.parametrize(
    ("format", "codec", "crf"),
    (
        ("auto", "h264", 18),
        ("mkv", "h264", 18),
    ),
)
def test_save_video_accepts_flat_dynamic_native_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format: str,
    codec: str,
    crf: int,
) -> None:
    default, _, _ = _mounts(tmp_path, monkeypatch)
    video = _video()
    extension = "mp4" if format == "auto" else format
    result = SaveVideo.execute(
        video=video,
        filename_prefix=f"flat/{extension}",
        format=format,
        **{
            "format.codec": codec,
            "format.codec.encoding": "re-encode",
            "format.codec.encoding.crf": crf,
        },
    )
    assert result["video"] is video
    path = next(default.glob(f"flat/{extension}_*.{extension}"))
    _, stream, frame_count = _tags(path)
    assert stream.codec_context.name == "h264"
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
        SaveVideo.execute(
            video=_video(), filename_prefix=prefix, format=format, **{"format.codec": codec}
        )
    assert not list(default.rglob("*"))
    assert not list(alternate.rglob("*"))
    assert not list(readonly.rglob("*"))


def test_save_video_refuses_readonly_mount_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default, alternate, readonly = _mounts(tmp_path, monkeypatch)
    with pytest.raises(AssetError, match="mount"):
        SaveVideo.execute(
            video=_video(), filename_prefix="video/clip", format="mp4", mount="readonly"
        )
    assert not list(default.rglob("*"))
    assert not list(alternate.rglob("*"))
    assert not list(readonly.rglob("*"))

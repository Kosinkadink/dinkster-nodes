"""LoadVideo admits indexed video assets without copying their source bytes."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from dinkster_api.v1 import (
    AssetError,
    AssetIntegrityError,
    AssetRef,
)
from dinkster_assets import LocalAssetRef, MountDef, MountTable, digest_file
from dinkster_values import video_source
from dinkster_video.video_load import LoadVideo


def _frames() -> np.ndarray:
    return np.array(
        [
            [
                [[3, 17, 29], [41, 53, 67], [71, 83, 97]],
                [[101, 113, 127], [131, 149, 157], [163, 173, 181]],
            ],
            [
                [[191, 199, 211], [223, 227, 229], [233, 239, 241]],
                [[251, 2, 5], [7, 11, 13], [19, 23, 31]],
            ],
        ],
        dtype=np.uint8,
    )


def _write_clip(path: Path) -> tuple[bytes, np.ndarray]:
    frames = _frames()
    with av.open(path, "w") as container:
        stream = container.add_stream("huffyuv", rate=Fraction(30_000, 1_001))
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "rgb24"
        for pixels in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path.read_bytes(), frames


def _asset(
    tmp_path: Path, *, indexed: bool
) -> tuple[MountTable, AssetRef | LocalAssetRef, bytes, np.ndarray]:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "asymmetric.avi"
    original, frames = _write_clip(path)
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="video", path=source), kind="media/video")
    if indexed:
        table.scan("video")
    asset = table.bind_name(path.name, "media/video")
    assert isinstance(asset, AssetRef if indexed else LocalAssetRef)
    return table, asset, original, frames


def _decoded_pixels(video: object) -> np.ndarray:
    source = video_source(video)
    with source.open() as handle, av.open(handle) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def test_load_video_admits_granted_sources_without_copying_and_rejects_untrusted_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table, local, original, expected = _asset(tmp_path, indexed=False)
    path = tmp_path / "source" / "asymmetric.avi"
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    monkeypatch.setenv("DINKSTER_PACK_SCRATCH", str(tmp_path / "scratch"))

    with pytest.raises(AssetError, match="must be indexed"):
        LoadVideo.execute(file=local)
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        LoadVideo.execute(file=path)
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        LoadVideo.execute(file=str(path))

    table.scan("video")
    selected = table.bind_name(path.name, "media/video")
    assert isinstance(selected, AssetRef)
    loaded_from_local = LoadVideo.execute(file=local)["VIDEO"]
    loaded = LoadVideo.execute(file=selected)["VIDEO"]
    assert video_source(loaded_from_local).digest == selected.digest
    source = video_source(loaded)
    assert isinstance(source, AssetRef)
    assert source.digest == digest_file(path)
    assert source.local_path() == path
    assert source.read_bytes() == original
    np.testing.assert_array_equal(_decoded_pixels(loaded), expected)
    assert loaded["probe"]["fps"] == Fraction(30_000, 1_001)
    assert loaded["probe"]["width"] == 3
    assert loaded["probe"]["height"] == 2
    assert loaded["probe"]["frame_count"] == 2
    assert not (tmp_path / "vault").exists()
    assert not (tmp_path / "scratch").exists()
    assert path.read_bytes() == original

    path.write_bytes(original + b"changed")
    with pytest.raises(AssetIntegrityError):
        LoadVideo.execute(file=selected)

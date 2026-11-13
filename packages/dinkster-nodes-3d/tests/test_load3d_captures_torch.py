"""Authored viewport captures through native Load3D."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
import torch
from dinkster_3d.load3d_captures import Load3D
from dinkster_api.v1 import (
    AbsentOutput,
    AssetRef,
    AssetVault,
    digest_bytes,
    effective_video_facts,
)
from dinkster_assets import LocalAssetRef, MountDef, MountTable
from PIL import Image


@pytest.fixture
def captures(tmp_path: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray, AssetVault]:
    vault = AssetVault(tmp_path / "vault")
    pixels = np.array([[[13, 51, 239, 0], [81, 127, 9, 64], [203, 17, 109, 255]]], dtype=np.uint8)
    normals = np.array([[[31, 191, 67]], [[251, 5, 137]]], dtype=np.uint8)
    state: dict[str, object] = {
        "camera_info": {
            "position": {"x": 3.0, "y": -2.0, "z": 7.0},
            "target": {"x": -1.0, "y": 4.0, "z": 0.0},
            "zoom": 1.75,
            "cameraType": "perspective",
        },
        "recording": "",
    }
    for name, array in (("image", pixels), ("mask", pixels), ("normal", normals)):
        stream = BytesIO()
        Image.fromarray(array).save(stream, format="PNG")
        data = stream.getvalue()
        digest = digest_bytes(data)
        with vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        state[name] = AssetRef(digest, name + ".png", len(data), "image/png", resolver=vault)
    return state, pixels, normals, vault


def _assert_captures(result: Mapping[str, Any], pixels: np.ndarray, normals: np.ndarray) -> None:
    torch.testing.assert_close(
        torch.as_tensor(result["image"]), torch.tensor(pixels[None, ..., :3] / 255).float()
    )
    torch.testing.assert_close(
        torch.as_tensor(result["mask"]), 1 - torch.tensor(pixels[None, ..., 3] / 255).float()
    )
    torch.testing.assert_close(
        torch.as_tensor(result["normal"]), torch.tensor(normals[None] / 255).float()
    )


@pytest.mark.parametrize("placement", [[], None, [{"scale": [2, 3, 5]}]])
def test_load3d_keeps_capture_pixels_alpha_and_placement(captures, placement: object) -> None:
    state, pixels, normals, _vault = captures
    if placement != []:
        state["model_3d_info"] = placement
    result = Load3D.execute(model_file="none", image=state, width=37, height=91)
    _assert_captures(result, pixels, normals)
    assert result["mesh_path"] == ""
    assert result["model_3d"] is None
    assert result["recording_video"] == AbsentOutput("no recording")
    assert result["model_3d_info"] == placement
    assert result["camera_info"] is state["camera_info"]


@pytest.mark.parametrize("format_id", ["gltf", "fbx"])
def test_load3d_retains_nested_selection_and_declared_model_bytes(
    captures, tmp_path: Path, format_id: str
) -> None:
    state, pixels, normals, _vault = captures
    root = tmp_path / "models"
    (root / "nested").mkdir(parents=True)
    primary = (
        b'{"asset":{"version":"2.0"},"buffers":[{"uri":"mesh.bin"}]}'
        if format_id == "gltf"
        else b"original opaque fbx bytes"
    )
    name = "nested/source." + format_id
    (root / name).write_bytes(primary)
    (root / "nested/mesh.bin").write_bytes(b"distinct declared buffer bytes")
    mounts = MountTable(tmp_path / "mounts.json")
    mounts.add(MountDef("models", root, kind="media/model3d"))
    source = mounts.bind_name(name, "media/model3d")
    assert isinstance(source, LocalAssetRef)
    result = Load3D.execute(model_file=source, image=state, width=37, height=91)
    _assert_captures(result, pixels, normals)
    assert result["mesh_path"] == name
    packet = result["model_3d"]
    assert packet["source"] is source
    assert source.read_bytes() == primary
    assert {key: value.read_bytes() for key, value in packet["resources"].items()} == (
        {"mesh.bin": b"distinct declared buffer bytes"} if format_id == "gltf" else {}
    )


def test_load3d_retains_recorded_video_source_and_timing(captures) -> None:
    state, pixels, normals, vault = captures
    stream = BytesIO()
    with av.open(stream, "w", format="matroska") as container:
        video = container.add_stream("ffv1", rate=7)
        video.width, video.height, video.pix_fmt = 3, 1, "bgr0"
        for packet in video.encode(av.VideoFrame.from_ndarray(pixels[..., :3], format="rgb24")):
            container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
    data = stream.getvalue()
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    recording = AssetRef(digest, "recording.mkv", len(data), "video/x-matroska", resolver=vault)
    state["recording"] = recording
    result = Load3D.execute(model_file=None, image=state)
    _assert_captures(result, pixels, normals)
    assert result["recording_video"]["source"] is recording
    facts = effective_video_facts(result["recording_video"])
    assert (facts["width"], facts["height"], facts["fps"]) == (3, 1, Fraction(7))
    assert recording.read_bytes() == data

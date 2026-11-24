"""Authored viewport captures through native Load3D."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest
import torch
from dinkster_3d import register_types
from dinkster_3d.load3d_captures import Load3D
from dinkster_api.v1 import (
    LOAD_3D_TYPE,
    AbsentOutput,
    AssetRef,
    AssetVault,
    digest_bytes,
    effective_video_facts,
)
from dinkster_assets import LibraryStore, LocalAssetRef, MountDef, MountTable
from dinkster_engine import EngineEvent
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_server.library import ServerLibrary
from dinkster_server.preflight import bind_model_names
from dinkster_values import ABSENT_REASON_META_KEY, TypeRegistry, register_core_types
from geometry_pack_fixtures import geometry_pack_specs
from PIL import Image

from dinkster.compose import ServingComposer


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


EARLIER = "These pictures show an earlier view. Queue the run from the editor to take new ones."
NOT_DRAWN = (
    "No pictures: the view was not drawn before this run. That happens when the model "
    "arrives by a wire, or when the run was not queued from the editor."
)


@pytest.mark.parametrize(
    ("case", "warning"),
    [
        ("earlier view", EARLIER),
        ("same view", None),
        ("same view, model by reference", None),
        ("same view, local model by reference", None),
        ("other model of the same name", EARLIER),
        ("no pictures", NOT_DRAWN),
        ("no pictures, no model", "No pictures: no model is chosen."),
    ],
)
def test_load3d_run_warns_on_earlier_view_and_gives_nothing_without_pictures(
    captures, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, warning: str | None
) -> None:
    state, pixels, normals, _vault = captures
    root = tmp_path / "models"
    root.mkdir()
    (root / "scene.gltf").write_bytes(b'{"asset":{"version":"2.0"}}')
    snapshot = tmp_path / "mounts.json"
    mounts = MountTable(snapshot, index_root=tmp_path / "indexes")
    mounts.add(MountDef("models", root, kind="media/model3d"))
    mounts.scan("models")
    environment = {
        "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
        "DINKSTER_ASSET_VAULT": str(tmp_path / "vault"),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    data = (root / "scene.gltf").read_bytes()
    reference = AssetRef(digest_bytes(data), "scene.gltf", len(data), "model/gltf+json").to_wire()
    pictured = {
        "model_file": "scene.gltf",
        "camera_info": state["camera_info"],
        "model_3d_info": None,
        "width": 37,
        "height": 91,
    }
    value: dict[str, object] = {
        **state,
        **{name: cast(AssetRef, state[name]).to_wire() for name in ("image", "mask", "normal")},
        "pictures_view": pictured,
    }
    if case == "earlier view":
        # The view was turned after its pictures were taken.
        value["camera_info"] = {**cast(dict[str, object], state["camera_info"]), "zoom": 2.5}
    if case.startswith("no pictures"):
        for name in ("image", "mask", "normal", "pictures_view"):
            del value[name]
    selection: object = "none" if case.endswith("no model") else "scene.gltf"
    if "model by reference" in case or "same name" in case:
        selection = reference
        pictured["model_file"] = reference
    if case == "same view, local model by reference":
        # A file added after the folder was indexed is granted by binding, not digest.
        (root / "later.gltf").write_bytes(data)
        local = mounts.bind_name("later.gltf", "media/model3d")
        assert isinstance(local, LocalAssetRef)
        selection = pictured["model_file"] = local.to_wire()
    if case == "other model of the same name":
        # The pictures were drawn from another file that has the same name.
        pictured["model_file"] = {**reference, "digest": digest_bytes(b"another scene")}
    schema = Load3D.define_schema()
    graph, missing = bind_model_names(
        Graph(
            {
                "load": GraphNode(
                    schema.node_type,
                    {
                        "model_file": selection,
                        "image": TypedLiteral(LOAD_3D_TYPE, value),
                        "width": 37,
                        "height": 91,
                    },
                )
            }
        ),
        {schema.node_type: schema},
        ServerLibrary(
            AssetVault(tmp_path / "vault"),
            LibraryStore(tmp_path / "library.sqlite"),
            resolver=mounts,
        ),
    )
    assert not missing
    events: list[EngineEvent] = []

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)
        composer = ServingComposer(registry=registry, worker_env=environment)
        try:
            for spec in geometry_pack_specs(in_process=True):
                await composer.add_pack(spec)
            result = await composer.composition.make_engine(events.append).run(graph, ["load"])
            out = result.outputs["load"]
            if case.startswith("no pictures"):
                for name in ("image", "mask", "normal"):
                    assert out[name].type_id == "core.absent"
                    assert out[name].meta.get(ABSENT_REASON_META_KEY) == warning
            else:
                _assert_captures(
                    {name: out[name].resolve() for name in ("image", "mask", "normal")},
                    pixels,
                    normals,
                )
            assert out["camera_info"].resolve() == value["camera_info"]
            assert (out["model_3d"].resolve() is None) == (selection == "none")
        finally:
            await composer.close()

    asyncio.run(scenario())
    warnings = [
        cast(Mapping[str, object], event.detail["data"])["message"]
        for event in events
        if event.kind == "node_event"
        and event.detail.get("name") == "log"
        and cast(Mapping[str, object], event.detail["data"])["level"] == "warning"
    ]
    assert warnings == ([] if warning is None else [warning])

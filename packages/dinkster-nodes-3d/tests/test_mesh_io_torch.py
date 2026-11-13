"""Mesh import exercises real file bytes, transforms and ordinary asset transport."""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_3d.mesh_io import Get3DComponents, MeshToFile3D, SaveGLB
from dinkster_api.v1 import (
    MESH_TYPE,
    AssetRef,
    ExportSnapshot,
    TriangleMeshBatch,
    TypeRegistry,
    digest_bytes,
    materialize_mesh,
    register_inference_types,
)
from dinkster_assets import AssetVault, MountDef, MountTable
from dinkster_protocol import use_export_snapshot
from dinkster_schema.reporting import capture_saved_artifacts
from PIL import Image


def export_mesh() -> TriangleMeshBatch[Any]:
    return TriangleMeshBatch(
        vertices=torch.tensor(
            [
                [[1.0, 2.0, 3.0], [-2.0, 1.0, 0.0], [0.0, 4.0, -1.0], [90.0, 90.0, 90.0]],
                [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0], [8.0, 8.0, 8.0]],
            ]
        ),
        faces=torch.tensor([[[2, 0, 1], [3, 3, 3]], [[0, 1, 2], [1, 2, 3]]]),
        vertex_counts=torch.tensor([3, 4]),
        face_counts=torch.tensor([1, 2]),
        uvs=torch.tensor([[[0.0, 0.0], [0.9, 0.1], [0.2, 0.8], [2.0, 2.0]]] * 2),
        normals=torch.tensor([[[0.0, 1.0, 0.0]]] * 4).transpose(0, 1).repeat(2, 1, 1),
        tangents=torch.tensor([[[1.0, 0.0, 0.0, -1.0]]] * 4).transpose(0, 1).repeat(2, 1, 1),
        vertex_colors=torch.tensor([[[0.2, 0.4, 0.6, 0.8]]] * 4).transpose(0, 1).repeat(2, 1, 1),
        texture=torch.full((2, 2, 3, 3), 0.3),
        metallic_roughness=torch.full((2, 2, 3, 3), 0.5),
        normal_map=torch.full((2, 2, 3, 3), 0.7),
        emissive=torch.full((2, 2, 3, 3), 0.9),
        occlusion_in_mr=True,
        material={"normal_scale": 0.6, "emissive_strength": 2.0, "double_sided": False},
    )


def assert_export(value: Any) -> bytes:
    assert value["format"] == "glb"
    source = value["source"]
    assert isinstance(source, AssetRef) and source.name.endswith(".glb")
    data = source.read_bytes()
    assert source.digest == digest_bytes(data) and source.size == len(data)
    assert struct.unpack_from("<4sII", data) == (b"glTF", 2, len(data))
    json_length = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20 : 20 + json_length])
    binary = data[28 + json_length :]
    assert document["accessors"][0]["count"] == 3
    assert document["accessors"][1]["count"] == 3
    assert struct.unpack_from("<9f", binary) == (1.0, 2.0, 3.0, -2.0, 1.0, 0.0, 0.0, 4.0, -1.0)
    assert struct.unpack_from("<3I", binary, 36) == (2, 0, 1)
    attributes = document["meshes"][0]["primitives"][0]["attributes"]
    assert set(attributes) == {"POSITION", "TEXCOORD_0", "COLOR_0", "NORMAL", "TANGENT"}
    assert len(document["images"]) == 4
    material = document["materials"][0]
    assert material["normalTexture"]["scale"] == 0.6
    assert material["doubleSided"] is False
    assert material["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"] == 2.0
    assert (
        material["occlusionTexture"]["index"]
        == material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]
    )
    return data


@pytest.mark.parametrize("portable", [False, True])
def test_mesh_export_first_padded_item_and_public_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    portable: bool,
) -> None:
    monkeypatch.delenv("DINKSTER_PACK_SCRATCH", raising=False)
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    original = export_mesh()
    vertices, faces = original.vertices.clone(), original.faces.clone()
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(MESH_TYPE)
    value = spec.decode(spec.encode(original)) if portable else original
    assert_export(MeshToFile3D.execute(mesh=value)["model_3d"])
    assert "Got 2; using first" in caplog.text
    torch.testing.assert_close(original.vertices, vertices, rtol=0, atol=0)
    torch.testing.assert_close(original.faces, faces, rtol=0, atol=0)


def test_mesh_export_empty_and_missing_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DINKSTER_PACK_SCRATCH", raising=False)
    monkeypatch.delenv("DINKSTER_ASSET_VAULT", raising=False)
    source = export_mesh()
    with pytest.raises(RuntimeError, match="produced-output store"):
        MeshToFile3D.execute(mesh=source)
    source.face_counts[0] = 0
    with pytest.raises(ValueError, match="MeshToFile3D: mesh is empty"):
        MeshToFile3D.execute(mesh=source)


@pytest.mark.parametrize("disable_metadata", [False, True])
def test_save_glb_skips_empty_item_and_preserves_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    disable_metadata: bool,
) -> None:
    from dinkster_inference.cli_args import args

    output = tmp_path / "output"
    output.mkdir()
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef("chosen", output, mode="readwrite"))
    table.scan("chosen")
    table.select_output_mount("chosen")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(tmp_path / "mounts.json"))
    monkeypatch.setattr(args, "disable_metadata", disable_metadata)
    mesh = export_mesh()
    mesh.face_counts[0] = 0
    submitted = ExportSnapshot(prompt={}, extra_pnginfo={"workflow": {"id": 73}})
    with use_export_snapshot(submitted), capture_saved_artifacts() as claims:
        assert SaveGLB.execute(mesh=mesh, filename_prefix="batch/model") == {}
    assert len(claims) == 1
    path = output / "batch/model_00001_.glb"
    assert path.is_file() and "skipping empty mesh at batch index 0" in caplog.text
    data = path.read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20 : 20 + length])
    assert document["accessors"][0]["count"] == 4
    assert document["accessors"][1]["count"] == 6
    assert struct.unpack_from("<3f", data, 28 + length) == (8.0, 0.0, 0.0)
    if disable_metadata:
        assert "extras" not in document["asset"]
    else:
        assert document["asset"]["extras"] == {"prompt": "{}", "workflow": '{"id": 73}'}
    assert claims[0]["digest"] == digest_bytes(data)


def test_save_glb_copies_primary_file_bytes_and_requires_mount_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_assets import AssetError

    monkeypatch.delenv("DINKSTER_MOUNTS_SNAPSHOT", raising=False)
    with pytest.raises(AssetError, match="DINKSTER_MOUNTS_SNAPSHOT"):
        SaveGLB.execute(mesh={})
    vault = AssetVault(tmp_path / "vault")
    payload = b'{"asset":{"version":"2.0"},"buffers":[{"uri":"data.bin"}]}'
    digest = digest_bytes(payload)
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    output, alternate = tmp_path / "output", tmp_path / "alternate"
    output.mkdir()
    alternate.mkdir()
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    for name, root in (("output", output), ("alternate", alternate)):
        table.add(MountDef(name, root, mode="readwrite"))
        table.scan(name)
    table.select_output_mount("output")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(tmp_path / "mounts.json"))
    source = AssetRef(digest, "source.gltf", len(payload), "model/gltf+json", resolver=vault)
    with capture_saved_artifacts() as claims:
        SaveGLB.execute(mesh={"format": "gltf", "source": source}, mount="alternate")
    assert (alternate / "3d/ComfyUI_00001_.gltf").read_bytes() == payload
    assert len(claims) == 1 and claims[0]["digest"] == digest
    assert not list(output.rglob("*.gltf"))


def scene() -> tuple[dict[str, Any], bytes]:
    positions = struct.pack("<9f", -1, 0, 2, 1, 0, 2, 0, 2, -1)
    normals = struct.pack("<9f", *([1, 1, 0] * 3))
    tangents = struct.pack("<12f", *([1, 0, 0, 1] * 3))
    uvs = struct.pack("<6f", 0.1, 0.2, 0.8, 0.3, 0.4, 0.9)
    colors = bytes([255, 0, 128, 64, 0, 255, 128, 192, 128, 64, 255, 255])
    payload = positions + normals + tangents + uvs + colors
    sizes = [len(value) for value in (positions, normals, tangents, uvs, colors)]
    views, offset = [], 0
    for size in sizes:
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": size})
        offset += size
    accessors = [
        {"bufferView": index, "componentType": 5126, "count": 3, "type": kind}
        for index, kind in enumerate(("VEC3", "VEC3", "VEC4", "VEC2", "VEC4"))
    ]
    accessors[4].update(componentType=5121, normalized=True)
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(payload), "uri": "geometry.bin"}],
        "bufferViews": views,
        "accessors": accessors,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TANGENT": 2,
                            "TEXCOORD_0": 3,
                            "COLOR_0": 4,
                        },
                        "material": 0,
                    }
                ]
            }
        ],
        "nodes": [{"translation": [3, 5, 7], "children": [1]}, {"mesh": 0, "scale": [-2, 3, 1]}],
        "scenes": [{"nodes": [0]}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.2, 0.4, 0.6, 0.8],
                    "metallicFactor": 0.3,
                    "roughnessFactor": 0.7,
                },
                "doubleSided": True,
                "extensions": {"KHR_materials_unlit": {}},
            }
        ],
    }
    return gltf, payload


def embedded_scene() -> bytes:
    gltf, payload = scene()
    gltf["buffers"][0]["uri"] = (
        "data:application/octet-stream;base64," + base64.b64encode(payload).decode()
    )
    return json.dumps(gltf).encode()


def local_file(
    tmp_path: Path, data: bytes, format_id: str, resources: dict[str, bytes] | None = None
) -> dict[str, Any]:
    mounts = MountTable()
    mounts.add(MountDef(id="mesh-inputs", path=tmp_path), kind="model/mesh")
    source = tmp_path / ("model." + format_id)
    source.write_bytes(data)
    refs = {}
    for name, value in (resources or {}).items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        refs[name] = mounts.bind_name(path.name, "model/mesh")
    return {
        "format": format_id,
        "source": mounts.bind_name(source.name, "model/mesh"),
        "resources": refs,
    }


def assert_scene(value: object) -> None:
    mesh = materialize_mesh(value)
    torch.testing.assert_close(
        mesh.vertices,
        torch.tensor([[[5, 5, 9], [1, 5, 9], [3, 11, 6]]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(mesh.faces, torch.tensor([[[2, 1, 0]]]), rtol=0, atol=0)
    expected_normal = torch.tensor([-0.5, 1 / 3, 0.0])
    expected_normal /= expected_normal.norm()
    torch.testing.assert_close(mesh.normals, expected_normal.expand(1, 3, 3), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        mesh.tangents, torch.tensor([[[-1, 0, 0, -1]]] * 3).transpose(0, 1).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        mesh.uvs, torch.tensor([[[0.1, 0.2], [0.8, 0.3], [0.4, 0.9]]]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        mesh.vertex_colors,
        torch.tensor([[[255, 0, 128, 64], [0, 255, 128, 192], [128, 64, 255, 255]]]).float() / 255,
        rtol=0,
        atol=0,
    )
    assert mesh.unlit and mesh.material == {
        "base_color_factor": [0.2, 0.4, 0.6, 0.8],
        "metallic_factor": 0.3,
        "roughness_factor": 0.7,
        "double_sided": True,
    }


@pytest.mark.parametrize("format_id", ["gltf", "glb", "embedded"])
def test_transformed_scene_material_and_attributes(tmp_path: Path, format_id: str) -> None:
    gltf, payload = scene()
    if format_id == "embedded":
        data, resources = embedded_scene(), {}
        format_id = "gltf"
    elif format_id == "glb":
        del gltf["buffers"][0]["uri"]
        encoded = json.dumps(gltf).encode()
        encoded += b" " * (-len(encoded) % 4)
        payload += b"\x00" * (-len(payload) % 4)
        data = struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(payload))
        data += struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
        data += struct.pack("<II", len(payload), 0x004E4942) + payload
        resources = {}
    else:
        data, resources = json.dumps(gltf).encode(), {"geometry.bin": payload}
    source = local_file(tmp_path, data, format_id, resources)
    assert_scene(Get3DComponents.execute(model_3d=source)["mesh"])
    assert source["source"].read_bytes() == data


def test_merge_missing_attributes_and_first_material(tmp_path: Path) -> None:
    gltf, payload = scene()
    second = {"attributes": {"POSITION": 0}, "material": 1}
    gltf["meshes"][0]["primitives"].append(second)
    gltf["materials"].append({"pbrMetallicRoughness": {"metallicFactor": 0.99}})
    source = local_file(tmp_path, json.dumps(gltf).encode(), "gltf", {"geometry.bin": payload})
    mesh = Get3DComponents.execute(model_3d=source)["mesh"]
    torch.testing.assert_close(mesh.faces, torch.tensor([[[2, 1, 0], [5, 4, 3]]]), rtol=0, atol=0)
    assert mesh.normals is None and mesh.tangents is None
    torch.testing.assert_close(mesh.uvs[:, 3:], torch.zeros(1, 3, 2), rtol=0, atol=0)
    torch.testing.assert_close(mesh.vertex_colors[:, 3:], torch.ones(1, 3, 4), rtol=0, atol=0)
    assert mesh.material["metallic_factor"] == 0.3


def test_strided_sparse_positions_and_instance_transforms(tmp_path: Path) -> None:
    gltf, payload = scene()
    interleaved = struct.pack("<12f", -1, 0, 2, 91, 1, 0, 2, 92, 0, 2, -1, 93)
    payload = interleaved + payload[36:]
    gltf["bufferViews"][0].update(byteLength=48, byteStride=16)
    for view in gltf["bufferViews"][1:]:
        view["byteOffset"] += 12
    index_offset = len(payload)
    payload += b"\x01\x00\x00\x00" + struct.pack("<3f", 4, 1, -2)
    gltf["bufferViews"].extend(
        [
            {"buffer": 0, "byteOffset": index_offset, "byteLength": 1},
            {"buffer": 0, "byteOffset": index_offset + 4, "byteLength": 12},
        ]
    )
    gltf["accessors"][0]["sparse"] = {
        "count": 1,
        "indices": {"bufferView": 5, "componentType": 5121},
        "values": {"bufferView": 6},
    }
    instance_offset = len(payload)
    payload += struct.pack("<6f", 0, 0, 0, 1, -1, 2)
    gltf["bufferViews"].append({"buffer": 0, "byteOffset": instance_offset, "byteLength": 24})
    gltf["accessors"].append({"bufferView": 7, "componentType": 5126, "count": 2, "type": "VEC3"})
    gltf["nodes"][1]["extensions"] = {"EXT_mesh_gpu_instancing": {"attributes": {"TRANSLATION": 5}}}
    gltf["extensionsRequired"] = ["EXT_mesh_gpu_instancing"]
    gltf["buffers"][0]["byteLength"] = len(payload)
    source = local_file(tmp_path, json.dumps(gltf).encode(), "gltf", {"geometry.bin": payload})
    mesh = Get3DComponents.execute(model_3d=source)["mesh"]
    torch.testing.assert_close(
        mesh.vertices,
        torch.tensor(
            [[[5, 5, 9], [-5, 8, 5], [3, 11, 6], [3, 2, 11], [-7, 5, 7], [1, 8, 8]]],
            dtype=torch.float32,
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(mesh.faces, torch.tensor([[[2, 1, 0], [5, 4, 3]]]), rtol=0, atol=0)


def test_external_texture_maps_and_material_factors(tmp_path: Path) -> None:
    gltf, payload = scene()
    image_path = tmp_path / "texture.png"
    Image.new("RGBA", (2, 1), (51, 102, 204, 17)).save(image_path)
    gltf["images"] = [{"uri": "textures/base%20color.png"}]
    gltf["textures"] = [{"source": 0}]
    material = gltf["materials"][0]
    material["pbrMetallicRoughness"].update(
        baseColorTexture={"index": 0}, metallicRoughnessTexture={"index": 0}
    )
    material.update(
        normalTexture={"index": 0, "scale": 0.6},
        emissiveTexture={"index": 0},
        emissiveFactor=[0.1, 0.3, 0.5],
        occlusionTexture={"index": 0, "strength": 0.4},
    )
    material["extensions"]["KHR_materials_emissive_strength"] = {"emissiveStrength": 2.5}
    source = local_file(
        tmp_path,
        json.dumps(gltf).encode(),
        "gltf",
        {
            "geometry.bin": payload,
            "textures/base color.png": image_path.read_bytes(),
        },
    )
    mesh = Get3DComponents.execute(model_3d=source)["mesh"]
    expected = torch.tensor([[[[0.2, 0.4, 0.8], [0.2, 0.4, 0.8]]]])
    for value in (mesh.texture, mesh.metallic_roughness, mesh.normal_map, mesh.emissive):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert mesh.occlusion_in_mr and mesh.unlit
    assert mesh.material["normal_scale"] == 0.6
    assert mesh.material["occlusion_strength"] == 0.4
    assert mesh.material["emissive_factor"] == [0.1, 0.3, 0.5]
    assert mesh.material["emissive_strength"] == 2.5


@pytest.mark.parametrize(
    "uri", ["geometry.bin", "../geometry.bin", "/geometry.bin", "https://example.invalid/model.bin"]
)
def test_gltf_cannot_read_undeclared_resources(tmp_path: Path, uri: str) -> None:
    gltf, payload = scene()
    gltf["buffers"][0]["uri"] = uri
    source = local_file(tmp_path, json.dumps(gltf).encode(), "gltf")
    (tmp_path / "geometry.bin").write_bytes(payload)
    with pytest.raises(ValueError, match="not granted|URI scheme"):
        Get3DComponents.execute(model_3d=source)


def test_obj_and_ascii_binary_stl_geometry(tmp_path: Path) -> None:
    obj = b"v 0 0 0\nv 2 0 0\nv 0 3 1\nvt .2 .8\nvt .7 .3\nf -3/1 -2/2 -1/1\n"
    mesh = Get3DComponents.execute(model_3d=local_file(tmp_path, obj, "obj"))["mesh"]
    expected = torch.tensor([[[0, 0, 0], [2, 0, 0], [0, 3, 1]]], dtype=torch.float32)
    torch.testing.assert_close(mesh.vertices, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        mesh.uvs, torch.tensor([[[0.2, 0.2], [0.7, 0.7], [0.2, 0.2]]]), rtol=0, atol=1e-7
    )
    ascii_stl = (
        b"solid mesh\nfacet normal 0 0 2\nouter loop\nvertex 0 0 0\n"
        b"vertex 2 0 0\nvertex 0 3 1\nendloop\nendfacet\nendsolid"
    )
    binary = b"\x00" * 80 + struct.pack("<I12fH", 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 1, 0)
    for data in (ascii_stl, binary):
        mesh = Get3DComponents.execute(model_3d=local_file(tmp_path, data, "stl"))["mesh"]
        torch.testing.assert_close(mesh.vertices, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            mesh.normals, torch.tensor([[[0, 0, 1]]] * 3).transpose(0, 1).float(), rtol=0, atol=0
        )

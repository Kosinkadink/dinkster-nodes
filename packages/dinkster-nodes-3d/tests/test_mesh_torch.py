"""CPU geometry consumers preserve complete portable mesh and voxel values."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any

import pytest
import torch
from dinkster_3d.mesh import GetMeshInfo, MergeMeshes, RotateMesh
from dinkster_3d.voxel_mesh import VoxelToMesh, VoxelToMeshBasic
from dinkster_api.v1 import (
    MESH_TYPE,
    VOXEL_TYPE,
    TriangleMeshBatch,
    TypeRegistry,
    materialize_mesh,
    materialize_voxel,
    register_inference_types,
)


def mesh() -> TriangleMeshBatch[Any]:
    return TriangleMeshBatch(
        torch.tensor([[[1.0, 2.0, 3.0], [-2.0, 5.0, 7.0]]], dtype=torch.float64),
        torch.tensor([[[0, 1, 0]]]),
        normals=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64),
        tangents=torch.tensor([[[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 1.0]]], dtype=torch.float64),
        texture=torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3),
        material={"roughnessFactor": 0.3, "baseColorFactor": [0.2, 0.4, 0.6, 1.0]},
        unlit=True,
        occlusion_in_mr=True,
    )


@pytest.mark.parametrize("portable", [False, True])
def test_rotation_preserves_materials_normals_tangent_handedness_and_source(portable: bool) -> None:
    source = mesh()
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(MESH_TYPE)
    value = spec.decode(spec.encode(source)) if portable else source
    result = RotateMesh.execute(
        mesh=value,
        mode="euler_xyz",
        **{"mode.angle_x": 0.0, "mode.angle_y": 0.0, "mode.angle_z": 90.0},
    )["mesh"]
    assert isinstance(result, TriangleMeshBatch)
    expected = torch.tensor([[[-2.0, 1.0, 3.0], [-5.0, -2.0, 7.0]]], dtype=torch.float64)
    torch.testing.assert_close(result.vertices, expected, rtol=0, atol=1e-15)
    torch.testing.assert_close(
        result.normals,
        torch.tensor([[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]], dtype=torch.float64),
        rtol=0,
        atol=1e-15,
    )
    torch.testing.assert_close(result.tangents[..., 3], source.tangents[..., 3], rtol=0, atol=0)
    torch.testing.assert_close(source.vertices, mesh().vertices, rtol=0, atol=0)
    assert result.material == source.material
    assert result.unlit and result.occlusion_in_mr
    torch.testing.assert_close(result.texture, source.texture, rtol=0, atol=0)
    quaternion = RotateMesh.execute(
        mesh=value,
        mode="quaternion",
        **{"mode.qw": 3.0, "mode.qx": 0.0, "mode.qy": 0.0, "mode.qz": 3.0},
    )["mesh"]
    assert isinstance(quaternion, TriangleMeshBatch)
    torch.testing.assert_close(quaternion.vertices, expected, rtol=0, atol=4e-15)
    assert materialize_mesh(source) is source
    assert (
        RotateMesh.execute(
            mesh=source,
            mode="euler_xyz",
            **dict.fromkeys(("mode.angle_x", "mode.angle_y", "mode.angle_z"), 0.0),
        )["mesh"]
        is source
    )
    with pytest.raises(ValueError, match="zero magnitude"):
        RotateMesh.execute(
            mesh=source,
            mode="quaternion",
            **dict.fromkeys(("mode.qw", "mode.qx", "mode.qy", "mode.qz"), 0.0),
        )


def test_materialization_retains_every_mesh_field_and_voxel_layout() -> None:
    source = mesh()
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(MESH_TYPE)
    restored = materialize_mesh(spec.decode(spec.encode(source)))
    for field in fields(source):
        left, right = getattr(restored, field.name), getattr(source, field.name)
        if isinstance(right, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert left == right
    voxel = {
        "data": torch.tensor([[1, 3, 7], [9, 5, 2]], dtype=torch.int32),
        "voxel_colors": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "resolution": 64,
    }
    live = materialize_voxel(voxel)
    assert live.data is voxel["data"] and live.voxel_colors is voxel["voxel_colors"]
    spec = registry.spec(VOXEL_TYPE)
    restored_voxel = materialize_voxel(spec.decode(spec.encode(voxel)))
    torch.testing.assert_close(restored_voxel.data, voxel["data"], rtol=0, atol=0)
    torch.testing.assert_close(restored_voxel.voxel_colors, voxel["voxel_colors"], rtol=0, atol=0)
    assert restored_voxel.resolution == 64


def test_mesh_info_uses_real_counts_not_padding() -> None:
    value = TriangleMeshBatch(
        torch.zeros(2, 10000, 3),
        torch.zeros(2, 9, 3, dtype=torch.int64),
        vertex_counts=torch.tensor([10000, 2]),
        face_counts=torch.tensor([9, 1]),
        texture=torch.zeros(2, 3, 5, 3),
    )
    result = GetMeshInfo.execute(mesh=value)
    assert result["MESH"] is value
    assert result["STRING"] == (
        "Batch:      2 meshes\nVertices:   10,002 (10.0K) total\nFaces:      10 total\n"
        "  [0]      10,000 verts  \u00b7           9 faces\n"
        "  [1]           2 verts  \u00b7           1 faces\nAttributes: texture 3\u00d75"
    )
    assert math.isfinite(value.vertices.sum().item())


@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize("attributes", [False, True])
def test_merge_first_items_excludes_padding_offsets_faces_and_fills_attributes(
    portable: bool, attributes: bool
) -> None:
    first = TriangleMeshBatch(
        torch.arange(24, dtype=torch.float64).reshape(2, 4, 3),
        torch.tensor([[[2, 0, 1], [99, 99, 99]], [[3, 2, 1], [1, 0, 2]]]),
        vertex_counts=torch.tensor([3, 4]),
        face_counts=torch.tensor([1, 2]),
        vertex_colors=torch.full((2, 4, 3), 0.25) if attributes else None,
        normals=torch.ones(2, 4, 3),
        material={"roughnessFactor": 0.4},
    )
    second = TriangleMeshBatch(
        torch.tensor([[[21.0, 22.0, 23.0], [31.0, 32.0, 33.0]]]),
        torch.tensor([[[1, 0, 1]]]),
        uvs=torch.tensor([[[0.2, 0.3], [0.7, 0.8]]]) if attributes else None,
        vertex_colors=torch.full((1, 2, 4), 0.75) if attributes else None,
        texture=torch.full((1, 2, 3, 3), 0.5) if attributes else None,
    )
    third = TriangleMeshBatch(
        torch.tensor([[[41.0, 42.0, 43.0]]]),
        torch.tensor([[[0, 0, 0]]]),
        texture=torch.ones(1, 1, 1, 3) if attributes else None,
    )
    sources = (first, second, third)
    snapshots = [(value.vertices.clone(), value.faces.clone()) for value in sources]
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(MESH_TYPE)
    values = [spec.decode(spec.encode(value)) if portable else value for value in sources]
    result = materialize_mesh(
        MergeMeshes.execute(meshes=dict(zip(("mesh10", "mesh2", "mesh7"), values, strict=True)))[
            "mesh"
        ]
    )
    torch.testing.assert_close(
        result.vertices,
        torch.tensor(
            [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [21, 22, 23], [31, 32, 33], [41, 42, 43]]],
            dtype=torch.float64,
        ),
        rtol=0,
        atol=0,
    )
    assert result.faces.tolist() == [[[2, 0, 1], [4, 3, 4], [5, 5, 5]]]
    assert result.vertices.device.type == "cpu" and result.faces.device.type == "cpu"
    assert result.vertex_counts is None and result.face_counts is None
    assert result.normals is None and result.material is None and not result.unlit
    if attributes:
        torch.testing.assert_close(
            result.uvs,
            torch.tensor([[[0, 0], [0, 0], [0, 0], [0.2, 0.3], [0.7, 0.8], [0, 0]]]).double(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.vertex_colors,
            torch.tensor([[[0.25, 0.25, 0.25, 1]] * 3 + [[0.75] * 4] * 2 + [[1] * 4]]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(result.texture, torch.full((1, 2, 3, 3), 0.5), rtol=0, atol=0)
    else:
        assert result.uvs is None and result.vertex_colors is None and result.texture is None
    for value, (vertices, faces) in zip(sources, snapshots, strict=True):
        torch.testing.assert_close(value.vertices, vertices, rtol=0, atol=0)
        torch.testing.assert_close(value.faces, faces, rtol=0, atol=0)
    with pytest.raises(ValueError, match="MergeMeshes: need at least one mesh"):
        MergeMeshes.execute(meshes={})


@pytest.mark.parametrize("portable", [False, True])
def test_voxel_basic_threshold_orientation_and_padded_batch(portable: bool) -> None:
    data = torch.zeros(2, 2, 3, 4)
    data[0, 0, 1, 2] = 0.6
    data[1, 0, 1, 2] = 0.7
    source = data.clone()
    voxel = {"data": data}
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(VOXEL_TYPE)
    value = spec.decode(spec.encode(voxel)) if portable else voxel
    result = VoxelToMeshBasic.execute(voxel=value, threshold=0.6)["MESH"]
    assert isinstance(result, TriangleMeshBatch)
    assert result.vertices.shape == (2, 24, 3)
    assert result.faces.shape == (2, 12, 3)
    assert result.vertex_counts.tolist() == [1, 24]
    assert result.face_counts.tolist() == [1, 12]
    torch.testing.assert_close(
        result.vertices[0, 0], torch.tensor([-1.0, -1.0, -1.0]), rtol=0, atol=0
    )
    # First exposed face is +x in source z/y/x coordinates, then converted to x/y/z.
    torch.testing.assert_close(
        result.vertices[1, :4],
        torch.tensor(
            [
                [0.5, -0.5, -1.0],
                [0.5, 0.0, -1.0],
                [0.5, 0.0, -0.5],
                [0.5, -0.5, -0.5],
            ]
        ),
        rtol=0,
        atol=0,
    )
    assert result.faces[1, :2].tolist() == [[0, 1, 2], [0, 2, 3]]
    assert result.faces.dtype == torch.float32  # Empty first row determines packed dtype in F.
    torch.testing.assert_close(data, source, rtol=0, atol=0)
    regular = VoxelToMesh.execute(
        voxel={"data": data[1:].repeat(2, 1, 1, 1)}, algorithm="basic", threshold=0.6
    )["MESH"]
    assert isinstance(regular, TriangleMeshBatch)
    assert regular.vertex_counts is None and regular.face_counts is None
    assert regular.faces.dtype == torch.int64


def test_surface_net_small_and_empty_grids_do_not_depend_on_progress_divisors() -> None:
    data = torch.zeros(1, 2, 2, 2)
    data[0, 0, 0, 0] = 1.0
    result = VoxelToMesh.execute(voxel={"data": data}, algorithm="surface net", threshold=0.5)[
        "MESH"
    ]
    assert isinstance(result, TriangleMeshBatch)
    assert result.vertices.shape == (1, 8, 3)
    assert result.faces.shape == (1, 12, 3)
    assert result.faces.dtype == torch.int64
    assert torch.isfinite(result.vertices).all()
    empty = VoxelToMesh.execute(
        voxel={"data": torch.zeros_like(data)}, algorithm="surface net", threshold=0.5
    )["MESH"]
    assert isinstance(empty, TriangleMeshBatch)
    assert empty.vertices.shape == (1, 0, 3) and empty.faces.shape == (1, 0, 3)

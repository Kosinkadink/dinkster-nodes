"""Mesh re-extraction retains field choices, colors and ragged batch boundaries."""

from typing import Any

import pytest
import torch
from dinkster_3d.mesh_remesh import RemeshMesh
from dinkster_api.v1 import (
    TriangleMeshBatch,
    materialize_mesh,
)


def source(*, open_top: bool = False) -> Any:
    vertices = torch.tensor(
        [[x * 0.31, y * 0.22, z * 0.38] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)],
        dtype=torch.float64,
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [1, 2, 3],
            [0, 1, 4],
            [1, 5, 4],
            [2, 6, 3],
            [3, 6, 7],
            [0, 4, 2],
            [2, 4, 6],
            [1, 3, 5],
            [3, 7, 5],
            [4, 5, 6],
            [5, 7, 6],
        ],
        dtype=torch.int32,
    )
    if open_top:
        faces = faces[:-2]
    return TriangleMeshBatch(
        vertices=vertices[None],
        faces=faces[None],
        vertex_colors=(vertices + 0.5)[None].float(),
        uvs=torch.zeros(1, 8, 2),
        normals=torch.ones(1, 8, 3),
        texture=torch.full((1, 2, 3, 3), 0.3),
        material={"doubleSided": True},
    )


def options() -> dict[str, Any]:
    return dict(
        resolution=32,
        band=1.0,
        project_back=0.0,
        fix_poles=False,
        smooth_iters=0,
        drop_small_components=0.0,
        precluster_max_verts=0,
    )


@pytest.mark.parametrize(
    "mode,controls",
    [
        ("udf", {}),
        ("sdf", {}),
        ("sdf", {"sign_mode.manifold": True}),
        ("solid", {"sign_mode.fill": 0.0}),
    ],
)
def test_fields_retain_colors_and_leave_input_unchanged(
    mode: str, controls: dict[str, Any]
) -> None:
    mesh = source()
    saved = mesh.vertices.clone()
    result = materialize_mesh(
        RemeshMesh.execute(mesh=mesh, sign_mode=mode, **options(), **controls)["mesh"]
    )
    assert result.vertices.dtype == torch.float32 and result.faces.dtype == torch.int32
    assert result.vertices.device.type == "cpu"
    assert result.face_counts[0] > 12 and result.vertex_counts[0] > 8
    assert torch.isfinite(result.vertices).all() and torch.isfinite(result.vertex_colors).all()
    assert int(result.faces.max()) < int(result.vertex_counts[0])
    assert result.vertex_colors.amin() >= 0 and result.vertex_colors.amax() <= 1
    assert result.material == mesh.material
    torch.testing.assert_close(result.texture, mesh.texture, rtol=0, atol=0)
    assert result.uvs is None and result.normals is None and result.tangents is None
    torch.testing.assert_close(mesh.vertices, saved, rtol=0, atol=0)
    assert mesh.uvs is not None and mesh.normals is not None


def test_projected_vertices_retain_interpolated_colors() -> None:
    args = options()
    args["project_back"] = 1.0
    result = materialize_mesh(RemeshMesh.execute(mesh=source(), sign_mode="udf", **args)["mesh"])
    torch.testing.assert_close(result.vertex_colors, result.vertices + 0.5, rtol=0, atol=1e-6)
    distance_to_planes = (result.vertices.abs() - torch.tensor([0.31, 0.22, 0.38])).abs()
    assert float(distance_to_planes.amin(-1).amax()) < 1e-6


def test_preclustering_averages_colors_and_removes_duplicate_and_degenerate_faces() -> None:
    mesh = source()
    expected = materialize_mesh(RemeshMesh.execute(mesh=mesh, sign_mode="sdf", **options())["mesh"])
    duplicated = source()
    duplicated.vertices = mesh.vertices.repeat_interleave(2, dim=1)
    duplicated.vertex_colors = mesh.vertex_colors.repeat_interleave(2, dim=1)
    duplicated.vertex_colors[:, ::2] -= 0.125
    duplicated.vertex_colors[:, 1::2] += 0.125
    duplicated.faces = torch.cat(
        [mesh.faces * 2, mesh.faces * 2 + 1, torch.tensor([[[0, 1, 2]]], dtype=torch.int32)],
        dim=1,
    )
    duplicated.uvs = duplicated.normals = None
    args = options()
    args["precluster_max_verts"] = 4
    result = materialize_mesh(RemeshMesh.execute(mesh=duplicated, sign_mode="sdf", **args)["mesh"])
    torch.testing.assert_close(result.vertices, expected.vertices, rtol=0, atol=0)
    torch.testing.assert_close(result.faces, expected.faces, rtol=0, atol=0)
    torch.testing.assert_close(result.vertex_colors, expected.vertex_colors, rtol=0, atol=1e-6)


def test_solid_fill_changes_openings_and_ragged_batches() -> None:
    mesh = source(open_top=True)
    filled = materialize_mesh(
        RemeshMesh.execute(mesh=mesh, sign_mode="solid", **options(), **{"sign_mode.fill": 0.0})[
            "mesh"
        ]
    )
    open_mesh = materialize_mesh(
        RemeshMesh.execute(mesh=mesh, sign_mode="solid", **options(), **{"sign_mode.fill": 1000.0})[
            "mesh"
        ]
    )
    assert filled.face_counts.tolist() != open_mesh.face_counts.tolist()
    full = source()
    full.vertices = full.vertices.repeat(2, 1, 1)
    full.faces = full.faces.repeat(2, 1, 1)
    full.vertex_colors = full.vertex_colors.repeat(2, 1, 1)
    full.vertex_counts = torch.tensor([8, 8])
    full.face_counts = torch.tensor([12, 10])
    full.texture = full.uvs = full.normals = None
    batched = materialize_mesh(
        RemeshMesh.execute(mesh=full, sign_mode="solid", **options(), **{"sign_mode.fill": 0.0})[
            "mesh"
        ]
    )
    nv, nf = int(batched.vertex_counts[1]), int(batched.face_counts[1])
    torch.testing.assert_close(batched.vertices[1, :nv], filled.vertices[0], rtol=0, atol=0)
    torch.testing.assert_close(batched.faces[1, :nf], filled.faces[0], rtol=0, atol=0)
    assert (batched.vertices[1, nv:] == 0).all() and (batched.faces[1, nf:] == 0).all()

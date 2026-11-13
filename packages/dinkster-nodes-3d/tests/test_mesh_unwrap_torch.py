"""UV charts retain seam attributes and exclude padded and degenerate triangles."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_3d.mesh_unwrap import UnwrapMesh
from dinkster_api.v1 import (
    MESH_TYPE,
    TriangleMeshBatch,
    TypeRegistry,
    materialize_mesh,
    register_inference_types,
)


def source() -> Any:
    return TriangleMeshBatch(
        vertices=torch.tensor(
            [
                [
                    [0, 0, 0],
                    [2, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 3],
                    [-1, 0, 0],
                    [4, 0, 0],
                    [4, 1, 0],
                ],
                [[5, 0, 0], [7, 0, 0], [5, 3, 0]] + [[999, 999, 999]] * 5,
            ],
            dtype=torch.float64,
        ),
        faces=torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 6, 7]], [[0, 1, 2], [7, 7, 7], [7, 7, 7]]]),
        vertex_counts=torch.tensor([8, 3]),
        face_counts=torch.tensor([3, 1]),
        vertex_colors=torch.arange(64, dtype=torch.float32).reshape(2, 8, 4) / 64,
        texture=torch.full((2, 2, 3, 3), 0.3),
    )


def assert_unwrapped(value: object) -> None:
    mesh = materialize_mesh(value)
    original = source()
    assert mesh.vertices.dtype == mesh.uvs.dtype == torch.float32
    assert mesh.faces.dtype == torch.int32
    assert mesh.vertices.device.type == "cpu"
    assert mesh.vertex_counts.tolist() == [6, 3]
    assert mesh.face_counts.tolist() == [2, 1]
    for item, (nv, nf) in enumerate(zip([6, 3], [2, 1], strict=True)):
        vertices, faces = mesh.vertices[item, :nv], mesh.faces[item, :nf].long()
        expected = original.vertices[item, original.faces[item, :nf]].float()
        torch.testing.assert_close(vertices[faces], expected, rtol=0, atol=0)
        uv = mesh.uvs[item, :nv]
        assert torch.isfinite(uv).all() and uv.min() >= 0 and uv.max() <= 1
        triangles = uv[faces]
        a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        assert ((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).abs() > 1e-4).all()
        for index, position in enumerate(vertices):
            matches = (original.vertices[item].float() == position).all(1).nonzero().flatten()
            # Quantized duplicates select the last original vertex's attributes.
            torch.testing.assert_close(
                mesh.vertex_colors[item, index],
                original.vertex_colors[item, matches[-1]],
                rtol=0,
                atol=0,
            )
    torch.testing.assert_close(mesh.texture, original.texture, rtol=0, atol=0)


@pytest.mark.parametrize("segmenter,portable", [("pec", False), ("adaptive", True)])
def test_seams_weld_mapping_and_variable_batches(segmenter: str, portable: bool) -> None:
    mesh = source()
    original = mesh.vertices.clone()
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(MESH_TYPE)
    value = spec.decode(spec.encode(mesh)) if portable else mesh
    result = UnwrapMesh.execute(
        mesh=value, segmenter=segmenter, resolution=64, padding=2, weld_distance=0.0
    )
    assert_unwrapped(result["mesh"])
    torch.testing.assert_close(mesh.vertices, original, rtol=0, atol=0)
    assert mesh.uvs is None

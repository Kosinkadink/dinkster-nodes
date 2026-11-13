"""Atlas pixels distinguish chart connectivity, boundary edges and tile seams."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_3d.uv_atlas import RenderUVAtlas
from dinkster_api.v1 import (
    MESH_TYPE,
    TriangleMeshBatch,
    TypeRegistry,
    materialize_image,
    register_inference_types,
)


def source() -> Any:
    # Two triangles share an edge; the third is a separate chart across tile 64.
    uvs = [
        [0.05, 0.05],
        [0.45, 0.05],
        [0.05, 0.45],
        [0.45, 0.45],
        [0.55, 0.55],
        [0.95, 0.55],
        [0.55, 0.95],
    ]
    return TriangleMeshBatch(
        vertices=torch.zeros(2, 7, 3),
        faces=torch.tensor([[[0, 1, 2], [1, 3, 2], [4, 5, 6]], [[0, 0, 0]] * 3]),
        uvs=torch.tensor([uvs, [[0, 0]] * 7], dtype=torch.float64),
        vertex_counts=torch.tensor([7, 0]),
        face_counts=torch.tensor([3, 0]),
    )


def assert_atlas(value: object) -> None:
    image = materialize_image(value)
    assert image.shape == (1, 128, 128, 3)
    assert image.dtype == torch.float32 and image.device.type == "cpu"
    blue = torch.tensor([0.121568627, 0.466666667, 0.705882353])
    pale_blue = torch.tensor([0.682352941, 0.780392157, 0.909803922])
    # Seed 42 permutes two charts to [1, 0]; shared-edge triangles keep one color.
    for y, x in ((15, 15), (45, 45), (31, 31)):
        torch.testing.assert_close(image[0, y, x], pale_blue, rtol=0, atol=0)
    torch.testing.assert_close(image[0, 80, 80], blue, rtol=0, atol=0)
    torch.testing.assert_close(image[0, 64, 64], torch.full((3,), 0.13), rtol=0, atol=0)
    torch.testing.assert_close(image[0, 6, 20], torch.zeros(3), rtol=0, atol=0)


@pytest.mark.parametrize("portable", [False, True])
def test_charts_boundaries_and_first_batch(portable: bool) -> None:
    mesh = source()
    before = mesh.uvs.clone()
    registry = TypeRegistry()
    register_inference_types(registry)
    codec = registry.spec(MESH_TYPE)
    value = codec.decode(codec.encode(mesh)) if portable else mesh
    assert_atlas(RenderUVAtlas.execute(mesh=value, resolution=128)["image"])
    torch.testing.assert_close(mesh.uvs, before, rtol=0, atol=0)


def test_unbatched_uvs_empty_faces_degenerate_and_missing_uvs() -> None:
    mesh = source()
    mesh.uvs = mesh.uvs[0]
    assert_atlas(RenderUVAtlas.execute(mesh=mesh, resolution=128)["image"])
    mesh.faces = torch.zeros(1, 0, 3, dtype=torch.int64)
    result = RenderUVAtlas.execute(mesh=mesh, resolution=64)["image"]
    torch.testing.assert_close(result, torch.full((1, 64, 64, 3), 0.13), rtol=0, atol=0)
    mesh.faces = torch.tensor([[[0, 0, 0]]])
    result = RenderUVAtlas.execute(mesh=mesh, resolution=64)["image"]
    torch.testing.assert_close(result, torch.full((1, 64, 64, 3), 0.13), rtol=0, atol=0)
    mesh.uvs = None
    with pytest.raises(RuntimeError, match="mesh has no UVs"):
        RenderUVAtlas.execute(mesh=mesh, resolution=64)


def test_clipped_uvs_and_first_overlapping_chart() -> None:
    mesh = TriangleMeshBatch(
        vertices=torch.zeros(1, 6, 3),
        faces=torch.tensor([[[0, 1, 2], [3, 4, 5]]]),
        uvs=torch.tensor([[[-1, -1], [2, -1], [-1, 2], [0, 0], [1, 0], [0, 1]]]),
    )
    image = RenderUVAtlas.execute(mesh=mesh, resolution=64)["image"]
    torch.testing.assert_close(
        image[0, 15, 15], torch.tensor([0.682352941, 0.780392157, 0.909803922]), rtol=0, atol=0
    )
    torch.testing.assert_close(image[0, 0, 20], torch.zeros(3), rtol=0, atol=0)
    torch.testing.assert_close(image[0, 50, 50], torch.full((3,), 0.13), rtol=0, atol=0)

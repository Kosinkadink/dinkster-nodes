"""Public mesh shading, counted geometry and ordinary worker transport."""

from __future__ import annotations

import pytest
import torch
from dinkster_3d.mesh_render import RenderMesh
from dinkster_api.v1 import TriangleMeshBatch


def source() -> TriangleMeshBatch:
    vertices = torch.tensor(
        [
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [100.0, 100.0, 100.0],
            ]
        ]
    )
    return TriangleMeshBatch(
        vertices,
        torch.tensor([[[0, 1, 2], [0, 2, 3], [4, 4, 4]]]),
        vertex_counts=torch.tensor([4]),
        face_counts=torch.tensor([2]),
        vertex_colors=torch.tensor([[[0.25, 0.0625, 1.0]] * 5]),
        uvs=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]),
        texture=torch.tensor([[[[0.1, 0.7, 0.3]]]]),
    )


def parameters() -> dict:
    return dict(
        mode="auto",
        width=80,
        height=64,
        background="#204080",
        camera_info={
            "position": {"x": 0.0, "y": 0.0, "z": 3.0},
            "target": {"x": 0.0, "y": 0.0, "z": 0.0},
            "zoom": 1.0,
            "fov": 40.0,
            "cameraType": "perspective",
        },
    )


@pytest.mark.parametrize("mode", ["auto", "texture", "vertex colors", "solid", "normal", "depth"])
def test_render_mesh_shading_and_counted_padding(mode: str) -> None:
    mesh = source()
    args = parameters()
    args["mode"] = mode
    result = RenderMesh.execute(mesh=mesh, **args)
    image, mask = result["IMAGE"], result["MASK"]
    assert image.shape == (1, 64, 80, 3) and mask.shape == (1, 64, 80)
    assert image.device.type == mask.device.type == "cpu"
    assert image.dtype == mask.dtype == torch.float32
    assert mask[0, 32, 40] == 1 and mask[0, 0, 0] == 0
    torch.testing.assert_close(image[0, 0, 0], torch.tensor([32, 64, 128]) / 255)
    if mode in ("auto", "texture"):
        expected = torch.tensor([0.1, 0.7, 0.3])
    elif mode == "vertex colors":
        expected = torch.tensor([0.25, 0.0625, 1.0]).pow(1.0 / 2.2)
    elif mode == "normal":
        expected = torch.tensor([0.5, 0.5, 1.0])
    elif mode == "depth":
        assert image[0, 32, 40, 0] > image[0, 32, 60, 0]
        return
    else:
        assert 0.71 < image[0, 32, 40, 0] <= 0.72
        return
    torch.testing.assert_close(image[0, 32, 40], expected)


def test_render_mesh_single_triangle() -> None:
    mesh = source()
    mesh.vertices[0, 2] = torch.tensor([0.0, 1.0, 0.0])
    mesh.vertex_counts = torch.tensor([3])
    mesh.face_counts = torch.tensor([1])
    result = RenderMesh.execute(mesh=mesh, **parameters())
    image, mask = result["IMAGE"], result["MASK"]
    assert mask[0, 32, 40] == 1 and mask[0, 0, 0] == 0
    torch.testing.assert_close(image[0, 32, 40], torch.tensor([0.1, 0.7, 0.3]))
    torch.testing.assert_close(image[0, 0, 0], torch.tensor([32, 64, 128]) / 255)


def test_render_mesh_camera_zoom_transform_and_auto_fallback() -> None:
    mesh = source()
    args = parameters()
    original = RenderMesh.execute(mesh=mesh, **args)["MASK"]
    args["camera_info"]["zoom"] = 0.5
    wide = RenderMesh.execute(mesh=mesh, **args)["MASK"]
    assert 0 < wide.sum() < original.sum()
    args["model_3d_info"] = [
        {
            "scale": {"x": 0.5, "y": 1.0, "z": 1.0},
            "position": {"x": 0.2, "y": 0.0, "z": 0.0},
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    ]
    transformed = RenderMesh.execute(mesh=mesh, **args)["MASK"]
    assert transformed.sum() < wide.sum() and not torch.equal(wide, transformed)
    mesh.texture = None
    fallback = RenderMesh.execute(mesh=mesh, **parameters())["IMAGE"]
    torch.testing.assert_close(
        fallback[0, 32, 40], torch.tensor([0.25, 0.0625, 1.0]).pow(1.0 / 2.2)
    )
    mesh.vertex_colors = None
    args = parameters()
    args["camera_info"] = None
    default = RenderMesh.execute(mesh=mesh, **args)
    assert default["MASK"].sum() > 0
    assert torch.isfinite(default["IMAGE"]).all()

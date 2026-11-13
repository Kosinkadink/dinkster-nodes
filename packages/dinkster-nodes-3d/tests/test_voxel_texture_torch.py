"""Sparse interpolation, PBR channel layout and ordinary-host voxel texture baking."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_3d.voxel_texture import BakeTextureFromVoxel
from test_mesh_bake_torch import plane


def fixture() -> tuple[Any, dict]:
    mesh = plane(extent=0.25)
    mesh.uvs = mesh.uvs * 0.5 + 0.25
    voxel = {
        "data": torch.tensor([[0, 0, 0], [1, 0, 0]]),
        "voxel_colors": torch.tensor([[0.2, 0.4, 0.6, 0.1, 0.8, 0], [0.8, 0.6, 0.4, 0.9, 0.3, 1]]),
        "resolution": 2,
    }
    return mesh, voxel


def test_voxel_bake_normalized_corners_pbr_projection_and_empty_batches() -> None:
    mesh, voxel = fixture()
    result = BakeTextureFromVoxel.execute(mesh=mesh, voxel_colors=voxel, texture_size=64)
    # Only the x-axis varies: missing y/z corners must not dilute occupied attributes.
    x_weight = torch.tensor([1.0 / 64, 63.0 / 64])
    expected = torch.lerp(voxel["voxel_colors"][0], voxel["voxel_colors"][1], x_weight[:, None])
    torch.testing.assert_close(result["base_color"][0, 32, [16, 47]], expected[:, :3])
    torch.testing.assert_close(result["metallic"][0, 32, [16, 47], 0], expected[:, 3])
    torch.testing.assert_close(result["roughness"][0, 32, [16, 47], 0], expected[:, 4])
    torch.testing.assert_close(result["base_color"][0, 32, 0], expected[0, :3])
    reference = plane(extent=0.25)
    reference.vertices[..., 0] = -0.25
    reference.vertices[0, 1:3, 2] = 0.5
    projected = BakeTextureFromVoxel.execute(
        mesh=mesh, voxel_colors=voxel, texture_size=64, reference_mesh=reference
    )
    torch.testing.assert_close(
        projected["base_color"], voxel["voxel_colors"][0, :3].expand(1, 64, 64, 3)
    )
    mesh.vertices, mesh.faces, mesh.uvs = (
        mesh.vertices.repeat(3, 1, 1),
        mesh.faces.repeat(3, 1, 1),
        mesh.uvs.repeat(3, 1, 1),
    )
    mesh.vertex_counts, mesh.face_counts = torch.tensor([4, 3, 4]), torch.tensor([2, 0, 2])
    voxel["data"] = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]])
    batched = BakeTextureFromVoxel.execute(mesh=mesh, voxel_colors=voxel, texture_size=64)
    for key, image in batched.items():
        torch.testing.assert_close(image[:1], result[key])
        assert torch.count_nonzero(image[1:]) == 0
        assert image.dtype == torch.float32 and image.device.type == "cpu"


def test_voxel_bake_nearest_fallback_rgb_and_missing_uv() -> None:
    mesh, voxel = fixture()
    mesh.vertices += 10
    voxel["voxel_colors"] = voxel["voxel_colors"][:, :3]
    result = BakeTextureFromVoxel.execute(mesh=mesh, voxel_colors=voxel, texture_size=64)
    torch.testing.assert_close(result["base_color"], voxel["voxel_colors"][1].expand(1, 64, 64, 3))
    assert not torch.count_nonzero(result["metallic"]) and not torch.count_nonzero(
        result["roughness"]
    )
    mesh.uvs = None
    with pytest.raises(ValueError, match="no UVs"):
        BakeTextureFromVoxel.execute(mesh=mesh, voxel_colors=voxel, texture_size=64)


def test_voxel_bake_large_sparse_keys_and_many_distant_texels() -> None:
    mesh, voxel = fixture()
    # Cubic keys exceed int32; more than 8192 stragglers use the spatial-tree tail.
    voxel["resolution"] = 4096
    voxel["data"] = torch.tensor([[4095, 4095, 4095], [2048, 1024, 512]])
    mesh.vertices += 10
    mesh.uvs = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    result = BakeTextureFromVoxel.execute(mesh=mesh, voxel_colors=voxel, texture_size=128)
    torch.testing.assert_close(
        result["base_color"], voxel["voxel_colors"][0, :3].expand(1, 128, 128, 3)
    )
    torch.testing.assert_close(
        result["metallic"], voxel["voxel_colors"][0, 3].expand(1, 128, 128, 3)
    )


def test_voxel_bake_oom_fallback_does_not_hide_other_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_3d import voxel_texture

    mesh, voxel = fixture()
    voxel["data"] = voxel["data"][:1]
    voxel["voxel_colors"] = voxel["voxel_colors"][:1]
    original = voxel_texture._trilinear

    def exhausted(positions: Any, *args: Any) -> Any:
        if positions.dtype == torch.float32:
            raise torch.OutOfMemoryError("injected allocation failure")
        return original(positions, *args)

    monkeypatch.setattr(voxel_texture, "_trilinear", exhausted)
    result = BakeTextureFromVoxel.execute(mesh=mesh, voxel_colors=voxel, texture_size=64)
    torch.testing.assert_close(
        result["base_color"], voxel["voxel_colors"][0, :3].expand(1, 64, 64, 3)
    )

    def invalid(*args: Any) -> Any:
        raise ValueError("invalid voxel attributes")

    monkeypatch.setattr(voxel_texture, "_trilinear", invalid)
    with pytest.raises(ValueError, match="invalid voxel attributes"):
        BakeTextureFromVoxel.execute(mesh=mesh, voxel_colors=voxel, texture_size=64)

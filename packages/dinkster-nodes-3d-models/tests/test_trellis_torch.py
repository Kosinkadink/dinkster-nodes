"""Texture-stage sparse batches and metadata survive ordinary native pack hosts."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_3d_models.trellis import Trellis2TextureStage


def stage_inputs(*, padded: bool = True) -> dict[str, Any]:
    # Hand-derived: batch zero has two tokens; batch one has one. The third
    # padded token in each batch must not become a sparse shape feature.
    coords = torch.tensor([[1, 1, 2, 3], [0, 3, 2, 1], [0, 0, 1, 2]], dtype=torch.int32)
    samples = (
        torch.arange(12).reshape(2, 2, 3, 1).float()
        if padded
        else torch.tensor([[0.0, 3.0], [1.0, 4.0], [6.0, 9.0]])
    )
    return {
        "positive": [[torch.ones(1, 2, 3), {"keep": "positive"}]],
        "negative": [[torch.zeros(1, 2, 3), {"keep": "negative"}]],
        "shape_latent": {"samples": samples, "coords": coords, "coord_resolution": 8},
    }


@pytest.mark.parametrize("padded", (False, True))
def test_texture_stage_preserves_sparse_layout_and_inputs(padded: bool) -> None:
    inputs = stage_inputs(padded=padded)
    before = inputs["shape_latent"]["samples"].clone()
    result = Trellis2TextureStage.execute(**inputs)
    latent = result["LATENT"]
    assert latent["samples"].shape == (2, 32, 2, 1)
    assert latent["samples"].dtype == torch.float32
    assert torch.count_nonzero(latent["samples"]) == 0
    assert latent["coords"] is inputs["shape_latent"]["coords"]
    assert torch.equal(latent["coord_counts"], torch.tensor([2, 1]))
    assert latent["coord_resolution"] == 8
    assert latent["model_frame"] == "z_up"
    expected = torch.tensor([[0.0, 3.0], [1.0, 4.0], [6.0, 9.0]])
    for name in ("positive", "negative"):
        entry = result[name][0]
        assert entry[0] is inputs[name][0][0]
        assert entry[1]["keep"] == name
        assert entry[1]["trellis2_generation_mode"] == "texture_generation"
        assert torch.equal(entry[1]["trellis2_shape_slat"], expected)
        assert torch.equal(entry[1]["trellis2_coord_counts"], torch.tensor([2, 1]))
        assert entry[1]["trellis2_model_frame"] == "z_up"
        assert inputs[name][0][1] == {"keep": name}
    assert torch.equal(inputs["shape_latent"]["samples"], before)


@pytest.mark.parametrize("resolution", (None, 8))
def test_texture_stage_projection_and_frame_override(resolution: int | None) -> None:
    inputs = stage_inputs()
    if resolution is None:
        del inputs["shape_latent"]["coord_resolution"]
    inputs["shape_latent"]["model_frame"] = "authored_frame"
    # Hand-derived: constant maps project the same two channel values at every
    # coordinate. A different fallback map exposes selecting the wrong stage.
    inputs["positive"][0][1]["proj_feat_pack"] = {
        "mesh_scale": torch.ones(1),
        "transform_matrix": torch.eye(4).unsqueeze(0),
        "camera_angle_x": torch.ones(1),
        "feature_map": torch.full((1, 2, 2, 2), 99.0),
        "stages": {
            "tex_1024": {
                "feature_map": torch.tensor([2.0, 7.0]).reshape(1, 2, 1, 1).expand(1, 2, 2, 2),
                "image_resolution": 4,
            }
        },
    }
    result = Trellis2TextureStage.execute(**inputs)
    assert result["LATENT"]["model_frame"] == "authored_frame"
    for name in ("positive", "negative"):
        metadata = result[name][0][1]
        assert metadata["trellis2_model_frame"] == "authored_frame"
        if resolution is None:
            assert "trellis2_proj_feats" not in metadata
            assert "coord_resolution" not in result["LATENT"]
        else:
            assert torch.equal(metadata["trellis2_proj_feats"], torch.tensor([[2.0, 7.0]] * 3))
    del inputs["shape_latent"]["model_frame"]
    assert Trellis2TextureStage.execute(**inputs)["LATENT"]["model_frame"] == "y_up"


@pytest.mark.parametrize(
    ("coords", "message"),
    (
        (torch.zeros(3, 3), "shape"),
        (torch.zeros(0, 4), "empty"),
        (torch.tensor([[-1, 0, 0, 0]]), "non-negative"),
        (torch.tensor([[0, 0, 0, 0], [2, 0, 0, 0]]), "Non-contiguous"),
    ),
)
def test_texture_stage_rejects_invalid_coord_batches(coords: Any, message: str) -> None:
    inputs = stage_inputs()
    inputs["shape_latent"]["coords"] = coords
    with pytest.raises(ValueError, match=message):
        Trellis2TextureStage.execute(**inputs)

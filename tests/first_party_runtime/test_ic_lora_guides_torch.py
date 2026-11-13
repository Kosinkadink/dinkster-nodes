"""IC-LoRA metadata and image guides through native public contracts.

Expected-value provenance: ComfyUI comfy_extras/nodes_lt.py,
GetICLoRAParameters.execute and LTXVAddGuide.execute/attach_guide_latent,
at F bc5f880dbdd43e426077f42ccec53bb577ffee9f. Metadata cases below
name exact inputs; max(1, round(float(first suffix-matching value))) gives
their hand-derived expectations. Guide inputs are a (1,128,3,4,6) zero
latent, one (8,12) RGB image filled with 0.75, VAE scales (8,2,2),
factor 2.5 -> 2, frame_idx=-1 -> 16, and strength=1.5. Hand-derived:
one encoded guide frame appends to three video frames; sparse dilation
places 0.75 at even rows/columns, zero elsewhere. The guide mask is
1-1.5=-0.5 at occupied positions and -1-1.5=-2.5 elsewhere.
Controlled VAE/Linear model fixtures do not claim real-model inference.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)
from dinkster_api.v1 import register_inference_types as register_inference_types
from dinkster_ltx.ltx_guides import GetICLoRAParameters


def model_with_metadata(metadata: Any) -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    patcher = ModelPatcher(torch.nn.Linear(1, 1), torch.device("cpu"), torch.device("cpu"))
    patcher.set_attachments("lora_metadata", metadata)
    return patcher


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (None, 1),
        ({}, 1),
        ({"unrelated": "9"}, 1),
        ({"a.reference_downscale_factor": "2.5", "reference_downscale_factor": "5"}, 2),
        ({"reference_downscale_factor": "3.5"}, 4),
        ({"reference_downscale_factor": -7}, 1),
        ({"reference_downscale_factor": "bad", "b.reference_downscale_factor": 8}, 1),
        ({"reference_downscale_factor": None}, 1),
        ({"reference_downscale_factor": "nan"}, 1),
    ],
)
def test_metadata_first_match_rounding_and_fallback(metadata: Any, expected: int) -> None:
    patcher = model_with_metadata(metadata)
    assert GetICLoRAParameters.execute(iclora_model=patcher) == {
        "iclora_parameters": {"reference_downscale_factor": expected}
    }
    assert patcher.get_attachment("lora_metadata") is metadata


class VideoVae:
    downscale_index_formula = (8, 2, 2)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels[::8, ::2, ::2, 0][None, None].repeat(1, 128, 1, 1, 1)


def check_guide(values: dict[str, Any]) -> None:
    latent = materialize_latent(values["latent"])
    assert latent["samples"].shape == (1, 128, 4, 4, 6)
    assert torch.count_nonzero(latent["samples"][:, :, :3]) == 0
    expected = torch.zeros(1, 128, 4, 6)
    expected[..., ::2, ::2] = 0.75
    assert torch.equal(latent["samples"][:, :, 3], expected)
    mask = torch.full((1, 1, 4, 6), -2.5)
    mask[..., ::2, ::2] = -0.5
    assert torch.equal(latent["noise_mask"][:, :, 3], mask)
    assert torch.equal(latent["noise_mask"][:, :, :3], torch.ones(1, 1, 3, 4, 6))
    for name in ("positive", "negative"):
        metadata = values[name][0][1]
        assert metadata["keep"] == name
        assert metadata["keyframe_idxs"].shape == (1, 3, 24, 2)
        assert metadata["keyframe_idxs"][0, 0, :, 0].tolist() == [16] * 24
        entry = metadata["guide_attention_entries"][0]
        assert entry["latent_shape"] == [1, 2, 3]
        assert entry["strength"] == 1.5


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ic-guide-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("model", MODEL_TYPE),
                    ("vae", VAE_TYPE),
                    ("latent", LATENT_TYPE),
                    ("image", IMAGE_TYPE),
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            model=model_with_metadata({"reference_downscale_factor": "2.5"}),
            vae=VideoVae(),
            latent={"samples": torch.zeros(1, 128, 3, 4, 6)},
            image=torch.full((1, 8, 12, 3), 0.75),
            positive=[[torch.ones(1), {"keep": "positive"}]],
            negative=[[torch.zeros(1), {"keep": "negative"}]],
        )


class Consumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ic-guide-fixture.consumer",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("latent", LATENT_TYPE),
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                )
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, **values: Any) -> dict[str, object]:
        check_guide(values)
        return cls.outputs(summary="sparse-guide")


HOST_NODES = (Source, Consumer)

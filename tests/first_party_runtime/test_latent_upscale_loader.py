"""Schema and loader dispatch contracts for latent-upscale assets."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_UPSCALE_MODEL_TYPE,
    AssetRef,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_assets import digest_file
from dinkster_models.latent_upscale_loader import LatentUpscaleModelLoader
from safetensors.torch import save_file


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, _digest: str) -> Path:
        return self.path


def _asset(path: Path) -> AssetRef:
    return AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=_Resolver(path))


def test_loader_rejects_unbound_non_asset_values() -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        LatentUpscaleModelLoader.execute(model_name=object())


@pytest.mark.parametrize("model_format", ("720p", "1080p", "ltx"))
def test_real_safetensors_checkpoints_load_the_expected_pinned_models(
    tmp_path: Path, model_format: str
) -> None:
    initialize_inference_process()
    from dinkster_inference import ops
    from dinkster_inference.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel
    from dinkster_inference.ldm.lightricks.latent_upsampler import LatentUpsampler
    from dinkster_inference.model_patcher import CoreModelPatcher

    if model_format == "720p":
        source = HunyuanVideo15SRModel(
            "720p",
            {
                "in_channels": 2,
                "out_channels": 2,
                "hidden_channels": 2,
                "num_blocks": 1,
                "global_residual": False,
            },
        )
        state = source.get_sd()
        metadata = None
    elif model_format == "1080p":
        source = HunyuanVideo15SRModel(
            "1080p", {"z_channels": 4, "out_channels": 4, "block_out_channels": (32,)}
        )
        state = {
            key.replace("nin_shortcut.conv", "nin_shortcut", 1): value
            for key, value in source.get_sd().items()
        }
        metadata = None
    else:
        source = LatentUpsampler.from_config(
            {
                "in_channels": 4,
                "mid_channels": 32,
                "num_blocks_per_stage": 1,
                "dims": 2,
                "spatial_upsample": True,
                "temporal_upsample": False,
            },
            operations=ops.disable_weight_init,
        )
        state = source.state_dict()
        metadata = {"config": __import__("json").dumps(source.config())}
    state = {key: torch.full_like(value, 0.125) for key, value in state.items()}
    checkpoint = tmp_path / f"{model_format}.safetensors"
    save_file(state, checkpoint, metadata=metadata)

    loaded = LatentUpscaleModelLoader.execute(model_name=_asset(checkpoint))["LATENT_UPSCALE_MODEL"]
    if model_format == "ltx":
        assert type(loaded) is CoreModelPatcher
        actual = loaded.model.state_dict()
    else:
        assert type(loaded) is HunyuanVideo15SRModel
        actual = loaded.get_sd()
    expected = (
        {key.replace("nin_shortcut", "nin_shortcut.conv", 1): value for key, value in state.items()}
        if model_format == "1080p"
        else state
    )
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


class LatentUpscaleConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="latent-upscale-fixture.consumer",
            inputs=(InputSpec("model", TypeExpr.concrete(LATENT_UPSCALE_MODEL_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model: object) -> dict[str, object]:
        model_format = os.environ["LATENT_UPSCALE_FORMAT"]
        hunyuan = importlib.import_module("dinkster_inference.ldm.hunyuan_video.upsampler")
        patcher = importlib.import_module("dinkster_inference.model_patcher")
        if model_format == "ltx":
            assert type(model) is patcher.CoreModelPatcher
            state = model.model.state_dict()
            assert next(iter(state.values())).dtype is torch.float32
            actual_type = type(model.model).__name__
        else:
            assert type(model) is hunyuan.HunyuanVideo15SRModel
            state = model.get_sd()
            assert type(model.model).__name__ == (
                "SRModel3DV2" if model_format == "720p" else "Upsampler"
            )
            actual_type = type(model).__name__
        assert state
        for value in state.values():
            torch.testing.assert_close(value, torch.full_like(value, 0.125), rtol=0, atol=0)
        return cls.outputs(summary=json.dumps({"format": model_format, "type": actual_type}))

"""CPU contracts for native latent upscale consumers."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from dinkster_api.v1 import (
    LATENT_TYPE,
    LATENT_UPSCALE_MODEL_TYPE,
    VAE_TYPE,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_hunyuan.latent_upscale import HunyuanVideo15LatentUpscaleWithModel
from dinkster_ltx.latent_upscale import LTXVLatentUpsampler

_LATENT = TypeExpr.concrete(LATENT_TYPE)
_UPSCALE_MODEL = TypeExpr.concrete(LATENT_UPSCALE_MODEL_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


def _hunyuan_model():
    initialize_inference_process()
    from dinkster_inference.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel

    model = HunyuanVideo15SRModel(
        "720p", {"in_channels": 1, "out_channels": 1, "hidden_channels": 1, "num_blocks": 0}
    )
    with torch.no_grad():
        model.model.in_conv.conv.weight.zero_()
        model.model.in_conv.conv.weight[0, 0, 2, 1, 1] = 2
        model.model.in_conv.conv.bias.fill_(3)
        model.model.out_conv.conv.weight.zero_()
        model.model.out_conv.conv.weight[0, 0, 2, 1, 1] = -0.5
        model.model.out_conv.conv.bias.fill_(7)
    return model


def test_native_runtime_models_execute_cpu_latent_upscale_contracts() -> None:
    initialize_inference_process()
    from dinkster_inference.model_patcher import CoreModelPatcher

    hunyuan = _hunyuan_model()
    assert type(hunyuan.patcher) is CoreModelPatcher
    source = {
        "samples": torch.arange(1 * 1 * 2 * 3 * 5, dtype=torch.float32).reshape(1, 1, 2, 3, 5)
    }
    result = HunyuanVideo15LatentUpscaleWithModel.execute(
        model=hunyuan,
        samples=source,
        upscale_method="nearest-exact",
        width=96,
        height=64,
        crop="disabled",
    )["LATENT"]
    assert result["samples"].shape == (1, 1, 2, 4, 6)
    assert result["samples"].dtype is torch.float32
    resized = source["samples"][..., (0, 1, 1, 2), :][..., (0, 1, 2, 2, 3, 4)]
    torch.testing.assert_close(result["samples"], 5.5 - resized, rtol=0, atol=0)

    class Statistics:
        def un_normalize(self, value: torch.Tensor) -> torch.Tensor:
            return value + 3

        def normalize(self, value: torch.Tensor) -> torch.Tensor:
            return value - 1

    ltx = CoreModelPatcher(torch.nn.Identity(), torch.device("cpu"), torch.device("cpu"))
    assert type(ltx) is CoreModelPatcher
    input_latent = {"samples": torch.tensor([[[[2.0]]]], dtype=torch.float64), "noise_mask": "drop"}
    vae = SimpleNamespace(first_stage_model=SimpleNamespace(per_channel_statistics=Statistics()))
    upsampled = LTXVLatentUpsampler.execute(samples=input_latent, upscale_model=ltx, vae=vae)[
        "LATENT"
    ]
    assert upsampled["samples"].dtype is torch.float64
    torch.testing.assert_close(upsampled["samples"], torch.tensor([[[[4.0]]]], dtype=torch.float64))
    assert "noise_mask" not in upsampled


def test_hunyuan_one_zero_dimension_uses_width_height_in_the_core_order(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    class Model:
        def resample_latent(self, latent: torch.Tensor) -> torch.Tensor:
            return latent + 3

    monkeypatch.setattr(
        "dinkster_hunyuan.latent_upscale.initialize_inference_process", lambda: None
    )
    monkeypatch.setattr(
        "dinkster_hunyuan.latent_upscale.importlib",
        SimpleNamespace(
            import_module=lambda name: SimpleNamespace(
                common_upscale=lambda tensor, width, height, _method, _crop: (
                    calls.append((width, height)) or tensor.new_ones((1, 2, height, width))
                )
            )
        ),
    )
    source = {"samples": torch.zeros((1, 2, 3, 5), dtype=torch.float64), "noise_mask": object()}
    output = HunyuanVideo15LatentUpscaleWithModel.execute(
        model=Model(), samples=source, upscale_method="bilinear", width=0, height=96, crop="center"
    )["LATENT"]
    assert calls == [(10, 6)]
    assert set(output) == {"samples"}
    assert output["samples"].shape == (1, 2, 6, 10)
    assert output["samples"].dtype is torch.float32

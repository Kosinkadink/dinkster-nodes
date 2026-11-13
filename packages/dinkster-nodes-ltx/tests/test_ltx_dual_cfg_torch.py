"""CPU tests of LTXV dual guidance with controlled engine predictions."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_ltx.ltx_dual_cfg import LTXVDualCFGGuider

initialize_inference_process()
samplers = importlib.import_module("dinkster_inference.samplers")
NestedTensor = importlib.import_module("dinkster_inference.nested_tensor").NestedTensor


def guider(video_cfg: float, audio_cfg: float):
    model = SimpleNamespace(
        model_options={"transformer_options": {"keep": 19}}, is_dynamic=lambda: False
    )
    positive = [[torch.tensor([[3.0]]), {"tag": "positive"}]]
    negative = [[torch.tensor([[-2.0]]), {"tag": "negative"}]]
    value = LTXVDualCFGGuider.execute(
        model=model,
        positive=positive,
        negative=negative,
        video_cfg=video_cfg,
        audio_cfg=audio_cfg,
    )["GUIDER"]
    value.inner_model = None
    value.conds = value.original_conds
    assert positive[0][1] == {"tag": "positive"}
    assert negative[0][1] == {"tag": "negative"}
    assert model.model_options == {"transformer_options": {"keep": 19}}
    return value


@pytest.mark.parametrize(
    "video_cfg,audio_cfg", [(2.0, 5.0), (1.0, 3.0), (4.0, 1.0), (0.0, 2.0), (1.0, 0.0)]
)
def test_dual_guidance_uses_per_sample_video_split_and_preserves_inputs(
    video_cfg: float, audio_cfg: float
) -> None:
    value = guider(video_cfg, audio_cfg)
    video = torch.zeros(2, 1, 1, 2, 3)
    audio = torch.zeros(2, 2, 1, 2)
    latent = NestedTensor((video, audio))
    assert value.sample(None, latent, None, torch.tensor([])) is latent
    x = torch.arange(20.0).reshape(2, 1, 10) + 31
    cond = torch.arange(20.0).reshape(2, 1, 10) / 2 - 4
    uncond = torch.arange(20.0).flip(0).reshape(2, 1, 10) / 4 + 2
    cond_before, uncond_before = cond.clone(), uncond.clone()

    def previous(_):
        raise AssertionError("Dual guidance must replace this callback for this prediction only")

    def predictions(args):
        assert args["conds"][0] is value.conds["positive"]
        assert args["conds"][1] is value.conds["negative"]
        return cond, uncond

    options = {
        "sampler_cfg_function": previous,
        "sampler_calc_cond_batch_function": predictions,
        "transformer_options": {"keep": 23},
    }
    output = value.predict_noise(x, torch.tensor([0.5]), options, seed=41)
    expected = uncond + (cond - uncond) * torch.tensor([video_cfg] * 6 + [audio_cfg] * 4)
    assert torch.equal(output, expected)
    assert torch.equal(cond, cond_before)
    assert torch.equal(uncond, uncond_before)
    assert options == {
        "sampler_cfg_function": previous,
        "sampler_calc_cond_batch_function": predictions,
        "transformer_options": {"keep": 23},
    }

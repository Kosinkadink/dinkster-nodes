from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_INT,
    LATENT_TYPE,
    LATENT_UPSCALE_MODEL_TYPE,
    ComboOption,
    ComboWidget,
    Input,
    Node,
    NodeSchema,
    NumberWidget,
    Output,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
    with_identity_replacement,
)

_COMBO = TypeExpr.concrete(CORE_COMBO)

_INT = TypeExpr.concrete(CORE_INT)

_LATENT = TypeExpr.concrete(LATENT_TYPE)

_LATENT_UPSCALE_MODEL = TypeExpr.concrete(LATENT_UPSCALE_MODEL_TYPE)

_UPSCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp")


def _combo(name: str, options: tuple[str, ...], default: str | None = None) -> Input:
    return Input(
        name,
        _COMBO,
        default=default,
        widget=ComboWidget(tuple(ComboOption(value) for value in options)),
    )


class HunyuanVideo15LatentUpscaleWithModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.hunyuan.hunyuan_video15_latent_upscale_with_model",
            display_name="Hunyuan Video 1.5 Latent Upscale With Model",
            category="model/latent/hunyuan video",
            body=(
                Input("model", _LATENT_UPSCALE_MODEL),
                Input("samples", _LATENT),
                _combo("upscale_method", _UPSCALE_METHODS, "bilinear"),
                Input("width", _INT, default=1280, widget=NumberWidget(min=0, max=16384, step=8)),
                Input("height", _INT, default=720, widget=NumberWidget(min=0, max=16384, step=8)),
                _combo("crop", ("disabled", "center")),
                Output("LATENT", _LATENT),
            ),
        )
        return with_identity_replacement(schema, "HunyuanVideo15LatentUpscaleWithModel")

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        samples: object,
        upscale_method: str,
        width: int,
        height: int,
        crop: str,
    ) -> Mapping[str, object]:
        if width == 0 and height == 0:
            return cls.outputs(LATENT=samples)
        source = materialize_latent(samples)
        latent = source["samples"]
        if width == 0:
            height = max(64, height)
            width = max(64, round(latent.shape[-1] * height / latent.shape[-2]))
        elif height == 0:
            width = max(64, width)
            height = max(64, round(latent.shape[-2] * width / latent.shape[-1]))
        else:
            width, height = max(64, width), max(64, height)
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        upscaled = utils.common_upscale(latent, width // 16, height // 16, upscale_method, crop)
        return cls.outputs(LATENT={"samples": model.resample_latent(upscaled).cpu().float()})

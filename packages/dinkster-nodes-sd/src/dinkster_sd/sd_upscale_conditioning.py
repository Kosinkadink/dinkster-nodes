"""Native SD 4x upscaler conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    IMAGE_TYPE,
    LATENT_TYPE,
    Input,
    Node,
    NodeSchema,
    NumberWidget,
    Output,
    Section,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    with_identity_replacement,
)
from dinkster_pack_helpers.conditioning import conditioning_set_values

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)

_FLOAT = TypeExpr.concrete(CORE_FLOAT)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)

_LATENT = TypeExpr.concrete(LATENT_TYPE)


def _float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
    step: float,
) -> Input:
    return Input(
        name,
        _FLOAT,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


class SD_4XUpscale_Conditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.sd.sd_4_x_upscale_conditioning",
            display_name="SD_4XUpscale_Conditioning",
            category="model/conditioning/stable diffusion upscaler",
            body=(
                Input("images", _IMAGE),
                Input("positive", _CONDITIONING),
                Input("negative", _CONDITIONING),
                _float("scale_ratio", 4.0, 0.0, 10.0, 0.01),
                Section(
                    "advanced",
                    label="Advanced",
                    collapsed=True,
                    rows=(
                        _float(
                            "noise_augmentation",
                            0.0,
                            0.0,
                            1.0,
                            0.001,
                        ),
                    ),
                ),
                Output("positive", _CONDITIONING, label="positive"),
                Output("negative", _CONDITIONING, label="negative"),
                Output("latent", _LATENT, label="latent"),
            ),
        )
        return with_identity_replacement(schema, "SD_4XUpscale_Conditioning")

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        scale_ratio: float = 4.0,
        noise_augmentation: float = 0.0,
    ) -> Mapping[str, object]:
        image = materialize_image(images)
        width = max(1, round(image.shape[-2] * scale_ratio))
        height = max(1, round(image.shape[-3] * scale_ratio))
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        pixels = utils.common_upscale(
            image.movedim(-1, 1) * 2.0 - 1.0, width // 4, height // 4, "bilinear", "center"
        )
        torch = importlib.import_module("torch")
        values = {"concat_image": pixels, "noise_augmentation": noise_augmentation}
        return cls.outputs(
            positive=conditioning_set_values(positive, values),
            negative=conditioning_set_values(negative, values),
            latent={"samples": torch.zeros((image.shape[0], 4, height // 4, width // 4))},
        )

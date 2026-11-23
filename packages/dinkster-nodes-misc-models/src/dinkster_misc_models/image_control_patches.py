"""Qwen, Z-Image, and USO image-control model patches."""

from __future__ import annotations

from dinkster_api.v1 import (
    CORE_FLOAT,
    IMAGE_TYPE,
    MASK_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    Input,
    NodeSchema,
    NumberWidget,
    Output,
    Section,
    TypeExpr,
    with_identity_replacement,
)
from dinkster_qwen.image_control_patches import QwenImageDiffsynthControlnet

_MODEL = TypeExpr.concrete(MODEL_TYPE)

_MODEL_PATCH = TypeExpr.concrete(MODEL_PATCH_TYPE)

_VAE = TypeExpr.concrete(VAE_TYPE)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)

_MASK = TypeExpr.concrete(MASK_TYPE)

_FLOAT = TypeExpr.concrete(CORE_FLOAT)


def _range() -> tuple[Input, Input]:
    return (
        Input(
            "start_percent",
            _FLOAT,
            default=0.0,
            required=False,
            widget=NumberWidget(min=0.0, max=1.0, step=0.001),
        ),
        Input(
            "end_percent",
            _FLOAT,
            default=1.0,
            required=False,
            widget=NumberWidget(min=0.0, max=1.0, step=0.001),
        ),
    )


class ZImageFunControlnet(QwenImageDiffsynthControlnet):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.misc.models.z_image_fun_controlnet",
            display_name="Apply Fun ControlNet",
            category="model/patch",
            search_terms=(
                "z-image controlnet",
                "qwen image 2.1 controlnet",
                "controlnet union",
                "fun controlnet",
            ),
            body=(
                Input("model", _MODEL),
                Input("model_patch", _MODEL_PATCH),
                Input("vae", _VAE),
                Input(
                    "strength",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
                Input("image", _IMAGE, required=False),
                Input("inpaint_image", _IMAGE, required=False),
                Input("mask", _MASK, required=False),
                Section("advanced", label="Advanced", collapsed=True, rows=(*_range(),)),
                Output("MODEL", _MODEL),
            ),
        )
        return with_identity_replacement(schema, "ZImageFunControlnet")

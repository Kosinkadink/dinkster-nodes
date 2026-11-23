"""SDXL text encoding with size and aesthetic conditioning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    Input,
    Node,
    NodeSchema,
    NumberWidget,
    Output,
    Section,
    StringWidget,
    TypeExpr,
    with_identity_replacement,
)


class CLIPTextEncodeSDXLRefiner(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.sd.clip_text_encode_sdxl_refiner",
            display_name="CLIP Text Encode (SDXL Refiner)",
            category="model/conditioning/stable diffusion",
            body=(
                Input(
                    "ascore",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=6.0,
                    widget=NumberWidget(min=0.0, max=1000.0, step=0.01),
                ),
                Input(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=0, max=16384),
                ),
                Input(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=0, max=16384),
                ),
                Input(
                    "text",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                Input("clip", TypeExpr.concrete(CLIP_TYPE)),
                Output("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
        )
        return with_identity_replacement(schema, "CLIPTextEncodeSDXLRefiner")

    @classmethod
    def execute(
        cls, *, clip: Any, ascore: float, width: int, height: int, text: str
    ) -> Mapping[str, object]:
        tokens = clip.tokenize(text)
        return cls.outputs(
            CONDITIONING=clip.encode_from_tokens_scheduled(
                tokens, add_dict={"aesthetic_score": ascore, "width": width, "height": height}
            )
        )


class CLIPTextEncodeSDXL(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.sd.clip_text_encode_sdxl",
            display_name="CLIP Text Encode (SDXL)",
            category="model/conditioning/stable diffusion",
            body=(
                Input("clip", TypeExpr.concrete(CLIP_TYPE)),
                Input(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=0, max=16384),
                ),
                Input(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=0, max=16384),
                ),
                Input(
                    "target_width",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=0, max=16384),
                ),
                Input(
                    "target_height",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=0, max=16384),
                ),
                Input(
                    "text_g",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                Input(
                    "text_l",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                Section(
                    "advanced",
                    label="Advanced",
                    collapsed=True,
                    rows=(
                        Input(
                            "crop_w",
                            TypeExpr.concrete(CORE_INT),
                            default=0,
                            widget=NumberWidget(min=0, max=16384),
                        ),
                        Input(
                            "crop_h",
                            TypeExpr.concrete(CORE_INT),
                            default=0,
                            widget=NumberWidget(min=0, max=16384),
                        ),
                    ),
                ),
                Output("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
        )
        return with_identity_replacement(schema, "CLIPTextEncodeSDXL")

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        width: int,
        height: int,
        crop_w: int,
        crop_h: int,
        target_width: int,
        target_height: int,
        text_g: str,
        text_l: str,
    ) -> Mapping[str, object]:
        tokens = clip.tokenize(text_g)
        tokens["l"] = clip.tokenize(text_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        return cls.outputs(
            CONDITIONING=clip.encode_from_tokens_scheduled(
                tokens,
                add_dict={
                    "width": width,
                    "height": height,
                    "crop_w": crop_w,
                    "crop_h": crop_h,
                    "target_width": target_width,
                    "target_height": target_height,
                },
            )
        )

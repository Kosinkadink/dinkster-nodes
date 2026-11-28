"""Core text encoding and CLIP layer selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CLIP_VISION_OUTPUT_TYPE,
    CLIP_VISION_TYPE,
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    ComboWidget,
    Input,
    Node,
    NodeSchema,
    NumberWidget,
    Output,
    Section,
    StringWidget,
    TypeExpr,
    materialize_image,
    with_identity_replacement,
)


class CLIPTextEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.conditioning.clip_text_encode",
                display_name="CLIP Text Encode (Prompt)",
                category="model/conditioning",
                search_terms=(
                    "text",
                    "prompt",
                    "text prompt",
                    "positive prompt",
                    "negative prompt",
                    "encode text",
                    "text encoder",
                    "encode prompt",
                ),
                description=(
                    "Encodes a text prompt using a CLIP model into an embedding that can be "
                    "used to guide the diffusion model towards generating specific images."
                ),
                body=(
                    Input(
                        "text",
                        TypeExpr.concrete(CORE_STRING),
                        tooltip="The text to be encoded.",
                        widget=StringWidget(multiline=True, dynamic_prompts=True),
                    ),
                    Input(
                        "clip",
                        TypeExpr.concrete(CLIP_TYPE),
                        tooltip="The CLIP model used for encoding the text.",
                    ),
                    Output(
                        "CONDITIONING",
                        TypeExpr.concrete(CONDITIONING_TYPE),
                        tooltip=(
                            "A conditioning containing the embedded text used to guide "
                            "the diffusion model towards generating specific images."
                        ),
                    ),
                ),
            ),
            "CLIPTextEncode",
        )

    @classmethod
    def execute(cls, *, clip: Any, text: str) -> Mapping[str, object]:
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\n"
                "If the clip is from a checkpoint loader node your checkpoint does not "
                "contain a valid clip or text encoder model."
            )
        tokens = clip.tokenize(text)
        return cls.outputs(CONDITIONING=clip.encode_from_tokens_scheduled(tokens))


class CLIPSetLastLayer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.conditioning.clip_set_last_layer",
                display_name="CLIP Set Last Layer",
                category="model/conditioning",
                body=(
                    Input("clip", TypeExpr.concrete(CLIP_TYPE)),
                    Section(
                        "advanced",
                        label="Advanced",
                        collapsed=True,
                        rows=(
                            Input(
                                "stop_at_clip_layer",
                                TypeExpr.concrete(CORE_INT),
                                default=-1,
                                widget=NumberWidget(min=-24, max=-1, step=1),
                            ),
                        ),
                    ),
                    Output("CLIP", TypeExpr.concrete(CLIP_TYPE)),
                ),
            ),
            "CLIPSetLastLayer",
        )

    @classmethod
    def execute(cls, *, clip: Any, stop_at_clip_layer: int) -> Mapping[str, object]:
        selected = clip.clone()
        selected.clip_layer(stop_at_clip_layer)
        return cls.outputs(CLIP=selected)


class CLIPVisionEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.conditioning.clip_vision_encode",
                display_name="CLIP Vision Encode",
                category="model/conditioning",
                body=(
                    Input("clip_vision", TypeExpr.concrete(CLIP_VISION_TYPE)),
                    Input("image", TypeExpr.concrete(IMAGE_TYPE)),
                    Input(
                        "crop",
                        TypeExpr.concrete(CORE_COMBO),
                        default="center",
                        widget=ComboWidget(("center", "none")),
                    ),
                    Output("CLIP_VISION_OUTPUT", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)),
                ),
            ),
            "CLIPVisionEncode",
        )

    @classmethod
    def execute(cls, *, clip_vision: Any, image: Any, crop: str) -> Mapping[str, object]:
        return cls.outputs(
            CLIP_VISION_OUTPUT=clip_vision.encode_image(
                materialize_image(image), crop=crop == "center"
            )
        )

"""Flux prompt streams and conditioning metadata transforms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_STRING,
    ComboOption,
    ComboWidget,
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
from dinkster_pack_helpers.conditioning import conditioning_set_values


class CLIPTextEncodeFlux(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.flux.clip_text_encode_flux",
            category="model/conditioning/flux",
            body=(
                Input("clip", TypeExpr.concrete(CLIP_TYPE)),
                Input(
                    "clip_l",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                Input(
                    "t5xxl",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                Input(
                    "guidance",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=3.5,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                Output("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
        )
        return with_identity_replacement(schema, "CLIPTextEncodeFlux")

    @classmethod
    def execute(
        cls, *, clip: Any, clip_l: str, t5xxl: str, guidance: float
    ) -> Mapping[str, object]:
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]
        return cls.outputs(
            CONDITIONING=clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance})
        )


class FluxGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.flux.flux_guidance",
            category="model/conditioning/flux",
            body=(
                Input("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                Input(
                    "guidance",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=3.5,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                Output("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
        )
        return with_identity_replacement(schema, "FluxGuidance")

    @classmethod
    def execute(cls, *, conditioning: Any, guidance: float) -> Mapping[str, object]:
        return cls.outputs(
            CONDITIONING=conditioning_set_values(conditioning, {"guidance": guidance})
        )


class FluxDisableGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.flux.flux_disable_guidance",
            category="model/conditioning/flux",
            description=(
                "This node completely disables the guidance embed on Flux and Flux like models"
            ),
            body=(
                Input("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                Output("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
        )
        return with_identity_replacement(schema, "FluxDisableGuidance")

    @classmethod
    def execute(cls, *, conditioning: Any) -> Mapping[str, object]:
        return cls.outputs(CONDITIONING=conditioning_set_values(conditioning, {"guidance": None}))


class FluxKontextMultiReferenceLatentMethod(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.flux.flux_kontext_multi_reference_latent_method",
            display_name="Edit Model Reference Method",
            category="model/conditioning/flux",
            body=(
                Input("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                Section(
                    "advanced",
                    label="Advanced",
                    collapsed=True,
                    rows=(
                        Input(
                            "reference_latents_method",
                            TypeExpr.concrete(CORE_COMBO),
                            widget=ComboWidget(
                                tuple(
                                    ComboOption(name)
                                    for name in (
                                        "offset",
                                        "index",
                                        "uxo/uno",
                                        "index_timestep_zero",
                                        "index_timestep_zero_isolated",
                                        "index_timestep_zero_text_start",
                                    )
                                )
                            ),
                        ),
                    ),
                ),
                Output("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
        )
        return with_identity_replacement(schema, "FluxKontextMultiReferenceLatentMethod")

    @classmethod
    def execute(cls, *, conditioning: Any, reference_latents_method: str) -> Mapping[str, object]:
        if "uxo" in reference_latents_method or "uso" in reference_latents_method:
            reference_latents_method = "uxo"
        return cls.outputs(
            CONDITIONING=conditioning_set_values(
                conditioning, {"reference_latents_method": reference_latents_method}
            )
        )

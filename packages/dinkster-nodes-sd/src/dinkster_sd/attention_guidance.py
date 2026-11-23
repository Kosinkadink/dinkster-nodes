"""Attention-guidance model patches executed by the inference sampler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_STRING,
    MODEL_TYPE,
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
from dinkster_sampling.attention_guidance import SkipLayerGuidanceDiT

_FLOAT = TypeExpr.concrete(CORE_FLOAT)

_MODEL = TypeExpr.concrete(MODEL_TYPE)

_STRING = TypeExpr.concrete(CORE_STRING)


class SkipLayerGuidanceSD3(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.sd.skip_layer_guidance_sd3",
            category="advanced/guidance",
            description=(
                "Generic version of SkipLayerGuidance node that can be used on every DiT model."
            ),
            body=(
                Input("model", _MODEL),
                Input(
                    "scale", _FLOAT, default=3.0, widget=NumberWidget(min=0.0, max=10.0, step=0.1)
                ),
                Section(
                    "advanced",
                    label="Advanced",
                    collapsed=True,
                    rows=(
                        Input(
                            "layers",
                            _STRING,
                            default="7, 8, 9",
                            widget=StringWidget(multiline=False),
                        ),
                        Input(
                            "start_percent",
                            _FLOAT,
                            default=0.01,
                            widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                        ),
                        Input(
                            "end_percent",
                            _FLOAT,
                            default=0.15,
                            widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                        ),
                    ),
                ),
                Output("MODEL", _MODEL),
            ),
        )
        return with_identity_replacement(schema, "SkipLayerGuidanceSD3")

    @classmethod
    def execute(
        cls, *, model: Any, layers: str, scale: float, start_percent: float, end_percent: float
    ) -> Mapping[str, object]:
        return SkipLayerGuidanceDiT.execute(
            model=model,
            double_layers=layers,
            single_layers="",
            scale=scale,
            start_percent=start_percent,
            end_percent=end_percent,
            rescaling_scale=0.0,
        )

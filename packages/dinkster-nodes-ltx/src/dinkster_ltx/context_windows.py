from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_INT,
    MODEL_TYPE,
    ComboWidget,
    Input,
    NodeSchema,
    NumberWidget,
    Output,
    Section,
    TypeExpr,
    with_identity_replacement,
)
from dinkster_models.context_windows import ContextWindowsManual

_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)

_COMBO = TypeExpr.concrete(CORE_COMBO)

_INT = TypeExpr.concrete(CORE_INT)

_MODEL = TypeExpr.concrete(MODEL_TYPE)

_SCHEDULES = ("standard_static", "standard_uniform", "looped_uniform", "batched")

_FUSE_METHODS = ("pyramid", "relative", "flat", "overlap-linear")


class LTXVContextWindows(ContextWindowsManual):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.ltx.ltxv_context_windows",
            display_name="LTXV Context Windows",
            category="model/patch/ltxv",
            description="Set context windows for LTXV-like models.",
            body=(
                Input("model", _MODEL),
                Input(
                    "context_length",
                    _INT,
                    default=145,
                    widget=NumberWidget(min=1, max=16384, step=8),
                ),
                Input("context_overlap", _INT, default=40, widget=NumberWidget(min=0, step=8)),
                Input(
                    "context_schedule",
                    _COMBO,
                    default="standard_uniform",
                    widget=ComboWidget(_SCHEDULES),
                ),
                Input("fuse_method", _COMBO, default="pyramid", widget=ComboWidget(_FUSE_METHODS)),
                Input("retain_first_frame", _BOOLEAN, default=False),
                Section(
                    "advanced",
                    label="Advanced",
                    collapsed=True,
                    rows=(
                        Input("context_stride", _INT, default=1, widget=NumberWidget(min=1)),
                        Input(
                            "closed_loop",
                            _BOOLEAN,
                            default=False,
                        ),
                        Input(
                            "freenoise",
                            _BOOLEAN,
                            default=True,
                        ),
                        Input(
                            "split_conds_to_windows",
                            _BOOLEAN,
                            default=False,
                        ),
                    ),
                ),
                Output("MODEL", _MODEL),
            ),
        )
        return with_identity_replacement(schema, "LTXVContextWindows")

    @classmethod
    def execute(
        cls,
        *,
        context_length: int,
        context_overlap: int,
        retain_first_frame: bool = False,
        **kwargs: Any,
    ) -> Mapping[str, object]:
        retain = "0" if retain_first_frame else ""
        return super().execute(
            context_length=max(((context_length - 1) // 8) + 1, 1),
            context_overlap=max(context_overlap // 8, 0),
            dim=2,
            cond_retain_index_list=retain,
            latent_retain_index_list=retain,
            **kwargs,
        )

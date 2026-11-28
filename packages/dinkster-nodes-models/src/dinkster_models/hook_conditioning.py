"""Native conditioning and CLIP hook nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    HOOKS_TYPE,
    MASK_TYPE,
    TIMESTEPS_RANGE_TYPE,
    ComboWidget,
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

from .hook_scheduling import retain_hook_weights

_CATEGORY = "advanced/hooks"


_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)


_CLIP = TypeExpr.concrete(CLIP_TYPE)


_HOOKS = TypeExpr.concrete(HOOKS_TYPE)


_MASK = TypeExpr.concrete(MASK_TYPE)


_TIMESTEPS_RANGE = TypeExpr.concrete(TIMESTEPS_RANGE_TYPE)


_FLOAT = TypeExpr.concrete(CORE_FLOAT)


_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)


_COMBO = TypeExpr.concrete(CORE_COMBO)


def _hooks() -> Any:
    initialize_inference_process()
    return importlib.import_module("dinkster_inference.hooks")


def _properties_inputs(*inputs: Input) -> tuple[Input, ...]:
    return (
        *inputs,
        Input(
            "strength",
            _FLOAT,
            default=1.0,
            widget=NumberWidget(min=0.0, max=10.0, step=0.01),
        ),
        Input(
            "set_cond_area",
            _COMBO,
            default="default",
            widget=ComboWidget(("default", "mask bounds")),
        ),
        Input("mask", _MASK, required=False),
        Input("hooks", _HOOKS, required=False),
        Input("timesteps", _TIMESTEPS_RANGE, required=False),
    )


def _retain_conditioning_weights(
    results: list[list[Any]], originals: Sequence[Sequence[Any]], hooks: object | None
) -> list[list[Any]]:
    if hooks is not None:
        for result, original in zip(results, originals, strict=True):
            for (_, metadata), (_, previous) in zip(result, original, strict=True):
                retain_hook_weights(metadata.get("hooks"), previous.get("hooks"), hooks)
    return results


def _properties(
    conds: Sequence[Sequence[Any]],
    strength: float,
    set_cond_area: str,
    mask: object | None,
    hooks: object | None,
    timesteps: tuple[float, float] | None,
) -> list[list[Any]]:
    results = _hooks().set_conds_props(
        list(conds),
        strength,
        set_cond_area,
        mask=None if mask is None else materialize_image(mask),
        hooks=hooks,
        timesteps_range=timesteps,
    )
    return _retain_conditioning_weights(results, conds, hooks)


class ConditioningSetProperties(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.models.conditioning_set_properties",
                display_name="Cond Set Props",
                category=f"{_CATEGORY}/cond single",
                body=(
                    *_properties_inputs(Input("cond_NEW", _CONDITIONING)),
                    Output("conditioning", _CONDITIONING),
                ),
            ),
            "ConditioningSetProperties",
        )

    @classmethod
    def execute(
        cls,
        *,
        cond_NEW: Sequence[Sequence[Any]],
        strength: float = 1.0,
        set_cond_area: str = "default",
        mask: object | None = None,
        hooks: object | None = None,
        timesteps: tuple[float, float] | None = None,
    ) -> Mapping[str, object]:
        return cls.outputs(
            conditioning=_properties([cond_NEW], strength, set_cond_area, mask, hooks, timesteps)[0]
        )


class ConditioningSetPropertiesAndCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.models.conditioning_set_properties_and_combine",
                display_name="Cond Set Props Combine",
                category=f"{_CATEGORY}/cond single",
                body=(
                    *_properties_inputs(
                        Input("cond", _CONDITIONING), Input("cond_NEW", _CONDITIONING)
                    ),
                    Output("conditioning", _CONDITIONING),
                ),
            ),
            "ConditioningSetPropertiesAndCombine",
        )

    @classmethod
    def execute(
        cls,
        *,
        cond: Sequence[Sequence[Any]],
        cond_NEW: Sequence[Sequence[Any]],
        strength: float = 1.0,
        set_cond_area: str = "default",
        mask: object | None = None,
        hooks: object | None = None,
        timesteps: tuple[float, float] | None = None,
    ) -> Mapping[str, object]:
        result = _hooks().set_conds_props_and_combine(
            [cond],
            [cond_NEW],
            strength,
            set_cond_area,
            mask=None if mask is None else materialize_image(mask),
            hooks=hooks,
            timesteps_range=timesteps,
        )
        return cls.outputs(
            conditioning=_retain_conditioning_weights(result, [[*cond, *cond_NEW]], hooks)[0]
        )


class PairConditioningSetProperties(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.models.pair_conditioning_set_properties",
                display_name="Cond Pair Set Props",
                category=f"{_CATEGORY}/cond pair",
                body=(
                    *_properties_inputs(
                        Input("positive_NEW", _CONDITIONING),
                        Input("negative_NEW", _CONDITIONING),
                    ),
                    Output("positive", _CONDITIONING),
                    Output("negative", _CONDITIONING),
                ),
            ),
            "PairConditioningSetProperties",
        )

    @classmethod
    def execute(
        cls,
        *,
        positive_NEW: Sequence[Sequence[Any]],
        negative_NEW: Sequence[Sequence[Any]],
        strength: float = 1.0,
        set_cond_area: str = "default",
        mask: object | None = None,
        hooks: object | None = None,
        timesteps: tuple[float, float] | None = None,
    ) -> Mapping[str, object]:
        positive, negative = _properties(
            [positive_NEW, negative_NEW], strength, set_cond_area, mask, hooks, timesteps
        )
        return cls.outputs(positive=positive, negative=negative)


class PairConditioningSetPropertiesAndCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.models.pair_conditioning_set_properties_and_combine",
                display_name="Cond Pair Set Props Combine",
                category=f"{_CATEGORY}/cond pair",
                body=(
                    *_properties_inputs(
                        Input("positive", _CONDITIONING),
                        Input("negative", _CONDITIONING),
                        Input("positive_NEW", _CONDITIONING),
                        Input("negative_NEW", _CONDITIONING),
                    ),
                    Output("positive", _CONDITIONING),
                    Output("negative", _CONDITIONING),
                ),
            ),
            "PairConditioningSetPropertiesAndCombine",
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        positive_NEW: Sequence[Sequence[Any]],
        negative_NEW: Sequence[Sequence[Any]],
        strength: float = 1.0,
        set_cond_area: str = "default",
        mask: object | None = None,
        hooks: object | None = None,
        timesteps: tuple[float, float] | None = None,
    ) -> Mapping[str, object]:
        results = _hooks().set_conds_props_and_combine(
            [positive, negative],
            [positive_NEW, negative_NEW],
            strength,
            set_cond_area,
            mask=None if mask is None else materialize_image(mask),
            hooks=hooks,
            timesteps_range=timesteps,
        )
        positive, negative = _retain_conditioning_weights(
            results, [[*positive, *positive_NEW], [*negative, *negative_NEW]], hooks
        )
        return cls.outputs(positive=positive, negative=negative)


class ConditioningSetDefaultCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.models.conditioning_set_default_combine",
                display_name="Cond Set Default Combine",
                category=f"{_CATEGORY}/cond single",
                body=(
                    Input("cond", _CONDITIONING),
                    Input("cond_DEFAULT", _CONDITIONING),
                    Input("hooks", _HOOKS, required=False),
                    Output("conditioning", _CONDITIONING),
                ),
            ),
            "ConditioningSetDefaultCombine",
        )

    @classmethod
    def execute(
        cls,
        *,
        cond: Sequence[Sequence[Any]],
        cond_DEFAULT: Sequence[Sequence[Any]],
        hooks: object | None = None,
    ) -> Mapping[str, object]:
        result = _hooks().set_default_conds_and_combine([cond], [cond_DEFAULT], hooks=hooks)
        return cls.outputs(
            conditioning=_retain_conditioning_weights(result, [[*cond, *cond_DEFAULT]], hooks)[0]
        )


class PairConditioningSetDefaultCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.models.pair_conditioning_set_default_combine",
                display_name="Cond Pair Set Default Combine",
                category=f"{_CATEGORY}/cond pair",
                body=(
                    Input("positive", _CONDITIONING),
                    Input("negative", _CONDITIONING),
                    Input("positive_DEFAULT", _CONDITIONING),
                    Input("negative_DEFAULT", _CONDITIONING),
                    Input("hooks", _HOOKS, required=False),
                    Output("positive", _CONDITIONING),
                    Output("negative", _CONDITIONING),
                ),
            ),
            "PairConditioningSetDefaultCombine",
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        positive_DEFAULT: Sequence[Sequence[Any]],
        negative_DEFAULT: Sequence[Sequence[Any]],
        hooks: object | None = None,
    ) -> Mapping[str, object]:
        results = _hooks().set_default_conds_and_combine(
            [positive, negative], [positive_DEFAULT, negative_DEFAULT], hooks=hooks
        )
        positive, negative = _retain_conditioning_weights(
            results, [[*positive, *positive_DEFAULT], [*negative, *negative_DEFAULT]], hooks
        )
        return cls.outputs(positive=positive, negative=negative)


class PairConditioningCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.models.pair_conditioning_combine",
                display_name="Cond Pair Combine",
                category=f"{_CATEGORY}/cond pair",
                body=(
                    Input("positive_A", _CONDITIONING),
                    Input("negative_A", _CONDITIONING),
                    Input("positive_B", _CONDITIONING),
                    Input("negative_B", _CONDITIONING),
                    Output("positive", _CONDITIONING),
                    Output("negative", _CONDITIONING),
                ),
            ),
            "PairConditioningCombine",
        )

    @classmethod
    def execute(
        cls,
        *,
        positive_A: Sequence[Sequence[Any]],
        negative_A: Sequence[Sequence[Any]],
        positive_B: Sequence[Sequence[Any]],
        negative_B: Sequence[Sequence[Any]],
    ) -> Mapping[str, object]:
        positive, negative = _hooks().set_conds_props_and_combine(
            [positive_A, negative_A], [positive_B, negative_B]
        )
        return cls.outputs(positive=positive, negative=negative)


class SetClipHooks(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.models.set_clip_hooks",
                display_name="Set CLIP Hooks",
                category=f"{_CATEGORY}/clip",
                body=(
                    Input("clip", _CLIP),
                    Input("hooks", _HOOKS, required=False),
                    Section(
                        "advanced",
                        label="Advanced",
                        collapsed=True,
                        rows=(
                            Input(
                                "apply_to_conds",
                                _BOOLEAN,
                                default=True,
                            ),
                            Input(
                                "schedule_clip",
                                _BOOLEAN,
                                default=False,
                            ),
                        ),
                    ),
                    Output("clip", _CLIP),
                ),
                occupies=("gpu",),
            ),
            "SetClipHooks",
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        apply_to_conds: bool = True,
        schedule_clip: bool = False,
        hooks: object | None = None,
    ) -> Mapping[str, object]:
        if hooks is None:
            return cls.outputs(clip=clip)
        inference_hooks = _hooks()
        hook_group = cast(Any, hooks)
        result = clip.clone(disable_dynamic=True)
        if apply_to_conds:
            result.apply_hooks_to_conds = hook_group
        result.patcher.forced_hooks = retain_hook_weights(hook_group.clone(), hook_group)
        result.use_clip_schedule = schedule_clip
        if not result.use_clip_schedule:
            result.patcher.forced_hooks.set_keyframes_on_hooks(None)
        result.patcher.register_all_hook_patches(
            hook_group, inference_hooks.create_target_dict(inference_hooks.EnumWeightTarget.Clip)
        )
        return cls.outputs(clip=result)


class ConditioningTimestepsRange(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.models.conditioning_timesteps_range",
                display_name="Timesteps Range",
                category=_CATEGORY,
                body=(
                    Input(
                        "start_percent",
                        _FLOAT,
                        default=0.0,
                        widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                    ),
                    Input(
                        "end_percent",
                        _FLOAT,
                        default=1.0,
                        widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                    ),
                    Output("TIMESTEPS_RANGE", _TIMESTEPS_RANGE),
                    Output("BEFORE_RANGE", _TIMESTEPS_RANGE),
                    Output("AFTER_RANGE", _TIMESTEPS_RANGE),
                ),
            ),
            "ConditioningTimestepsRange",
        )

    @classmethod
    def execute(
        cls, *, start_percent: float = 0.0, end_percent: float = 1.0
    ) -> Mapping[str, object]:
        return cls.outputs(
            TIMESTEPS_RANGE=(start_percent, end_percent),
            BEFORE_RANGE=(0.0, start_percent),
            AFTER_RANGE=(end_percent, 1.0),
        )

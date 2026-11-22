"""Model and text-encoder merging through the inference patcher contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar, cast

from dinkster_api.v1 import (
    CLIP_TYPE,
    CORE_FLOAT,
    MODEL_TYPE,
    Formula,
    Group,
    Input,
    InputFamilyMapping,
    InputFamilyMember,
    Item,
    MappingSource,
    Members,
    Named,
    Node,
    NodeSchema,
    NumberWidget,
    Output,
    ReplacementCase,
    ReplacementRule,
    TypeExpr,
    with_identity_replacement,
)


def _coefficient(name: str, *, subtract: bool = False) -> Input:
    return Input(
        name,
        TypeExpr.concrete(CORE_FLOAT),
        default=1.0,
        widget=NumberWidget(
            min=-10.0 if subtract else 0.0, max=10.0 if subtract else 1.0, step=0.01
        ),
    )


def _merge_schema(node_type: str, *, clip: bool = False, coefficient: str = "") -> NodeSchema:
    type_id = CLIP_TYPE if clip else MODEL_TYPE
    prefix = "clip" if clip else "model"
    inputs = (
        Input(prefix + "1", TypeExpr.concrete(type_id)),
        Input(prefix + "2", TypeExpr.concrete(type_id)),
    )
    if coefficient:
        inputs += (_coefficient(coefficient, subtract=coefficient == "multiplier"),)
    return NodeSchema(
        node_type=node_type,
        category="model/merging",
        body=(
            *inputs,
            Output("CLIP" if clip else "MODEL", TypeExpr.concrete(type_id)),
        ),
    )


def _merge(
    first: Any, second: Any, patch_strength: float, model_strength: float, *, clip: bool = False
) -> Any:
    merged = first.clone()
    patches = second.get_key_patches() if clip else second.get_key_patches("diffusion_model.")
    for key in patches:
        if clip and (key.endswith(".position_ids") or key.endswith(".logit_scale")):
            continue
        merged.add_patches({key: patches[key]}, patch_strength, model_strength)
    return merged


class ModelMergeSimple(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            _merge_schema("dinkster.models.model_merge_simple", coefficient="ratio"),
            "ModelMergeSimple",
        )

    @classmethod
    def execute(cls, *, model1: Any, model2: Any, ratio: float) -> Mapping[str, object]:
        return cls.outputs(MODEL=_merge(model1, model2, 1.0 - ratio, ratio))


class ModelMergeSubtract(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            _merge_schema("dinkster.models.model_merge_subtract", coefficient="multiplier"),
            "ModelMergeSubtract",
        )

    @classmethod
    def execute(cls, *, model1: Any, model2: Any, multiplier: float) -> Mapping[str, object]:
        return cls.outputs(MODEL=_merge(model1, model2, -multiplier, multiplier))


class ModelMergeAdd(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            _merge_schema("dinkster.models.model_merge_add"), "ModelMergeAdd"
        )

    @classmethod
    def execute(cls, *, model1: Any, model2: Any) -> Mapping[str, object]:
        return cls.outputs(MODEL=_merge(model1, model2, 1.0, 1.0))


class CLIPMergeSimple(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            _merge_schema("dinkster.models.clip_merge_simple", clip=True, coefficient="ratio"),
            "CLIPMergeSimple",
        )

    @classmethod
    def execute(cls, *, clip1: Any, clip2: Any, ratio: float) -> Mapping[str, object]:
        return cls.outputs(CLIP=_merge(clip1, clip2, 1.0 - ratio, ratio, clip=True))


class CLIPMergeSubtract(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            _merge_schema(
                "dinkster.models.clip_merge_subtract", clip=True, coefficient="multiplier"
            ),
            "CLIPMergeSubtract",
        )

    @classmethod
    def execute(cls, *, clip1: Any, clip2: Any, multiplier: float) -> Mapping[str, object]:
        return cls.outputs(CLIP=_merge(clip1, clip2, -multiplier, multiplier, clip=True))


class CLIPMergeAdd(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            _merge_schema("dinkster.models.clip_merge_add", clip=True), "CLIPMergeAdd"
        )

    @classmethod
    def execute(cls, *, clip1: Any, clip2: Any) -> Mapping[str, object]:
        return cls.outputs(CLIP=_merge(clip1, clip2, 1.0, 1.0, clip=True))


def _block_key(name: str) -> str:
    """A block's member key: its prefix with the last dot dropped and dots as underscores."""
    return name.removesuffix(".").replace(".", "_")


class ModelMergeBlocks(Node):
    """Merges two models with one ratio per block prefix.

    ``BLOCKS`` lists ComfyUI's input ids, which are the model key prefixes the
    ratios apply to. Each becomes a member of the fixed ``ratios`` group, in the
    same order, labelled with its prefix."""

    NODE_TYPE: ClassVar[str] = "dinkster.models.model_merge_blocks"
    SOURCE: ClassVar[str] = "ModelMergeBlocks"
    CATEGORY: ClassVar[str] = "model/merging"
    DESCRIPTION: ClassVar[str] = ""
    BLOCKS: ClassVar[tuple[str, ...]] = ("input", "middle", "out")

    @classmethod
    def define_schema(cls) -> NodeSchema:
        keys = tuple(_block_key(name) for name in cls.BLOCKS)
        schema = with_identity_replacement(
            NodeSchema(
                node_type=cls.NODE_TYPE,
                category=cls.CATEGORY,
                description=cls.DESCRIPTION,
                tables={"blocks": dict(zip(keys, cls.BLOCKS, strict=True))},
                body=(
                    Input("model1", TypeExpr.concrete(MODEL_TYPE)),
                    Input("model2", TypeExpr.concrete(MODEL_TYPE)),
                    Group(
                        "ratios",
                        value=_coefficient("ratio"),
                        keys=Named(keys),
                        min_members=len(keys),
                        max_members=len(keys),
                        label="Block ratios",
                        member_label=Formula("blocks[key()]"),
                    ),
                    Output("MODEL", TypeExpr.concrete(MODEL_TYPE)),
                ),
            ),
            cls.SOURCE,
        )
        # ComfyUI stores each ratio under its block prefix; import each into its member.
        rule = ReplacementRule(
            from_type=cls.SOURCE,
            cases=(
                ReplacementCase.build(
                    cls.NODE_TYPE,
                    inputs={name: MappingSource.copy(name) for name in ("model1", "model2")},
                    outputs={"MODEL": "MODEL"},
                    input_families={
                        "ratios": InputFamilyMapping.from_members(
                            *(
                                InputFamilyMember.build(
                                    key, inputs={"ratio": MappingSource.copy(name)}
                                )
                                for key, name in zip(keys, cls.BLOCKS, strict=True)
                            )
                        )
                    },
                ),
            ),
        )
        return replace(schema, replacements=(rule,))

    @classmethod
    def execute(cls, *, model1: Any, model2: Any, ratios: Members[Item]) -> Mapping[str, object]:
        prefixes = dict(zip((_block_key(name) for name in cls.BLOCKS), cls.BLOCKS, strict=True))
        by_prefix = {prefixes[member.key]: cast("float", member.value) for member in ratios}
        merged = model1.clone()
        patches = model2.get_key_patches("diffusion_model.")
        default_ratio = next(iter(by_prefix.values()))
        for key in patches:
            ratio = default_ratio
            unet_key = key[len("diffusion_model.") :]
            longest = 0
            for prefix, candidate in by_prefix.items():
                if unet_key.startswith(prefix) and longest < len(prefix):
                    ratio = candidate
                    longest = len(prefix)
            merged.add_patches({key: patches[key]}, 1.0 - ratio, ratio)
        return cls.outputs(MODEL=merged)

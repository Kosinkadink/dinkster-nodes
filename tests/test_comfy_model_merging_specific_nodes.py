from __future__ import annotations

from typing import cast

import pytest
from dinkster_api.v1 import Formula, Item, Members, Named, NodeSchema
from dinkster_graph import Graph, GraphNode, migrate_node_replacements
from dinkster_models import NODES
from dinkster_models.model_merging_specific import (
    ModelMergeAuraflow,
    ModelMergeCosmos7B,
    ModelMergeCosmos14B,
    ModelMergeCosmosPredict2_2B,
    ModelMergeCosmosPredict2_14B,
    ModelMergeFlux1,
    ModelMergeKrea2,
    ModelMergeLTXV,
    ModelMergeMochiPreview,
    ModelMergeQwenImage,
    ModelMergeSD1,
    ModelMergeSD2,
    ModelMergeSD3_2B,
    ModelMergeSD35_Large,
    ModelMergeSDXL,
    ModelMergeWAN2_1,
)

from tests.body_rows import groups

MODEL_SPECIFIC_MERGE_NODES = (
    ModelMergeSD1,
    ModelMergeSD2,
    ModelMergeSDXL,
    ModelMergeSD3_2B,
    ModelMergeAuraflow,
    ModelMergeFlux1,
    ModelMergeSD35_Large,
    ModelMergeMochiPreview,
    ModelMergeLTXV,
    ModelMergeCosmos7B,
    ModelMergeCosmos14B,
    ModelMergeWAN2_1,
    ModelMergeCosmosPredict2_2B,
    ModelMergeCosmosPredict2_14B,
    ModelMergeQwenImage,
    ModelMergeKrea2,
)


def _ratios(schema: NodeSchema) -> Named:
    keys = groups(schema)[0].keys
    assert isinstance(keys, Named)
    return keys


class _Model:
    def __init__(self, patches: dict[str, float], weights: dict[str, float] | None = None) -> None:
        self.patches = patches
        self.weights = {} if weights is None else weights

    def clone(self) -> _Model:
        return _Model({}, self.weights.copy())

    def get_key_patches(self, prefix: str) -> dict[str, float]:
        assert prefix == "diffusion_model."
        return self.patches

    def add_patches(self, patches: dict[str, float], first: float, second: float) -> None:
        for key, patch in patches.items():
            self.weights[key] = self.weights[key] * first + patch * second


def test_specific_model_merge_schemas_have_stable_public_contracts() -> None:
    expected = (
        ("ModelMergeSD1", 30),
        ("ModelMergeSD2", 30),
        ("ModelMergeSDXL", 24),
        ("ModelMergeSD3_2B", 30),
        ("ModelMergeAuraflow", 43),
        ("ModelMergeFlux1", 63),
        ("ModelMergeSD35_Large", 44),
        ("ModelMergeMochiPreview", 53),
        ("ModelMergeLTXV", 33),
        ("ModelMergeCosmos7B", 34),
        ("ModelMergeCosmos14B", 42),
        ("ModelMergeWAN2_1", 46),
        ("ModelMergeCosmosPredict2_2B", 33),
        ("ModelMergeCosmosPredict2_14B", 41),
        ("ModelMergeQwenImage", 66),
        ("ModelMergeKrea2", 38),
    )
    assert (
        tuple(
            (node.__name__, len(_ratios(node.schema()).names))
            for node in MODEL_SPECIFIC_MERGE_NODES
        )
        == expected
    )
    expected_ids = (
        "dinkster.models.model_merge_sd1",
        "dinkster.models.model_merge_sd2",
        "dinkster.models.model_merge_sdxl",
        "dinkster.models.model_merge_sd3_2_b",
        "dinkster.models.model_merge_auraflow",
        "dinkster.models.model_merge_flux1",
        "dinkster.models.model_merge_sd35_large",
        "dinkster.models.model_merge_mochi_preview",
        "dinkster.models.model_merge_ltxv",
        "dinkster.models.model_merge_cosmos7_b",
        "dinkster.models.model_merge_cosmos14_b",
        "dinkster.models.model_merge_wan2_1",
        "dinkster.models.model_merge_cosmos_predict2_2_b",
        "dinkster.models.model_merge_cosmos_predict2_14_b",
        "dinkster.models.model_merge_qwen_image",
        "dinkster.models.model_merge_krea2",
    )
    for node, node_type in zip(MODEL_SPECIFIC_MERGE_NODES, expected_ids, strict=True):
        schema = node.schema()
        assert schema.node_type == node_type
        assert NODES.count(node) == 1
        assert schema.category == "model/merging/model specific"
        assert tuple(input_spec.id for input_spec in schema.inputs) == ("model1", "model2")
        group = groups(schema)[0]
        names = _ratios(schema).names
        assert (group.id, group.label, group.min_members, group.max_members) == (
            "ratios",
            "Block ratios",
            len(names),
            len(names),
        )
        assert not group.reorderable
        # Each member is labelled with the ComfyUI input id it stands for.
        assert tuple(cast("dict[str, str]", schema.tables["blocks"])) == names
        assert group.member_label == Formula("blocks[key()]")

    assert MODEL_SPECIFIC_MERGE_NODES[11].schema().description == (
        "1.3B model has 30 blocks, 14B model has 40 blocks. "
        "Image to video model has the extra img_emb."
    )


def test_sd2_has_its_own_node_identity_with_sd1_coefficients() -> None:
    assert ModelMergeSD2 is not ModelMergeSD1
    assert ModelMergeSD2.schema().node_type == "dinkster.models.model_merge_sd2"
    assert ModelMergeSD2.schema().body == ModelMergeSD1.schema().body
    assert ModelMergeSD2.schema().tables == ModelMergeSD1.schema().tables


def test_specific_merge_matches_block_prefixes_and_the_first_ratio_by_default() -> None:
    source = _Model(
        {
            "diffusion_model.input_blocks.1.weight": 100.0,
            "diffusion_model.input_blocks.8.weight": 200.0,
            "diffusion_model.elsewhere.weight": 300.0,
        }
    )
    merged = cast(
        _Model,
        ModelMergeSDXL.execute(
            model1=_Model(
                {},
                {
                    "diffusion_model.input_blocks.1.weight": 10.0,
                    "diffusion_model.input_blocks.8.weight": 20.0,
                    "diffusion_model.elsewhere.weight": 30.0,
                },
            ),
            model2=source,
            ratios=Members(
                Item(key, ratio)
                for key, ratio in (
                    ("time_embed", 0.2),
                    ("input_blocks_1", 0.4),
                    ("input_blocks_8", 0.8),
                )
            ),
        )["MODEL"],
    )
    assert merged.weights == pytest.approx(
        {
            "diffusion_model.input_blocks.1.weight": 46.0,
            "diffusion_model.input_blocks.8.weight": 164.0,
            "diffusion_model.elsewhere.weight": 84.0,
        }
    )


def test_comfy_block_ratios_import_into_their_members() -> None:
    schema = ModelMergeSD1.schema()
    ratios = {name: 0.01 * position for position, name in enumerate(ModelMergeSD1.BLOCKS)}
    graph = Graph({"merge": GraphNode("ModelMergeSD1", dict(ratios))})
    migrated = migrate_node_replacements(graph, {schema.node_type: schema}).nodes["merge"]
    assert isinstance(migrated, GraphNode)
    assert migrated.node_type == "dinkster.models.model_merge_sd1"
    assert dict(migrated.inputs) == {
        "ratios.time_embed": 0.0,
        "ratios.label_emb": 0.01,
        **{f"ratios.input_blocks_{index}": 0.01 * (2 + index) for index in range(12)},
        **{f"ratios.middle_block_{index}": 0.01 * (14 + index) for index in range(3)},
        **{f"ratios.output_blocks_{index}": 0.01 * (17 + index) for index in range(12)},
        "ratios.out": 0.01 * 29,
    }


@pytest.mark.parametrize("node", MODEL_SPECIFIC_MERGE_NODES)
def test_all_specific_merge_nodes_use_model_specific_category(node: type[ModelMergeSD1]) -> None:
    assert node.schema().category == "model/merging/model specific"

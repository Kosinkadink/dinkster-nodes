"""Public Gaussian splat inspection contracts without Torch execution."""

import tomllib
from pathlib import Path
from typing import cast

import pytest
from dinkster_3d import NODES
from dinkster_3d.gaussian_splat import GetSplatCount, MergeSplat, TransformSplat
from dinkster_3d.splat_render import RenderSplat
from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    SPLAT_TYPE,
    Input,
    NodeSchema,
    Numbered,
    Output,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_engine import ParentGraphBudget, admit_parent_graph
from dinkster_graph import Graph, GraphNode, Link, graph_to_wire, validate
from dinkster_schema import ElaborationError, elaborate

from tests.body_rows import advanced_ids, groups


def test_render_splat_schema_and_native_registration() -> None:
    schema = RenderSplat.schema()
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert [port.id for port in schema.inputs] == [
        "splat",
        "width",
        "height",
        "frames",
        "sharpen",
        "render_style",
        "background",
        "bg_image",
        "camera_info",
        "splat_scale",
        "headlight_shading",
        "opacity_threshold",
    ]
    assert advanced_ids(schema.body) == ("splat_scale", "headlight_shading", "opacity_threshold")
    assert schema.inputs[0].type.types == (SPLAT_TYPE,)
    assert [port.id for port in schema.inputs if not port.required] == ["bg_image", "camera_info"]
    assert [port.id for port in schema.outputs] == ["IMAGE", "MASK"]
    assert sum(node is RenderSplat for node in NODES) == 1
    manifest = Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-3d/dinkster-pack.toml"
    assert "dinkster.3d" in tomllib.loads(manifest.read_text())["pack"]["namespaces"]


def test_splat_count_schema_and_production_claim_are_canonical() -> None:
    schema = GetSplatCount.schema()
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert schema.node_type == "dinkster.3d.get_splat_count" and schema.category == "3d/splat"
    assert [(port.id, port.type.types, port.required) for port in schema.inputs] == [
        ("splat", (SPLAT_TYPE,), True)
    ]
    assert [(port.id, port.type.types) for port in schema.outputs] == [
        ("splat", (SPLAT_TYPE,)),
        ("count", (CORE_INT,)),
    ]
    assert sum(node is GetSplatCount for node in NODES) == 1
    manifest = Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-3d/dinkster-pack.toml"
    assert "dinkster.3d" in tomllib.loads(manifest.read_text())["pack"]["namespaces"]


def test_transform_splat_public_schema_and_production_claim() -> None:
    schema = TransformSplat.schema()
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert schema.node_type == "dinkster.3d.transform_splat" and schema.category == "3d/splat"
    assert schema.inputs[0].type.types == (SPLAT_TYPE,) and schema.inputs[0].required
    assert [(port.id, port.default, port.type.types) for port in schema.inputs[1:]] == [
        (f"{operation}_{axis}", default, (CORE_FLOAT,))
        for operation, default in (("translate", 0.0), ("rotate", 0.0), ("scale", 1.0))
        for axis in "xyz"
    ]
    assert [(port.id, port.type.types) for port in schema.outputs] == [("splat", (SPLAT_TYPE,))]
    assert sum(node is TransformSplat for node in NODES) == 1
    manifest = Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-3d/dinkster-pack.toml"
    assert "dinkster.3d" in tomllib.loads(manifest.read_text())["pack"]["namespaces"]


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        (("splat0", "splat1"), (("splat0", True), ("splat1", True))),
        # The first positions are the required ones, whatever their keys.
        (("splat10", "splat2", "splat0"), (("splat10", True), ("splat2", True), ("splat0", False))),
    ],
)
def test_merge_splat_required_initial_slots_and_optional_authored_tail(
    members: tuple[str, ...],
    expected: tuple[tuple[str, bool], ...],
) -> None:
    schema = schema_from_wire(schema_to_wire(MergeSplat.schema()))
    family = groups(schema)[0]
    assert (
        family.id,
        family.min_members,
        family.max_members,
        cast(Numbered, family.keys).prefix,
    ) == (
        "splats",
        2,
        32,
        "splat",
    )
    value = family.value
    assert isinstance(value, Input)
    assert value.required and family.optional_after_minimum
    assert value.type == TypeExpr.concrete(SPLAT_TYPE)
    assert [(port.id, port.type.types) for port in schema.outputs] == [("splat", (SPLAT_TYPE,))]
    assert sum(node is MergeSplat for node in NODES) == 1
    manifest = Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-3d/dinkster-pack.toml"
    assert "dinkster.3d" in tomllib.loads(manifest.read_text())["pack"]["namespaces"]
    with pytest.raises(ElaborationError, match="needs at least 2 members"):
        elaborate(schema, ())
    effective = elaborate(schema, tuple(f"splats.{member}" for member in members))
    assert [(port.id, port.required) for port in effective.inputs] == [
        (f"splats.{member}", required) for member, required in expected
    ]
    required = [f"splats.{member}" for member, needed in expected if needed]
    source = NodeSchema("test.source", body=(Output("splat", TypeExpr.concrete(SPLAT_TYPE)),))
    inputs = {member: Link("source", "splat") for member in required}
    graph = Graph(
        {
            "source": GraphNode(source.node_type),
            "merge": GraphNode("dinkster.3d.merge_splat", inputs),
        }
    )
    schemas = {source.node_type: source, schema.node_type: effective}
    assert validate(graph, schemas, ("merge",)) == []
    admitted = admit_parent_graph(
        graph,
        graph_to_wire(Graph({})),
        namespace="test.splats",
        origins=(),
        virtual_joins=(),
        budget=ParentGraphBudget(),
        schemas=schemas,
    )
    assert graph_to_wire(admitted.graph) == graph_to_wire(graph)
    for missing in required:
        incomplete = Graph(
            {
                **graph.nodes,
                "merge": GraphNode(
                    "dinkster.3d.merge_splat",
                    {key: value for key, value in inputs.items() if key != missing},
                ),
            }
        )
        assert [
            (item.code, item.input_id) for item in validate(incomplete, schemas, ("merge",))
        ] == [("missing-input", missing)]

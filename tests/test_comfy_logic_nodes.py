"""Native logic preserves truthiness, short circuit, and lazy branch demand."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from dinkster_api.v1 import Item, Members
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral, migrate_node_replacements
from dinkster_schema import build_node_types, build_schemas
from dinkster_utilities.logic import ComfyAndNode, ComfyNotNode, ComfyOrNode, CustomCombo
from dinkster_utilities.primitive import PrimitiveInt, PrimitiveString
from dinkster_utilities.switch import Switch
from dinkster_values import TypeRegistry, is_absent, register_core_types
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario


def _members(values: Mapping[str, object]) -> Members[Item]:
    return Members(Item(key, value) for key, value in values.items())


@pytest.mark.parametrize(("choice", "index"), [("Final", 2), ("Quick", 0), ("Detailed", 1)])
def test_custom_combo_index_is_the_choice_position_among_its_options(
    choice: str, index: int
) -> None:
    options = ["Quick", "Detailed", "Final", "Detailed"]
    extras = {"index": 7, "2": "unused", "label": 5}
    assert CustomCombo.execute(choice=choice, options=options, **extras) == {
        "STRING": choice,
        "INDEX": index,
    }


@pytest.mark.parametrize("options", [["Quick", "Detailed"], []])
def test_custom_combo_refuses_a_choice_that_is_not_one_of_its_options(options: list[str]) -> None:
    with pytest.raises(ValueError, match=r'^Custom Combo\'s choice "Fast" is not one of its'):
        CustomCombo.execute(choice="Fast", options=options, index=0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), (0, True), ([], True), ({}, True), ("False", False), (-3, False), ([0], False)],
)
def test_not_uses_truthiness(value: object, expected: bool) -> None:
    assert ComfyNotNode.execute(value=value) == {"BOOLEAN": expected}


@pytest.mark.parametrize(
    ("values", "conjunction", "disjunction"),
    [
        ({}, True, False),
        ({"value0": "False", "value1": -3}, True, True),
        ({"value0": [0], "value1": []}, False, True),
        ({"value0": {}, "value1": 0}, False, False),
        ({"value0": False, "value1": True, "value2": False}, False, True),
    ],
)
def test_mapping_values_determine_truth(
    values: dict[str, object], conjunction: bool, disjunction: bool
) -> None:
    assert ComfyAndNode.execute(values=_members(values)) == {"BOOLEAN": conjunction}
    assert ComfyOrNode.execute(values=_members(values)) == {"BOOLEAN": disjunction}


@pytest.mark.parametrize("source", ["ComfySwitchNode", "dinkster.utilities.comfy_switch_node"])
@pytest.mark.parametrize("switch", [False, True])
@pytest.mark.parametrize("connected", [False, True])
@pytest.mark.parametrize("cache_enabled", [False, True])
def test_an_if_else_switch_opens_as_the_switch_and_demands_only_the_selected_arm(
    source: str, switch: bool, connected: bool, cache_enabled: bool
) -> None:
    async def scenario() -> None:
        nodes = [Switch, PrimitiveInt]
        schemas = build_schemas(nodes)
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        selected = "on_true" if switch else "on_false"
        unselected = "on_false" if switch else "on_true"
        inputs: dict[str, object] = {"switch": switch, unselected: Link("unused", "INT")}
        if connected:
            inputs[selected] = Link("selected", "INT")
        graph = migrate_node_replacements(
            Graph(
                {
                    "unused": GraphNode("dinkster.utilities.primitive_int", {"value": 29}),
                    "selected": GraphNode("dinkster.utilities.primitive_int", {"value": -17}),
                    "switch": GraphNode(source, inputs),
                }
            ),
            schemas,
        )
        opened = graph.nodes["switch"]
        assert isinstance(opened, GraphNode)
        assert opened.node_type == "dinkster.utilities.switch"
        assert opened.inputs["pick"] == TypedLiteral("STRING", "branch2" if switch else "branch1")
        for _ in range(2):
            result = await engine.run(graph, ["switch"], cache_enabled=cache_enabled)
            value = result.outputs["switch"]["value"]
            if connected:
                assert value.resolve() == -17
            else:
                assert is_absent(value)
            assert "unused" not in result.executed
            assert "selected" not in result.executed or connected

    run_scenario(scenario())


@pytest.mark.parametrize("switch", [False, True])
@pytest.mark.parametrize("connected", [False, True])
@pytest.mark.parametrize("cache_enabled", [False, True])
def test_switch_routes_a_yes_or_no_to_the_second_or_first_branch(
    switch: bool, connected: bool, cache_enabled: bool
) -> None:
    async def scenario() -> None:
        nodes = [Switch, PrimitiveInt]
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        selected = "branch.branch2" if switch else "branch.branch1"
        unselected = "branch.branch1" if switch else "branch.branch2"
        inputs: dict[str, object] = {
            "pick": TypedLiteral("BOOLEAN", switch),
            unselected: Link("unused", "INT"),
        }
        if connected:
            inputs[selected] = Link("selected", "INT")
        graph = Graph(
            {
                "unused": GraphNode("dinkster.utilities.primitive_int", {"value": 29}),
                "selected": GraphNode("dinkster.utilities.primitive_int", {"value": -17}),
                "switch": GraphNode(
                    "dinkster.utilities.switch",
                    inputs,
                    output_members={"branch": ("branch1", "branch2")},
                ),
            }
        )
        for _ in range(2):
            result = await engine.run(graph, ["switch"], cache_enabled=cache_enabled)
            value = result.outputs["switch"]["value"]
            if connected:
                assert value.resolve() == -17
            else:
                assert is_absent(value)
            assert "unused" not in result.executed
            assert "selected" not in result.executed or connected

    run_scenario(scenario())


@pytest.mark.parametrize(("text", "expected"), [("Detailed", 37), ("branch2", 5), ("Quick", 11)])
def test_switch_picks_the_branch_a_text_wire_names(text: str, expected: int) -> None:
    async def scenario() -> None:
        nodes = [Switch, PrimitiveInt, PrimitiveString]
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        values = {"branch3": 11, "branch1": 37, "branch2": 5}
        graph = Graph(
            {
                **{
                    key: GraphNode("dinkster.utilities.primitive_int", {"value": value})
                    for key, value in values.items()
                },
                "text": GraphNode("dinkster.utilities.primitive_string", {"value": text}),
                "switch": GraphNode(
                    "dinkster.utilities.switch",
                    {
                        "pick": Link("text", "STRING"),
                        "names": TypedLiteral("list<STRING>", ["Quick", "Detailed", "Final"]),
                        **{f"branch.{key}": Link(key, "INT") for key in values},
                    },
                    output_members={"branch": tuple(values)},
                ),
            }
        )
        result = await engine.run(graph, ["switch"])
        assert result.outputs["switch"]["value"].resolve() == expected
        assert [key for key in values if key in result.executed] == [
            key for key, value in values.items() if value == expected
        ]

    run_scenario(scenario())

"""Native loop expansion executes body occurrences through the engine."""

import asyncio

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, RegionNode, RegionOutput, TypedLiteral
from dinkster_schema import (
    ABSENT,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_utilities import NODES as UTILITY_NODES
from dinkster_utilities.loop_lists import ListElement, MakeList
from dinkster_utilities.math import ComfyMathExpression
from dinkster_utilities.primitive import PrimitiveBoolean, PrimitiveInt
from dinkster_utilities.switch import ComfySwitchNode
from dinkster_values import TypeRegistry, is_absent, register_core_types
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario

LOOP_NODES = tuple(n for n in UTILITY_NODES if n.__module__ == "dinkster_utilities.loops")

NODES = (
    *LOOP_NODES,
    ComfyMathExpression,
    PrimitiveInt,
    PrimitiveBoolean,
    ComfySwitchNode,
    MakeList,
    ListElement,
)


def make_engine() -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=build_schemas(NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=MemoryLRUCache(),
    )


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize(
    ("count", "accumulate", "expected"),
    [
        (0, False, []),
        (4, False, [10]),
        (4, True, [1, 3, 6, 10]),
    ],
)
def test_carried_value_and_empty_loop(
    cache_enabled: bool, count: int, accumulate: bool, expected: list[int]
) -> None:
    graph = Graph(
        {
            "start": GraphNode(
                "dinkster.utilities.start_loop",
                {
                    "mode.num_iterations": count,
                    "initial_iteration_value": TypedLiteral("INT", 0),
                },
                slot_variants={"mode": "simple"},
            ),
            "body": GraphNode(
                "dinkster.utilities.comfy_math_expression",
                {
                    "expression": "a + b + 1",
                    "values.a": Link("start", "current_iteration_value"),
                    "values.b": Link("start", "iteration_index"),
                },
            ),
            "end": GraphNode(
                "dinkster.utilities.end_loop",
                {
                    "output_value": Link("body", "INT"),
                    "next_iteration_value": Link("body", "INT"),
                    "accumulate": accumulate,
                },
            ),
        }
    )

    async def scenario() -> None:
        engine = make_engine()
        for _ in range(2):
            result = await asyncio.wait_for(
                engine.run(graph, ["end"], cache_enabled=cache_enabled), 10
            )
            assert result.outputs["end"]["outputs"].resolve() == expected

    run_scenario(scenario())


@pytest.mark.parametrize("inner_cache", [False, True])
def test_nested_loop_keeps_each_occurrences_results(inner_cache: bool) -> None:
    graph = Graph(
        {
            "outer": GraphNode(
                "dinkster.utilities.start_loop",
                {"mode.num_iterations": 2},
                slot_variants={"mode": "simple"},
            ),
            "inner": GraphNode(
                "dinkster.utilities.start_loop",
                {
                    "mode.num_iterations": 3,
                    "parent_iteration": Link("outer", "iteration_index"),
                    "cache_iterations": inner_cache,
                },
                slot_variants={"mode": "simple"},
            ),
            "body": GraphNode(
                "dinkster.utilities.comfy_math_expression",
                {
                    "expression": "a * 10 + b",
                    "values.a": Link("outer", "iteration_index"),
                    "values.b": Link("inner", "iteration_index"),
                },
            ),
            "inner_end": GraphNode(
                "dinkster.utilities.end_loop",
                {"output_value": Link("body", "INT"), "accumulate": True},
            ),
            "outer_end": GraphNode(
                "dinkster.utilities.end_loop",
                {"output_value": Link("inner_end", "outputs"), "accumulate": True},
            ),
        }
    )

    async def scenario() -> None:
        engine = make_engine()
        for run in range(2):
            result = await asyncio.wait_for(engine.run(graph, ["outer_end"]), 10)
            assert result.outputs["outer_end"]["outputs"].resolve() == [[0, 1, 2], [10, 11, 12]]
            executed_body = [node for node in result.executed if node.endswith("_body")]
            assert len(executed_body) == (0 if run and inner_cache else 6)

    run_scenario(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("absent_input", ["output_value", "terminations.0"])
def test_absent_loop_dependency_completes(cache_enabled: bool, absent_input: str) -> None:
    class Empty(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            integer = TypeExpr.concrete("INT")
            return NodeSchema(
                "test.empty",
                inputs=(InputSpec("index", integer),),
                outputs=(OutputSpec("value", integer, optional=True),),
            )

        @classmethod
        def execute(cls, index: int):
            return cls.outputs(value=ABSENT)

    nodes = (*NODES, Empty)
    registry = TypeRegistry()
    register_core_types(registry)
    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
    )
    graph = Graph(
        {
            "start": GraphNode(
                "dinkster.utilities.start_loop",
                {"mode.num_iterations": 1},
                slot_variants={"mode": "simple"},
            ),
            "empty": GraphNode("test.empty", {"index": Link("start", "iteration_index")}),
            "end": GraphNode(
                "dinkster.utilities.end_loop",
                {
                    "output_value": Link("start", "iteration_index"),
                    absent_input: Link("empty", "value"),
                },
            ),
        }
    )

    async def scenario() -> None:
        for _ in range(2):
            result = await asyncio.wait_for(
                engine.run(graph, ["end"], cache_enabled=cache_enabled), 3
            )
            output = result.outputs["end"]["outputs"]
            if absent_input == "output_value":
                assert is_absent(output)
            else:
                assert output.resolve() == [0]

    run_scenario(scenario())


def test_loop_inside_map_can_override_its_end() -> None:
    body = Graph(
        {
            "start": GraphNode(
                "dinkster.utilities.start_loop",
                {"mode.num_iterations": Link("$region", "count")},
                slot_variants={"mode": "simple"},
            ),
            "end": GraphNode(
                "dinkster.utilities.end_loop",
                {"output_value": Link("start", "iteration_index"), "accumulate": True},
            ),
        }
    )
    graph = Graph(
        {
            "region": RegionNode(
                kind="map",
                ports={"count": TypeExpr.concrete("INT")},
                inputs={"count": [2]},
                element_ports=("count",),
                body=body,
                outputs={"values": RegionOutput(Link("end", "outputs"))},
            )
        }
    )
    result = run_scenario(asyncio.wait_for(make_engine().run(graph, ["region"]), 3))
    assert result.outputs["region"]["values"].resolve() == [[0, 1]]

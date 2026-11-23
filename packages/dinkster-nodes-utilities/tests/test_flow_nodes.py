"""Or else, Has value and Only if treat only nothing as nothing and ask for
an input only when they need it."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import pytest
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_INT,
    CORE_STRING,
    AbsentOutput,
    Input,
    Node,
    NodeSchema,
    Output,
    TypeExpr,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionError, GraphValidationError, RunResult
from dinkster_graph import Graph, GraphNode, Link
from dinkster_schema import build_node_types, build_schemas
from dinkster_utilities.flow import HasValue, OnlyIf, OrElse
from dinkster_values import TypeRegistry, is_absent, register_core_types
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario

INT = TypeExpr.concrete(CORE_INT)


class Produce(Node):
    """An INT whose runs are recorded by node label."""

    ran: ClassVar[list[str]] = []

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.produce",
            body=(
                Input("label", TypeExpr.concrete(CORE_STRING)),
                Input("value", INT),
                Output("value", INT),
            ),
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, label: str, value: int) -> Mapping[str, object]:
        cls.ran.append(label)
        return cls.outputs(value=value)


class Nothing(Node):
    """Gives nothing on purpose."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.nothing",
            body=(Output("value", INT, optional=True),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(value=AbsentOutput("no mask"))


class Fail(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(node_type="test.fail", body=(Output("value", INT),))

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        raise RuntimeError("producer broke")


class Inpaint(Node):
    """Needs a mask; skipped when the mask is nothing."""

    ran: ClassVar[list[int]] = []

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.inpaint",
            body=(Input("image", INT), Input("mask", INT), Output("image", INT)),
        )

    @classmethod
    def execute(cls, *, image: int, mask: int) -> Mapping[str, object]:
        cls.ran.append(image)
        return cls.outputs(image=image * 1000 + mask)


class Literal(Node):
    """One stored value of a chosen type, for 0, No, empty text and an empty list."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.literal",
            body=(
                Input("kind", TypeExpr.concrete(CORE_STRING)),
                Output("int", INT, optional=True),
                Output("bool", TypeExpr.concrete(CORE_BOOLEAN), optional=True),
                Output("text", TypeExpr.concrete(CORE_STRING), optional=True),
                Output("list", TypeExpr.list_of(INT), optional=True),
            ),
        )

    @classmethod
    def execute(cls, *, kind: str) -> Mapping[str, object]:
        values: dict[str, object] = {
            "int": AbsentOutput(),
            "bool": AbsentOutput(),
            "text": AbsentOutput(),
            "list": AbsentOutput(),
        }
        values[kind] = {"int": 0, "bool": False, "text": "", "list": []}[kind]
        return cls.outputs(**values)


NODES = (OrElse, HasValue, OnlyIf, Produce, Nothing, Fail, Inpaint, Literal)


@pytest.fixture(autouse=True)
def _reset_runs() -> None:
    Produce.ran.clear()
    Inpaint.ran.clear()


def _engine() -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=build_schemas(NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=MemoryLRUCache(),
    )


def _produce(label: str, value: int) -> GraphNode:
    return GraphNode("test.produce", {"label": label, "value": value})


def _run(nodes: Mapping[str, GraphNode], targets: list[str]) -> RunResult:
    return run_scenario(_engine().run(Graph(dict(nodes)), targets))


def _reason(result: RunResult, node: str, output: str = "value") -> str:
    value = result.outputs[node][output]
    assert is_absent(value)
    return str(value.meta.get("absent.reason"))


# -- Or else ---------------------------------------------------------------------


def test_or_else_stops_at_a_first_input_of_zero() -> None:
    result = _run(
        {
            "first": _produce("first", 0),
            "next": _produce("next", 2),
            "or_else": GraphNode(
                "dinkster.utilities.or_else",
                {"values.1": Link("first", "value"), "values.2": Link("next", "value")},
            ),
        },
        ["or_else"],
    )
    assert result.outputs["or_else"]["value"].resolve() == 0
    assert Produce.ran == ["first"]


def test_or_else_moves_past_nothing_and_past_an_input_without_a_wire() -> None:
    result = _run(
        {
            "nothing": GraphNode("test.nothing"),
            "third": _produce("third", 3),
            "fourth": _produce("fourth", 4),
            "or_else": GraphNode(
                "dinkster.utilities.or_else",
                {
                    "values.1": Link("nothing", "value"),
                    "values.3": Link("third", "value"),
                    "values.4": Link("fourth", "value"),
                },
            ),
        },
        ["or_else"],
    )
    assert result.outputs["or_else"]["value"].resolve() == 3
    assert Produce.ran == ["third"]


def test_or_else_follows_the_stored_order_not_the_key_numbers() -> None:
    result = _run(
        {
            "a": _produce("a", 10),
            "b": _produce("b", 20),
            "or_else": GraphNode(
                "dinkster.utilities.or_else",
                {"values.2": Link("b", "value"), "values.1": Link("a", "value")},
            ),
        },
        ["or_else"],
    )
    assert result.outputs["or_else"]["value"].resolve() == 20
    assert Produce.ran == ["b"]


def test_a_losing_input_still_runs_for_another_node_that_needs_it() -> None:
    result = _run(
        {
            "first": _produce("first", 1),
            "next": _produce("next", 2),
            "or_else": GraphNode(
                "dinkster.utilities.or_else",
                {"values.1": Link("first", "value"), "values.2": Link("next", "value")},
            ),
        },
        ["or_else", "next"],
    )
    assert result.outputs["or_else"]["value"].resolve() == 1
    assert sorted(Produce.ran) == ["first", "next"]


def test_a_failure_behind_the_first_input_fails_the_run_without_falling_back() -> None:
    with pytest.raises(ExecutionError) as failed:
        _run(
            {
                "broken": GraphNode("test.fail"),
                "next": _produce("next", 2),
                "or_else": GraphNode(
                    "dinkster.utilities.or_else",
                    {"values.1": Link("broken", "value"), "values.2": Link("next", "value")},
                ),
            },
            ["or_else"],
        )
    assert "producer broke" in failed.value.error.message
    assert Produce.ran == []


def test_every_input_nothing_gives_nothing_with_the_reason_and_is_cached() -> None:
    engine = _engine()
    graph = Graph(
        {
            "nothing": GraphNode("test.nothing"),
            "or_else": GraphNode(
                "dinkster.utilities.or_else", {"values.1": Link("nothing", "value")}
            ),
        }
    )
    first = run_scenario(engine.run(graph, ["or_else"]))
    assert _reason(first, "or_else") == "No input has a value"
    assert "or_else" in first.executed
    assert first.skipped == ()
    again = run_scenario(engine.run(graph, ["or_else"]))
    assert "or_else" in again.cached


# -- Has value -----------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["int", "bool", "text", "list"])
def test_zero_no_empty_text_and_an_empty_list_have_a_value(kind: str) -> None:
    result = _run(
        {
            "literal": GraphNode("test.literal", {"kind": kind}),
            "has": GraphNode("dinkster.utilities.has_value", {"value": Link("literal", kind)}),
        },
        ["has"],
    )
    assert result.outputs["has"]["has_value"].resolve() is True


def test_nothing_and_no_wire_have_no_value() -> None:
    wired = _run(
        {
            "nothing": GraphNode("test.nothing"),
            "has": GraphNode("dinkster.utilities.has_value", {"value": Link("nothing", "value")}),
        },
        ["has"],
    )
    assert wired.outputs["has"]["has_value"].resolve() is False
    assert "has" in wired.executed
    unwired = _run({"has": GraphNode("dinkster.utilities.has_value")}, ["has"])
    assert unwired.outputs["has"]["has_value"].resolve() is False


def test_has_value_does_not_turn_a_failure_into_no() -> None:
    with pytest.raises(ExecutionError) as failed:
        _run(
            {
                "broken": GraphNode("test.fail"),
                "has": GraphNode(
                    "dinkster.utilities.has_value", {"value": Link("broken", "value")}
                ),
            },
            ["has"],
        )
    assert "producer broke" in failed.value.error.message


# -- Only if -----------------------------------------------------------------------


def test_condition_no_gives_nothing_without_asking_for_value() -> None:
    result = _run(
        {
            "upscale": _produce("upscale", 4),
            "only_if": GraphNode(
                "dinkster.utilities.only_if",
                {"value": Link("upscale", "value"), "condition": False},
            ),
        },
        ["only_if"],
    )
    assert _reason(result, "only_if") == "Condition is false"
    assert "only_if" in result.executed
    assert Produce.ran == []


def test_condition_yes_passes_value_on_unchanged() -> None:
    result = _run(
        {
            "upscale": _produce("upscale", 4),
            "only_if": GraphNode(
                "dinkster.utilities.only_if",
                {"value": Link("upscale", "value"), "condition": True},
            ),
        },
        ["only_if", "upscale"],
    )
    assert result.outputs["only_if"]["value"] == result.outputs["upscale"]["value"]


def test_condition_nothing_skips_only_if_instead_of_reading_no() -> None:
    result = _run(
        {
            "nothing": GraphNode("test.nothing"),
            "upscale": _produce("upscale", 4),
            "only_if": GraphNode(
                "dinkster.utilities.only_if",
                {"value": Link("upscale", "value"), "condition": Link("nothing", "value")},
            ),
        },
        ["only_if"],
    )
    assert result.skipped == ("only_if",)
    assert is_absent(result.outputs["only_if"]["value"])
    assert Produce.ran == []


def test_condition_yes_with_value_unplugged_gives_nothing_without_an_error() -> None:
    result = _run(
        {"only_if": GraphNode("dinkster.utilities.only_if", {"condition": True})}, ["only_if"]
    )
    assert is_absent(result.outputs["only_if"]["value"])
    assert "only_if" in result.executed


def test_two_copies_with_different_conditions_give_different_outputs() -> None:
    result = _run(
        {
            "image": _produce("image", 7),
            "on": GraphNode(
                "dinkster.utilities.only_if", {"value": Link("image", "value"), "condition": True}
            ),
            "off": GraphNode(
                "dinkster.utilities.only_if", {"value": Link("image", "value"), "condition": False}
            ),
        },
        ["on", "off"],
    )
    assert result.outputs["on"]["value"].resolve() == 7
    assert is_absent(result.outputs["off"]["value"])


# -- Named acceptance ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mask", "expected", "inpainted"),
    [("mask", 5009, [5]), ("no_mask", 5, [])],
    ids=["mask plugged in", "mask is nothing"],
)
def test_a_stage_that_can_be_left_out(mask: str, expected: int, inpainted: list[int]) -> None:
    result = _run(
        {
            "load": _produce("load", 5),
            "mask": _produce("mask", 9),
            "no_mask": GraphNode("test.nothing"),
            "inpaint": GraphNode(
                "test.inpaint", {"image": Link("load", "value"), "mask": Link(mask, "value")}
            ),
            "or_else": GraphNode(
                "dinkster.utilities.or_else",
                {"values.1": Link("inpaint", "image"), "values.2": Link("load", "value")},
            ),
        },
        ["or_else"],
    )
    assert result.outputs["or_else"]["value"].resolve() == expected
    assert Inpaint.ran == inpainted


@pytest.mark.parametrize(
    ("condition", "expected", "skipped"), [(False, 5, ("inpaint",)), (True, 5009, ())]
)
def test_a_mask_behind_only_if_saves_the_original_when_condition_is_no(
    condition: bool, expected: int, skipped: tuple[str, ...]
) -> None:
    result = _run(
        {
            "load": _produce("load", 5),
            "mask": _produce("mask", 9),
            "only_if": GraphNode(
                "dinkster.utilities.only_if",
                {"value": Link("mask", "value"), "condition": condition},
            ),
            "inpaint": GraphNode(
                "test.inpaint", {"image": Link("load", "value"), "mask": Link("only_if", "value")}
            ),
            "or_else": GraphNode(
                "dinkster.utilities.or_else",
                {"values.1": Link("inpaint", "image"), "values.2": Link("load", "value")},
            ),
        },
        ["or_else"],
    )
    assert result.outputs["or_else"]["value"].resolve() == expected
    assert result.skipped == skipped


def test_an_unplugged_mask_still_refuses_the_run() -> None:
    # A muted producer compiles as no wire, so this also covers a muted mask.
    with pytest.raises(GraphValidationError, match="missing-input"):
        _run(
            {
                "load": _produce("load", 5),
                "inpaint": GraphNode("test.inpaint", {"image": Link("load", "value")}),
                "or_else": GraphNode(
                    "dinkster.utilities.or_else",
                    {"values.1": Link("inpaint", "image"), "values.2": Link("load", "value")},
                ),
            },
            ["or_else"],
        )
    assert Produce.ran == []


@pytest.mark.parametrize(("condition", "expected", "upscaled"), [(False, 3, []), (True, 3000, [3])])
def test_a_stage_behind_a_switch(condition: bool, expected: int, upscaled: list[int]) -> None:
    result = _run(
        {
            "load": _produce("load", 3),
            "upscale": GraphNode("test.inpaint", {"image": Link("load", "value"), "mask": 0}),
            "only_if": GraphNode(
                "dinkster.utilities.only_if",
                {"value": Link("upscale", "image"), "condition": condition},
            ),
            "or_else": GraphNode(
                "dinkster.utilities.or_else",
                {"values.1": Link("only_if", "value"), "values.2": Link("load", "value")},
            ),
        },
        ["or_else"],
    )
    assert result.outputs["or_else"]["value"].resolve() == expected
    assert Inpaint.ran == upscaled

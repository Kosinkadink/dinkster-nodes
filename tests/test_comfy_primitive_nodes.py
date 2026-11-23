"""Primitive nodes preserve the supplied value, without numeric or text coercion."""

from __future__ import annotations

import math
import sys
from typing import cast

import pytest
from dinkster_api.v1 import schema_to_wire
from dinkster_utilities.primitive import (
    PrimitiveBoolean,
    PrimitiveFloat,
    PrimitiveInt,
    PrimitiveString,
)


@pytest.mark.parametrize("value", ["", "  text\r\nnext\t", "\U0001f600e\u0301"])
def test_text_preserves_whitespace_and_codepoints(value: str) -> None:
    assert PrimitiveString.execute(value=value) == {"STRING": value}


def test_text_is_one_line_or_multiline_and_opens_comfy_multiline_text() -> None:
    body = cast(list[dict[str, object]], schema_to_wire(PrimitiveString.schema())["body"])
    widget = cast(dict[str, object], body[0]["widget"])
    assert widget["default"] == "single_line"
    assert widget["userSwitchable"] is True
    assert [
        (r["id"], r["widget"]) for r in cast(list[dict[str, object]], widget["representations"])
    ] == [
        ("single_line", {"type": "STRING", "multiline": False}),
        ("multiline", {"type": "STRING", "multiline": True}),
    ]
    sources = {rule.from_type for rule in PrimitiveString.schema().replacements}
    assert sources == {"PrimitiveString", "PrimitiveStringMultiline"}


@pytest.mark.parametrize(
    ("node", "terms", "limits"),
    [
        (PrimitiveInt, ("integer", "number"), ["minimum", "maximum", "step", "default"]),
        (PrimitiveFloat, ("number", "decimal"), ["minimum", "maximum", "step", "default"]),
        (PrimitiveString, ("string",), ["default"]),
        (PrimitiveBoolean, ("bool", "toggle"), ["default"]),
    ],
)
def test_typed_primitive_is_found_by_its_words_and_declares_its_limits(
    node: type[PrimitiveInt], terms: tuple[str, ...], limits: list[str]
) -> None:
    schema = node.schema()
    assert schema.search_terms[: len(terms)] == terms
    wire = schema_to_wire(schema)
    body = cast(list[dict[str, object]], wire["body"])
    value_type = cast(dict[str, object], body[0]["type"])
    section = body[1]
    assert (section["kind"], section["label"], section["collapsed"]) == ("section", "Limits", True)
    rows = cast(list[dict[str, object]], section["rows"])
    assert [row["id"] for row in rows] == limits
    assert all(row["required"] is False and row["type"] == value_type for row in rows)
    output = body[2]
    assert output["knownValue"] == {"input": "value"}
    assert output["type"] == value_type


def test_limits_do_not_change_the_value() -> None:
    assert PrimitiveInt.execute(value=20, minimum=0, maximum=8, step=1, default=4) == {"INT": 20}


@pytest.mark.parametrize("value", [-sys.maxsize, -17, 0, 2**53 + 1, sys.maxsize])
def test_integer_preserves_large_and_negative_values(value: int) -> None:
    result = PrimitiveInt.execute(value=value)
    assert result == {"INT": value}
    assert type(result["INT"]) is int


@pytest.mark.parametrize("value", [-0.0, -0.125, 0.1, 1e100])
def test_float_preserves_fraction_and_sign(value: float) -> None:
    result = PrimitiveFloat.execute(value=value)
    assert result == {"FLOAT": value}
    assert type(result["FLOAT"]) is float
    assert math.copysign(1.0, result["FLOAT"]) == math.copysign(1.0, value)


@pytest.mark.parametrize("value", [True, False])
def test_boolean_remains_boolean(value: bool) -> None:
    result = PrimitiveBoolean.execute(value=value)
    assert result == {"BOOLEAN": value}
    assert type(result["BOOLEAN"]) is bool


@pytest.mark.parametrize("node", [PrimitiveInt, PrimitiveFloat])
def test_numeric_controller_and_bounds_survive_serialization(
    node: type[PrimitiveInt] | type[PrimitiveFloat],
) -> None:
    wire = schema_to_wire(node.schema())
    body = cast(list[dict[str, object]], wire["body"])
    assert body[0]["widget"] == {
        "type": "NUMBER",
        "min": "-9223372036854775807" if node is PrimitiveInt else -9.223372036854776e18,
        "max": "9223372036854775807" if node is PrimitiveInt else 9.223372036854776e18,
        **({} if node is PrimitiveInt else {"step": 0.1}),
    }
    assert body[0].get("afterRun") == ("fixed" if node is PrimitiveInt else None)

"""Primitive value nodes: Primitive, Int, Float, Text, Boolean and Choice.

They are one implementation found under several names in the add-node
search. Each holds one value field and a Limits section of ordinary optional
fields of its own: minimum, maximum, step and default for Int and Float, and
default for the others. The output's known value names these fields as its
limits; an empty limit is derived by the editor from the fields the
primitive feeds. Choice takes its options, and Primitive its type,
label and field, from the fields they feed. The value is known before the
run, so the editor writes it into the fields it feeds instead of running the
node.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import replace

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    Input,
    Node,
    NodeSchema,
    NumberWidget,
    Output,
    OutputKnownValue,
    Section,
    StringWidget,
    TypeExpr,
    WidgetRepresentation,
    WidgetRepresentations,
    with_identity_replacement,
)

_INT_WIDGET = NumberWidget(min=-sys.maxsize, max=sys.maxsize)
_FLOAT_WIDGET = NumberWidget(min=-float(sys.maxsize), max=float(sys.maxsize), step=0.1)
_TEXT_WIDGET = StringWidget(multiline=False)
_TEXT_REPRESENTATIONS = WidgetRepresentations(
    representations=(
        WidgetRepresentation("single_line", _TEXT_WIDGET, "Single line"),
        WidgetRepresentation("multiline", StringWidget(multiline=True), "Multiline"),
    ),
    default="single_line",
    user_switchable=True,
)


_LIMIT_KEYS = {"minimum": "min", "maximum": "max", "step": "step", "default": "default"}


def _limit(
    id: str, label: str, type_id: str | TypeExpr, widget: NumberWidget | StringWidget | None
) -> Input:
    value_type = TypeExpr.concrete(type_id) if isinstance(type_id, str) else type_id
    return Input(id, value_type, required=False, widget=widget, label=label)


def _primitive_schema(
    node_type: str,
    display_name: str,
    output: Output,
    value: Input,
    limits: tuple[Input, ...],
    search_terms: tuple[str, ...],
    comfy_types: tuple[str, ...] = (),
) -> NodeSchema:
    section = (Section("limits", limits, label="Limits", collapsed=True),) if limits else ()
    known_value = OutputKnownValue(
        "value", limits={_LIMIT_KEYS[limit.id]: limit.id for limit in limits}
    )
    schema = NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category="utilities/primitive",
        search_terms=search_terms,
        body=(value, *section, replace(output, known_value=known_value)),
    )
    if not comfy_types:
        return schema
    declared = [with_identity_replacement(schema, comfy_type) for comfy_type in comfy_types]
    return replace(
        declared[-1],
        search_terms=tuple(dict.fromkeys(t for d in declared for t in d.search_terms)),
        replacements=tuple(rule for d in declared for rule in d.replacements),
    )


class _Primitive(Node):
    """The one implementation: the output is the value field."""

    @classmethod
    def execute(cls, *, value: object, **_limits: object) -> Mapping[str, object]:
        return cls.outputs(**{cls.schema().outputs[0].id: value})


def _number_limits(type_id: str, widget: NumberWidget) -> tuple[Input, ...]:
    step = NumberWidget(min=1, max=sys.maxsize) if type_id == CORE_INT else NumberWidget(min=0.0)
    return (
        _limit("minimum", "Minimum", type_id, widget),
        _limit("maximum", "Maximum", type_id, widget),
        _limit("step", "Step", type_id, step),
        _limit("default", "Default", type_id, widget),
    )


class PrimitiveInt(_Primitive):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _primitive_schema(
            "dinkster.utilities.primitive_int",
            "Int",
            Output(CORE_INT, TypeExpr.concrete(CORE_INT)),
            Input("value", TypeExpr.concrete(CORE_INT), widget=_INT_WIDGET, after_run="fixed"),
            _number_limits(CORE_INT, _INT_WIDGET),
            ("integer", "number"),
            ("PrimitiveInt",),
        )


class PrimitiveFloat(_Primitive):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _primitive_schema(
            "dinkster.utilities.primitive_float",
            "Float",
            Output(CORE_FLOAT, TypeExpr.concrete(CORE_FLOAT)),
            Input("value", TypeExpr.concrete(CORE_FLOAT), widget=_FLOAT_WIDGET),
            _number_limits(CORE_FLOAT, _FLOAT_WIDGET),
            ("number", "decimal"),
            ("PrimitiveFloat",),
        )


class PrimitiveString(_Primitive):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        # ComfyUI's multiline text node is this node with the multiline representation.
        return _primitive_schema(
            "dinkster.utilities.primitive_string",
            "Text",
            Output(CORE_STRING, TypeExpr.concrete(CORE_STRING)),
            Input("value", TypeExpr.concrete(CORE_STRING), widget=_TEXT_REPRESENTATIONS),
            (_limit("default", "Default", CORE_STRING, _TEXT_WIDGET),),
            ("string",),
            ("PrimitiveString", "PrimitiveStringMultiline"),
        )


class PrimitiveBoolean(_Primitive):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _primitive_schema(
            "dinkster.utilities.primitive_boolean",
            "Boolean",
            Output(CORE_BOOLEAN, TypeExpr.concrete(CORE_BOOLEAN)),
            Input("value", TypeExpr.concrete(CORE_BOOLEAN)),
            (_limit("default", "Default", CORE_BOOLEAN, None),),
            ("bool", "toggle"),
            ("PrimitiveBoolean",),
        )


class PrimitiveChoice(_Primitive):
    """A choice whose options are those of the dropdown fields it feeds."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _primitive_schema(
            "dinkster.utilities.primitive_choice",
            "Choice",
            Output(CORE_COMBO, TypeExpr.concrete(CORE_COMBO)),
            Input("value", TypeExpr.concrete(CORE_COMBO)),
            (_limit("default", "Default", CORE_COMBO, None),),
            ("dropdown", "combo", "select", "options"),
        )


class Primitive(_Primitive):
    """A value of the type of the first field it feeds, shown with that
    field's label and widget; a color or an asset as well as a number or a
    text. The stored value carries its type."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        value_type = TypeExpr.variable("value")
        return _primitive_schema(
            "dinkster.utilities.primitive",
            "Primitive",
            Output("value", value_type, label="Value"),
            Input("value", value_type, label="Value"),
            (_limit("default", "Default", value_type, None),),
            ("value", "constant"),
        )

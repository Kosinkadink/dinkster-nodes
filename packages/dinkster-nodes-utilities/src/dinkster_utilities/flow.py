"""Nodes that let a graph handle nothing on purpose.

Only nothing counts as nothing: 0, false, empty text and an empty list are
values. An input with no wire counts as nothing on these nodes.
"""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    ABSENT,
    CORE_BOOLEAN,
    CORE_INT,
    AbsentOutput,
    Ask,
    BooleanWidget,
    Button,
    Count,
    Formula,
    Group,
    Input,
    Item,
    Members,
    Node,
    NodeSchema,
    Note,
    Numbered,
    NumberWidget,
    Output,
    Rule,
    Set,
    TypeExpr,
    Value,
    list_children,
    member_id,
    report_log,
)

T = TypeExpr.variable("T")
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)


_ABSENT_TYPE = "core.absent"


def _has_value(input_values: Mapping[str, Value], input_id: str) -> bool:
    value = input_values.get(input_id)
    return value is not None and value.type_id != _ABSENT_TYPE


class OrElse(Node):
    """The first input, top to bottom, that has a value.

    Inputs are asked for one at a time, so an input below the winner is never
    asked for and a node that only feeds it does not run."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.utilities.or_else",
            display_name="Or else",
            category="utilities/flow",
            description="Gives the first input, top to bottom, that has a value.",
            body=(
                Group(
                    "values",
                    value=Input("value", T, required=False, on_absent="accept", lazy=True),
                    keys=Numbered("", start=1),
                    grows="button",
                    min_members=2,
                    reorderable=True,
                    add_label="Add input",
                    member_label=Formula('"First" if index() == 0 else "Next"'),
                    open_name=False,
                ),
                Output("value", T, optional=True, label="Value"),
            ),
            search_terms=("coalesce", "fallback", "first non-empty", "default"),
            input_values=True,
        )

    @classmethod
    def check_lazy_status(
        cls, *, values: Members[Item], input_values: Mapping[str, Value]
    ) -> tuple[str, ...]:
        for member in values:
            input_id = member_id("values", member.key)
            if input_id not in input_values:
                return (input_id,)
            if _has_value(input_values, input_id):
                return ()
        return ()

    @classmethod
    def execute(
        cls, *, values: Members[Item], input_values: Mapping[str, Value]
    ) -> Mapping[str, object]:
        for member in values:
            input_id = member_id("values", member.key)
            if _has_value(input_values, input_id):
                return cls.outputs(value=input_values[input_id])
        return cls.outputs(value=AbsentOutput("No input has a value"))


class HasValue(Node):
    """Yes when the input has a value, No when it is nothing or has no wire."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.utilities.has_value",
            display_name="Has value",
            category="utilities/flow",
            description="Yes when the input has a value, No when it is nothing.",
            body=(
                Input("value", T, required=False, on_absent="accept", label="Value"),
                Output("has_value", BOOLEAN, label="Has value"),
            ),
            search_terms=("is empty", "is nothing", "exists", "is set"),
            input_values=True,
        )

    @classmethod
    def execute(
        cls, *, input_values: Mapping[str, Value], value: object = None
    ) -> Mapping[str, object]:
        del value
        return cls.outputs(has_value=_has_value(input_values, "value"))


class OnlyIf(Node):
    """Passes Value on when Condition is Yes and gives nothing when it is No.

    With Condition No, Value is never asked for, so the stage behind it does
    not run."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.utilities.only_if",
            display_name="Only if",
            category="utilities/flow",
            description="Passes Value on when Condition is Yes, and gives nothing when it is No.",
            body=(
                Input(
                    "value",
                    T,
                    required=False,
                    on_absent="accept",
                    lazy=True,
                    label="Value",
                ),
                Input(
                    "condition",
                    BOOLEAN,
                    default=True,
                    widget=BooleanWidget(label_on="Yes", label_off="No"),
                    label="Condition",
                ),
                Output("value", T, optional=True, label="Value"),
            ),
            search_terms=("gate", "when", "if", "condition"),
            input_values=True,
        )

    @classmethod
    def check_lazy_status(
        cls, *, condition: bool, input_values: Mapping[str, Value], value: object = ABSENT
    ) -> tuple[str, ...]:
        # A wired Value not asked for yet arrives as None; with no wire the
        # parameter keeps its default.
        if condition and value is None and "value" not in input_values:
            return ("value",)
        return ()

    @classmethod
    def execute(
        cls, *, condition: bool, input_values: Mapping[str, Value], value: object = ABSENT
    ) -> Mapping[str, object]:
        del value
        if not condition:
            return cls.outputs(value=AbsentOutput("Condition is false"))
        if _has_value(input_values, "value"):
            return cls.outputs(value=input_values["value"])
        return cls.outputs(value=ABSENT)


class Unpack(Node):
    """Take a list apart without letting a run change the stored output count."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        integer = TypeExpr.concrete(CORE_INT)
        limits = NumberWidget(min=0, max=512, step=1)
        return NodeSchema(
            node_type="dinkster.utilities.unpack",
            display_name="Unpack",
            category="utilities/flow",
            description="Gives one item of a list on each output, using the count you set.",
            body=(
                Input("list", TypeExpr.list_of(T), required=False, on_absent="skip", label="List"),
                Input(
                    "count",
                    integer,
                    default=0,
                    widget=limits,
                    hidden=True,
                    pin=False,
                    list_count_of="list",
                ),
                Note(
                    "empty",
                    "Connect a list or record",
                    visible=Formula("not linked(list) or count == 0"),
                ),
                Group(
                    "items",
                    value=Output("item", T, optional=True),
                    keys=Numbered("", start=0),
                    grows=Count(Formula("count")),
                    member_label=Formula('"Item " + str(index())'),
                    label="Items",
                ),
                Button(
                    "set_count",
                    "Set count",
                    (
                        Ask(
                            "Set count",
                            (Input("new_count", integer, default=0, widget=limits, label="Count"),),
                        ),
                        Set("count", Formula("new_count")),
                    ),
                ),
            ),
            input_values=True,
            search_terms=("split list", "items"),
            validate=(Rule(Formula("0 <= count <= 512"), "Count must be from 0 to 512."),),
        )

    @classmethod
    def execute(
        cls, *, count: int, items: Members[Item], input_values: Mapping[str, Value]
    ) -> Mapping[str, object]:
        if "list" not in input_values:
            return cls.outputs(
                items={member.key: AbsentOutput("No list is connected") for member in items}
            )
        children = list_children(input_values["list"])
        assert children is not None
        if len(children) != count:
            report_log(
                "warning",
                f"Expected {count} items; received {len(children)}",
                data={
                    "code": "unpack_count_mismatch",
                    "expected": count,
                    "received": len(children),
                },
            )
        return cls.outputs(
            items={
                member.key: children[index] if index < len(children) else ABSENT
                for index, member in enumerate(items)
            }
        )

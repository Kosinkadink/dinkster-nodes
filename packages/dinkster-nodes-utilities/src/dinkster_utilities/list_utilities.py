"""Native list construction and indexing nodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dinkster_api.v1 import (
    CORE_INT,
    Group,
    Input,
    Item,
    Members,
    Node,
    NodeSchema,
    Numbered,
    Output,
    TypeExpr,
    Value,
    list_children,
    member_id,
    with_identity_replacement,
)

_ITEM = TypeExpr.wildcard()
_JOINED = TypeExpr.variable("T")


def _execution_items(value: Value) -> tuple[Value, ...]:
    if TypeExpr.runtime_cardinality(value.type_id) != "list":
        return (value,)
    children = list_children(value)
    assert children is not None
    return children


class CreateList(Node):
    """Join values and lists of one type into one flat list.

    A single value adds one item; a list adds its items in order. ComfyUI's
    CreateList joins the same way, so its family names are kept and it imports
    as this node.
    """

    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.create_list",
            display_name="Create List",
            category="utilities",
            body=(
                Group(
                    "inputs",
                    value=Input("input", _JOINED, accepts_list=True),
                    min_members=1,
                    optional_after_minimum=True,
                    keys=Numbered("input", start=0),
                ),
                Output("list", TypeExpr.list_of(_JOINED)),
            ),
            search_terms=(
                "Image Iterator",
                "Text Iterator",
                "Iterator",
                "flatten",
                "combine",
                "merge",
                "concat",
            ),
            input_values=True,
        )
        return with_identity_replacement(schema, "CreateList")

    @classmethod
    def execute(
        cls, *, inputs: Members[Item], input_values: Mapping[str, Value]
    ) -> Mapping[str, object]:
        items: list[Value] = []
        for member in inputs:
            items.extend(_execution_items(input_values[member_id("inputs", member.key)]))
        return cls.outputs(list=items)


class GetItemFromList(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.get_item_from_list",
            display_name="Get Item From List",
            category="utilities",
            body=(
                Input("list", _ITEM),
                Input(
                    "index",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    accepts_list=True,
                ),
                Output("item", _ITEM, label="*"),
            ),
            input_values=True,
        )
        return with_identity_replacement(schema, "GetItemFromList")

    @classmethod
    def execute(
        cls, *, list: object, index: object, input_values: Mapping[str, Value]
    ) -> Mapping[str, object]:
        selected_index = _execution_items(input_values["index"])[0]
        if selected_index.type_id != CORE_INT:
            raise TypeError("index requires an INT or a list of INT values")
        children = _execution_items(input_values["list"])
        return cls.outputs(item=children[cast("int", selected_index.resolve())])

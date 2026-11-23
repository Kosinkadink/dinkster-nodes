"""Eager boolean nodes using Python truthiness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    ComboWidget,
    Group,
    Input,
    Item,
    Members,
    Node,
    NodeSchema,
    Numbered,
    Output,
    TypeExpr,
    with_identity_replacement,
)


class ComfyNotNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.comfy_not_node",
            display_name="Not",
            category="utilities/logic",
            description=(
                "Logical NOT operation. Returns true if the value is falsy. "
                "Uses Python's rules for truthiness."
            ),
            body=(
                Input("value", TypeExpr.wildcard()),
                Output("BOOLEAN", TypeExpr.concrete(CORE_BOOLEAN)),
            ),
        )
        return with_identity_replacement(schema, "ComfyNotNode")

    @classmethod
    def execute(cls, *, value: object) -> Mapping[str, object]:
        return cls.outputs(BOOLEAN=not value)


class ComfyAndNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.comfy_and_node",
            display_name="And",
            category="utilities/logic",
            description=(
                "Logical AND operation. Returns true if all of the values are truthy. "
                "Uses Python's rules for truthiness."
            ),
            body=(
                Group(
                    "values",
                    value=Input("value", TypeExpr.wildcard()),
                    min_members=1,
                    max_members=10,
                    optional_after_minimum=True,
                    keys=Numbered("value", start=0),
                ),
                Output("BOOLEAN", TypeExpr.concrete(CORE_BOOLEAN)),
            ),
        )
        return with_identity_replacement(schema, "ComfyAndNode")

    @classmethod
    def execute(cls, *, values: Members[Item]) -> Mapping[str, object]:
        return cls.outputs(BOOLEAN=all(member.value for member in values))


class ComfyOrNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.comfy_or_node",
            display_name="Or",
            category="utilities/logic",
            description=(
                "Logical OR operation. Returns true if any of the values are truthy. "
                "Uses Python's rules for truthiness."
            ),
            body=(
                Group(
                    "values",
                    value=Input("value", TypeExpr.wildcard()),
                    min_members=1,
                    max_members=10,
                    optional_after_minimum=True,
                    keys=Numbered("value", start=0),
                ),
                Output("BOOLEAN", TypeExpr.concrete(CORE_BOOLEAN)),
            ),
        )
        return with_identity_replacement(schema, "ComfyOrNode")

    @classmethod
    def execute(cls, *, values: Members[Item]) -> Mapping[str, object]:
        return cls.outputs(BOOLEAN=any(member.value for member in values))


class CustomCombo(Node):
    """A choice among options the author types. The editor sends those options,
    in order, to ``options``; INDEX is the choice's position among them,
    counting from 0."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.custom_combo",
            display_name="Custom Combo",
            category="utilities",
            body=(
                Input(
                    "choice",
                    TypeExpr.concrete(CORE_COMBO),
                    widget=ComboWidget(options=(), editable_options=True, options_input="options"),
                ),
                Input(
                    "options",
                    TypeExpr.list_of(TypeExpr.concrete(CORE_STRING)),
                    required=False,
                    hidden=True,
                ),
                Output("STRING", TypeExpr.concrete(CORE_STRING)),
                Output("INDEX", TypeExpr.concrete(CORE_INT)),
            ),
            additional_inputs=TypeExpr.wildcard(),
        )
        return with_identity_replacement(schema, "CustomCombo")

    @classmethod
    def execute(
        cls, *, choice: str, options: Sequence[str] = (), **_: object
    ) -> Mapping[str, object]:
        if choice not in options:
            raise ValueError(f'Custom Combo\'s choice "{choice}" is not one of its options.')
        return cls.outputs(STRING=choice, INDEX=list(options).index(choice))

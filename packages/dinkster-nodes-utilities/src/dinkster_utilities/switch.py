"""The Switch, which the engine routes, and ComfyUI's If/Else Switch."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    ABSENT,
    CORE_BOOLEAN,
    CORE_INT,
    Group,
    Input,
    Node,
    NodeSchema,
    Numbered,
    Output,
    SelectorSpec,
    TypeExpr,
    with_identity_replacement,
)


class Switch(Node):
    """Pick, then the branches in their stored order, then Value.

    The engine passes on the branch the pick names without running the node.
    A stored pick is a branch's key, so renaming or moving a branch never
    changes it. A wired whole number is a position from 0, and a yes or no
    picks between exactly two branches, no first."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        value_type = TypeExpr.variable("value")
        return NodeSchema(
            node_type="dinkster.utilities.switch",
            display_name="Switch",
            category="utilities/logic",
            body=(
                Input("pick", TypeExpr.union(CORE_INT, CORE_BOOLEAN), label="Pick", required=False),
                Group(
                    "branch",
                    value=Input("branch", value_type, required=False, lazy=True),
                    keys=Numbered("branch", start=1),
                    grows="button",
                    min_members=1,
                    reorderable=True,
                ),
                Output("value", value_type, label="Value"),
            ),
            selector=SelectorSpec("pick", "branch"),
            search_terms=("If/Else Switch", "If", "Else", "Select", "Route"),
        )

    @classmethod
    def execute(cls, **_: object) -> Mapping[str, object]:
        raise AssertionError("the engine routes a Switch without running it")


class ComfySwitchNode(Node):
    """ComfyUI's If/Else Switch, kept so its workflows open. Search offers the
    Switch instead. Only the input the switch names is demanded."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        value_type = TypeExpr.variable("switch")
        schema = NodeSchema(
            node_type="dinkster.utilities.comfy_switch_node",
            display_name="If/Else Switch",
            category="utilities/logic",
            body=(
                Input("switch", TypeExpr.concrete(CORE_BOOLEAN)),
                Input("on_false", value_type, required=False, lazy=True, on_absent="skip"),
                Input("on_true", value_type, required=False, lazy=True, on_absent="skip"),
                Output("output", value_type, optional=True),
            ),
            search_visibility="hidden",
        )
        return with_identity_replacement(schema, "ComfySwitchNode")

    @classmethod
    def check_lazy_status(
        cls, *, switch: bool, on_false: object = ABSENT, on_true: object = ABSENT
    ) -> tuple[str, ...]:
        selected = "on_true" if switch else "on_false"
        return (selected,) if (on_true if switch else on_false) is None else ()

    @classmethod
    def execute(
        cls, *, switch: bool, on_false: object = ABSENT, on_true: object = ABSENT
    ) -> Mapping[str, object]:
        value = on_true if switch else on_false
        return cls.outputs(output=ABSENT if value is None else value)

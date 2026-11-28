"""The Switch, which the engine routes and which ComfyUI's If/Else Switch opens as."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    Group,
    Input,
    InputFamilyMapping,
    InputFamilyMember,
    MappingSource,
    Node,
    NodeSchema,
    Numbered,
    Output,
    ReplacementCase,
    ReplacementMigration,
    ReplacementPredicate,
    ReplacementRule,
    SelectorSpec,
    TypeExpr,
)


def _if_else_cases() -> tuple[ReplacementCase, ...]:
    """ComfyUI's If/Else Switch as the Switch: a wired switch picks, a stored
    true picks the second branch and anything else the first; on_false and
    on_true become the two branches and output becomes Value."""
    branches = {
        "branch": InputFamilyMapping.from_members(
            InputFamilyMember.build("branch1", inputs={"branch": MappingSource.copy("on_false")}),
            InputFamilyMember.build("branch2", inputs={"branch": MappingSource.copy("on_true")}),
        )
    }
    return tuple(
        ReplacementCase.build(
            "dinkster.utilities.switch",
            when=when,
            inputs={"pick": pick},
            input_families=branches,
            outputs={"value": "output"},
        )
        for when, pick in (
            (ReplacementPredicate.input_connected("switch"), MappingSource.link("switch")),
            (ReplacementPredicate.value_equals("switch", True), MappingSource.constant("branch2")),
            (None, MappingSource.constant("branch1")),
        )
    )


class Switch(Node):
    """Pick, then the branches in their stored order, then Value.

    The engine passes on the branch the pick names without running the node.
    A stored pick is a branch's key, so renaming or moving a branch never
    changes it. A wired whole number is a position from 0, and a yes or no
    picks between exactly two branches, no first. Wired text or a choice
    picks the branch with exactly that name, then the branch with that key;
    the editor sends the branch names to the hidden ``names`` input only when
    that text is not known before the run."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        value_type = TypeExpr.variable("value")
        return NodeSchema(
            node_type="dinkster.utilities.switch",
            display_name="Switch",
            category="utilities/logic",
            body=(
                Input(
                    "pick",
                    TypeExpr.union(CORE_INT, CORE_BOOLEAN, CORE_STRING, CORE_COMBO),
                    label="Pick",
                    required=False,
                ),
                Input(
                    "names",
                    TypeExpr.list_of(TypeExpr.concrete(CORE_STRING)),
                    required=False,
                    hidden=True,
                ),
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
            selector=SelectorSpec("pick", "branch", names="names"),
            search_terms=("If/Else Switch", "If", "Else", "Select", "Route"),
            # A ComfyUI workflow, and a Dinkster document saved while the If/Else
            # Switch was its own node. Nothing installed declares either source's inputs.
            replacements=tuple(
                ReplacementRule(
                    source,
                    _if_else_cases(),
                    migration=ReplacementMigration(("switch", "on_false", "on_true")),
                )
                for source in ("ComfySwitchNode", "dinkster.utilities.comfy_switch_node")
            ),
        )

    @classmethod
    def execute(cls, **_: object) -> Mapping[str, object]:
        raise AssertionError("the engine routes a Switch without running it")

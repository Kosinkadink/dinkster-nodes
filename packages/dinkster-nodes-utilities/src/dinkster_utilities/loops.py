"""Loop boundaries expand into ordinary, occurrence-scoped graph nodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_api.v1 import (
    ABSENT,
    CORE_BOOLEAN,
    CORE_INT,
    CORE_STRING,
    Choice,
    Expansion,
    ExpansionBuilder,
    ExpansionNode,
    Group,
    Input,
    Node,
    NodeSchema,
    Numbered,
    NumberWidget,
    Option,
    Output,
    RunExecutionContext,
    Section,
    StringWidget,
    TypeExpr,
    report_progress,
    with_identity_replacement,
)

INT = TypeExpr.concrete(CORE_INT)
BOOL = TypeExpr.concrete(CORE_BOOLEAN)
STRING = TypeExpr.concrete(CORE_STRING)
ANY = TypeExpr.wildcard()
ITEM = TypeExpr.variable("list_item")
CARRY = TypeExpr.variable("carried_value")
OUTPUT = TypeExpr.variable("output_value")
DEPENDENCIES = Group(
    "dependencies",
    value=Input("dependency", ANY, hidden=True, on_absent="accept"),
    keys=Numbered("dependency", start=0),
)


def _outputs() -> tuple[Output, ...]:
    return (
        Output("iteration_index", INT, optional=True),
        Output("is_first", BOOL),
        Output("is_last", BOOL),
        Output("list_item", ITEM, optional=True),
        Output("current_iteration_value", CARRY, optional=True),
    )


def _link(value: object) -> Mapping[str, str] | None:
    if isinstance(value, Mapping) and "$link" in value:
        return cast("Mapping[str, str]", value["$link"])
    return None


class StartLoop(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            "dinkster.utilities.start_loop",
            display_name="Start Loop",
            category="utilities/looping",
            body=(
                Choice(
                    "mode",
                    options=(
                        Option(
                            "simple",
                            rows=(
                                Input("num_iterations", INT, default=4, widget=NumberWidget(min=0)),
                            ),
                        ),
                        Option(
                            "For",
                            rows=(
                                Input("start_iteration_index", INT, default=0),
                                Input(
                                    "max_iteration",
                                    INT,
                                    default=4,
                                    widget=NumberWidget(max=0xFFFFFFFFFFFFFFFF),
                                ),
                                Input("step", INT, default=1, widget=NumberWidget(min=1)),
                            ),
                        ),
                        Option("List", rows=(Input("list", TypeExpr.list_of(ITEM)),)),
                    ),
                ),
                Input("parent_iteration", INT, required=False, pin_first=True),
                Input("initial_iteration_value", CARRY, required=False),
                Section(
                    "advanced",
                    label="Advanced",
                    collapsed=True,
                    rows=(
                        Input(
                            "cache_iterations",
                            BOOL,
                            default=False,
                        ),
                    ),
                ),
                *_outputs(),
            ),
            execution_context=True,
            may_expand_graph=True,
            idempotent=False,
        )
        return with_identity_replacement(schema, "StartLoop")

    @classmethod
    async def execute(
        cls,
        *,
        execution_context: RunExecutionContext,
        mode: str = "simple",
        cache_iterations: bool = False,
        parent_iteration: object = None,
        initial_iteration_value: object = None,
        **options: Any,
    ) -> Expansion:
        if mode == "simple":
            values = list(range(options.get("mode.num_iterations", 4)))
        elif mode == "For":
            step = options.get("mode.step", 1)
            if step == 0:
                raise ValueError("Start Loop step must not be 0")
            values = list(
                range(
                    options.get("mode.start_iteration_index", 0),
                    options.get("mode.max_iteration", 4),
                    step,
                )
            )
        else:
            values = list(range(len(options["mode.list"])))
        document = cast("Any", await execution_context.get_graph())
        start_id = execution_context.node_id
        pair = document["loops"][start_id]
        end_id: str = pair["end"]
        body: Sequence[str] = pair["body"]
        nodes = document["nodes"]
        end_inputs = nodes[end_id]["inputs"]
        accumulate: object = end_inputs.get("accumulate", False)
        dynamic_accumulate = type(accumulate) is not bool
        builder = ExpansionBuilder(cache_policy="reuse" if cache_iterations else "rerun")

        def external(value: object) -> object:
            link = _link(value)
            return builder.external(link["node"], link["output"]) if link is not None else value

        previous_carry: object = external(nodes[start_id]["inputs"].get("initial_iteration_value"))
        dependencies: list[object] = []
        previous_progress: object = None
        results: list[object] = []
        for position, index in enumerate(values):
            iteration = builder.node(
                "dinkster.utilities.loop_iteration",
                f"iteration_{position}",
                iteration_index=index,
                is_first=position == 0,
                is_last=position == len(values) - 1,
                reuse_cache=cache_iterations,
                **{f"dependencies.{i}": value for i, value in enumerate(dependencies)},
            )
            iteration.set_override_display_id(start_id, external=True)
            if previous_carry is not None:
                iteration.set_input("current_iteration_value", previous_carry)
            if mode == "List":
                element = builder.node(
                    "dinkster.utilities.list_element",
                    f"item_{position}",
                    list=external(nodes[start_id]["inputs"]["mode.list"]),
                    index=position,
                )
                iteration.set_input("list_item", element.out("item"))
            copies: dict[str, ExpansionNode] = {}
            for node_id in body:
                original = nodes[node_id]
                copy = builder.node(original["nodeType"], f"{position}_{node_id}")
                copy.set_override_display_id(node_id, external=True)
                copy.output_members.update(original.get("outputMembers", {}))
                copy.slot_variants.update(original.get("slotVariants", {}))
                copies[node_id] = copy

            def copied(
                value: object,
                iteration: ExpansionNode = iteration,
                copies: Mapping[str, ExpansionNode] = copies,
            ) -> object:
                link = _link(value)
                if link is None:
                    return value
                if link["node"] == start_id:
                    return iteration.out(link["output"])
                if link["node"] in copies:
                    return copies[link["node"]].out(link["output"])
                return external(value)

            for node_id, copy in copies.items():
                for name, value in nodes[node_id]["inputs"].items():
                    copy.set_input(name, copied(value))
            dependencies = []
            if _link(end_inputs.get("output_value")) is not None:
                output = copied(end_inputs["output_value"])
                dependencies.append(output)
                if accumulate is True or dynamic_accumulate:
                    results.append(output)
                elif position == len(values) - 1:
                    results = [output]
            if _link(end_inputs.get("next_iteration_value")) is not None:
                previous_carry = cast("object", copied(end_inputs["next_iteration_value"]))
                dependencies.append(previous_carry)
            dependencies.extend(
                copied(value)
                for name, value in end_inputs.items()
                if name.startswith("terminations.") and _link(value) is not None
            )
            progress = builder.node(
                "dinkster.utilities.loop_progress",
                f"progress_{position}",
                start_id=start_id,
                position=position + 1,
                total=len(values),
                **{f"dependencies.{i}": value for i, value in enumerate(dependencies)},
            )
            if previous_progress is not None:
                progress.set_input("previous_progress", previous_progress)
            progress.set_override_display_id(start_id, external=True)
            previous_progress = progress.out("INT")
        if results:
            collected = builder.node(
                "dinkster.utilities.make_list",
                "collected",
                **{f"items.{i}": value for i, value in enumerate(results)},
            )
            result_value: object = collected.out("list")
            if dynamic_accumulate:
                final = builder.node(
                    "dinkster.utilities.make_list", "final", **{"items.0": results[-1]}
                )
                selection = builder.node(
                    "dinkster.utilities.comfy_switch_node",
                    "accumulation",
                    switch=external(accumulate),
                    on_false=final.out("list"),
                    on_true=result_value,
                )
                result_value = selection.out("output")
        else:
            result_type = document["outputTypes"][end_id]["outputs"]
            if result_type is None:
                if "output_value" in end_inputs:
                    raise ValueError("empty loop output type cannot be resolved")
                result_type = "list<core.absent>"
            result_value = {"$typed": {"type": result_type, "value": []}}
        result = builder.node(
            "dinkster.utilities.loop_result",
            "result",
            close_id=end_id,
            result=result_value,
            **{f"dependencies.{i}": value for i, value in enumerate(dependencies)},
        )
        if previous_progress is not None:
            result.set_input("progress", previous_progress)
        await execution_context.inhibit(body)
        await execution_context.override_node(
            end_id,
            {
                name: value
                for name, value in end_inputs.items()
                if name not in ("output_value", "next_iteration_value")
                and not name.startswith("terminations.")
            },
        )
        report_progress(0, len(values), text=f"Iteration 0 / {len(values)}")
        return builder.finalize(
            iteration_index=ABSENT,
            is_first=False,
            is_last=not values,
            list_item=ABSENT,
            current_iteration_value=ABSENT,
        )


class LoopIteration(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            "dinkster.utilities.loop_iteration",
            category="sd",
            body=(
                Input("iteration_index", INT),
                Input("is_first", BOOL),
                Input("is_last", BOOL),
                Input("list_item", ITEM, required=False),
                Input("current_iteration_value", CARRY, required=False),
                Input("reuse_cache", BOOL),
                DEPENDENCIES,
                *_outputs(),
            ),
            execution_context=True,
        )
        return with_identity_replacement(schema, "LoopIteration")

    @classmethod
    def execute(
        cls, *, execution_context: RunExecutionContext, **kwargs: object
    ) -> Mapping[str, object]:
        return {output.id: execution_context.inputs.get(output.id, ABSENT) for output in _outputs()}


class LoopProgress(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            "dinkster.utilities.loop_progress",
            category="sd",
            body=(
                Input("start_id", STRING, widget=StringWidget(multiline=False)),
                Input("position", INT),
                Input("total", INT),
                Input("previous_progress", INT, required=False, hidden=True),
                DEPENDENCIES,
                Output("INT", INT),
            ),
            output_node=True,
            idempotent=False,
        )
        return with_identity_replacement(schema, "LoopProgress")

    @classmethod
    def execute(
        cls, *, start_id: str, position: int, total: int, **kwargs: object
    ) -> Mapping[str, object]:
        report_progress(position, total, text=f"Iteration {position} / {total}")
        return cls.outputs(INT=position)


class LoopResult(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            "dinkster.utilities.loop_result",
            category="sd",
            body=(
                Input("close_id", STRING, widget=StringWidget(multiline=False)),
                Input("result", TypeExpr.list_of(ANY), hidden=True, on_absent="accept"),
                Input("progress", INT, required=False, hidden=True),
                DEPENDENCIES,
            ),
            output_node=True,
            idempotent=False,
            execution_context=True,
        )
        return with_identity_replacement(schema, "LoopResult")

    @classmethod
    async def execute(
        cls, *, close_id: str, execution_context: RunExecutionContext, **kwargs: object
    ) -> Mapping[str, object]:
        await execution_context.publish_run_value(
            close_id, {"outputs": execution_context.inputs["result"]}
        )
        return {}


class EndLoop(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            "dinkster.utilities.end_loop",
            display_name="End Loop",
            category="utilities/looping",
            body=(
                Input("output_value", OUTPUT, required=False),
                Input("next_iteration_value", CARRY, required=False),
                Input("accumulate", BOOL, default=False),
                Group(
                    "terminations",
                    value=Input("termination", ANY, required=False),
                    max_members=50,
                    keys=Numbered("termination", start=0),
                ),
                Output("outputs", TypeExpr.list_of(OUTPUT), optional=True),
            ),
            execution_context=True,
        )
        return with_identity_replacement(schema, "EndLoop")

    @classmethod
    async def execute(
        cls, *, execution_context: RunExecutionContext, **kwargs: object
    ) -> Mapping[str, object]:
        return await execution_context.await_run_value(execution_context.node_id)

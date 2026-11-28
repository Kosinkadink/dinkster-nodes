"""Public schemas and engine runs of native hook scheduling nodes."""

from typing import Any

import pytest
from dinkster_api.v1 import (
    CORE_FLOAT,
    HOOK_KEYFRAMES_TYPE,
    HOOKS_TYPE,
    Curve,
    NumberWidget,
    TypeExpr,
    TypeRegistry,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral
from dinkster_models import NODES
from dinkster_models.hook_scheduling import (
    CombineHooks2,
    CombineHooks4,
    CombineHooks8,
    CreateHookKeyframe,
    CreateHookKeyframesFromFloats,
    CreateHookKeyframesInterpolated,
    HookKeyframesFromCurve,
    SetHookKeyframes,
)
from dinkster_models.types import register_types
from dinkster_schema import build_node_types, build_schemas
from dinkster_utilities.curve_editor import SampleCurve
from dinkster_utilities.list_utilities import CreateList
from dinkster_values import register_core_types
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario
from tests.body_rows import advanced_ids


def test_hook_scheduling_nodes_publish_the_fixed_core_ids_and_sockets() -> None:
    expected = {
        "dinkster.models.set_hook_keyframes": ("hooks", "hook_kf"),
        "dinkster.models.create_hook_keyframe": ("strength_mult", "start_percent", "prev_hook_kf"),
        "dinkster.models.create_hook_keyframes_interpolated": (
            "strength_start",
            "strength_end",
            "interpolation",
            "start_percent",
            "end_percent",
            "keyframes_count",
            "prev_hook_kf",
        ),
        "dinkster.models.create_hook_keyframes_from_floats": (
            "floats_strength",
            "start_percent",
            "end_percent",
            "prev_hook_kf",
        ),
        "dinkster.models.combine_hooks2": ("hooks_A", "hooks_B"),
        "dinkster.models.combine_hooks4": ("hooks_A", "hooks_B", "hooks_C", "hooks_D"),
        "dinkster.models.combine_hooks8": (
            "hooks_A",
            "hooks_B",
            "hooks_C",
            "hooks_D",
            "hooks_E",
            "hooks_F",
            "hooks_G",
            "hooks_H",
        ),
    }
    advanced = {
        "dinkster.models.create_hook_keyframes_interpolated": ("print_keyframes",),
        "dinkster.models.create_hook_keyframes_from_floats": ("print_keyframes",),
    }
    nodes = (
        SetHookKeyframes,
        CreateHookKeyframe,
        CreateHookKeyframesInterpolated,
        CreateHookKeyframesFromFloats,
        CombineHooks2,
        CombineHooks4,
        CombineHooks8,
    )
    assert tuple(node.schema().node_type for node in nodes) == tuple(expected)
    for node in nodes:
        schema = node.schema()
        assert node in NODES
        shown = advanced.get(schema.node_type, ())
        assert tuple(input.id for input in schema.inputs) == (*expected[schema.node_type], *shown)
        assert advanced_ids(schema.body) == shown
        assert all(
            input.type == TypeExpr.concrete(HOOKS_TYPE)
            for input in schema.inputs
            if input.id.startswith("hooks_")
        )

    set_keyframes, create_one, interpolated, from_floats, *_ = (node.schema() for node in nodes)
    assert set_keyframes.inputs[0].type == TypeExpr.concrete(HOOKS_TYPE)
    assert set_keyframes.inputs[1].type == TypeExpr.concrete(HOOK_KEYFRAMES_TYPE)
    assert create_one.outputs[0].type == TypeExpr.concrete(HOOK_KEYFRAMES_TYPE)
    assert interpolated.outputs[0].id == from_floats.outputs[0].id == "HOOK_KF"
    assert from_floats.inputs[0].type == TypeExpr.concrete(CORE_FLOAT)
    assert from_floats.inputs[0].accepts_list
    assert from_floats.inputs[0].pin_first
    assert from_floats.inputs[0].default == -1
    assert from_floats.inputs[0].widget == NumberWidget(min=-1, step=0.001)
    assert interpolated.inputs[5].widget == NumberWidget(min=2, max=100, step=1)


def test_keyframes_from_floats_take_a_wired_float_list_or_one_float() -> None:
    async def run() -> dict[str, list[tuple[float, float, int]]]:
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)
        nodes = (CreateList, CreateHookKeyframesFromFloats)
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        from_floats = "dinkster.models.create_hook_keyframes_from_floats"
        graph = Graph(
            nodes={
                "strengths": GraphNode(
                    "dinkster.utilities.create_list",
                    {
                        "inputs.input0": TypedLiteral("FLOAT", 0.25),
                        "inputs.input1": TypedLiteral("FLOAT", 1.5),
                        "inputs.input2": TypedLiteral("FLOAT", -1.0),
                    },
                ),
                "listed": GraphNode(
                    from_floats,
                    {
                        "floats_strength": Link("strengths", "list"),
                        "start_percent": 0.2,
                        "end_percent": 0.8,
                    },
                ),
                "single": GraphNode(
                    from_floats,
                    {"floats_strength": 0.75, "start_percent": 0.1, "end_percent": 0.9},
                ),
            }
        )
        result = await engine.run(graph, ["listed", "single"])
        keyframes: dict[str, list[tuple[float, float, int]]] = {}
        for name in ("listed", "single"):
            group: Any = result.outputs[name]["HOOK_KF"].resolve()
            keyframes[name] = [
                (float(keyframe.strength), keyframe.start_percent, keyframe.guarantee_steps)
                for keyframe in group.keyframes
            ]
        return keyframes

    keyframes = run_scenario(run())
    # One keyframe per strength at evenly spaced percents; only the first guarantees a step.
    assert keyframes["listed"] == [
        (0.25, pytest.approx(0.2), 1),
        (1.5, pytest.approx(0.5), 0),
        (-1.0, pytest.approx(0.8), 0),
    ]
    assert keyframes["single"] == [(0.75, pytest.approx(0.1), 1)]


def test_curve_nodes_drive_hook_keyframe_schedules() -> None:
    async def run() -> dict[str, Any]:
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)
        nodes = (SampleCurve, CreateHookKeyframesFromFloats, HookKeyframesFromCurve)
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        graph = Graph(
            nodes={
                "sample": GraphNode(
                    "dinkster.utilities.sample_curve",
                    {
                        "curve": Curve(((0.2, 1), (0.5, 4), (0.8, -2)), "linear").to_record(),
                        "count": 5,
                    },
                ),
                "sampled": GraphNode(
                    "dinkster.models.create_hook_keyframes_from_floats",
                    {
                        "floats_strength": Link("sample", "values"),
                        "start_percent": 0.2,
                        "end_percent": 0.8,
                    },
                ),
                "points": GraphNode(
                    "dinkster.models.hook_keyframes_from_curve",
                    {"curve": Curve(((0.8, -2), (0.2, 1), (0.5, 4)), "step").to_record()},
                ),
                "append": GraphNode(
                    "dinkster.models.hook_keyframes_from_curve",
                    {
                        "curve": Curve(((0.9, 7),), "step").to_record(),
                        "prev_hook_kf": Link("points", "HOOK_KF"),
                    },
                ),
            }
        )
        result = await engine.run(graph, ["sampled", "points", "append"])
        return {name: result.outputs[name]["HOOK_KF"].resolve() for name in result.outputs}

    groups = run_scenario(run())
    expected = {
        "sampled": [(0.2, 1), (0.35, 2.5), (0.5, 4), (0.65, 1), (0.8, -2)],
        "points": [(0.2, 1), (0.5, 4), (0.8, -2)],
        "append": [(0.2, 1), (0.5, 4), (0.8, -2), (0.9, 7)],
    }
    for name, pairs in expected.items():
        actual = [(key.start_percent, key.strength) for key in groups[name].keyframes]
        assert len(actual) == len(pairs)
        for pair, wanted in zip(actual, pairs, strict=True):
            assert pair == pytest.approx(wanted)

    import importlib
    from types import SimpleNamespace

    import torch

    sampling = importlib.import_module("dinkster_inference.model_sampling").ModelSamplingDiscrete()
    sigmas = torch.tensor([sampling.percent_to_sigma(percent) for percent in (0.1, 0.4, 0.7, 0.9)])
    for name, strengths in (("sampled", [1, 2.5, 1, -2]), ("points", [1, 1, 4, -2])):
        group = groups[name]
        group.initialize_timesteps(SimpleNamespace(model_sampling=sampling))
        actual_strengths = []
        for sigma in sigmas:
            group.prepare_current_keyframe(float(sigma), {"sample_sigmas": sigmas})
            actual_strengths.append(group.strength)
        assert actual_strengths == pytest.approx(strengths)

    from dinkster_api.v1 import CurveWidget

    for node in (SampleCurve, HookKeyframesFromCurve):
        widget = node.schema().inputs[0].widget
        assert isinstance(widget, CurveWidget)
        assert widget.histogram == ""

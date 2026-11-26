"""First-party primitive execution and portable color values across workers."""

from __future__ import annotations

import dataclasses

import dinkster_api.v1 as api
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, Invocation
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral
from dinkster_schema import build_node_types, build_schemas
from dinkster_utilities import NODES, register_types
from dinkster_utilities.utilities import PreviewAny
from dinkster_values import EncodedPayload, TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario


def test_color_registration_is_portable_and_idempotent() -> None:
    registry = TypeRegistry()
    api.register_inference_types(registry)
    api.register_inference_types(registry)
    assert (api.COLOR_TYPE, api.COLORS_TYPE) == (
        "COLOR",
        "COLORS",
    )
    for type_id, source in (
        (api.COLOR_TYPE, "#ff001280"),
        (api.COLORS_TYPE, []),
        (api.COLORS_TYPE, ["blue", "#00ff00", "blue"]),
    ):
        spec = registry.spec(type_id)
        encoded = spec.encode(source)
        assert encoded.startswith(b"json:")
        decoded = spec.decode(encoded)
        assert decoded == source
        assert (
            registry.wrap(type_id, decoded).fingerprint
            == registry.wrap(type_id, source).fingerprint
        )
    assert (
        registry.wrap(api.COLORS_TYPE, ["red", "blue"]).fingerprint
        != registry.wrap(api.COLORS_TYPE, ["blue", "red"]).fingerprint
    )


def test_first_party_primitives_execute_and_feed_preview() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)
        engine = Engine(
            schemas=build_schemas(NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(NODES), registry),
            cache=MemoryLRUCache(),
        )
        cases = (
            ("i", "primitive_int", "INT", "INT", 7),
            ("f", "primitive_float", "FLOAT", "FLOAT", 0.25),
            ("s", "primitive_string", "STRING", "STRING", "hello"),
            ("m", "primitive_string", "STRING", "STRING", "a\nb"),
            ("b", "primitive_boolean", "BOOLEAN", "BOOLEAN", True),
            ("c", "primitive_choice", "COMBO", "COMBO", "Detailed"),
            ("g", "primitive", "value", "FLOAT", 0.5),
            ("k", "primitive", "value", api.COLOR_TYPE, "#3366cc80"),
        )
        nodes = {
            key: GraphNode(
                f"dinkster.utilities.{leaf}",
                {"value": TypedLiteral(type_id, value) if leaf == "primitive" else value},
            )
            for key, leaf, _output, type_id, value in cases
        }
        nodes["p"] = GraphNode("dinkster.utilities.preview_any", {"source": Link("i", "INT")})
        result = await engine.run(Graph(nodes=nodes), list(nodes))
        for key, _leaf, output, type_id, value in cases:
            actual = result.outputs[key][output]
            assert actual.type_id == type_id
            assert actual.resolve() == value
        assert result.outputs["p"]["STRING"].resolve() == "7"

    run_scenario(scenario())


def test_preview_text_formats_like_comfy_preview_any() -> None:
    for source, text in (
        (None, "None"),
        ("already text", "already text"),
        (7, "7"),
        (0.5, "0.5"),
        (True, "True"),
        ({"a": 1}, '{\n    "a": 1\n}'),
        ([1, 2], "[\n    1,\n    2\n]"),
    ):
        assert PreviewAny.execute(source=source) == {"STRING": text}

    class OnlyStr:
        def __str__(self) -> str:
            return "stringified"

    assert PreviewAny.execute(source=OnlyStr()) == {"STRING": "stringified"}

    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("no")

    assert PreviewAny.execute(source=Hostile()) == {
        "STRING": "source exists, but could not be serialized."
    }


def test_unresolvable_payload_fails_as_node_error() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = InProcessWorker(build_node_types((PreviewAny,)), registry)
        donor = registry.wrap("STRING", "x")
        alien = dataclasses.replace(
            donor,
            type_id="alien.tensor",
            payload=EncodedPayload(type_id="alien.tensor", data=b"\x00", decoder=None),
        )
        schema = PreviewAny.schema()
        result = await worker.invoke(
            Invocation(
                invocation_id="i1",
                node_id="p",
                node_type=schema.node_type,
                inputs={"source": alien},
                effective_schema=schema,
            )
        )
        assert result.error is not None
        assert result.error.node_id == "p"
        assert "alien.tensor" in result.error.message
        assert "not registered" in result.error.message
        assert "Traceback" not in result.error.message

    run_scenario(scenario())

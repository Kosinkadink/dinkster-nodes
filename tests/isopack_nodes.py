"""Isolated incremental-composition fixture."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    report_event,
    report_preview,
    report_progress,
)
from dinkster_values import CORE_FLOAT, CORE_INT, CORE_STRING, CustomWidgetDescriptor, TypeRegistry

STRING = TypeExpr.concrete(CORE_STRING)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
INT = TypeExpr.concrete(CORE_INT)
BLOB = TypeExpr.concrete("iso.blob")


def blob_rendition_parameters(
    parameters: Mapping[str, str], _metadata: Mapping[str, object]
) -> Mapping[str, str]:
    prefix = parameters.get("prefix", "blob").strip()
    if not prefix:
        raise ValueError("prefix must be non-empty")
    return {"prefix": prefix}


def render_blob(blob: object, parameters: Mapping[str, str]) -> bytes:
    assert isinstance(blob, Mapping)
    return f"{parameters['prefix']}:{blob['n']}:{blob['data']}".encode()


def blob_metadata(blob: object) -> Mapping[str, object]:
    assert isinstance(blob, Mapping)
    return {"size": blob["n"]}


def register_types(registry: TypeRegistry) -> None:
    registry.register("iso.blob", meta=blob_metadata)
    registry.register_rendition(
        "iso.blob",
        "summary",
        mime=lambda metadata: "text/plain" if metadata["size"] else "text/x-empty",
        render=render_blob,
        version="1",
        parameters=("prefix",),
        defaults={"prefix": "blob"},
        limits={"prefixLength": 32},
        normalize=blob_rendition_parameters,
    )


class Chatty(Node):
    """Reports progress, a binary preview, and a pack event while running -
    the cross-boundary streaming test subject. ``text`` carries the node's
    input value so concurrent invocations' events are distinguishable."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.chatty",
            display_name="Chatty",
            category="test",
            inputs=(
                InputSpec("value", STRING),
                InputSpec("seconds", FLOAT, default=0.0),
            ),
            outputs=(OutputSpec("value", STRING),),
            io_bound=True,
        )

    @classmethod
    async def execute(cls, *, value: str, seconds: float) -> Mapping[str, object]:
        report_progress(1, 2, text=value)
        if seconds:
            await asyncio.sleep(seconds)
        report_progress(2, 2, text=value)
        report_preview(b"\x89fakepng:" + value.encode(), mime="image/png", width=4, height=2)
        report_event("isopack.stage", {"stage": "done", "value": value})
        return cls.outputs(value=value.upper())


class BlobOut(Node):
    """Produces a value of a type the engine process has not registered."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.blob_out",
            display_name="Blob Out",
            category="test",
            inputs=(
                InputSpec(
                    "size",
                    INT,
                    default=8,
                    widget=CustomWidgetDescriptor(
                        "isopack.size",
                        {"min": 1, "max": 12, "unit": "bytes"},
                    ),
                ),
            ),
            outputs=(OutputSpec("blob", BLOB),),
        )

    @classmethod
    def execute(cls, *, size: int) -> Mapping[str, object]:
        return cls.outputs(blob={"n": size, "data": "x" * size})


NODES = [Chatty]
SERVER_NODES = [Chatty, BlobOut]

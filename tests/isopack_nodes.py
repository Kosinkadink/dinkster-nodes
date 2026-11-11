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
from dinkster_values import CORE_FLOAT, CORE_STRING

STRING = TypeExpr.concrete(CORE_STRING)
FLOAT = TypeExpr.concrete(CORE_FLOAT)


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


NODES = [Chatty]

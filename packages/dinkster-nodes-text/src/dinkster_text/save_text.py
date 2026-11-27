"""Save text content to a mounted output target."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_STRING,
    MOUNT_TYPE,
    AssetError,
    AssetWriter,
    ComboWidget,
    Editor,
    Input,
    MountSnapshotWriter,
    MountWidget,
    Node,
    NodeSchema,
    Output,
    SaveTargetWidget,
    StringWidget,
    TypeExpr,
    attach_saved_artifact,
    with_identity_replacement,
)

_STRING = TypeExpr.concrete(CORE_STRING)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_FORMATS = ("txt", "csv", "md", "json")


def _writer() -> AssetWriter:
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
    return AssetWriter(MountSnapshotWriter(snapshot))


def _render(text: str, format: str) -> bytes:
    if format not in _FORMATS:
        raise ValueError(f"Unsupported text format: {format!r}")
    if format == "json":
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    return text.encode("utf-8")


class SaveText(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.text.save_text",
            display_name="Save Text",
            category="text",
            description="Save text content to a file in the output directory.",
            search_terms=("save text", "write text", "export text"),
            body=(
                Input("text", _STRING, pin_first=True, widget=StringWidget(multiline=False)),
                Input(
                    "filename_prefix",
                    _STRING,
                    default="ComfyUI",
                    widget=StringWidget(multiline=False),
                ),
                Input(
                    "format",
                    _COMBO,
                    default="txt",
                    widget=ComboWidget(options=_FORMATS),
                ),
                Input(
                    "mount",
                    TypeExpr.union(MOUNT_TYPE, CORE_STRING),
                    required=False,
                    default=None,
                    widget=MountWidget(),
                ),
                Editor(
                    "target",
                    SaveTargetWidget(writes={"mount": "mount", "name": "filename_prefix"}),
                ),
                Output("text", _STRING),
            ),
            idempotent=False,
            output_node=True,
        )
        return with_identity_replacement(schema, "SaveText")

    @classmethod
    def execute(
        cls,
        *,
        text: str,
        filename_prefix: str = "ComfyUI",
        format: str = "txt",
        mount: str | None = None,
    ) -> Mapping[str, object]:
        data = _render(text, format)
        writer = _writer()
        ref = writer.save_bytes(
            writer.output_target(filename_prefix, mount=mount, image_width=1, image_height=1),
            data,
            suffix=f".{format}",
            media_type="application/json" if format == "json" else "text/plain",
        )
        attach_saved_artifact(ref.to_wire())
        return cls.outputs(text=text)


__all__ = ["SaveText"]

"""Layered-file import and library export nodes."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping
from io import BytesIO

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_COMBO,
    IMAGE_DOCUMENT_MEDIA_TYPE,
    SAVE_TARGET_TYPE,
    AssetRef,
    AssetWidget,
    AssetWriter,
    ComboWidget,
    ImageDocument,
    InputSpec,
    MountSnapshotWriter,
    Node,
    NodeSchema,
    OutputSpec,
    SaveTargetWidget,
    TypeExpr,
    attach_saved_artifact,
    current_export_snapshot,
    export_ora,
    export_psd,
    initialize_inference_process,
    load_document,
)

from .compositor_types import LAYERS_TYPE

LAYERS = TypeExpr.concrete(LAYERS_TYPE)
ASSET = TypeExpr.concrete(ASSET_TYPE)


class LoadLayers(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.layers.load",
            editor_role="layers-load",
            display_name="Load Layers",
            category="image/io",
            inputs=(InputSpec("document", ASSET, widget=AssetWidget()),),
            outputs=(OutputSpec("layers", LAYERS),),
        )

    @classmethod
    def execute(cls, *, document: AssetRef) -> Mapping[str, object]:
        loaded = load_document(document.read_bytes())
        return cls.outputs(
            layers=ImageDocument(
                loaded.data, lambda wire: AssetRef.from_wire(wire, document.resolver)
            )
        )


class SaveLayers(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.layers.save",
            display_name="Save Layers",
            category="image/io",
            inputs=(
                InputSpec("layers", LAYERS),
                InputSpec("target", TypeExpr.concrete(SAVE_TARGET_TYPE), widget=SaveTargetWidget()),
                InputSpec(
                    "format",
                    TypeExpr.concrete(CORE_COMBO),
                    required=False,
                    default="native",
                    widget=ComboWidget(options=("native", "ora", "psd")),
                ),
            ),
            outputs=(OutputSpec("document", ASSET),),
            output_node=True,
            idempotent=False,
        )

    @classmethod
    def execute(
        cls, *, layers: ImageDocument, target: object, format: str = "native"
    ) -> Mapping[str, object]:
        if format not in ("native", "ora", "psd"):
            raise ValueError("layer export format must be native, ora, or psd")
        initialize_inference_process()
        submitted = (
            None
            if importlib.import_module("dinkster_inference.cli_args").args.disable_metadata
            else current_export_snapshot()
        )
        metadata: dict[str, object] = {}
        if submitted is not None:
            if submitted.prompt is not None:
                metadata["prompt"] = dict(submitted.prompt)
            if submitted.extra_pnginfo is not None:
                metadata["extraPnginfo"] = dict(submitted.extra_pnginfo)
        if metadata and format != "psd":
            record = layers.to_record()
            extensions = record.setdefault("extensions", {})
            key, counter = "dinkster.exportSnapshot", 0
            while key in extensions:
                counter += 1
                key = f"dinkster.exportSnapshot.{counter}"
            # Document number canonicalization must not round opaque seeds or workflow values.
            extensions[key] = {"json": json.dumps(metadata)}
            layers = ImageDocument.from_record(record, bind=layers.bind)
        data = (
            layers.data
            if format == "native"
            else export_ora(layers)
            if format == "ora"
            else export_psd(layers)
        )
        suffix, media_type = {
            "native": (".json", IMAGE_DOCUMENT_MEDIA_TYPE),
            "ora": (".ora", "image/openraster"),
            "psd": (".psd", "image/vnd.adobe.photoshop"),
        }[format]
        writer = AssetWriter(MountSnapshotWriter(os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")))
        if metadata and format == "psd":
            sidecar = json.dumps(metadata, indent=2).encode("utf-8")
            references = writer.save_named_streams(
                target,
                {"document.psd": BytesIO(data), "document.psd.metadata.json": BytesIO(sidecar)},
                limit=max(len(data), len(sidecar)),
                media_types={
                    "document.psd": media_type,
                    "document.psd.metadata.json": "application/json",
                },
            )
            reference = references["document.psd"]
            for saved in references.values():
                attach_saved_artifact(saved.to_wire())
        else:
            reference = writer.save_bytes(target, data, suffix=suffix, media_type=media_type)
            attach_saved_artifact(reference.to_wire())
        return cls.outputs(document=reference)

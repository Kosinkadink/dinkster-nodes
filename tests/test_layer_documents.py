from __future__ import annotations

import io
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from dinkster_api.v1 import Node, NodeSchema, OutputSpec, TypeExpr, register_media_types
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionRuntime
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral
from dinkster_image.compositor import AddLayer, CreateLayeredImage, LayersFromBoundingBoxes
from dinkster_image.layer_document import EditLayers, FlattenLayers, SplitLayers
from dinkster_image.layer_io import LoadLayers, SaveLayers
from dinkster_image.types import Region, register_image_types
from dinkster_image_document.compat import from_comfy_layers, register_comfy_layers
from dinkster_image_document.document import (
    ImageDocument,
    append_raster,
    apply_commands,
    empty_document,
    flatten,
    raster_png,
)
from dinkster_image_document.format import (
    IMAGE_DOCUMENT_MEDIA_TYPE,
    INLINE_RESOURCE_LIMIT,
    InvalidDocument,
    affine_from_components,
    canonical_json,
)
from dinkster_image_document.interchange import load_document
from dinkster_protocol import InvocationResult, SavedArtifact
from dinkster_schema import build_node_types, build_schemas
from dinkster_values import (
    TypeRegistry,
    annotate_image,
    annotate_mask,
    media_semantics,
    register_core_types,
)
from dinkster_workers import InProcessWorker
from dinkster_workers.boundary import ValueCodec, decode_result_outputs, encode_result
from dinkster_workers.saved_artifacts import SavedArtifactAuthority
from PIL import Image

from tests.async_scenarios import run_scenario

pytestmark = pytest.mark.usefixtures("saved_artifact_context")


@pytest.fixture
def layer_document_pack(tmp_path: Path) -> Path:
    (tmp_path / "layer_document_fixture.py").write_text(
        "from dinkster_api.v1 import register_media_types\n"
        "from dinkster_image.compositor import AddLayer\n"
        "from dinkster_image.layer_document import EditLayers, FlattenLayers\n"
        "from dinkster_image.layer_io import SaveLayers\n"
        "from dinkster_image.types import register_image_types\n"
        "NODES = (AddLayer, EditLayers, FlattenLayers, SaveLayers)\n"
        "def register_types(registry):\n"
        "    register_media_types(registry)\n"
        "    register_image_types(registry)\n",
        encoding="ascii",
    )
    manifest = tmp_path / "layer-document.toml"
    manifest.write_text(
        '[pack]\nname="dinkster-image"\npublisher="dinkster"\n'
        'namespaces=["dinkster.layers"]\n'
        'type-namespaces=["dinkster.layers", "comfy.LAYERS", "comfy.COMPOSITOR"]\n'
        "[pack.sandbox]\nwritable-mounts=true\n"
        '[pack.entry]\nnodes="layer_document_fixture:NODES"\n'
        'types="layer_document_fixture:register_types"\n',
        encoding="ascii",
    )
    return manifest


def document() -> ImageDocument:
    return append_raster(
        empty_document(2, 1),
        raster_png(np.array([[[1, 0, 0, 1], [0, 1, 0, 0.5]]], dtype=np.float32)),
    )


def large_document() -> ImageDocument:
    from dinkster_image.types import register_image_types

    value = empty_document(64, 64)
    rng = np.random.default_rng(0)
    for index in range(50):
        output = io.BytesIO()
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(output, "PNG")
        raster = output.getvalue()
        assert len(raster) > INLINE_RESOURCE_LIMIT
        value = append_raster(value, raster, name=f"Layer {index}")
    registry = TypeRegistry()
    register_image_types(registry)
    spec = registry.spec("dinkster.layers")
    assert spec.meta is not None
    meta = spec.meta(value)
    size = len(spec.encode(value)) + len(canonical_json(meta).encode())
    assert size < 1024 * 1024
    assert all("inline" not in resource for resource in value.to_record()["resources"].values())
    assert b"/tmp" not in value.data
    assert len(value.to_record()["resources"]) == 50
    print(f"50x64x64 document plus metadata: {size} bytes")
    return value


def test_native_and_layer_wire_share_one_canonical_document() -> None:
    value = document()
    registry = TypeRegistry()
    register_image_types(registry)
    register_comfy_layers(registry)
    for type_id in ("dinkster.layers", "comfy.LAYERS"):
        spec = registry.spec(type_id)
        encoded = spec.encode(value)
        assert encoded == value.data
        assert cast(ImageDocument, spec.decode(encoded)).data == encoded
    assert registry.equivalent_type("comfy.LAYERS") == "dinkster.layers"
    record = value.to_record()
    record["canvas"]["width"] = 100
    assert value.to_record()["canvas"]["width"] == 2


@pytest.mark.parametrize("type_id", ["dinkster.layers", "comfy.LAYERS"])
def test_document_boundary_preserves_external_and_auxiliary_references(
    type_id, monkeypatch
) -> None:
    from dinkster_assets import AssetError, AssetRef, AssetVault, digest_bytes
    from dinkster_assets.value import value_asset_references
    from dinkster_values import default_encode

    record = append_raster(
        document(), raster_png(np.full((1, 2, 3), 0.25, dtype=np.float32))
    ).to_record()
    record["resources"]["r0"].pop("inline")
    orphan = dict(record["resources"]["r0"], id="r4", digest=digest_bytes(b"orphan"), byteSize=6)
    record["resources"]["r4"] = orphan
    record["allocation"]["nextOrdinal"] = 5
    expected = ImageDocument.from_record(record)

    def forbid_io(*args, **kwargs):
        pytest.fail("encoded document admission must not read, render or publish resources")

    monkeypatch.setattr(ImageDocument, "read_resource", forbid_io)
    monkeypatch.setattr(ImageDocument, "render", forbid_io)
    monkeypatch.setattr(AssetVault, "writer", forbid_io)
    registry = TypeRegistry()
    register_image_types(registry)
    wrapped = registry.wrap(type_id, expected)
    codec = ValueCodec(registry)
    blobs, segments = [], []
    wire, _ = codec.encode(wrapped, blobs, segments)
    auxiliary = AssetRef(digest_bytes(b"auxiliary"), "source.json", 9)
    entries = {
        **wrapped.meta.entries,
        "asset_refs": [
            auxiliary.to_wire(),
            *reversed(cast(list[object], wrapped.meta.get("asset_refs"))),
        ],
    }
    blobs[cast(int, wire["metaBlob"])] = default_encode(entries)
    received, _ = codec.decode(wire, blobs, [])
    assert cast(ImageDocument, received.resolve()).data == expected.data
    assert received.fingerprint == wrapped.fingerprint
    references = value_asset_references({"document": received})
    assert {digest: ref.size for digest, ref in references.items()} == {
        record["resources"]["r0"]["digest"]: record["resources"]["r0"]["byteSize"],
        orphan["digest"]: 6,
        auxiliary.digest: 9,
    }
    if type_id == "comfy.LAYERS":
        assert (
            value_asset_references({"document": received}, complete_registry=registry) == references
        )
    else:
        with pytest.raises(AssetError, match="incomplete"):
            value_asset_references({"document": received}, complete_registry=registry)
    first = TypeRegistry()
    first_spec = first.register(type_id)
    register_image_types(first)
    assert first.spec(type_id) is first_spec
    assert not first.spec(type_id).asset_references_complete


@pytest.mark.parametrize("type_id", ["dinkster.layers", "comfy.LAYERS"])
@pytest.mark.parametrize(
    "invalid",
    ["omitted", "last", "size", "digest", "conflict", "malformed", "noncanonical", "payload"],
)
def test_document_boundary_refuses_uncovered_resources_before_exposure(type_id, invalid) -> None:
    from dinkster_assets import digest_bytes
    from dinkster_values import default_encode
    from dinkster_workers.boundary import EncodedPayloadValidationError

    record = document().to_record()
    record["resources"]["r0"].pop("inline")
    record["resources"]["r2"] = dict(
        record["resources"]["r0"], id="r2", digest=digest_bytes(b"orphan"), byteSize=6
    )
    record["allocation"]["nextOrdinal"] = 3
    registry = TypeRegistry()
    register_image_types(registry)
    wrapped = registry.wrap(type_id, ImageDocument.from_record(record))
    codec = ValueCodec(registry)
    blobs, segments = [], []
    wire, _ = codec.encode(wrapped, blobs, segments)
    metadata = dict(wrapped.meta.entries)
    refs = [dict(ref) for ref in cast(list[dict[str, object]], metadata["asset_refs"])]
    if invalid == "omitted":
        refs.clear()
    elif invalid == "last":
        refs.pop()
    elif invalid == "size":
        refs[0]["size"] = cast(int, refs[0]["size"]) + 1
    elif invalid == "digest":
        refs[0]["digest"] = digest_bytes(b"wrong")
    elif invalid == "conflict":
        refs.append({**refs[0], "size": cast(int, refs[0]["size"]) + 1})
    elif invalid == "malformed":
        metadata["asset_refs"] = "not a list"
    else:
        payload = cast(dict[str, object], wire["payload"])
        blobs[cast(int, payload["blob"])] = (
            json.dumps(record, indent=2).encode()
            if invalid == "noncanonical"
            else b"not a document"
        )
    if invalid != "malformed":
        metadata["asset_refs"] = refs
    blobs[cast(int, wire["metaBlob"])] = default_encode(metadata)
    with pytest.raises(EncodedPayloadValidationError):
        codec.decode(wire, blobs, [])


@pytest.mark.parametrize("type_id", ["dinkster.layers", "comfy.LAYERS"])
def test_document_rendition_needs_no_raster_resources(
    type_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = document().to_record()
    for resource in record["resources"].values():
        resource.pop("inline")
    expected = ImageDocument.from_record(record)

    def forbid_raster(*args, **kwargs):
        pytest.fail("document retrieval must not render or read raster resources")

    monkeypatch.setattr(ImageDocument, "render", forbid_raster)
    monkeypatch.setattr(ImageDocument, "read_resource", forbid_raster)
    registry = TypeRegistry()
    register = register_image_types if type_id == "dinkster.layers" else register_comfy_layers
    register(registry)
    register(registry)
    spec = registry.spec(type_id)
    value = registry.wrap(type_id, spec.decode(spec.encode(expected)))
    rendition = registry.render(value, "document")
    assert rendition.mime == IMAGE_DOCUMENT_MEDIA_TYPE
    assert rendition.data == expected.data
    assert [s.kind for s in registry.renditions_of(type_id) if s.default] == [
        "image" if type_id == "dinkster.layers" else "document"
    ]


class LayerNodeSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.layer-node-source",
            outputs=(
                OutputSpec("image", TypeExpr.concrete("IMAGE")),
                OutputSpec(
                    "mask",
                    TypeExpr.concrete("MASK"),
                    mask_polarity="transparency",
                    mask_semantic="alpha",
                ),
            ),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(
            image=annotate_image(
                np.array([[[[0.5, 0, 0, 0.5]]]], dtype=np.float32),
                alpha="premultiplied",
                color={"primaries": 9, "transfer": 16, "range": 1, "matrix": 9, "bit_depth": 10},
            ),
            mask=annotate_mask(
                np.zeros((1, 1, 1), dtype=np.float32),
                polarity="transparency",
                semantic="alpha",
            ),
        )


def execute_graph_node(node: type[Node], inputs: dict[str, object]) -> dict[str, object]:
    registry = TypeRegistry()
    register_core_types(registry)
    register_media_types(registry)
    register_image_types(registry)
    nodes = [node]
    worker = InProcessWorker(build_node_types(nodes), registry)

    async def validate_saved(
        node_id: str, claims: Sequence[Mapping[str, object]], started_at_ns: int
    ) -> tuple[SavedArtifact, ...]:
        return SavedArtifactAuthority(os.environ["DINKSTER_MOUNTS_SNAPSHOT"]).capture_mounted(
            claims, node_id, started_at_ns
        )

    runtime = ExecutionRuntime(worker=worker, validate_saved_artifacts=validate_saved)
    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=worker,
        pin_execution=lambda: runtime,
        cache=MemoryLRUCache(),
    )

    async def run() -> dict[str, object]:
        graph = Graph(nodes={"node": GraphNode(node.define_schema().node_type, inputs)})
        result = await engine.run(graph, ["node"])
        return {name: value.resolve() for name, value in result.outputs["node"].items()}

    return run_scenario(run())


def test_split_layer_integer_regions_survive_the_worker_boundary() -> None:
    registry = TypeRegistry()
    register_core_types(registry)
    register_image_types(registry)
    direct = cast("list[Region]", SplitLayers.execute(layers=document())["regions"])
    assert [region.to_record() for region in direct] == [{"x": 0, "y": 0, "width": 2, "height": 1}]
    assert all(
        type(coordinate) is int
        for coordinate in (direct[0].x, direct[0].y, direct[0].width, direct[0].height)
    )

    value = registry.wrap("list<REGION>", direct)
    header, blobs, segments = encode_result(
        ValueCodec(registry, use_shm=False, accept_shm=False),
        InvocationResult(outputs={"regions": value}),
        "invocation",
        0.0,
    )
    assert not segments
    outputs, _ = decode_result_outputs(
        ValueCodec(registry, use_shm=False, accept_shm=False), header, blobs, []
    )
    decoded = cast("list[Region]", outputs["regions"].resolve())
    assert [region.to_record() for region in decoded] == [{"x": 0, "y": 0, "width": 2, "height": 1}]
    assert all(
        type(coordinate) is int
        for coordinate in (decoded[0].x, decoded[0].y, decoded[0].width, decoded[0].height)
    )


def test_node_build_edit_split_selectors_and_api_flatten_match() -> None:
    source = np.array([[[[1, 0, 0], [0, 1, 0]]]], dtype=np.float32)
    expected_value = AddLayer.execute(image=source)["layers"]
    value = execute_graph_node(AddLayer, {"image": TypedLiteral("IMAGE", source.tolist())})[
        "layers"
    ]
    assert isinstance(value, ImageDocument)
    assert isinstance(expected_value, ImageDocument)
    assert value.data == expected_value.data
    record = value.to_record()
    identifier = record["rootLayerIds"][0]
    components = record["layers"][identifier]["transform"]["components"]
    edited = EditLayers.execute(
        layers=value,
        commands=json.dumps(
            [
                {
                    "op": "transform",
                    "id": identifier,
                    "components": {**components, "flipHorizontal": True},
                }
            ]
        ),
    )["layers"]
    assert isinstance(edited, ImageDocument)
    result = FlattenLayers.execute(layers=edited)
    np.testing.assert_array_equal(
        result["image"], np.array([[[[0, 1, 0], [1, 0, 0]]]], dtype=np.float32)
    )
    split_value = append_raster(
        edited, raster_png(np.array([[[0, 0, 1], [0, 0, 1]]], dtype=np.float32))
    )
    split_ids = split_value.to_record()["rootLayerIds"]
    split_value = apply_commands(split_value, [{"op": "group", "ids": split_ids}])
    expected_split = cast(dict[str, Any], SplitLayers.execute(layers=split_value))
    split = cast(
        dict[str, Any],
        execute_graph_node(
            SplitLayers, {"layers": TypedLiteral("dinkster.layers", split_value.to_record())}
        ),
    )
    assert len(split["images"]) == len(split["masks"]) == len(split["regions"]) == 2
    for name in ("images", "masks"):
        for actual, expected in zip(split[name], expected_split[name], strict=True):
            np.testing.assert_array_equal(actual, expected)
    assert split["regions"] == expected_split["regions"]
    np.testing.assert_array_equal(split["images"][0], result["image"])
    np.testing.assert_array_equal(split["masks"][0], result["transparency_mask"])
    assert split["regions"][0].width == 2


def test_edit_layers_node_exposes_structural_and_layer_commands() -> None:
    value = AddLayer.execute(image=np.array([[[[1, 0, 0], [0, 1, 0]]]], dtype=np.float32))["layers"]
    value = AddLayer.execute(
        layers=value,
        image=np.array([[[[0, 0, 1], [0, 0, 1]]]], dtype=np.float32),
        name="Top",
    )["layers"]
    assert isinstance(value, ImageDocument)
    record = value.to_record()
    base, top = record["rootLayerIds"]
    components = {
        **record["layers"][top]["transform"]["components"],
        "x": 1,
        "width": 1,
        "rotation": 0.25,
        "flipHorizontal": True,
        "flipVertical": True,
    }
    commands = [
        {"op": "reorder", "ids": [top, base]},
        {"op": "group", "ids": [top, base], "name": "Node Group"},
        {
            "op": "layer",
            "id": top,
            "changes": {"name": "Edited", "blendMode": "screen", "opacity": 32768},
        },
        {
            "op": "layer",
            "id": base,
            "changes": {"clipping": "clip-to-previous"},
        },
        {"op": "transform", "id": top, "components": components},
    ]
    expected = apply_commands(value, commands)
    edited = execute_graph_node(
        EditLayers,
        {
            "layers": TypedLiteral("dinkster.layers", value.to_record()),
            "commands": json.dumps(commands),
        },
    )["layers"]
    assert isinstance(edited, ImageDocument)
    assert edited.data == expected.data
    record = edited.to_record()
    group_id = record["rootLayerIds"][0]
    assert record["layers"][group_id]["name"] == "Node Group"
    assert record["layers"][group_id]["childLayerIds"] == [top, base]
    assert [record["layers"][identifier]["z_index"] for identifier in (top, base)] == [0, 1]
    assert record["layers"][top]["name"] == "Edited"
    assert record["layers"][top]["blendMode"] == "screen"
    assert record["layers"][top]["opacity"] == 32768
    assert record["layers"][base]["clipping"] == "clip-to-previous"
    transform = record["layers"][top]["transform"]
    assert transform["components"] == components
    assert {key: transform[key] for key in ("a", "b", "c", "d", "tx", "ty")} == (
        affine_from_components(components)
    )
    selected = execute_graph_node(
        FlattenLayers,
        {
            "layers": TypedLiteral("dinkster.layers", edited.to_record()),
            "selector": f"layer:{top}",
        },
    )
    expected_image, expected_mask = flatten(edited, f"layer:{top}")
    np.testing.assert_array_equal(selected["image"], expected_image)
    np.testing.assert_array_equal(selected["transparency_mask"], expected_mask)

    remove = [{"op": "remove", "id": group_id}]
    expected_removed = apply_commands(edited, remove)
    removed = execute_graph_node(
        EditLayers,
        {
            "layers": TypedLiteral("dinkster.layers", edited.to_record()),
            "commands": json.dumps(remove),
        },
    )["layers"]
    assert isinstance(removed, ImageDocument)
    assert removed.data == expected_removed.data
    assert removed.to_record()["layers"] == {}


@pytest.mark.parametrize("combine", ("multiply", "add", "subtract", "intersect"))
@pytest.mark.parametrize("channel", ("alpha", "luminance"))
def test_edit_layers_node_exposes_raster_mask_commands(combine: str, channel: str) -> None:
    value = document()
    record = value.to_record()
    owner = record["rootLayerIds"][0]
    resource = record["layers"][owner]["resourceId"]
    commands = [
        {
            "op": "add_mask",
            "ownerLayerId": owner,
            "mask": {
                "resourceId": resource,
                "sourceRect": {"x": 0, "y": 0, "width": 2, "height": 1},
                "enabled": False,
                "invert": True,
                "opacity": 32768,
                "combineMode": combine,
                "channel": channel,
            },
        }
    ]
    expected = apply_commands(value, commands)
    edited = execute_graph_node(
        EditLayers,
        {
            "layers": TypedLiteral("dinkster.layers", value.to_record()),
            "commands": json.dumps(commands),
        },
    )["layers"]
    assert isinstance(edited, ImageDocument)
    assert edited.data == expected.data
    record = edited.to_record()
    mask_id = record["layers"][owner]["maskIds"][0]
    assert record["masks"][mask_id] == {
        "id": mask_id,
        "kind": "raster",
        "ownerLayerId": owner,
        "resourceId": resource,
        "sourceRect": {"x": 0, "y": 0, "width": 2, "height": 1},
        "enabled": False,
        "invert": True,
        "opacity": 32768,
        "transform": {"a": 1_000_000, "b": 0, "c": 0, "d": 1_000_000, "tx": 0, "ty": 0},
        "combineMode": combine,
        "channel": channel,
    }
    selected = execute_graph_node(
        FlattenLayers,
        {
            "layers": TypedLiteral("dinkster.layers", edited.to_record()),
            "selector": f"mask:{mask_id}",
        },
    )
    expected_image, expected_mask = flatten(edited, f"mask:{mask_id}")
    np.testing.assert_array_equal(selected["image"], expected_image)
    np.testing.assert_array_equal(selected["transparency_mask"], expected_mask)

    components = {
        "x": 1,
        "y": 0,
        "width": 1,
        "height": 1,
        "rotation": 0,
        "flipHorizontal": True,
        "flipVertical": False,
        "sourceWidth": 2,
        "sourceHeight": 1,
    }
    enable = [
        {"op": "mask", "id": mask_id, "changes": {"enabled": True, "invert": False}},
        {"op": "transform", "kind": "mask", "id": mask_id, "components": components},
    ]
    expected_enabled = apply_commands(edited, enable)
    enabled = execute_graph_node(
        EditLayers,
        {
            "layers": TypedLiteral("dinkster.layers", edited.to_record()),
            "commands": json.dumps(enable),
        },
    )["layers"]
    assert isinstance(enabled, ImageDocument)
    assert enabled.data == expected_enabled.data
    assert enabled.to_record()["masks"][mask_id]["enabled"] is True
    assert enabled.to_record()["masks"][mask_id]["invert"] is False
    assert enabled.to_record()["masks"][mask_id]["transform"]["components"] == components
    remove = [{"op": "remove_mask", "id": mask_id}]
    expected_removed = apply_commands(enabled, remove)
    removed = execute_graph_node(
        EditLayers,
        {
            "layers": TypedLiteral("dinkster.layers", enabled.to_record()),
            "commands": json.dumps(remove),
        },
    )["layers"]
    assert isinstance(removed, ImageDocument)
    assert removed.data == expected_removed.data
    assert removed.to_record()["masks"] == {}
    assert removed.to_record()["layers"][owner]["maskIds"] == []


def test_layer_creation_normalizes_premultiplied_pixels_before_png_storage() -> None:
    source = LayerNodeSource.execute()
    expected = AddLayer.execute(image=source["image"], mask=source["mask"])["layers"]
    registry = TypeRegistry()
    register_core_types(registry)
    register_media_types(registry)
    register_image_types(registry)
    nodes: list[type[Node]] = [LayerNodeSource, AddLayer]
    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
    )

    async def run() -> ImageDocument:
        result = await engine.run(
            Graph(
                nodes={
                    "source": GraphNode("test.layer-node-source"),
                    "add": GraphNode(
                        "dinkster.layers.add",
                        {"image": Link("source", "image"), "mask": Link("source", "mask")},
                    ),
                }
            ),
            ["add"],
        )
        value = result.outputs["add"]["layers"].resolve()
        assert isinstance(value, ImageDocument)
        return value

    layered = run_scenario(run())
    assert isinstance(expected, ImageDocument)
    assert layered.data == expected.data
    record = layered.to_record()
    assert len(record["masks"]) == 1
    outputs = FlattenLayers.execute(layers=layered)
    assert media_semantics(outputs["image"])["color"] == {
        "primaries": 9,
        "transfer": 16,
        "range": 1,
        "matrix": 9,
        "bit_depth": 10,
    }
    assert media_semantics(outputs["transparency_mask"]) == {
        "polarity": "transparency",
        "semantic": "alpha",
    }
    actual = np.asarray(outputs["image"])
    np.testing.assert_array_equal(actual[..., :3], [[[[1, 0, 0]]]])
    np.testing.assert_allclose(actual[..., 3], 0.5, atol=1 / 255, rtol=0)


@pytest.mark.parametrize("format", ["native", "ora", "psd"])
def test_layer_file_nodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format: str) -> None:
    from dinkster_assets import AssetRef, AssetVault, digest_bytes

    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps({"mounts": [{"id": "out", "root": str(tmp_path), "mode": "readwrite"}]})
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    value = document()
    saved = cast(
        AssetRef,
        execute_graph_node(
            SaveLayers,
            {
                "layers": TypedLiteral("dinkster.layers", value.to_record()),
                "target": {"mount": "out", "prefix": "layers"},
                "format": format,
            },
        )["document"],
    )
    suffix, media_type = {
        "native": (".json", IMAGE_DOCUMENT_MEDIA_TYPE),
        "ora": (".ora", "image/openraster"),
        "psd": (".psd", "image/vnd.adobe.photoshop"),
    }[format]
    assert saved.name.endswith(suffix)
    assert saved.media_type == media_type
    content = (tmp_path / saved.name).read_bytes()
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest_bytes(content)) as writer:
        writer.write(content)
        writer.commit()
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    bound = AssetRef.from_wire(saved.to_wire(), vault)
    expected = LoadLayers.execute(document=bound)["layers"]
    loaded = execute_graph_node(LoadLayers, {"document": bound.to_wire()})["layers"]
    assert isinstance(loaded, ImageDocument)
    assert isinstance(expected, ImageDocument)
    assert loaded.data == expected.data
    if format != "psd":
        assert loaded.data == value.data
    assert loaded.render().png == value.render().png


@pytest.mark.parametrize(
    ("format", "mode", "media_type", "alpha_mode"),
    [
        ("PNG", "RGBA", "image/png", "straight"),
        ("JPEG", "RGB", "image/jpeg", "opaque"),
        ("WEBP", "RGBA", "image/webp", "straight"),
    ],
)
def test_load_layers_node_creates_documents_from_raster_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format: str,
    mode: str,
    media_type: str,
    alpha_mode: str,
) -> None:
    from dinkster_assets import AssetRef, AssetVault, digest_bytes

    output = io.BytesIO()
    image = Image.new(mode, (3, 2), (20, 40, 60, 128) if mode == "RGBA" else (20, 40, 60))
    image.save(output, format, **({"lossless": True} if format == "WEBP" else {}))
    payload = output.getvalue()
    digest = digest_bytes(payload)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    reference = AssetRef(
        digest, f"source.{format.lower()}", len(payload), media_type, resolver=vault
    )
    loaded = execute_graph_node(LoadLayers, {"document": reference.to_wire()})["layers"]
    assert isinstance(loaded, ImageDocument)
    expected = load_document(payload)
    assert loaded.data == expected.data
    record = loaded.to_record()
    assert (record["canvas"]["width"], record["canvas"]["height"]) == (3, 2)
    assert len(record["rootLayerIds"]) == 1
    resource = record["resources"][record["layers"][record["rootLayerIds"][0]]["resourceId"]]
    assert (resource["mediaType"], resource["alphaMode"], resource["byteSize"]) == (
        media_type,
        alpha_mode,
        len(payload),
    )
    assert loaded.read_resource(resource) == payload
    assert flatten(loaded)[0].shape == (1, 2, 3, 4 if alpha_mode == "straight" else 3)


def test_split_layers_uses_z_order_not_object_key_order() -> None:
    value = append_raster(document(), raster_png(np.ones((1, 2, 3))), z_index=10)
    record = value.to_record()
    base, clipped = record["rootLayerIds"]
    record["layers"][base]["z_index"] = -10
    record["rootLayerIds"] = [clipped, base]
    value = ImageDocument.from_record(record)
    grouped = apply_commands(value, [{"op": "group", "ids": [clipped]}])
    split = cast(dict[str, Any], SplitLayers.execute(layers=grouped))
    np.testing.assert_array_equal(split["images"][0], flatten(value, f"layer:{base}")[0])


def test_layer_aliases_execute() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/layer_document_goldens.json").read_text()
    )
    for case in fixture["cases"]:
        value = from_comfy_layers(case["layers"])
        assert value.to_record()["extensions"]["comfy"]["inputs"] == case["inputs"]
        result = cast(dict[str, Any], CreateLayeredImage.execute(layers=value, compositor={}))
        np.testing.assert_allclose(result["image"], case["image"], atol=1 / 255, rtol=0)
        np.testing.assert_allclose(result["transparency_mask"], case["mask"], atol=1 / 255, rtol=0)
        background, subject = case["layers"]["layers"]
        built = AddLayer.execute(image=np.asarray(background["image"], dtype=np.float32))["layers"]
        built = LayersFromBoundingBoxes.execute(
            layers=built,
            image=np.asarray(subject["image"], dtype=np.float32),
            mask=np.asarray(subject["mask"], dtype=np.float32),
            bboxes=json.dumps([{"x": 1, "y": 1, "width": 2, "height": 2}]),
        )["layers"]
        assert isinstance(built, ImageDocument)
        identifier = built.to_record()["rootLayerIds"][-1]
        built = apply_commands(
            built,
            [
                {
                    "op": "layer",
                    "id": identifier,
                    "changes": {"opacity": round(subject["opacity"] * 65535)},
                }
            ],
        )
        result = cast(
            dict[str, Any],
            CreateLayeredImage.execute(layers=built, compositor={}, color_space="linear"),
        )
        np.testing.assert_allclose(result["image"], case["image"], atol=1 / 255, rtol=0)
        np.testing.assert_allclose(result["transparency_mask"], case["mask"], atol=1 / 255, rtol=0)


@pytest.mark.parametrize("case_index", range(3))
def test_layer_node_source_goldens(case_index: int) -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/layer_document_goldens.json").read_text()
    )
    case = fixture["aliasCases"][case_index]
    inputs = case["inputs"]
    inputs["layers"] = from_comfy_layers(inputs["layers"])
    for key in ("image", "mask"):
        if key in inputs:
            inputs[key] = np.asarray(inputs[key], dtype=np.float32)
    if case["node"] == "AddLayer":
        inputs["flip_horizontal"] = inputs.pop("flip_h")
        value = AddLayer.execute(**inputs)["layers"]
    elif case["node"] == "LayersFromBoundingBoxes":
        value = LayersFromBoundingBoxes.execute(**inputs)["layers"]
        assert isinstance(value, ImageDocument)
        record = value.to_record()
        assert record["layers"][record["rootLayerIds"][-1]]["z_index"] == 11
    else:
        value = inputs["layers"]
    result = cast(dict[str, Any], CreateLayeredImage.execute(layers=value, color_space="linear"))
    np.testing.assert_allclose(result["image"], case["image"], atol=1 / 255, rtol=0)
    np.testing.assert_allclose(result["transparency_mask"], case["mask"], atol=1 / 255, rtol=0)


def test_legacy_layer_admission_does_not_publish_migrated_rasters(monkeypatch) -> None:
    from dinkster_assets import AssetError, AssetVault
    from dinkster_assets.value import value_asset_references
    from dinkster_image.compositor_types import (
        CompositorSourceLayer,
        LayerStack,
        encode_layer_stack,
    )
    from dinkster_workers.boundary import EncodedPayloadValidationError

    pixels = np.random.default_rng(41).random((1, 64, 64, 3), dtype=np.float32)
    legacy = encode_layer_stack(
        LayerStack(
            (
                CompositorSourceLayer(
                    pixels,
                    None,
                    "Noise",
                    0,
                    0,
                    64,
                    64,
                ),
            )
        )
    )
    registry = TypeRegistry()
    register_image_types(registry)
    codec = ValueCodec(registry)
    blobs, segments = [], []
    wire, _ = codec.encode(registry.wrap("dinkster.layers", document()), blobs, segments)
    payload = cast(dict[str, object], wire["payload"])
    blobs[cast(int, payload["blob"])] = legacy

    def forbid_publication(*args, **kwargs):
        pytest.fail("legacy validation must not publish migrated raster resources")

    monkeypatch.setattr(AssetVault, "writer", forbid_publication)
    received, _ = codec.decode(wire, blobs, [])
    with pytest.raises(AssetError, match="incomplete"):
        value_asset_references({"legacy": received}, complete_registry=registry)
    blobs[cast(int, payload["blob"])] = legacy + b"trailing"
    with pytest.raises(EncodedPayloadValidationError):
        codec.decode(wire, blobs, [])


def test_delta_codecs_bind_edits_and_reject_stale_documents() -> None:
    from dinkster_assets import digest_bytes
    from dinkster_image.layer_recipe import replay_compositor

    value = document()
    identifier = value.to_record()["rootLayerIds"][0]
    delta = {
        "version": 2,
        "documentDigest": digest_bytes(value.data),
        "commands": [{"op": "layer", "id": identifier, "changes": {"visible": False}}],
    }
    registry = TypeRegistry()
    register_image_types(registry)
    for type_id in ("COMPOSITOR", "comfy.COMPOSITOR"):
        spec = registry.spec(type_id)
        decoded = spec.decode(spec.encode(delta))
        assert decoded == delta
        edited, stale = replay_compositor(value, decoded, value)
        assert not stale
        assert np.all(flatten(edited)[1] == 1)
        unchanged, stale = replay_compositor(edited, decoded, edited)
        assert stale and unchanged.data == edited.data


@pytest.mark.parametrize("type_id", ["COMPOSITOR", "comfy.COMPOSITOR"])
def test_compositor_boundary_retains_auxiliary_references_without_retrieving_document(
    type_id, monkeypatch
) -> None:
    from dinkster_assets import AssetRef, AssetVault, digest_bytes
    from dinkster_assets.value import value_asset_references
    from dinkster_image.layer_recipe import replay_compositor
    from dinkster_values import default_encode

    value = document()
    identifier = value.to_record()["rootLayerIds"][0]
    delta = {
        "version": 2,
        "documentDigest": digest_bytes(value.data),
        "commands": [{"op": "layer", "id": identifier, "changes": {"visible": False}}],
    }
    registry = TypeRegistry()
    register_image_types(registry)
    codec = ValueCodec(registry)
    wrapped = registry.wrap(type_id, delta)
    blobs, segments = [], []
    wire, _ = codec.encode(wrapped, blobs, segments)
    expected = canonical_json(delta).encode()
    assert blobs[cast(int, cast(dict[str, object], wire["payload"])["blob"])] == expected
    auxiliary = AssetRef(digest_bytes(b"auxiliary"), "source.json", 9)
    blobs[cast(int, wire["metaBlob"])] = default_encode(
        {**wrapped.meta.entries, "asset_refs": [auxiliary.to_wire()]}
    )

    def forbid_io(*args, **kwargs):
        pytest.fail("compositor admission must not retrieve or publish a document")

    monkeypatch.setattr(ImageDocument, "read_resource", forbid_io)
    monkeypatch.setattr(AssetVault, "writer", forbid_io)
    received, _ = codec.decode(wire, blobs, [])
    assert received.fingerprint == wrapped.fingerprint
    assert received.resolve() == delta
    assert value_asset_references({"delta": received}, complete_registry=registry) == {
        auxiliary.digest: auxiliary
    }
    edited, stale = replay_compositor(value, received.resolve(), value)
    assert not stale
    assert edited.to_record()["layers"][identifier]["visible"] is False
    unchanged, stale = replay_compositor(edited, received.resolve(), edited)
    assert stale and unchanged.data == edited.data
    payload = cast(dict[str, object], wire["payload"])
    blobs[cast(int, payload["blob"])] = json.dumps(delta, indent=2).encode()
    noncanonical, _ = codec.decode(wire, blobs, [])
    assert noncanonical.resolve() == delta
    first = TypeRegistry()
    first_spec = first.register(type_id)
    register_image_types(first)
    assert first.spec(type_id) is first_spec
    assert not first_spec.asset_references_complete


@pytest.mark.parametrize("type_id", ["COMPOSITOR", "comfy.COMPOSITOR"])
@pytest.mark.parametrize("data", [b"not json", b"[]", b'{"unknown":true}'])
def test_compositor_boundary_refuses_invalid_encoding_before_exposure(type_id, data) -> None:
    from dinkster_workers.boundary import EncodedPayloadValidationError

    registry = TypeRegistry()
    register_image_types(registry)
    codec = ValueCodec(registry)
    blobs, segments = [], []
    wire, _ = codec.encode(registry.wrap(type_id, {}), blobs, segments)
    payload = cast(dict[str, object], wire["payload"])
    blobs[cast(int, payload["blob"])] = data
    with pytest.raises(EncodedPayloadValidationError):
        codec.decode(wire, blobs, [])


@pytest.mark.parametrize("type_id", ["COMPOSITOR", "comfy.COMPOSITOR"])
@pytest.mark.parametrize(
    "recipe",
    [
        {},
        {"version": 1, "canvas": {"w": 3, "h": 2}, "inputs": [], "layers": []},
        {"version": 2, "documentDigest": None, "commands": [{"op": "unknown"}]},
    ],
)
def test_compositor_boundary_preserves_existing_legacy_and_deferred_command_forms(
    type_id, recipe
) -> None:
    from dinkster_assets.value import value_asset_references
    from dinkster_image.layer_recipe import replay_compositor

    registry = TypeRegistry()
    register_image_types(registry)
    codec = ValueCodec(registry)
    expected = (
        {"version": 2, "documentDigest": None, "commands": []}
        if type_id == "COMPOSITOR" and recipe == {}
        else recipe
    )
    wrapped = registry.wrap(type_id, recipe)
    blobs, segments = [], []
    wire, _ = codec.encode(wrapped, blobs, segments)
    assert blobs[cast(int, cast(dict[str, object], wire["payload"])["blob"])] == (
        canonical_json(expected).encode()
    )
    received, _ = codec.decode(wire, blobs, [])
    assert received.resolve() == expected
    assert received.fingerprint == wrapped.fingerprint
    assert value_asset_references({"recipe": received}, complete_registry=registry) == {}
    if recipe.get("commands"):
        value = document()
        with pytest.raises(InvalidDocument, match="unknown document command"):
            replay_compositor(value, received.resolve(), value)


@pytest.mark.parametrize("color_space", ["perceptual", "linear"])
def test_compositor_preview_digest_binds_repeated_apply_to_input(monkeypatch, color_space) -> None:
    from dinkster_assets import digest_bytes
    from dinkster_image import compositor as module

    value = document()
    original = value.data
    identifier = value.to_record()["rootLayerIds"][0]
    events = []
    monkeypatch.setattr(module, "report_event", lambda name, data: events.append((name, data)))
    delta = {"version": 2, "documentDigest": None, "commands": []}
    for opacity in (65535, 32768, 16384):
        delta["commands"] = [{"op": "layer", "id": identifier, "changes": {"opacity": opacity}}]
        CreateLayeredImage.execute(layers=value, compositor=delta, color_space=color_space)
        name, state = events[-1]
        assert name == "dinkster.compositor.state"
        assert state["stale"] is False
        assert state["documentDigest"] == digest_bytes(original)
        assert state["document"]["layers"][identifier]["opacity"] == opacity
        assert state["document"]["canvas"]["compositing"] == (
            "linear-premultiplied-alpha" if color_space == "linear" else "premultiplied-alpha"
        )
        delta["documentDigest"] = state["documentDigest"]
    assert value.data == original

    changed = apply_commands(value, [{"op": "layer", "id": identifier, "changes": {"name": "New"}}])
    CreateLayeredImage.execute(layers=changed, compositor=delta, color_space=color_space)
    state = events[-1][1]
    assert state["stale"] is True
    assert state["documentDigest"] == digest_bytes(changed.data)
    assert state["document"]["layers"][identifier]["opacity"] == 65535

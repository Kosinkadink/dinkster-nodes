"""Image nodes resolve document resources and match the canonical HTTP renderer."""

from __future__ import annotations

import asyncio
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetRef, AssetVault, LibraryStore, digest_bytes
from dinkster_image.layer_document import FlattenLayers
from dinkster_image.layer_io import LoadLayers
from dinkster_server.library import ServerLibrary
from PIL import Image

from tests.async_scenarios import run_scenario

MEDIA_TYPE = "application/vnd.dinkster.image-document+json"
IDENTITY = {"a": 1_000_000, "b": 0, "c": 0, "d": 1_000_000, "tx": 0, "ty": 0}


def _png(pixels: list[list[tuple[int, int, int, int]]]) -> bytes:
    output = BytesIO()
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), "RGBA").save(output, "PNG")
    return output.getvalue()


def _resource(resource_id: str, data: bytes, width: int, height: int) -> dict[str, object]:
    return {
        "id": resource_id,
        "kind": "raster",
        "digest": digest_bytes(data),
        "byteSize": len(data),
        "mediaType": "image/png",
        "width": width,
        "height": height,
        "colorSpace": "srgb",
        "channelDepth": 8,
        "alphaMode": "straight",
    }


def _layer(
    layer_id: str,
    resource_id: str,
    width: int,
    height: int,
    *,
    blend: str = "normal",
    opacity: int = 65_535,
    transform: dict[str, int] | None = None,
    mask_ids: list[str] | None = None,
    clipping: str = "none",
) -> dict[str, object]:
    return {
        "id": layer_id,
        "kind": "raster",
        "name": layer_id,
        "visible": True,
        "opacity": opacity,
        "transform": transform or IDENTITY,
        "blendMode": blend,
        "clipping": clipping,
        "maskIds": mask_ids or [],
        "resourceId": resource_id,
        "sourceRect": {"x": 0, "y": 0, "width": width, "height": height},
    }


def _document(
    width: int,
    height: int,
    layers: dict[str, dict[str, object]],
    resources: dict[str, dict[str, object]],
    roots: list[str],
    *,
    masks: dict[str, dict[str, object]] | None = None,
    next_ordinal: int = 4,
) -> dict[str, object]:
    return {
        "format": "dinkster-image",
        "formatVersion": 1,
        "lineage": "render-test",
        "canvas": {
            "width": width,
            "height": height,
            "colorSpace": "srgb",
            "channelDepth": 8,
            "compositing": "premultiplied-alpha",
        },
        "allocation": {"nextOrdinal": next_ordinal},
        "rootLayerIds": roots,
        "layers": layers,
        "masks": masks or {},
        "resources": resources,
    }


def _canonical(document: dict[str, object]) -> bytes:
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")


def test_native_node_resolves_document_children_with_the_bound_resolver(tmp_path: Path) -> None:
    source = _png([[(90, 80, 70, 255)]])
    document = _document(
        1,
        1,
        {"l1": _layer("l1", "r0", 1, 1)},
        {"r0": _resource("r0", source, 1, 1)},
        ["l1"],
        next_ordinal=2,
    )
    body = _canonical(document)
    vault = AssetVault(tmp_path / "vault")
    for data in (source, body):
        with vault.writer(digest_bytes(data)) as writer:
            writer.write(data)
            writer.commit()
    reference = AssetRef(
        digest=digest_bytes(body),
        name="document.dinkster-image.json",
        size=len(body),
        media_type=MEDIA_TYPE,
        resolver=vault,
    )
    layers = LoadLayers.execute(document=reference)["layers"]
    output = FlattenLayers.execute(layers=layers)["image"]
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 1, 1, 3)
    assert np.allclose(output[0, 0, 0], [90 / 255, 80 / 255, 70 / 255])


def test_render_api_and_flatten_node_rendition_are_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_caches import MemoryLRUCache
    from dinkster_engine import Engine, EventListener
    from dinkster_graph import Graph, GraphNode, Link, graph_to_wire
    from dinkster_image.compositor import AddLayer
    from dinkster_image.ported.image_file_tools import LoadImage
    from dinkster_image.types import register_image_types
    from dinkster_schema import build_node_types, build_schemas
    from dinkster_server import create_app
    from dinkster_values import TypeRegistry, register_core_types
    from dinkster_workers import InProcessWorker

    nodes = [LoadImage, AddLayer, FlattenLayers]
    schemas = build_schemas(nodes)
    library = ServerLibrary(
        vault=AssetVault(tmp_path / "vault"),
        store=LibraryStore(tmp_path / "library.sqlite"),
    )
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(library.vault.root))

    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_image_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        client = TestClient(TestServer(create_app(factory, schemas, library=library)))
        await client.start_server()
        source = _png(
            [
                [(255, 0, 0, 255), (0, 255, 0, 128), (0, 0, 255, 0), (4, 8, 12, 64)],
                [(15, 31, 63, 255), (127, 95, 63, 192), (9, 7, 5, 1), (1, 2, 3, 254)],
                [(20, 40, 60, 80), (80, 60, 40, 20), (3, 5, 7, 9), (250, 240, 230, 220)],
            ]
        )
        try:
            uploaded = await client.post(
                "/api/assets/media?scope=local&kind=media/image&name=source.png",
                data=source,
                headers={"Content-Type": "image/png"},
            )
            assert uploaded.status == 201, await uploaded.text()
            asset = (await uploaded.json())["asset"]
            graph = Graph(
                nodes={
                    "load": GraphNode("dinkster.image.load_image", {"image": asset}),
                    "layers": GraphNode(
                        "dinkster.layers.add",
                        {
                            "image": Link("load", "IMAGE"),
                            "mask": Link("load", "MASK"),
                        },
                    ),
                    "flatten": GraphNode(
                        "dinkster.layers.flatten", {"layers": Link("layers", "layers")}
                    ),
                }
            )
            submitted = await client.post(
                "/api/jobs",
                json={
                    "clientId": "c1",
                    "jobId": "j1",
                    "graph": graph_to_wire(graph),
                    "targets": ["flatten"],
                },
            )
            assert submitted.status == 202, await submitted.text()
            async with asyncio.timeout(5):
                while True:
                    status = await (await client.get("/api/jobs/c1/j1")).json()
                    if status["state"] in ("completed", "failed", "cancelled"):
                        assert status["state"] == "completed", status
                        break
                    await asyncio.sleep(0.01)

            values = "/api/values?clientId=c1&jobId=j1"
            document_response = await client.get(
                values + "&nodeId=layers&outputId=layers&rendition=document"
            )
            assert document_response.status == 200, await document_response.text()
            document = await document_response.read()
            adopted = await client.post(
                "/api/assets/image-document?scope=local",
                data=document,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert adopted.status == 201, await adopted.text()
            rendered = await client.post(
                f"/api/assets/{digest_bytes(document)}/render?scope=local",
                json={"selector": "composite"},
            )
            assert rendered.status == 201, await rendered.text()
            render_asset = (await rendered.json())["asset"]
            render_response = await client.get(f"/api/assets/{render_asset['digest']}")
            assert render_response.status == 200, await render_response.text()

            flatten_response = await client.get(
                values + "&nodeId=flatten&outputId=image&rendition=png"
            )
            assert flatten_response.status == 200, await flatten_response.text()
            assert await flatten_response.read() == await render_response.read()
        finally:
            await client.close()

    run_scenario(scenario())

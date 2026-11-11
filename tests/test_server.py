"""Production layer and video nodes expose their registered HTTP renditions."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, cast

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_graph import Graph, GraphNode, graph_to_wire
from dinkster_schema import (
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_server import create_app
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario


def submit_body(graph: Graph, targets: list[str], **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "clientId": "c1",
        "jobId": "j1",
        "graph": graph_to_wire(graph),
        "targets": targets,
    }
    body.update(overrides)
    return body


async def finish_job(
    client: TestClient, graph: Graph, targets: list[str], job_id: str = "j1"
) -> Any:
    resp = await client.post("/api/jobs", json=submit_body(graph, targets, jobId=job_id))
    assert resp.status == 202
    async with asyncio.timeout(5):
        while True:
            status = await (await client.get(f"/api/jobs/c1/{job_id}")).json()
            if status["state"] in ("completed", "failed", "cancelled"):
                assert status["state"] == "completed"
                return status
            await asyncio.sleep(0.01)


def values_url(node_id: str, output_id: str = "out", job_id: str = "j1", **extra: str) -> str:
    params = {
        "clientId": "c1",
        "jobId": job_id,
        "nodeId": node_id,
        "outputId": output_id,
        **extra,
    }
    return "/api/values?" + "&".join(f"{k}={v}" for k, v in params.items())


@pytest.mark.parametrize("type_id", ["dinkster.layers", "comfy.LAYERS"])
def test_values_endpoint_layer_document_rendition(
    type_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np
    from dinkster_graph import TypedLiteral
    from dinkster_image.compositor import AddLayer
    from dinkster_image.types import register_image_types
    from dinkster_image_document.compat import register_comfy_layers
    from dinkster_image_document.document import ImageDocument
    from dinkster_image_document.format import IMAGE_DOCUMENT_MEDIA_TYPE

    pixels = np.array([[[[1, 0, 0, 1], [0, 1, 0, 0.5]]]], dtype=np.float32)
    expected = cast(ImageDocument, AddLayer.execute(image=pixels)["layers"]).data

    class CompatLayers(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.compat_layers",
                outputs=(OutputSpec("layers", TypeExpr.concrete("comfy.LAYERS")),),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            return AddLayer.execute(image=pixels)

    nodes: list[type[Node]] = [AddLayer] if type_id == "dinkster.layers" else [CompatLayers]
    schemas = build_schemas(nodes)

    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        if type_id == "dinkster.layers":
            register_image_types(registry)
        else:
            register_comfy_layers(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    def forbid_raster(*args, **kwargs):
        pytest.fail("document retrieval must not render or read raster resources")

    async def scenario() -> None:
        client = TestClient(TestServer(create_app(factory, schemas)))
        await client.start_server()
        try:
            node = (
                GraphNode(
                    "dinkster.layers.add",
                    {"image": TypedLiteral("IMAGE", pixels.tolist())},
                )
                if type_id == "dinkster.layers"
                else GraphNode("test.compat_layers", {})
            )
            await finish_job(client, Graph(nodes={"layers": node}), ["layers"])
            monkeypatch.setattr(ImageDocument, "render", forbid_raster)
            monkeypatch.setattr(ImageDocument, "read_resource", forbid_raster)
            url = values_url("layers", output_id="layers")
            resp = await client.get(url)
            assert resp.status == 200
            data = await resp.json()
            assert data["available"] is True
            assert data["descriptor"]["typeId"] == type_id
            assert "value" not in data["descriptor"]
            renditions = [
                {
                    "kind": "image",
                    "mime": "image/png",
                    "default": True,
                    "cacheKey": "image",
                }
            ]
            assert data["renditions"] == (renditions if type_id == "dinkster.layers" else []) + [
                {
                    "kind": "document",
                    "mime": IMAGE_DOCUMENT_MEDIA_TYPE,
                    "default": type_id == "comfy.LAYERS",
                    "cacheKey": "document",
                }
            ]

            resp = await client.get(url + "&rendition=document")
            assert resp.status == 200
            assert resp.headers["Content-Type"] == IMAGE_DOCUMENT_MEDIA_TYPE
            assert resp.headers["X-Dinkster-Type-Id"] == type_id
            assert resp.headers["X-Dinkster-Rendition"] == "document"
            fingerprint = data["descriptor"]["fingerprint"]
            assert resp.headers["X-Dinkster-Fingerprint"] == fingerprint
            etag = resp.headers["ETag"]
            assert etag == f'"{fingerprint}/document"'
            assert await resp.read() == expected

            monkeypatch.setattr(TypeRegistry, "render", forbid_raster)
            resp = await client.get(url + "&rendition=document", headers={"If-None-Match": etag})
            assert resp.status == 304
            assert resp.headers["ETag"] == etag
            assert await resp.read() == b""
        finally:
            await client.close()

    run_scenario(scenario())


def test_values_endpoint_omits_original_rendition_for_component_video() -> None:
    import numpy as np
    from dinkster_api.v1 import VIDEO_TYPE
    from dinkster_video import register_types as register_video_types
    from dinkster_video.video_components import CreateVideo

    class ComponentVideo(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.component_video",
                outputs=(OutputSpec("video", TypeExpr.concrete(VIDEO_TYPE)),),
            )

        @classmethod
        async def execute(cls) -> Mapping[str, object]:
            frames = np.zeros((2, 4, 4, 3), dtype=np.float32)
            return cls.outputs(video=CreateVideo.execute(images=frames, fps=4.0)["video"])

    nodes: list[type[Node]] = [ComponentVideo]
    schemas = build_schemas(nodes)

    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_video_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        app = create_app(factory, schemas)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            graph = Graph(nodes={"v": GraphNode("test.component_video", {})})
            await finish_job(client, graph, ["v"])
            url = values_url("v", output_id="video")
            response = await client.get(url)
            data = await response.json()
            assert response.status == 200
            assert data["descriptor"]["typeId"] == VIDEO_TYPE
            assert data["renditions"] == []

            response = await client.get(url + "&rendition=original")
            data = await response.json()
            assert (response.status, data["reason"]) == (406, "no-rendition")
            assert data["renditions"] == []
        finally:
            await client.close()

    run_scenario(scenario())

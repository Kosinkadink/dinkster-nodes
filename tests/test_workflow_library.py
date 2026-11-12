"""Native saves complete into the project catalog with their media grants."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    AssetVault,
    AssetWriter,
    LibraryStore,
    MountDef,
    MountTable,
    digest_bytes,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener, ExecutionRuntime
from dinkster_protocol import AssetFolderContext, SavedArtifact
from dinkster_schema import Node, NodeSchema, OutputSpec, TypeExpr, build_node_types, build_schemas
from dinkster_server import ServerLibrary, create_app
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario

STRING = TypeExpr.concrete("STRING")


@pytest.mark.parametrize("kind", ("media/image", "media/audio", "media/video", "asset/file"))
def test_native_media_saves_complete_into_kind_filtered_project_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import numpy as np
    from dinkster_audio.audio_save import SaveAudio
    from dinkster_image.ported.image_file_tools import SaveImage
    from dinkster_text.save_text import SaveText
    from dinkster_video.video_save import SaveVideo
    from dinkster_video_runtime import assemble_video
    from dinkster_workers.saved_artifacts import SavedArtifactAuthority

    root = tmp_path / "output"
    root.mkdir()
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, output_mount="renders")
    table.add(MountDef("renders", root, "readwrite"))
    table.scan("renders")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    class NativeSave(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.native_save",
                outputs=(OutputSpec("out", STRING),),
                output_node=True,
                idempotent=False,
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            if kind == "media/image":
                SaveImage.execute(
                    images=np.full((1, 3, 5, 3), 0.25, dtype=np.float32),
                    filename_prefix="saved/result",
                    mount="renders",
                )
            elif kind == "media/audio":
                SaveAudio.execute(
                    audio={
                        "waveform": np.zeros((1, 1, 800), dtype=np.float32),
                        "sample_rate": 8000,
                    },
                    filename_prefix="saved/result",
                    mount="renders",
                )
            elif kind == "media/video":
                video = assemble_video(np.full((2, 16, 16, 3), 0.75, dtype=np.float32), fps=2)
                SaveVideo.execute(
                    video=video,
                    filename_prefix="saved/result",
                    mount="renders",
                    format="mp4",
                    codec="h264",
                )
            else:
                SaveText.execute(
                    text="ordinary text, not encoded media",
                    filename_prefix="saved/result",
                    mount="renders",
                )
            return cls.outputs(out="saved")

    schemas = build_schemas([NativeSave])

    async def validate_saved(
        node_id: str, claims: Sequence[Mapping[str, object]], started_at_ns: int
    ) -> tuple[SavedArtifact, ...]:
        return SavedArtifactAuthority(snapshot).capture_mounted(claims, node_id, started_at_ns)

    def engine(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = InProcessWorker(build_node_types([NativeSave]), registry)
        runtime = ExecutionRuntime(worker=worker, validate_saved_artifacts=validate_saved)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=worker,
            pin_execution=lambda: runtime,
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"),
            store=LibraryStore(tmp_path / "catalog.sqlite"),
            local_resolver=table,
            resolver=table,
            upload_writer=AssetWriter(table),
            project_folders_for=lambda _project: AssetFolderContext(("renders",), "renders"),
        )
        client = TestClient(TestServer(create_app(engine, schemas, library=library)))
        await client.start_server()
        try:
            response = await client.post(
                "/api/jobs",
                json={
                    "clientId": "client",
                    "jobId": "save",
                    "project": "film",
                    "graph": {"nodes": {"save": {"nodeType": "test.native_save", "inputs": {}}}},
                    "targets": ["save"],
                },
            )
            assert response.status == 202
            submitted = await response.json()
            async with asyncio.timeout(10):
                while True:
                    status = await (await client.get("/api/jobs/client/save")).json()
                    if status["state"] in {"completed", "failed", "cancelled"}:
                        break
                    await asyncio.sleep(0)
            assert status["state"] == "completed", status
            (artifact,) = status["artifacts"]
            path = root / artifact["virtualPath"].split("/", 2)[2]
            assert artifact["digest"] == digest_bytes(path.read_bytes())
            response = await client.get(
                "/api/assets/catalog",
                params={
                    "scope": "local",
                    "project": "film",
                    "kind": kind,
                },
            )
            assert response.status == 200
            (item,) = (await response.json())["assets"]
            assert item["asset"]["digest"] == artifact["digest"]
            (record,) = library.store.query_assets("local", project="film", kind=kind)[0]
            assert dict(record.origin) == {
                "kind": "produced",
                "job": submitted["jobRef"],
                "node": "save",
            }
            assert record.kind == kind
            if kind != "asset/file":
                assert library.store.media_grant("local", artifact["digest"], kind) is not None
            assert library.store.query_assets("other", project="film", kind=kind) == []
            assert library.vault.digests() == []
        finally:
            await client.close()

    run_scenario(scenario())

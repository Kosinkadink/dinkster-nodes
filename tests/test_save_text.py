"""Mounted SaveText behavior under the native ComfyUI interface."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_api.v1 import MOUNT_TYPE, ComboWidget, StringWidget
from dinkster_assets import (
    AssetError,
    AssetVault,
    AssetWriter,
    LibraryStore,
    MountSnapshotWriter,
    digest_bytes,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionError, ExecutionRuntime
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral, graph_to_wire
from dinkster_protocol import AssetFolderContext, SavedArtifact
from dinkster_schema import build_node_types, build_schemas
from dinkster_server import ServerLibrary, create_app
from dinkster_text.save_text import SaveText
from dinkster_utilities import register_types
from dinkster_utilities.primitive import Mount, PrimitiveString
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker
from dinkster_workers.saved_artifacts import SavedArtifactAuthority

from dinkster.compose import PackSpec, ServingComposer
from tests.async_scenarios import run_scenario

pytestmark = pytest.mark.usefixtures("saved_artifact_context")


def test_save_text_schema_preserves_core_contract_and_appends_mount() -> None:
    schema = SaveText.schema()
    assert schema.node_type == "dinkster.text.save_text"
    assert schema.display_name == "Save Text"
    assert schema.category == "text"
    assert tuple(spec.id for spec in schema.inputs) == (
        "text",
        "filename_prefix",
        "format",
        "mount",
    )
    assert schema.inputs[0].pin_first
    assert schema.inputs[0].widget == StringWidget(multiline=False)
    assert schema.inputs[1].default == "ComfyUI"
    assert schema.inputs[2].default == "txt"
    assert isinstance(schema.inputs[2].widget, ComboWidget)
    assert isinstance(schema.inputs[2].widget.options, tuple)
    assert tuple(schema.inputs[2].widget.options) == ("txt", "csv", "md", "json")
    assert not schema.inputs[3].required
    assert schema.inputs[3].default is None
    assert tuple(spec.id for spec in schema.outputs) == ("text",)
    assert schema.output_node
    assert not schema.idempotent


def _snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    roots = {name: tmp_path / name for name in ("default", "alternate", "readonly")}
    for root in roots.values():
        root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {"id": "default", "root": str(roots["default"]), "mode": "readwrite"},
                    {"id": "alternate", "root": str(roots["alternate"]), "mode": "readwrite"},
                    {"id": "readonly", "root": str(roots["readonly"]), "mode": "read"},
                ],
                "outputMount": "default",
            }
        ),
        "utf-8",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    roots["snapshot"] = snapshot
    return roots


def test_save_text_writes_default_and_explicit_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _snapshot(tmp_path, monkeypatch)
    original = '{"greeting":"cafe"}\n'
    assert SaveText.execute(text=original, filename_prefix="notes/default", format="json") == {
        "text": original
    }
    assert SaveText.execute(
        text="alternate\n",
        filename_prefix="notes/%width%x%height%-%unknown%",
        format="md",
        mount="alternate",
    ) == {"text": "alternate\n"}
    assert (roots["default"] / "notes" / "default_00001.json").read_bytes() == (
        b'{\n  "greeting": "cafe"\n}'
    )
    assert (roots["alternate"] / "notes" / "1x1-%unknown%_00001.md").read_bytes() == b"alternate\n"


def test_save_text_collisions_and_security_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _snapshot(tmp_path, monkeypatch)
    SaveText.execute(text="one", filename_prefix="same")
    SaveText.execute(text="two", filename_prefix="same")
    assert (roots["default"] / "same_00001.txt").read_bytes() == b"one"
    assert (roots["default"] / "same_00002.txt").read_bytes() == b"two"
    with pytest.raises(AssetError, match="read-only"):
        SaveText.execute(text="blocked", mount="readonly")
    with pytest.raises(AssetError):
        SaveText.execute(text="blocked", filename_prefix="../escape")
    with pytest.raises(ValueError, match="Unsupported text format"):
        SaveText.execute(text="blocked", format="../evil")
    roots["snapshot"].write_text(json.dumps({"mounts": [], "outputMount": "default"}), "utf-8")
    with pytest.raises(AssetError, match="no ready mount"):
        SaveText.execute(text="revoked")


def test_save_text_accepts_shared_mount_and_name_wires_without_changing_legacy_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _snapshot(tmp_path, monkeypatch)

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)
        node_types = (SaveText, Mount, PrimitiveString)
        worker = InProcessWorker(build_node_types(node_types), registry)

        async def validate_saved(
            node_id: str, claims: Sequence[Mapping[str, object]], started_at_ns: int
        ) -> tuple[SavedArtifact, ...]:
            return SavedArtifactAuthority(roots["snapshot"]).capture_mounted(
                claims, node_id, started_at_ns
            )

        runtime = ExecutionRuntime(worker=worker, validate_saved_artifacts=validate_saved)
        engine = Engine(
            schemas=build_schemas(node_types),
            registry=registry,
            worker=worker,
            pin_execution=lambda: runtime,
            cache=MemoryLRUCache(),
        )
        graph = Graph(
            nodes={
                "mount": GraphNode("dinkster.utilities.mount", {"value": "alternate"}),
                "name": GraphNode("dinkster.utilities.primitive_string", {"value": "wired/a"}),
                "legacy_mount": GraphNode(
                    "dinkster.utilities.primitive_string", {"value": "alternate"}
                ),
                "first": GraphNode(
                    "dinkster.text.save_text",
                    {
                        "text": "first",
                        "mount": Link("mount", "mount"),
                        "filename_prefix": Link("name", "STRING"),
                    },
                ),
                "second": GraphNode(
                    "dinkster.text.save_text",
                    {"text": "second", "mount": Link("mount", "mount"), "filename_prefix": "b"},
                ),
                "legacy": GraphNode(
                    "dinkster.text.save_text",
                    {
                        "text": "legacy",
                        "mount": Link("legacy_mount", "STRING"),
                        "filename_prefix": "old",
                    },
                ),
                "typed": GraphNode(
                    "dinkster.text.save_text",
                    {
                        "text": "typed",
                        "mount": TypedLiteral(MOUNT_TYPE, "alternate"),
                        "filename_prefix": "chosen",
                    },
                ),
                "saved": GraphNode(
                    "SaveText",
                    {"text": "saved", "mount": "alternate", "filename_prefix": "restored"},
                ),
                "default": GraphNode("dinkster.text.save_text", {"text": "default"}),
            }
        )
        targets = ("first", "second", "legacy", "typed", "saved", "default")
        result = await engine.run(graph, list(targets))
        expected = {
            "first": ("alternate", "wired/a", b"first"),
            "second": ("alternate", "b", b"second"),
            "legacy": ("alternate", "old", b"legacy"),
            "typed": ("alternate", "chosen", b"typed"),
            "saved": ("alternate", "restored", b"saved"),
            "default": ("default", "ComfyUI", b"default"),
        }
        for node_id, (mount, prefix, data) in expected.items():
            assert result.outputs[node_id]["text"].resolve() == data.decode()
            assert (roots[mount] / f"{prefix}_00001.txt").read_bytes() == data
        assert {artifact.virtual_path for artifact in result.artifacts} == {
            f"mounts/{mount}/{prefix}_00001.txt" for mount, prefix, _data in expected.values()
        }
        for mount, name, message in (
            ("alternate", "../escape", "prefix"),
            ("readonly", "denied", "read-only"),
        ):
            refused = Graph(
                nodes={
                    "mount": GraphNode("dinkster.utilities.mount", {"value": mount}),
                    "name": GraphNode("dinkster.utilities.primitive_string", {"value": name}),
                    "save": GraphNode(
                        "dinkster.text.save_text",
                        {
                            "text": "blocked",
                            "mount": Link("mount", "mount"),
                            "filename_prefix": Link("name", "STRING"),
                        },
                    ),
                }
            )
            with pytest.raises(ExecutionError, match=message):
                await engine.run(refused, ["save"])
        assert not list(roots["readonly"].rglob("*.txt"))
        assert not (tmp_path / "escape_00001.txt").exists()

    run_scenario(scenario())


@pytest.mark.parametrize("in_process", (True, False))
def test_save_text_executes_through_the_ordinary_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    roots = _snapshot(tmp_path, monkeypatch)

    async def scenario() -> None:
        manifest = (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-text/dinkster-pack.toml"
        )
        composer = ServingComposer(
            worker_env={
                "DINKSTER_MOUNTS_SNAPSHOT": os.environ["DINKSTER_MOUNTS_SNAPSHOT"],
                "PYTHONPATH": str(tmp_path),
            }
        )
        try:
            await composer.add_pack(
                PackSpec(manifest=manifest, in_process=in_process, trust_reserved=True)
            )
            graph = Graph(
                nodes={
                    "save": GraphNode(
                        "dinkster.text.save_text",
                        {
                            "text": "manifest\n",
                            "filename_prefix": "manifest/%width%x%height%",
                            "format": "txt",
                        },
                    )
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for counter in range(1, 3):
                result = await engine.run(graph, ["save"])
                assert result.outputs["save"]["text"].resolve() == "manifest\n"
                assert len(result.artifacts) == 1
                artifact = result.artifacts[0]
                assert artifact.node_id == "save"
                assert artifact.virtual_path == f"mounts/default/manifest/1x1_{counter:05}.txt"
                assert artifact.digest == digest_bytes(b"manifest\n")
                assert artifact.size == len(b"manifest\n")
                assert artifact.media_type == "text/plain"
            alternate = Graph(
                nodes={
                    "save": GraphNode(
                        "dinkster.text.save_text",
                        {
                            "text": "",
                            "filename_prefix": "empty",
                            "mount": TypedLiteral(MOUNT_TYPE, "alternate"),
                        },
                    )
                }
            )
            result = await engine.run(alternate, ["save"])
            assert len(result.artifacts) == 1
            assert result.artifacts[0].virtual_path == "mounts/alternate/empty_00001.txt"
            assert result.artifacts[0].digest == digest_bytes(b"")
            assert result.artifacts[0].size == 0
            library = ServerLibrary(
                vault=AssetVault(tmp_path / "vault"),
                store=LibraryStore(tmp_path / "library.sqlite"),
                upload_writer=AssetWriter(MountSnapshotWriter(roots["snapshot"])),
                project_folders_for=lambda _project: AssetFolderContext(
                    ("default", "alternate"), "default"
                ),
            )
            app = create_app(
                lambda on_event: composer.composition.make_engine(on_event),
                composer.composition.schemas,
                library=library,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                response = await client.post(
                    "/api/jobs",
                    json={
                        "clientId": "save-client",
                        "jobId": "native-save",
                        "project": "film",
                        "graph": graph_to_wire(
                            Graph({"save": GraphNode("SaveText", graph.nodes["save"].inputs)})
                        ),
                        "targets": ["save"],
                    },
                )
                assert response.status == 202
                submitted = await response.json()
                async with asyncio.timeout(10):
                    while True:
                        status = await (
                            await client.get("/api/jobs/save-client/native-save")
                        ).json()
                        if status["state"] in {"completed", "failed", "cancelled"}:
                            break
                        await asyncio.sleep(0)
                assert status["state"] == "completed", status
                assert (
                    status["artifacts"][0]["virtualPath"] == "mounts/default/manifest/1x1_00003.txt"
                )
                (record,) = library.store.query_assets("local", project="film")[0]
                assert record.asset.digest == digest_bytes(b"manifest\n")
                assert record.asset.virtual_path == "mounts/default/manifest/1x1_00003.txt"
                assert dict(record.origin) == {
                    "kind": "produced",
                    "job": submitted["jobRef"],
                    "node": "save",
                }
                assert library.vault.digests() == []
                assert library.store.query_assets("local", project="another") == []
            finally:
                await client.close()
        finally:
            await composer.close()

    run_scenario(scenario())
    assert (roots["default"] / "manifest" / "1x1_00001.txt").read_bytes() == b"manifest\n"
    assert (roots["default"] / "manifest" / "1x1_00002.txt").read_bytes() == b"manifest\n"

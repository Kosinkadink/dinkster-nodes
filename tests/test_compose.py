"""Public node packs compose, reload and execute under their declared namespaces."""

from __future__ import annotations

import asyncio
import importlib.metadata
import os
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetError, AssetRef, AssetVault, LibraryStore, digest_bytes
from dinkster_assets.scratch import claim_inactive_produced_job
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral, graph_to_wire
from dinkster_registry import Lockfile
from dinkster_schema import Node
from dinkster_server import ServerLibrary, create_app
from dinkster_values import TypeRegistry, Value
from dinkster_values.storage import image_input
from dinkster_workers import load_manifest
from dinkster_workers.doctor import prepare_catalog
from dinkster_workers.manifest import resolve_entry

from dinkster.compose import (
    CompositionError,
    PackSpec,
    ServingComposer,
    compose_serving,
    default_pack_spec,
)
from dinkster.packs import first_party_pack_metadata
from tests.async_scenarios import run_scenario

TESTS_DIR = Path(__file__).parent
ATTENTION_PROVIDER = TESTS_DIR / "fixtures/attention_provider"
WORKER_ENV = {"PYTHONPATH": os.pathsep.join((str(ATTENTION_PROVIDER), str(TESTS_DIR)))}


@pytest.fixture
def first_party_specs() -> tuple[PackSpec, ...]:
    from dinkster import compose

    catalog = Path(compose.__file__).with_name("checkout-packs.lock")
    lock = Lockfile.from_record_json(catalog.read_text(encoding="utf-8"))
    assert lock.packs
    return tuple(default_pack_spec(pack.pack) for pack in lock.packs)


@pytest.fixture
def first_party_node_packs(first_party_specs: tuple[PackSpec, ...]) -> dict[str, str]:
    return {
        node.schema().node_type: manifest.name
        for spec in first_party_specs
        if (manifest := load_manifest(spec.manifest)).nodes_entry
        for node in cast("list[type[Node]]", resolve_entry(manifest.nodes_entry))
    }


def test_first_party_packs_do_not_require_standard_suite_distribution(
    monkeypatch: pytest.MonkeyPatch,
    first_party_node_packs: dict[str, str],
) -> None:
    from dinkster import compose

    real_distribution = importlib.metadata.distribution

    def distribution(name: str):
        if name == "dinkster-nodes-std":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_distribution(name)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    expected = tuple(sorted(set(first_party_node_packs.values()) | {"dinkster-nodes-remote"}))
    selected = tuple(sorted(first_party_pack_metadata(TESTS_DIR.parent)))
    assert selected == expected
    for pack_id in expected:
        spec = compose.default_pack_spec(pack_id)
        assert spec.packs is not None
        assert load_manifest(spec.manifest).name == pack_id


def test_compose_defaults_only(
    tmp_path: Path,
    first_party_specs: tuple[PackSpec, ...],
    first_party_node_packs: dict[str, str],
) -> None:
    """The explicit first-party selection provides the complete standard surface."""

    async def scenario() -> None:
        composition = await compose_serving(first_party_specs, include_default_packs=False)
        try:
            assert set(composition.schemas) == set(first_party_node_packs)
            assert not any(t.startswith("dev.") for t in composition.schemas)
            assert set(composition.packs) == (
                set(first_party_node_packs.values()) | {"dinkster-nodes-remote"}
            )
            assert composition.node_packs == first_party_node_packs
            locked = first_party_specs
            assert {
                next(iter(spec.packs))
                for spec in locked
                if spec.packs is not None and not spec.in_process
            } == set(composition.packs)
            expected_digests = {
                pack_id: info.artifact_digest
                for spec in locked
                if spec.packs is not None
                for pack_id, info in spec.packs.items()
            }
            assert {
                pack.pack: pack.artifact_digest for pack in composition.generation.packs
            } == expected_digests
            engine = composition.make_engine(lambda event: None)
            graph = Graph(
                nodes={
                    "g": GraphNode(
                        "dinkster.text.string_concatenate",
                        {"string_a": "left", "string_b": "right", "delimiter": ":"},
                    )
                }
            )
            result = await engine.run(graph, ["g"])
            assert result.outputs["g"]["STRING"].resolve() == "left:right"

            library = ServerLibrary(
                vault=AssetVault(tmp_path / "vault"),
                store=LibraryStore(tmp_path / "library.sqlite"),
                pack_assets=composition.asset_catalog,
            )
            app = create_app(
                composition.make_engine,
                composition.schemas,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
                library=library,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                response = await client.get("/api/nodes")
                payload = await response.json()
                assert response.status == 200, payload
                assert set(payload["nodes"]) == set(first_party_node_packs)
            finally:
                await client.close()
        finally:
            await composition.close()

    run_scenario(scenario())


def test_composed_server_end_to_end(tmp_path: Path) -> None:
    """One server: text nodes in-process, the pack isolated, both executing
    through the same queue; /api/nodes attributes each side truthfully and
    carries the manifest's declared presentation."""

    async def scenario() -> None:
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "isopack"\npublisher = "dinkster"\n'
            'type-namespaces = ["iso"]\nnamespaces = ["iso"]\n[pack.entry]\n'
            'nodes = "isopack_nodes:SERVER_NODES"\ntypes = "isopack_nodes:register_types"\n'
            '\n[pack.presentation]\ndisplay_name = "Iso Pack"\n'
            'abbr = "ISO"\nmark = "\\U0001F9EA"\ncolor = "#336699"\n',
            encoding="utf-8",
        )
        composition = await compose_serving(
            [
                replace(default_pack_spec("dinkster-text"), in_process=True),
                default_pack_spec("dinkster-models"),
                manifest,
            ],
            include_default_packs=False,
            worker_env=WORKER_ENV,
        )
        published_schemas = {
            node_type: schema
            for node_type, schema in composition.schemas.items()
            if node_type.startswith("iso.") or node_type == "dinkster.text.string_concatenate"
        }
        published_packs = {
            pack_id: info
            for pack_id, info in composition.packs.items()
            if pack_id in {"isopack", "dinkster-text"}
        }
        app = create_app(
            composition.make_engine,
            published_schemas,
            packs=published_packs,
            node_packs={
                node_type: pack_id
                for node_type, pack_id in composition.node_packs.items()
                if node_type in published_schemas
            },
            choices=composition.choices,
            lazy_choices=composition.lazy_choices,
        )

        async def close_composition(_: object) -> None:
            await composition.close()

        app.on_cleanup.append(close_composition)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            data = await (await client.get("/api/nodes")).json()
            assert data["packs"]["isopack"] == {
                "displayName": "Iso Pack",
                "abbr": "ISO",
                "mark": "\U0001f9ea",
                "color": "#336699",
                "source": f"local:{Path(manifest).parent.resolve()}",
            }
            assert data["nodes"]["iso.chatty"]["pack"] == "isopack"
            assert data["nodes"]["dinkster.text.string_concatenate"]["pack"] == "dinkster-text"
            blob_size = next(
                item
                for item in data["nodes"]["iso.blob_out"]["interface"]
                if item["role"] == "input" and item["id"] == "size"
            )
            assert blob_size["widget"] == {
                "type": "isopack.size",
                "min": 1,
                "max": 12,
                "unit": "bytes",
            }

            devices = await client.get("/api/choices/devices.gpu")
            assert devices.status == 200, await devices.text()
            assert "cpu" in await devices.json()

            graph = Graph(
                nodes={
                    "g": GraphNode(
                        "dinkster.text.string_concatenate",
                        {"string_a": "left", "string_b": "right", "delimiter": ":"},
                    ),
                    "c": GraphNode("iso.chatty", {"value": "hi"}),
                    "b": GraphNode("iso.blob_out", {"size": 3}),
                }
            )
            body = {
                "clientId": "c1",
                "jobId": "j1",
                "graph": graph_to_wire(graph),
                "targets": ["g", "c", "b"],
            }
            resp = await client.post("/api/jobs", json=body)
            assert resp.status == 202, await resp.text()
            status: dict[str, object] = {}
            for _ in range(200):
                status = await (await client.get("/api/jobs/c1/j1")).json()
                if status["state"] in ("completed", "failed"):
                    break
                await asyncio.sleep(0.05)
            assert status.get("state") == "completed", status

            for node_id, output_id, expected in (
                ("g", "STRING", "left:right"),
                ("c", "value", "HI"),
            ):
                response = await client.get(
                    f"/api/values?clientId=c1&jobId=j1&nodeId={node_id}&outputId={output_id}"
                )
                assert response.status == 200, await response.text()
                assert (await response.json())["descriptor"]["value"] == expected

            value_url = "/api/values?clientId=c1&jobId=j1&nodeId=b&outputId=blob"
            discovery = await client.get(value_url)
            assert discovery.status == 200, await discovery.text()
            value_data = await discovery.json()
            assert value_data["descriptor"]["typeId"] == "iso.blob"
            assert value_data["renditions"] == [
                {
                    "kind": "summary",
                    "mime": "text/plain",
                    "default": True,
                    "cacheKey": "summary/1",
                    "version": "1",
                    "parameters": ["prefix"],
                    "defaults": {"prefix": "blob"},
                    "limits": {"prefixLength": 32},
                }
            ]
            rendered = await client.get(value_url + "&rendition=summary/1&prefix=pack")
            assert rendered.status == 200, await rendered.text()
            assert rendered.content_type == "text/plain"
            assert await rendered.read() == b"pack:3:xxx"
        finally:
            await client.close()

    run_scenario(scenario())


def test_serving_composer_incremental(
    tmp_path: Path,
    first_party_specs: tuple[PackSpec, ...],
    first_party_node_packs: dict[str, str],
) -> None:
    """The diagnostic host starts empty and grows through ordinary pack deltas."""
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "isopack"\npublisher = "dinkster"\nnamespaces = ["iso"]\n'
        '[pack.entry]\nnodes = "isopack_nodes:NODES"\n'
        '[pack.presentation]\ndisplay_name = "Iso Pack"\n'
        'abbr = "ISO"\nmark = "\\U0001F9EA"\ncolor = "#336699"\n',
        encoding="utf-8",
    )

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            composition = composer.composition
            assert composition.schemas == {}
            assert composition.packs == {}
            engine = composition.make_engine(lambda event: None)

            app = create_app(composition.make_engine, composition.schemas)
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                assert await (await client.get("/api/health")).json() == {"ok": True}
                nodes = await (await client.get("/api/nodes")).json()
                assert nodes["nodes"] == {}
            finally:
                await client.close()

            standard_deltas = [
                await composer.add_pack(standard)
                for standard in composer.order_pack_entries(first_party_specs)
            ]
            assert set().union(*(delta.schemas for delta in standard_deltas)) == set(
                first_party_node_packs
            )
            for standard_delta in standard_deltas:
                engine.announce_schemas(standard_delta.schemas)
            result = await engine.run(
                Graph(
                    nodes={
                        "g": GraphNode(
                            "dinkster.text.string_concatenate",
                            {"string_a": "left", "string_b": "right", "delimiter": ":"},
                        )
                    }
                ),
                ["g"],
            )
            assert result.outputs["g"]["STRING"].resolve() == "left:right"

            delta = await composer.add_pack(manifest)
            assert delta.pack == "isopack"
            assert "iso.chatty" in delta.schemas
            assert delta.node_packs["iso.chatty"] == "isopack"
            assert delta.packs["isopack"].display_name == "Iso Pack"
            # Merged into the shared composition too (the all-at-once path).
            assert composition.schemas["iso.chatty"] is delta.schemas["iso.chatty"]
            assert composition.node_packs["iso.chatty"] == "isopack"

            # The engine made BEFORE the pack composed executes it after
            # announcement - no rebuild, no restart.
            engine.announce_schemas(delta.schemas)
            result = await engine.run(
                Graph(nodes={"c": GraphNode("iso.chatty", {"value": "hi"})}),
                ["c"],
            )
            assert result.outputs["c"]["value"].resolve() == "HI"
        finally:
            await composer.close()

    run_scenario(scenario())


def test_default_pack_artifact_files_have_explicit_line_ending_policy() -> None:
    repo_root = TESTS_DIR.parent
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "packages"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    tracked_paths = {Path(raw.decode("utf-8")) for raw in listed.split(b"\0") if raw}
    artifact_paths: set[Path] = set()
    metadata = first_party_pack_metadata(repo_root)
    assert metadata
    for pack_id, (module_name, _) in metadata.items():
        pack_root = next(
            path.parent.relative_to(repo_root)
            for path in (repo_root / "packages").glob("dinkster-nodes-*/dinkster-pack.toml")
            if tomllib.loads(path.read_text(encoding="utf-8"))["pack"]["name"] == pack_id
        )
        sidecar_root = pack_root
        manifest_path = sidecar_root / "dinkster-pack.toml"
        manifest = tomllib.loads((repo_root / manifest_path).read_text(encoding="utf-8"))
        docs = manifest.get("pack", {}).get("docs", {})
        docs_dir = docs.get("dir") if isinstance(docs, dict) else None
        module_root = pack_root / "src" / Path(*module_name.split("."))
        prefixes = [module_root, sidecar_root / "locales"]
        if isinstance(docs_dir, str):
            prefixes.append(sidecar_root / docs_dir)
        artifact_paths.add(manifest_path)
        module_init = module_root.parent / "__init__.py"
        if module_init in tracked_paths:
            artifact_paths.add(module_init)
        if module_root.parent.name == "dinkster_nodes_vision":
            artifact_paths.update(path for path in tracked_paths if path.parent == sidecar_root)
        artifact_paths.update(
            path
            for path in tracked_paths
            if any(path.is_relative_to(prefix) for prefix in prefixes)
        )

    checked = subprocess.run(
        [
            "git",
            "check-attr",
            "-z",
            "--stdin",
            "text",
            "eol",
            "--",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        input=b"".join(path.as_posix().encode("utf-8") + b"\0" for path in sorted(artifact_paths)),
    ).stdout.split(b"\0")
    attributes: dict[str, dict[str, str]] = {}
    for index in range(0, len(checked) - 1, 3):
        path, attribute, value = (part.decode("utf-8") for part in checked[index : index + 3])
        attributes.setdefault(path, {})[attribute] = value

    uncovered = [
        path.as_posix()
        for path in sorted(artifact_paths)
        if attributes[path.as_posix()]["eol"] != "lf"
        and attributes[path.as_posix()]["text"] != "unset"
    ]
    assert uncovered == [], (
        "pack artifact files must declare eol=lf for text or -text for binary: "
        + ", ".join(uncovered)
    )


@pytest.mark.parametrize("in_process", (False, True))
def test_pack_namespace_compose_reload_execute_and_reject_second_owner(
    tmp_path: Path, in_process: bool
) -> None:
    from dinkster_utilities import NODES

    claims = tuple(node.schema().node_type for node in NODES)
    manifest = TESTS_DIR.parent / "packages/dinkster-nodes-utilities/dinkster-pack.toml"
    assert load_manifest(manifest).namespaces == ("dinkster.utilities",)
    other_manifest = tmp_path / "dinkster-pack.toml"
    other_manifest.write_text(
        '[pack]\nname = "other-owner"\npublisher = "dinkster"\n'
        'namespaces = ["dinkster.utilities"]\n'
        '[pack.entry]\nnodes = "other_owner_fixture:NODES"\n'
    )
    (tmp_path / "other_owner_fixture.py").write_text(
        "from dinkster_utilities.primitive import PrimitiveString\nNODES = (PrimitiveString,)\n"
    )

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(
                PackSpec(manifest=manifest, in_process=in_process, trust_reserved=True)
            )
            assert set(composer.composition.schemas) == set(claims)
            assert set(composer.composition.node_packs.values()) == {"dinkster-utilities"}
            assert not composer.composition.lazy_choices
            if in_process:
                with pytest.raises(CompositionError, match="cannot reload in-process pack"):
                    await composer.reload_pack("dinkster-utilities")
            else:
                await composer.reload_pack("dinkster-utilities")
            assert set(composer.composition.schemas) == set(claims)
            assert not composer.composition.lazy_choices
            await composer.add_pack(
                PackSpec(
                    manifest=TESTS_DIR.parent / "packages/dinkster-nodes-text/dinkster-pack.toml",
                    in_process=in_process,
                    trust_reserved=True,
                )
            )
            combined_claims = set(composer.composition.schemas)
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "text": GraphNode(
                        "dinkster.utilities.primitive_string", {"value": "a\U0001f600"}
                    ),
                    "length": GraphNode(
                        "dinkster.text.string_length", {"string": Link("text", "STRING")}
                    ),
                    "false": GraphNode("dinkster.utilities.primitive_boolean", {"value": False}),
                    "not": GraphNode(
                        "dinkster.utilities.comfy_not_node", {"value": Link("length", "length")}
                    ),
                    "and": GraphNode(
                        "dinkster.utilities.comfy_and_node",
                        {
                            "values.value0": Link("length", "length"),
                            "values.value1": Link("false", "BOOLEAN"),
                        },
                    ),
                    "or": GraphNode(
                        "dinkster.utilities.comfy_or_node",
                        {
                            "values.value0": Link("false", "BOOLEAN"),
                            "values.value1": Link("length", "length"),
                        },
                    ),
                    "combo_default": GraphNode("dinkster.utilities.custom_combo", {"choice": ""}),
                    "combo_index": GraphNode("dinkster.utilities.primitive_int", {"value": 205}),
                    "combo": GraphNode(
                        "dinkster.utilities.custom_combo",
                        {
                            "choice": "outside-menu",
                            "index": Link("combo_index", "INT"),
                            "2": TypedLiteral("STRING", "unused sparse option"),
                            "205": Link("text", "STRING"),
                            "label": TypedLiteral("STRING", "arbitrary extra name"),
                            **{str(i): TypedLiteral("INT", i) for i in range(300, 405)},
                        },
                    ),
                }
            )
            targets = ["length", "not", "and", "or", "combo_default", "combo"]
            result = await engine.run(graph, targets)
            assert result.outputs["length"]["length"].resolve() == 2
            assert result.outputs["not"]["BOOLEAN"].resolve() is False
            assert result.outputs["and"]["BOOLEAN"].resolve() is False
            assert result.outputs["or"]["BOOLEAN"].resolve() is True
            assert result.outputs["combo_default"]["STRING"].resolve() == ""
            assert result.outputs["combo_default"]["INDEX"].resolve() == 0
            assert result.outputs["combo"]["STRING"].resolve() == "outside-menu"
            assert result.outputs["combo"]["INDEX"].resolve() == 205
            repeated = await engine.run(graph, targets)
            assert repeated.executed == ()
            assert "combo" in repeated.cached
            assert repeated.outputs["combo"]["INDEX"].resolve() == 205
            changed = Graph(
                nodes={
                    **graph.nodes,
                    "combo_index": GraphNode("dinkster.utilities.primitive_int", {"value": -7}),
                }
            )
            changed_result = await engine.run(changed, ["combo"])
            assert changed_result.outputs["combo"]["STRING"].resolve() == "outside-menu"
            assert changed_result.outputs["combo"]["INDEX"].resolve() == -7
            assert "combo" in changed_result.executed
            with pytest.raises(
                CompositionError, match="declared by both 'dinkster-utilities' and 'other-owner'"
            ):
                await composer.add_pack(PackSpec(manifest=other_manifest, trust_reserved=True))
            assert set(composer.composition.schemas) == combined_claims
        finally:
            await composer.close()

    run_scenario(scenario())


@pytest.mark.parametrize("ending", ["success", "failure", "cancel", "native failure"])
@pytest.mark.parametrize("other_job", ["other job", "../main job"])
def test_composed_job_scratch_stays_live_between_nodes_until_run_settles(
    tmp_path: Path, ending: str, other_job: str
) -> None:
    async def scenario() -> None:
        scratch = tmp_path / "scratch"
        entered = {job: asyncio.Event() for job in ("main", "other")}
        resume = {job: asyncio.Event() for job in entered}
        released: list[str] = []

        class Policy:
            async def select(
                self,
                node_type: str,
                inputs: Mapping[str, Value],
                candidates: tuple[str, Mapping[str, str]],
                **context: object,
            ) -> None:
                value = inputs["string_a"].resolve()
                if value in ("first-second", "other"):
                    label = "other" if value == "other" else "main"
                    entered[label].set()
                    await resume[label].wait()
                    if label == "main" and ending == "failure":
                        raise RuntimeError("dispatch failed")

            def release_run(self, run_id: str) -> None:
                with claim_inactive_produced_job(scratch, run_id) as inactive:
                    assert not inactive, "scratch released before native run settlement"
                released.append(run_id)
                if len(released) == 1 and ending == "native failure":
                    raise RuntimeError("native settlement failed")

        composer = ServingComposer(pack_scratch_root=scratch, native_policy=Policy())
        tasks: list[asyncio.Task[object]] = []
        try:
            await composer.add_pack(default_pack_spec("dinkster-text"))
            engine = composer.composition.make_engine(lambda event: None)
            with pytest.raises(AssetError, match="ownership state is unavailable"):
                with claim_inactive_produced_job(scratch, "unknown job"):
                    pytest.fail("missing ownership state was treated as an idle job")
            main = asyncio.create_task(
                engine.run(
                    Graph(
                        nodes={
                            "first": GraphNode(
                                "dinkster.text.string_concatenate",
                                {"string_a": "first", "string_b": "second", "delimiter": "-"},
                            ),
                            "second": GraphNode(
                                "dinkster.text.string_concatenate",
                                {
                                    "string_a": Link("first", "STRING"),
                                    "string_b": "third",
                                    "delimiter": ":",
                                },
                            ),
                        }
                    ),
                    ["second"],
                    run_id="../main job",
                    attempt_id=2,
                )
            )
            other = asyncio.create_task(
                composer.composition.make_engine(lambda event: None).run(
                    Graph(
                        nodes={
                            "other": GraphNode(
                                "dinkster.text.string_concatenate",
                                {"string_a": "other", "string_b": "job", "delimiter": "/"},
                            )
                        }
                    ),
                    ["other"],
                    run_id=other_job,
                )
            )
            tasks.extend((main, other))
            async with asyncio.timeout(20):
                await asyncio.gather(*(event.wait() for event in entered.values()))
                for job in ("../main job", other_job):
                    with claim_inactive_produced_job(scratch, job) as inactive:
                        assert not inactive
                if ending == "cancel":
                    main.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await main
                else:
                    resume["main"].set()
                    if ending in ("failure", "native failure"):
                        with pytest.raises(
                            Exception, match="dispatch failed|native settlement failed"
                        ):
                            await main
                    else:
                        assert (await main).outputs["second"][
                            "STRING"
                        ].resolve() == "first-second:third"
                with claim_inactive_produced_job(scratch, "../main job") as inactive:
                    assert inactive == (other_job != "../main job")
                with claim_inactive_produced_job(scratch, other_job) as inactive:
                    assert not inactive
                assert released == ["../main job"]
                resume["other"].set()
                assert (await other).outputs["other"]["STRING"].resolve() == "other/job"
                with claim_inactive_produced_job(scratch, other_job) as inactive:
                    assert inactive
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await composer.close()

    run_scenario(scenario())


def test_installed_pack_digest_matches_bundled_artifact(tmp_path: Path) -> None:
    from dinkster_registry import artifact_digest, unpack_artifact
    from dinkster_workers.pack_wheel import build_pack_archive

    from dinkster import compose

    source = Path(__file__).parent.parent / "packages/dinkster-nodes-image"
    output = tmp_path / "wheel"
    build_pack_archive(source, output)
    archive = next(output.glob("*.whl"))
    unpack_artifact(archive, tmp_path, artifact_digest(archive.read_bytes()))
    bundled = tmp_path / "dinkster_image"

    source_digest = compose._installed_pack_digest(
        source / "dinkster-pack.toml", source / "src/dinkster_image"
    )
    bundled_digest = compose._installed_pack_digest(bundled / "dinkster-pack.toml", None)

    assert source_digest == bundled_digest


def test_lazy_image_pack_resolves_assets_after_host_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    import numpy as np
    from PIL import Image

    pixels = np.zeros((3, 5, 3), dtype=np.uint8)
    pixels[..., 1] = 255
    encoded = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(encoded, format="PNG")
    data = encoded.getvalue()
    digest = digest_bytes(data)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    for key in ("DINKSTER_PACK_SCRATCH", "DINKSTER_MOUNTS_SNAPSHOT", "DINKSTER_ASSET_ROOT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    ref = AssetRef(digest, "input.png", len(data), "image/png")

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            image = replace(default_pack_spec("dinkster-image"), require_catalog=True)
            assert prepare_catalog(image.manifest, environment=os.environ).ok
            await composer.add_pack(image)

            engine = composer.composition.make_engine(lambda _event: None)
            result = await engine.run(
                Graph(
                    nodes={"load": GraphNode("dinkster.image.load_image", {"image": ref.to_wire()})}
                ),
                ["load"],
            )
            loaded = cast("np.ndarray", result.outputs["load"]["IMAGE"].resolve())
            np.testing.assert_array_equal(cast("np.ndarray", image_input(loaded))[0] * 255, pixels)
        finally:
            await composer.close()

    run_scenario(scenario())


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("verified", (False, True))
def test_models_pack_preserves_host_bound_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool, verified: bool
) -> None:
    from dinkster_api.v1 import ASSET_TYPE
    from dinkster_assets import (
        LocalAssetRef,
        MountDef,
        MountTable,
    )

    root = tmp_path / "models"
    root.mkdir()
    payload = b"bound model bytes"
    (root / "weights.bin").write_bytes(payload)
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "indexes")
    table.add(MountDef(id="models", path=root), kind="model/checkpoint")
    if verified:
        table.scan("models")
    ref = table.bind_name("weights.bin", "model/checkpoint")
    assert isinstance(ref, AssetRef if verified else LocalAssetRef)
    manifest = TESTS_DIR.parent / "packages/dinkster-nodes-models/dinkster-pack.toml"
    env = {
        **WORKER_ENV,
        "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
    }
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    async def scenario() -> None:
        registry = TypeRegistry()
        assert ASSET_TYPE not in registry
        composer = ServingComposer(registry=registry, worker_env=env)
        try:
            asset_type = registry.spec(ASSET_TYPE)
            await composer.add_pack(
                PackSpec(manifest, in_process=in_process, env=env, trust_reserved=True)
            )
            assert registry.spec(ASSET_TYPE) is asset_type
            assert set(composer.composition.lazy_choices) == {"devices.gpu", "devices.vae"}
            if not in_process:
                await composer.reload_pack("dinkster-models")
                assert set(composer.composition.lazy_choices) == {"devices.gpu", "devices.vae"}
                assert registry.spec(ASSET_TYPE) is asset_type
            for _ in range(2):
                bound = registry.wrap(ASSET_TYPE, ref.to_wire()).resolve()
                restored = registry.spec(ASSET_TYPE).decode(registry.spec(ASSET_TYPE).encode(bound))
                assert isinstance(restored, AssetRef if verified else LocalAssetRef)
                assert restored.local_path().read_bytes() == payload
        finally:
            await composer.close()

    run_scenario(scenario())


@pytest.mark.parametrize("in_process", (True, False))
def test_native_comfy_pack_registers_container_literals(in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            manifest = TESTS_DIR.parent / "packages/dinkster-nodes-text/dinkster-pack.toml"
            await composer.add_pack(
                PackSpec(manifest=manifest, in_process=in_process, trust_reserved=True)
            )
            result = await composer.composition.make_engine(lambda _event: None).run(
                Graph(
                    nodes={
                        "dictionary": GraphNode(
                            "dinkster.text.convert_dictionary_to_string",
                            {"dictionary": {"key": [3, True]}, "indent": 0},
                        ),
                        "array": GraphNode(
                            "dinkster.text.convert_array_to_string",
                            {"array": [3, {"key": False}], "indent": 0},
                        ),
                    }
                ),
                ["dictionary", "array"],
            )
            assert result.outputs["dictionary"]["STRING"].resolve() == '{"key": [3, true]}'
            assert result.outputs["array"]["STRING"].resolve() == '[3, {"key": false}]'
        finally:
            await composer.close()

    run_scenario(scenario())


@pytest.mark.parametrize("source", ["image-list", "video-asset"])
def test_subprocess_official_pack_values_through_jobs(tmp_path: Path, source: str) -> None:
    import io

    import av
    import numpy as np
    from PIL import Image

    async def scenario() -> None:
        vault = AssetVault(tmp_path / "vault")
        specs = tuple(
            replace(default_pack_spec(pack), in_process=False, python=sys.executable)
            for pack in (
                "dinkster-utilities",
                "dinkster-video",
                "dinkster-image",
            )
        )
        composition = await compose_serving(
            specs,
            include_default_packs=False,
            worker_env={"DINKSTER_ASSET_VAULT": str(tmp_path / "vault")},
        )
        try:
            library = ServerLibrary(vault=vault, store=LibraryStore(tmp_path / "library.sqlite"))
            app = create_app(
                composition.make_engine,
                composition.schemas,
                library=library,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
            )
            async with TestClient(TestServer(app)) as client:
                if source == "image-list":
                    nodes = {
                        "source": GraphNode(
                            "dinkster.image.empty_image",
                            {"width": 7, "height": 3, "batch_size": 1},
                        ),
                        "other": GraphNode(
                            "dinkster.image.empty_image",
                            {"width": 11, "height": 5, "batch_size": 2},
                        ),
                        "list": GraphNode(
                            "dinkster.utilities.make_list",
                            {"items.0": Link("other", "IMAGE"), "items.1": Link("source", "IMAGE")},
                        ),
                        "select": GraphNode(
                            "dinkster.utilities.list_element",
                            {"list": Link("list", "list"), "index": -1},
                        ),
                        "info": GraphNode(
                            "dinkster.image.get_image_size", {"image": Link("select", "item")}
                        ),
                    }
                    expected = {"width": 7, "height": 3, "batch_size": 1}
                else:
                    path = tmp_path / "input.mp4"
                    with av.open(str(path), "w") as container:
                        stream = container.add_stream("libx264rgb", rate=2)
                        stream.width, stream.height, stream.pix_fmt = 32, 16, "rgb24"
                        stream.options = {"crf": "0"}
                        for value in (31, 173):
                            frame = av.VideoFrame.from_ndarray(
                                np.full((16, 32, 3), value, dtype=np.uint8), format="rgb24"
                            )
                            container.mux(stream.encode(frame))
                        container.mux(stream.encode())
                    response = await client.post("/api/assets", data=path.read_bytes())
                    assert response.status == 201
                    uploaded = await response.json()
                    nodes = {
                        "source": GraphNode(
                            "dinkster.video.load_video",
                            {"file": TypedLiteral("ASSET", {"digest": uploaded["digest"]})},
                        ),
                        "frames": GraphNode(
                            "dinkster.video.get_video_components",
                            {"video": Link("source", "VIDEO")},
                        ),
                        "info": GraphNode(
                            "dinkster.image.get_image_size", {"image": Link("frames", "images")}
                        ),
                    }
                    expected = {"width": 32, "height": 16, "batch_size": 2}
                response = await client.post(
                    "/api/jobs",
                    json={
                        "clientId": "cross-pack",
                        "jobId": source,
                        "graph": graph_to_wire(Graph(nodes=nodes)),
                        "targets": ["info", "frames"] if source == "video-asset" else ["info"],
                    },
                )
                assert response.status == 202, await response.text()
                async with asyncio.timeout(60):
                    while True:
                        result = await (await client.get(f"/api/jobs/cross-pack/{source}")).json()
                        if result["state"] in {"completed", "failed", "cancelled"}:
                            break
                        await asyncio.sleep(0.01)
                assert result["state"] == "completed", result.get("error", result)
                assert {
                    name: result["outputs"]["info"][name]["value"] for name in expected
                } == expected
                if source == "video-asset":
                    response = await client.get(
                        "/api/values",
                        params={
                            "clientId": "cross-pack",
                            "jobId": source,
                            "nodeId": "frames",
                            "outputId": "images",
                            "rendition": "png",
                        },
                    )
                    assert response.status == 200, await response.text()
                    pixels = np.array(Image.open(io.BytesIO(await response.read())))
                    assert np.array_equal(pixels, np.full((16, 32, 3), 31, dtype=np.uint8))
        finally:
            await composition.close()

    run_scenario(scenario())


def test_image_pack_starts_with_canonical_inference_types(
    first_party_specs: tuple[PackSpec, ...],
) -> None:
    async def scenario() -> None:
        image = replace(default_pack_spec("dinkster-image"), require_catalog=True)
        assert prepare_catalog(image.manifest, environment=os.environ).ok
        composer = ServingComposer()
        try:
            specs = tuple(
                image if load_manifest(spec.manifest).name == "dinkster-image" else spec
                for spec in first_party_specs
            )
            for spec in composer.order_pack_entries(specs):
                await composer.add_pack(spec)
            result = await composer.composition.make_engine(lambda event: None).run(
                Graph(
                    nodes={
                        "image": GraphNode(
                            "dinkster.image.empty_image",
                            {"width": 7, "height": 3, "batch_size": 2, "color": 0x123456},
                        ),
                        "size": GraphNode(
                            "dinkster.image.get_image_size", {"image": Link("image", "IMAGE")}
                        ),
                    }
                ),
                ["size"],
            )
            assert result.executed == ("image", "size")
            assert {name: value.resolve() for name, value in result.outputs["size"].items()} == {
                "width": 7,
                "height": 3,
                "batch_size": 2,
            }
            assert result.diagnostics == ()
        finally:
            await composer.close()

    run_scenario(scenario())


@pytest.mark.parametrize(
    ("pack_id", "directory", "expected_version"),
    [
        ("dinkster-text", "dinkster-nodes-text", "0.0.3"),
        ("dinkster-image", "dinkster-nodes-image", "0.0.9"),
        ("dinkster-audio", "dinkster-nodes-audio", "0.0.4"),
        ("dinkster-3d", "dinkster-nodes-3d", "0.0.4"),
        ("dinkster-models", "dinkster-nodes-models", "0.0.6"),
        ("dinkster-nodes-remote", "dinkster-nodes-remote", "0.0.4"),
    ],
)
def test_default_pack_prefers_loaded_source_tree_over_bundled_copy(
    pack_id: str,
    directory: str,
    expected_version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import compose

    source_root = TESTS_DIR.parent / "packages" / directory
    lock = Lockfile.from_record_json(
        Path(compose.__file__).with_name("checkout-packs.lock").read_text()
    )
    monkeypatch.setattr(compose, "_default_pack_lock", lambda: lock)
    locked = lock.get(pack_id)
    assert locked is not None
    spec = default_pack_spec(pack_id)
    assert Path(spec.manifest).resolve() == (source_root / "dinkster-pack.toml").resolve()
    assert spec.packs is not None
    info = spec.packs[pack_id]
    assert info.source == locked.source
    assert info.version == expected_version
    assert info.artifact_digest == locked.artifact_digest


@pytest.mark.parametrize(
    "compositor_value, width",
    [
        ({"version": 2, "documentDigest": None, "commands": []}, 7),
        ({}, 7),
        (None, 7),
        (
            {
                "version": 2,
                "documentDigest": None,
                "commands": [{"op": "canvas", "changes": {"width": 5}}],
            },
            5,
        ),
    ],
)
def test_lazy_image_pack_composes_nonempty_layers_with_literal_delta(
    compositor_value: object, width: int
) -> None:
    import numpy as np

    async def scenario() -> None:
        image = replace(default_pack_spec("dinkster-image"), require_catalog=True)
        assert not image.in_process
        assert prepare_catalog(image.manifest, environment=os.environ).ok
        composer = ServingComposer()
        try:
            await composer.add_pack(image)
            assert composer._records["dinkster-image"].worker.cold
            result = await composer.composition.make_engine(lambda _event: None).run(
                Graph(
                    nodes={
                        "image": GraphNode(
                            "dinkster.image.empty_image",
                            {"width": 7, "height": 3, "batch_size": 1, "color": 0x123456},
                        ),
                        "layer": GraphNode(
                            "dinkster.image.add_layer", {"image": Link("image", "IMAGE")}
                        ),
                        "compose": GraphNode(
                            "dinkster.image.image_compositor",
                            {
                                "layers": Link("layer", "LAYERS"),
                                **(
                                    {"compositor": compositor_value}
                                    if compositor_value is not None
                                    else {}
                                ),
                            },
                        ),
                    }
                ),
                ["compose"],
            )
            pixels = np.asarray(image_input(result.outputs["compose"]["IMAGE"].resolve()))
            assert pixels.shape == (1, 3, width, 3)
            expected = np.broadcast_to(np.asarray([0x12, 0x34, 0x56]), pixels.shape)
            np.testing.assert_array_equal(np.rint(pixels * 255).astype(np.uint8), expected)
            assert result.diagnostics == ()
        finally:
            await composer.close()

    run_scenario(scenario())

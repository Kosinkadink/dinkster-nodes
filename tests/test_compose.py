"""Public node packs compose, reload and execute under their declared namespaces."""

from __future__ import annotations

import asyncio
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
from dinkster_server import ServerLibrary, create_app
from dinkster_values import TypeRegistry, Value
from dinkster_values.storage import image_input
from dinkster_workers import load_manifest
from dinkster_workers.doctor import prepare_catalog

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

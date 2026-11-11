from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest
from blake3 import blake3
from dinkster_workers import ManifestError
from dinkster_workers.pack_wheel import build_pack_archive

from scripts.build_official_packs import build as build_official_packs
from tests.async_scenarios import run_scenario


@pytest.fixture(scope="module")
def official_payloads(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ...]:
    # Consumers only read these wheels and unpack into their own test directories.
    return build_official_packs(
        Path(__file__).resolve().parents[1], tmp_path_factory.mktemp("official-payloads")
    )


def test_official_publication_declarations_match_archive_admission(
    tmp_path: Path, official_payloads: tuple[Path, ...]
) -> None:
    from dinkster_registry import Lockfile, unpack_artifact
    from dinkster_registry.archive_admission import run_ingest_integrity, run_manifest_admission

    from scripts.build_official_packs import library_metadata

    first = official_payloads
    catalog = Lockfile.from_record_json(first[-1].read_text())
    assert len(catalog.packs) == 37
    for pack in catalog.packs:
        package_name = pack.pack.replace("-", "_")
        archive = first[-1].parent / f"{package_name}-{pack.version}-py3-none-any.whl"
        data = archive.read_bytes()
        if pack.pack == "dinkster-pack-helpers":
            assert library_metadata(archive) == (
                "dinkster-pack-helpers",
                "0.0.3",
                (
                    "dinkster-api<0.1,>=0.0.2.dev0",
                    "dinkster-inference",
                    "numpy>=1.26",
                    "pillow>=10",
                    "torch",
                ),
            )
            assert f"blake3:{blake3(data).hexdigest()}" == pack.artifact_digest
            continue
        assert not run_ingest_integrity(data, len(data), pack.artifact_digest).blocked
        source = (
            unpack_artifact(archive, tmp_path / "sources" / pack.pack, pack.artifact_digest)
            / package_name
        )
        accepted = run_manifest_admission(
            source, pack.pack, pack.version, publisher="dinkster", engine_version="0.0.4.dev123"
        )
        assert not accepted.blocked, accepted.findings
        for engine in ("0.0.1", "0.1"):
            refused = run_manifest_admission(
                source, pack.pack, pack.version, publisher="dinkster", engine_version=engine
            )
            assert refused.blocked
            assert refused.findings[0].code == "manifest.invalid"


def test_official_sources_prepare_catalogs_and_execute_existing_node_ids(
    tmp_path: Path, official_payloads: tuple[Path, ...]
) -> None:
    import json
    from dataclasses import replace

    from dinkster_graph import graph_from_wire
    from dinkster_registry import Lockfile, artifact_digest, unpack_artifact
    from dinkster_workers import load_manifest
    from dinkster_workers.catalog import read_catalog
    from dinkster_workers.doctor import prepare_catalog
    from dinkster_workers.pack_wheel import pack_manifest_path

    from dinkster.compose import CompositionError, PackSpec, ServingComposer
    from dinkster.packs import pack_info_from_manifest

    saved = tmp_path / "workflow.json"
    original = (
        b'{"nodes":{"length":{"nodeType":"dinkster.text.string_length",'
        b'"inputs":{"string":"source packs"}},'
        b'"image":{"nodeType":"dinkster.image.empty_image",'
        b'"inputs":{"width":4,"height":5,"batch_size":2}},'
        b'"size":{"nodeType":"dinkster.image.get_image_size",'
        b'"inputs":{"image":{"$link":{"node":"image","output":"IMAGE"}}}}}}'
    )
    saved.write_bytes(original)
    payloads = official_payloads
    lock = Lockfile.from_record_json(payloads[-1].read_text())
    specs = []
    for entry in lock.packs:
        if entry.pack == "dinkster-pack-helpers":
            assert entry.claims == ()
            continue
        filename = f"{entry.pack.replace('-', '_')}-{entry.version}-py3-none-any.whl"
        source = unpack_artifact(
            payloads[-1].parent / filename,
            tmp_path / "sources" / entry.pack,
            entry.artifact_digest,
        )
        manifest = load_manifest(pack_manifest_path(source))
        assert read_catalog(manifest) is None
        report = prepare_catalog(manifest.path, environment={"PYTHONPATH": str(source)})
        assert report.ok, report
        assert read_catalog(manifest) is not None
        repacked = tmp_path / "repacked" / filename
        build_pack_archive(manifest.root, repacked)
        assert artifact_digest(repacked.read_bytes()) == entry.artifact_digest
        info = replace(
            pack_info_from_manifest(manifest),
            version=entry.version,
            artifact_digest=entry.artifact_digest,
            source=entry.source,
            publisher=entry.publisher,
        )
        specs.append(
            PackSpec(
                manifest=manifest.path,
                python=sys.executable,
                env={
                    "PYTHONPATH": str(source),
                    "DINKSTER_REMOTE_CATALOG_BASE": "",
                    "DINKSTER_REMOTE_GATEWAY_BASE": "",
                },
                packs={entry.pack: info},
                trust_reserved=True,
                require_catalog=True,
            )
        )

    async def execute() -> None:
        composer = ServingComposer()
        try:
            ordered = composer.order_pack_entries(specs)
            composer.validate_catalogs(ordered)
            for spec in ordered:
                await composer.add_pack(spec)
            composition = composer.composition
            assert composition.node_packs["dinkster.text.string_length"] == "dinkster-text"
            assert composition.node_packs["dinkster.image.empty_image"] == "dinkster-image"
            assert composition.packs["dinkster-text"].version == "0.0.3"
            assert composition.packs["dinkster-image"].version == "0.0.7"
            result = await composition.make_engine(lambda _event: None).run(
                graph_from_wire(json.loads(saved.read_bytes())), ["length", "size"]
            )
            assert result.diagnostics == ()
            assert result.outputs["length"]["length"].resolve() == 12
            assert result.outputs["size"]["width"].resolve() == 4
            assert result.outputs["size"]["height"].resolve() == 5
            assert result.outputs["size"]["batch_size"].resolve() == 2
            assert saved.read_bytes() == original
            source = tmp_path / "sources" / "dinkster-text" / "dinkster_text"
            (source / "changed.txt").write_text("source drift")
            with pytest.raises(CompositionError, match="missing or stale"):
                composer.validate_catalogs(ordered)
        finally:
            await composer.close()

    run_scenario(execute())


def test_official_artifacts_preserve_sources_and_existing_pack_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, official_payloads: tuple[Path, ...]
) -> None:
    from dinkster_registry import Lockfile, unpack_artifact
    from dinkster_workers import load_manifest

    root = Path(__file__).resolve().parents[1]
    first = official_payloads
    write_text = Path.write_text

    def windows_write_text(path, data, encoding=None, errors=None, newline=None):
        return write_text(
            path,
            data,
            encoding=encoding,
            errors=errors,
            newline="\r\n" if newline is None else newline,
        )

    with monkeypatch.context() as platform:
        platform.setattr(Path, "write_text", windows_write_text)
        platform.setenv("GIT_CONFIG_COUNT", "2")
        platform.setenv("GIT_CONFIG_KEY_0", "core.autocrlf")
        platform.setenv("GIT_CONFIG_VALUE_0", "true")
        platform.setenv("GIT_CONFIG_KEY_1", "core.eol")
        platform.setenv("GIT_CONFIG_VALUE_1", "crlf")
        second = build_official_packs(root, tmp_path / "second")
    assert [path.name for path in first] == [path.name for path in second]
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]
    assert b"\r\n" not in second[-1].read_bytes()
    lock = Lockfile.from_record_json(first[-1].read_text())
    expected = {
        "dinkster-audio",
        "dinkster-audio-models",
        "dinkster-conditioning",
        "dinkster-cosmos",
        "dinkster-flux",
        "dinkster-hidream",
        "dinkster-hunyuan",
        "dinkster-latent",
        "dinkster-llm",
        "dinkster-ltx",
        "dinkster-minimax",
        "dinkster-misc-models",
        "dinkster-models",
        "dinkster-3d",
        "dinkster-3d-models",
        "dinkster-image",
        "dinkster-utilities",
        "dinkster-nodes-remote",
        "dinkster-pack-helpers",
        "dinkster-qwen",
        "dinkster-sampling",
        "dinkster-sd",
        "dinkster-seedvr",
        "dinkster-text",
        "dinkster-upscale",
        "dinkster-video",
        "dinkster-vision-background-removal",
        "dinkster-vision-depth-anything-3",
        "dinkster-vision-marigold",
        "dinkster-vision-mediapipe",
        "dinkster-vision-moge",
        "dinkster-vision-rtdetr",
        "dinkster-vision-sam3",
        "dinkster-vision-sam3d-body",
        "dinkster-vision-sdpose",
        "dinkster-void",
        "dinkster-wan",
    }
    assert {pack.pack for pack in lock.packs} == expected
    core_requirements = {
        "dinkster-image": (
            "comfy-angle==0.1.1; sys_platform != 'darwin' or platform_machine == 'arm64'",
            "numpy>=1.26",
            "opencv-contrib-python-headless>=5.0,<5.1",
            "pillow>=10",
            "PyOpenGL>=3.1.8,<4",
            "av==18.1.0",
            "torch",
            "torchvision",
            "scipy",
            "kornia",
            "dinkster-inference",
        ),
        "dinkster-nodes-remote": ("httpx>=0.28,<1",),
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    for pack in lock.packs:
        assert pack.publisher == "dinkster"
        assert pack.source == f"git:Kosinkadink/dinkster-nodes@{commit}"
        package_name = pack.pack.replace("-", "_")
        archive = first[0].parent / f"{package_name}-{pack.version}-py3-none-any.whl"
        with zipfile.ZipFile(archive) as contents:
            assert not any("tests" in Path(name).parts for name in contents.namelist())
            if pack.pack == "dinkster-pack-helpers":
                assert pack.claims == ()
                assert not any(name.endswith("dinkster-pack.toml") for name in contents.namelist())
                for name in contents.namelist():
                    if name.startswith(f"{package_name}/"):
                        expected_source = subprocess.check_output(
                            ["git", "show", f"HEAD:packages/{pack.pack}/src/{name}"], cwd=root
                        )
                        assert contents.read(name) == expected_source
                continue
        extracted = (
            unpack_artifact(archive, tmp_path / "unpacked" / pack.pack, pack.artifact_digest)
            / package_name
        )
        assert b"\r\n" not in (extracted / "dinkster-pack.toml").read_bytes()
        manifest = load_manifest(extracted / "dinkster-pack.toml")
        assert manifest.name == pack.pack
        if pack.pack in core_requirements:
            assert manifest.requires == core_requirements[pack.pack]
        module = manifest.nodes_entry.partition(":")[0]
        relative = Path(*module.split(".")) / "__init__.py"
        if not pack.pack.startswith("dinkster-nodes-"):
            original = (
                Path("packages")
                / pack.pack.replace("dinkster-", "dinkster-nodes-", 1)
                / "src"
                / relative
            )
        else:
            original = Path("packages") / pack.pack / "src" / relative
        original_manifest = subprocess.check_output(
            ["git", "show", f"HEAD:{original.parents[2].as_posix()}/dinkster-pack.toml"],
            cwd=root,
        )
        assert pack.version == tomllib.loads(original_manifest.decode())["pack"]["version"]
        expected_source = subprocess.check_output(
            ["git", "show", f"HEAD:{original.as_posix()}"], cwd=root
        )
        assert (extracted / "__init__.py").read_bytes() == expected_source
        declared_templates = tomllib.loads((extracted / "dinkster-pack.toml").read_text())[
            "pack"
        ].get("templates", [])
        assert len(manifest.templates) == len(declared_templates)
        for template in manifest.templates:
            assert template.path.is_file()


@pytest.mark.parametrize(
    "pack", ["dinkster-image", "dinkster-nodes-remote", "dinkster-vision-sam3"]
)
def test_official_artifacts_reject_static_errors_in_built_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pack: str
) -> None:
    def build_invalid_artifact(pack_dir: Path, output: Path) -> str:
        if output.name.startswith(pack.replace("-", "_") + "-"):
            (pack_dir / "src" / pack.replace("-", "_") / "runtime_import.py").write_text(
                "import dinkster_engine\n"
            )
        return build_pack_archive(pack_dir, output)

    monkeypatch.setattr("scripts.build_official_packs.build_pack_archive", build_invalid_artifact)
    with pytest.raises(ValueError, match=rf"{pack}.*static doctor"):
        build_official_packs(Path(__file__).resolve().parents[1], tmp_path / "output")
    assert not (tmp_path / "output" / "official-packs.lock").exists()


@pytest.mark.parametrize("version", ["0.0.2.dev1+gabc", "../../escape"])
def test_official_artifacts_require_public_pack_versions(tmp_path: Path, version: str) -> None:
    with pytest.raises(ValueError, match="pack version"):
        build_official_packs(Path(__file__).resolve().parents[1], tmp_path, version=version)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("selection", ["all", "single", "unknown", "undeclared"])
def test_official_publication_uses_selected_manifest_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    from dinkster_registry import Lockfile, unpack_artifact
    from dinkster_workers import load_manifest

    versions = {
        "dinkster-nodes-comfy": "1.3.0",
        "dinkster-nodes-foundation": "2.7.1",
        "dinkster-nodes-image": "0.8.0",
        "dinkster-nodes-media-io": "3.0.0rc1",
        "dinkster-nodes-remote": "1.2.4",
        "dinkster-text": "0.1.0",
        "dinkster-upscale": "0.2.1",
        "dinkster-nodes-unlisted-family": "4.2.1",
    }
    tracked = io.BytesIO()
    with tarfile.open(fileobj=tracked, mode="w") as archive:
        for pack, version in versions.items():
            declaration = (
                ""
                if selection == "undeclared" and pack == "dinkster-nodes-foundation"
                else f'version = "{version}"\nengine = ">=0.0.2,<0.1"\n'
            )
            files = {
                "dinkster-pack.toml": (
                    f'[pack]\nname = "{pack}"\npublisher = "dinkster"\n'
                    + declaration
                    + f'[pack.entry]\nnodes = "{pack.replace("-", "_")}:NODES"\n'
                ),
                "pyproject.toml": f'[project]\nname = "{pack}"\nrequires-python = ">=3.12"\n',
                f"src/{pack.replace('-', '_')}/__init__.py": "NODES = []\n",
            }
            for name, text in files.items():
                data = text.encode()
                directory = (
                    pack.replace("dinkster-", "dinkster-nodes-", 1)
                    if pack in {"dinkster-text", "dinkster-upscale"}
                    else pack
                )
                entry = tarfile.TarInfo(f"packages/{directory}/{name}")
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))

    def read_git(command, **kwargs):
        return "a" * 40 if "rev-parse" in command else tracked.getvalue()

    monkeypatch.setattr("scripts.build_official_packs.subprocess.check_output", read_git)
    requested = () if selection == "all" else ("dinkster-nodes-foundation",)
    if selection == "unknown":
        requested = ("dinkster-nodes-not-published",)
    output = tmp_path / "published"
    if selection in {"unknown", "undeclared"}:
        message = "unknown official packs" if selection == "unknown" else "version and engine"
        with pytest.raises(ValueError, match=message):
            build_official_packs(tmp_path, output, packs=requested)
        assert not list(output.glob("*.whl"))
        assert not (output / "official-packs.lock").exists()
        return
    paths = build_official_packs(tmp_path, output, packs=requested)
    repeated = build_official_packs(tmp_path, tmp_path / "repeated", packs=requested)
    assert {path.name: path.read_bytes() for path in paths} == {
        path.name: path.read_bytes() for path in repeated
    }
    expected = versions if selection == "all" else {"dinkster-nodes-foundation": "2.7.1"}
    assert {path.name for path in paths} == {
        "official-packs.lock",
        *(
            f"{pack.replace('-', '_')}-{version}-py3-none-any.whl"
            for pack, version in expected.items()
        ),
    }
    lock = Lockfile.from_record_json(paths[-1].read_text())
    assert {pack.pack: pack.version for pack in lock.packs} == expected
    for pack in lock.packs:
        extracted = unpack_artifact(
            output / f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl",
            tmp_path / "unpacked" / pack.pack,
            pack.artifact_digest,
        ) / pack.pack.replace("-", "_")
        manifest = load_manifest(extracted / "dinkster-pack.toml")
        assert manifest.compatibility is not None
        assert manifest.compatibility.version == expected[pack.pack]
        assert manifest.compatibility.engine == ">=0.0.2,<0.1"


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        (("src/dinkster/serve.py", "pyproject.toml"), ()),
        (("packages/dinkster-nodes-foundation/src/nodes.py",), ("foundation",)),
        (("packages/dinkster-nodes-image/deleted.py",), ("image",)),
        (("packages/dinkster-nodes-unlisted-family/src/nodes.py",), ("unlisted",)),
        (("packages/dinkster-nodes-dev/src/nodes.py",), ()),
        (("packages/dinkster-nodes-generation/dinkster-pack.toml",), ()),
        (("packages/dinkster-nodes-vision/first_pack/dinkster-pack.toml",), ("first",)),
        (("packages/dinkster-nodes-vision/src/vision/first/nodes.py",), ("first",)),
        (("packages/dinkster-nodes-vision/src/vision/second/nodes.py",), ("second",)),
        (("packages/dinkster-nodes-vision/src/vision/shared.py",), ("first", "second")),
        (("packages/dinkster-nodes-vision/pyproject.toml",), ("first", "second")),
        (
            ("packages/dinkster-nodes-foundation/old.py", "packages/dinkster-nodes-image/new.py"),
            ("foundation", "image"),
        ),
    ],
)
def test_official_publication_selects_only_changed_committed_packs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    from scripts.select_official_packs import select_packs

    paths = {
        "foundation": "packages/dinkster-nodes-foundation/dinkster-pack.toml",
        "image": "packages/dinkster-nodes-image/dinkster-pack.toml",
        "first": "packages/dinkster-nodes-vision/first_pack/dinkster-pack.toml",
        "second": "packages/dinkster-nodes-vision/second_pack/dinkster-pack.toml",
        "unlisted": "packages/dinkster-nodes-unlisted-family/dinkster-pack.toml",
    }
    committed = {
        path: (
            f'[pack]\nname = "{name}"\npublisher = "dinkster"\n'
            f'version = "{version}"\nengine = ">=0.2,<1"\n'
            f'[pack.entry]\nnodes = "vision.{name}:NODES"\n'
        ).encode()
        for (name, path), version in zip(
            paths.items(), ("1.0", "7.2", "3.4rc1", "0.5", "4.2.1"), strict=True
        )
    }

    def git(command, **kwargs):
        if "rev-parse" in command:
            return "a" * 40 + "\n"
        if "ls-tree" in command:
            manifests = [
                path for path in committed if "b" * 40 not in command or path != paths["first"]
            ]
            return "\0".join(
                [*manifests, "packages/dinkster-nodes-dev/dinkster-pack.toml", ""]
            ).encode()
        if "show" in command:
            data = committed[command[-1].split(":", 1)[1]]
            return (
                data.replace(b'version = "1.0"', b'version = "0.9"')
                if command[-1].startswith("b" * 40)
                else data
            )
        if "diff" in command:
            return ("\0".join(changed) + "\0").encode()
        raise AssertionError(command)

    monkeypatch.setattr("scripts.select_official_packs.subprocess.check_output", git)
    (tmp_path / "dinkster-pack.toml").write_text("invalid uncommitted data")
    assert select_packs(tmp_path, before="b" * 40) == expected
    assert select_packs(tmp_path, before="b" * 40, version_changes_only=True) == tuple(
        pack for pack in expected if pack in {"foundation", "first"}
    )
    assert select_packs(tmp_path, before="0" * 40) == (
        "first",
        "foundation",
        "image",
        "second",
        "unlisted",
    )
    assert select_packs(tmp_path, tag="pack/unlisted/v4.2.1") == ("unlisted",)
    assert select_packs(tmp_path, tag="pack/first/v3.4rc1") == ("first",)
    for tag in ("v3.4rc1", "pack/first/v3.4", "pack/unknown/v3.4rc1", "pack/first/v3.4rc1/extra"):
        with pytest.raises(ValueError, match="pack tag"):
            select_packs(tmp_path, tag=tag)
    for before in ("main", "--all", "a" * 39, "z" * 40):
        with pytest.raises(ValueError, match="full Git commit"):
            select_packs(tmp_path, before=before)
    with pytest.raises(ValueError, match="select one"):
        select_packs(tmp_path)
    with pytest.raises(ValueError, match="select one"):
        select_packs(tmp_path, before="b" * 40, tag="pack/first/v3.4rc1")
    first = paths["first"]
    committed[first] = committed[first].replace(b'version = "3.4rc1"', b'version = "2.0"')
    with pytest.raises(ValueError, match="pack tag"):
        select_packs(tmp_path, tag="pack/first/v3.4rc1")
    committed[first] = committed[first].replace(b'engine = ">=0.2,<1"', b'engine = ""')
    with pytest.raises(ManifestError, match="engine"):
        select_packs(tmp_path, tag="pack/first/v2.0")

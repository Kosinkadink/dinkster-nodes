from __future__ import annotations

import hashlib
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
                "0.0.4",
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
            assert composition.packs["dinkster-image"].version == "0.0.9"
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
        (("README.md",), ()),
        ((".github/workflows/ci.yml",), ()),
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
    from scripts.select_official_packs import publication_selection, select_packs

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
            if kwargs.get("text"):
                return "\n".join(changed) + "\n"
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
    assert publication_selection(tmp_path, "b" * 40) == (
        ("first", "foundation", "image", "second", "unlisted")
        if changed
        in (
            ("src/dinkster/serve.py", "pyproject.toml"),
            (".github/workflows/ci.yml",),
            ("packages/dinkster-nodes-vision/pyproject.toml",),
        )
        else expected
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


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "unmarked",
        "newly-marked",
        "requires-pack",
        "pack-requires-pack",
        "manifest",
        "library-rejected",
    ],
)
def test_official_library_selection_and_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str | None
) -> None:
    from dinkster_registry import Lockfile

    from scripts.select_official_packs import select_packs

    files = {
        "packages/helpers/pyproject.toml": (
            '[project]\nname = "dinkster-test-helpers"\nversion = "0.0.1"\n'
            'requires-python = ">=3.12"\n'
            + (
                'dependencies = ["dinkster-test-consumer>=0.0.1"]\n'
                if problem == "requires-pack"
                else ""
            )
            + '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
            '[tool.hatch.build.targets.wheel]\npackages = ["src/dinkster_test_helpers"]\n'
            + ("" if problem == "unmarked" else "[tool.dinkster]\nofficial-library = true\n")
        ),
        "packages/helpers/src/dinkster_test_helpers/__init__.py": "VALUE = 17\n",
        "packages/dinkster-nodes-test/pyproject.toml": (
            '[project]\nname = "dinkster-test-consumer"\nversion = "0.0.1"\n'
        ),
        "packages/dinkster-nodes-test/dinkster-pack.toml": (
            '[pack]\nname = "dinkster-test-consumer"\nversion = "0.0.1"\n'
            'publisher = "dinkster"\nengine = ">=0.0.2,<0.1"\n'
            'requires = ["dinkster-test-helpers>=0.0.1"]\n'
            '[pack.entry]\nnodes = "dinkster_test_consumer:NODES"\n'
        ),
        "packages/dinkster-nodes-test/src/dinkster_test_consumer/__init__.py": "NODES = []\n",
    }
    if problem == "pack-requires-pack":
        manifest_path = "packages/dinkster-nodes-test/dinkster-pack.toml"
        files["packages/dinkster-nodes-supplier/dinkster-pack.toml"] = (
            files[manifest_path]
            .replace("dinkster-test-consumer", "dinkster-test-supplier")
            .replace("dinkster_test_consumer", "dinkster_test_supplier")
        )
        files["packages/dinkster-nodes-supplier/pyproject.toml"] = (
            '[project]\nname = "dinkster-test-supplier"\nversion = "0.0.1"\n'
        )
        files["packages/dinkster-nodes-supplier/src/dinkster_test_supplier/__init__.py"] = (
            "NODES = []\n"
        )
        files[manifest_path] = files[manifest_path].replace(
            'requires = ["dinkster-test-helpers>=0.0.1"]',
            'requires = ["dinkster-test-supplier>=0.0.1"]',
        )
    if problem == "manifest":
        files["packages/helpers/dinkster-pack.toml"] = files[
            "packages/dinkster-nodes-test/dinkster-pack.toml"
        ]
    tracked = io.BytesIO()
    with tarfile.open(fileobj=tracked, mode="w") as archive:
        for name, text in files.items():
            data = text.encode()
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))

    def git(command, **kwargs):
        if "rev-parse" in command:
            return "a" * 40
        if "ls-tree" in command:
            return ("\0".join(files) + "\0").encode()
        if "show" in command:
            content = files[command[-1].split(":", 1)[1]]
            if problem == "newly-marked" and command[-1].startswith("b" * 40):
                content = content.replace("official-library = true", "official-library = false")
            return content.encode()
        if "diff" in command:
            return b"packages/helpers/src/dinkster_test_helpers/__init__.py\0"
        if "archive" in command:
            return tracked.getvalue()
        raise AssertionError(command)

    monkeypatch.setattr("scripts.build_official_packs.subprocess.check_output", git)
    if problem == "manifest":
        with pytest.raises(ValueError, match="node manifest"):
            build_official_packs(tmp_path, tmp_path / "built")
        with pytest.raises(ValueError, match="duplicates"):
            select_packs(tmp_path, before="0" * 40)
        return
    expected = (
        ("dinkster-test-consumer",)
        + (() if problem == "unmarked" else ("dinkster-test-helpers",))
        + (("dinkster-test-supplier",) if problem == "pack-requires-pack" else ())
    )
    assert select_packs(tmp_path, before="0" * 40) == expected
    assert select_packs(tmp_path, before="b" * 40) == (
        () if problem == "unmarked" else ("dinkster-test-helpers",)
    )
    assert select_packs(tmp_path, before="b" * 40, version_changes_only=True) == (
        ("dinkster-test-helpers",) if problem == "newly-marked" else ()
    )
    if problem in {"requires-pack", "pack-requires-pack"}:
        with pytest.raises(ValueError, match="wheel.requires-pack"):
            build_official_packs(tmp_path, tmp_path / "built")
        return
    paths = build_official_packs(tmp_path, tmp_path / "built")
    repeated = build_official_packs(tmp_path, tmp_path / "repeated")
    assert {path.name: path.read_bytes() for path in paths} == {
        path.name: path.read_bytes() for path in repeated
    }
    catalog = Lockfile.from_record_json(paths[-1].read_text())
    assert tuple(pack.pack for pack in catalog.packs) == expected
    if problem != "unmarked":
        library = next(pack for pack in catalog.packs if pack.pack == "dinkster-test-helpers")
        assert library.claims == ()
        with zipfile.ZipFile(
            tmp_path / "built/dinkster_test_helpers-0.0.1-py3-none-any.whl"
        ) as wheel:
            assert wheel.read("dinkster_test_helpers/__init__.py") == b"VALUE = 17\n"
            assert not any(name.endswith("dinkster-pack.toml") for name in wheel.namelist())
    if problem not in {None, "library-rejected"}:
        return

    import json

    from dinkster_workers.pack_lock import PackLock

    from dinkster.registries import NamedRegistry, RegistryPublishError
    from scripts.publish_official_packs import publish_official_packs
    from tests.registry_fixtures import _serve_artifacts

    constraints = tmp_path / "constraints.txt"
    constraints.write_text("")
    native = build_official_packs(
        tmp_path,
        tmp_path / "native",
        packs=("dinkster-test-consumer",),
        variant="linux-cpu",
        python=Path(sys.executable),
        constraints=constraints,
    )
    assert native[-1].read_bytes() == paths[-1].read_bytes()
    lock = PackLock.read(tmp_path / "native/dinkster-test-consumer-0.0.1-linux-cpu.pack-lock.json")
    assert [package.name for package in lock.packages] == ["dinkster-test-helpers"]
    library_bytes = (tmp_path / "built/dinkster_test_helpers-0.0.1-py3-none-any.whl").read_bytes()
    assert lock.packages[0].wheels[0].url == (
        "https://registry.dinkster.org/static/wheels/"
        + hashlib.sha256(library_bytes).hexdigest()
        + "/dinkster_test_helpers-0.0.1-py3-none-any.whl"
    )
    base = {
        "format": "dinkster.base/1",
        "variant": "linux-cpu",
        "platform": {"os": "linux", "arch": "x86_64"},
        "python": ".".join(map(str, sys.version_info[:3])),
        "interpreter": {},
        "wheels": [],
        "replaces": [],
        "workerProtocol": 11,
        "nvidia": None,
        "archive": {"format": "tar+zstd"},
    }
    identity = hashlib.sha256(
        json.dumps(base, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    base["id"] = "sha256:" + identity
    base["archive"]["parts"] = []
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    (tmp_path / "release-manifest.json").write_text(
        json.dumps(
            {
                "format": "dinkster.engine/1",
                "repository": "Kosinkadink/Dinkster",
                "version": "0.0.4.dev123",
                "commit": "c" * 40,
                "workerProtocol": 11,
                "artifacts": [
                    {
                        "name": "constraints.txt",
                        "size": 0,
                        "sha256": hashlib.sha256(b"").hexdigest(),
                    }
                ],
                "dependencies": {"linux-cpu": {"base": base["id"], "wheels": []}},
            }
        )
    )
    responses = {
        f"/v1/publish/{name}/versions/0.0.1": json.dumps(
            {"status": "queued", "candidateId": name}
        ).encode()
        for name in expected
    }
    server = _serve_artifacts(responses, [])
    respond = vars(server.RequestHandlerClass)["respond"]
    submitted = []

    def registry_response(handler):
        if handler.command == "POST":
            name = handler.path.split("/")[3]
            if name == "dinkster-test-consumer":
                assert submitted == ["dinkster-test-helpers"]
            submitted.append(name)
            if problem == "library-rejected":
                responses[f"/v1/publish/{name}"] = json.dumps(
                    {
                        "status": "rejected",
                        "reason": "library refused",
                        "findings": [{"code": "wheel.invalid", "message": "refused"}],
                    }
                ).encode()
            else:
                responses[f"/v1/packs/{name}/versions/0.0.1"] = json.dumps(
                    {
                        "packName": name,
                        "version": "0.0.1",
                        "state": "active",
                        "artifact": {
                            "digest": next(
                                pack.artifact_digest for pack in catalog.packs if pack.pack == name
                            )
                        },
                    }
                ).encode()
        respond(handler)

    monkeypatch.setattr(server.RequestHandlerClass, "respond", registry_response)
    try:
        if problem == "library-rejected":
            with pytest.raises(RegistryPublishError, match="activation refused.*wheel.invalid"):
                publish_official_packs(
                    tmp_path,
                    tmp_path / "built",
                    [base_path],
                    [path for path in native if path.name.endswith(".pack-lock.json")],
                    tmp_path / "submitted",
                    registry=NamedRegistry("", f"http://127.0.0.1:{server.server_port}"),
                )
            assert submitted == ["dinkster-test-helpers"]
            return
        result = publish_official_packs(
            tmp_path,
            tmp_path / "built",
            [base_path],
            [path for path in native if path.name.endswith(".pack-lock.json")],
            tmp_path / "submitted",
            registry=NamedRegistry("", f"http://127.0.0.1:{server.server_port}"),
        )
        assert (
            [row["pack"] for row in result]
            == submitted
            == ["dinkster-test-helpers", "dinkster-test-consumer"]
        )
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "prepare-only",
        "published-identical",
        "check-published",
        "dependency-active",
        "dependency-levels",
        "dependency-levels-timeout",
        "dependency-external",
        "dependency-rejected",
        "dependency-timeout",
        "submitted-progress",
        "submitted-rejected",
        "submitted-timeout",
        "submitted-digest",
        "published-digest",
        "published-size",
        "published-lock",
        "published-lock-size",
        "published-base",
        "published-coverage",
        "published-publisher",
        "lookup-failed",
        "publish-conflict",
        "missing-base",
        "duplicate-base",
        "wrong-base",
        "engine-protocol",
        "engine-repository",
        "missing-lock",
        "duplicate-lock",
        "extra-lock",
        "pack",
        "variant",
        "python",
        "wheel-url",
        "index-url",
        "credentials",
        "empty-wheel",
        "boolean-size",
        "archive",
        "source-version",
        "publisher",
        "declaration",
    ],
)
def test_official_publication_requires_complete_supported_base_bindings(
    tmp_path: Path, problem: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import json

    from dinkster_registry import (
        ArtifactError,
        LockedPack,
        Lockfile,
        artifact_digest,
        build_artifact,
    )
    from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock

    from dinkster.registries import NamedRegistry, RegistryPublishError
    from scripts.publish_official_packs import publish_official_packs
    from tests.registry_fixtures import _serve_artifacts

    sources = tmp_path / "sources"
    sources.mkdir()
    packs = []
    versions = [("first-pack", "1.2.3"), ("second-pack", "7.4.0rc1")]
    chained = problem in {"dependency-levels", "dependency-levels-timeout"}
    if chained:
        versions.extend([("third-pack", "2.0"), ("fourth-pack", "3.0")])
    for name, version in versions:
        tree = tmp_path / name
        tree.mkdir()
        observed = "9.9" if name == "second-pack" and problem == "source-version" else version
        publisher = "other" if name == "second-pack" and problem == "publisher" else "dinkster"
        declaration = (
            ""
            if name == "second-pack" and problem == "declaration"
            else f'version = "{observed}"\nengine = ">=0.2,<1"\n'
        )
        (tree / "dinkster-pack.toml").write_text(
            f'[pack]\nname = "{name}"\npublisher = "{publisher}"\n'
            + declaration
            + f'[pack.entry]\nnodes = "{name.replace("-", "_")}:NODES"\n'
            + (
                '[pack.dependencies]\nthird-pack = ">=2,<3"\n'
                if name == "first-pack" and problem == "dependency-external"
                else '[pack.dependencies]\nsecond-pack = ">=7,<8"\n'
                if name == "first-pack" and problem and problem.startswith("dependency-")
                else '[pack.dependencies]\nthird-pack = ">=2,<3"\n'
                if name == "second-pack" and chained
                else ""
            )
        )
        module = tree / name.replace("-", "_")
        module.mkdir()
        (module / "__init__.py").write_text("raise RuntimeError('must not import')\n")
        archive = sources / f"{module.name}-{version}-py3-none-any.whl"
        if declaration:
            built = tmp_path / f"{module.name}-{observed}-py3-none-any.whl"
            build_pack_archive(tree, built)
            built.rename(archive)
        else:
            build_artifact(tree, archive)
        digest = artifact_digest(archive.read_bytes())
        packs.append(LockedPack(name, version, digest, "dinkster", ()))
    (sources / "official-packs.lock").write_text(Lockfile.of(packs).record_json())
    if problem == "archive":
        (sources / "second_pack-7.4.0rc1-py3-none-any.whl").write_bytes(b"changed")
    bases, supported, locks = [], {}, []
    expected_locks = {}
    variants = {
        "linux-cpu": ("linux", "3.12.9"),
        "linux-nvidia": ("linux", "3.12.9"),
        "win-cpu": ("windows", "3.14.10"),
        "win-nvidia": ("windows", "3.14.10"),
        "win-nvidia-cu126": ("windows", "3.14.10"),
        "mac-arm64": ("macos", "3.12.9"),
    }
    for variant, (system, python) in variants.items():
        base = {
            "format": "dinkster.base/1",
            "variant": variant,
            "platform": {"os": system, "arch": "arm64" if system == "macos" else "x86_64"},
            "python": python,
            "interpreter": {},
            "wheels": [],
            "replaces": [],
            "workerProtocol": 11,
            "nvidia": None,
            "archive": {"format": "tar+zstd"},
        }
        identity = hashlib.sha256(
            json.dumps(base, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        supported[variant] = identity
        base["id"] = "sha256:" + identity
        base["archive"]["parts"] = []
        path = tmp_path / f"{variant}.json"
        path.write_text(json.dumps(base))
        bases.append(path)
        for pack in packs:
            corrupt = pack.pack == "second-pack" and variant == "win-nvidia"
            path = tmp_path / f"{pack.pack}-{pack.version}-{variant}.pack-lock.json"
            PackLock(
                "wrong" if corrupt and problem == "pack" else pack.pack,
                "wrong" if corrupt and problem == "variant" else variant,
                "<3.14" if corrupt and problem == "python" else ">=3.12",
                ("fixture-dependency==2.0",),
                (
                    "file:///local/index"
                    if corrupt and problem == "index-url"
                    else "https://pypi.org/simple",
                ),
                (
                    LockedDistribution(
                        "fixture-dependency",
                        "2.0",
                        (
                            LockedWheel(
                                "fixture_dependency-2.0-py3-none-any.whl",
                                0
                                if corrupt and problem == "empty-wheel"
                                else True
                                if corrupt and problem == "boolean-size"
                                else 17,
                                "b" * 64,
                                "file:///local/wheel.whl"
                                if corrupt and problem == "wheel-url"
                                else "https://user:password@example.test/wheel.whl"
                                if corrupt and problem == "credentials"
                                else "https://example.test/wheel.whl",
                            ),
                        ),
                    ),
                ),
            ).write(path)
            locks.append(path)
            expected_locks[pack.pack, variant] = json.loads(path.read_bytes())
    if problem == "wrong-base":
        supported["win-nvidia"] = "c" * 64
    (tmp_path / "release-manifest.json").write_text(
        json.dumps(
            {
                "format": "dinkster.engine/1",
                "repository": "other/engine"
                if problem == "engine-repository"
                else "Kosinkadink/Dinkster",
                "version": "0.0.4.dev123",
                "commit": "c" * 40,
                "workerProtocol": 12 if problem == "engine-protocol" else 11,
                "artifacts": [
                    {
                        "name": "constraints.txt",
                        "size": 0,
                        "sha256": hashlib.sha256(b"").hexdigest(),
                    }
                ],
                "dependencies": {
                    variant: {"base": "sha256:" + identity, "wheels": []}
                    for variant, identity in supported.items()
                },
            }
        )
    )
    if problem == "missing-base":
        bases.pop()
    elif problem == "duplicate-base":
        bases.append(bases[-1])
    elif problem == "missing-lock":
        locks.pop()
    elif problem == "duplicate-lock":
        locks.append(locks[-1])
    elif problem == "extra-lock":
        extra = tmp_path / "unselected-9.0-win-nvidia.pack-lock.json"
        extra.write_bytes(locks[-1].read_bytes())
        locks.append(extra)
    requests = []
    files = {
        f"/v1/publish/{pack.pack}/versions/{pack.version}": json.dumps(
            {"status": "queued", "candidateId": pack.pack}
        ).encode()
        for pack in packs
    }
    if problem == "dependency-external":
        files["/v1/packs/third-pack/versions?limit=100"] = json.dumps(
            {"items": [{"version": "9.0", "state": "active"}], "nextCursor": "next"}
        ).encode()
        files["/v1/packs/third-pack/versions?limit=100&cursor=next"] = json.dumps(
            {"items": [{"version": "2.3", "state": "active"}], "nextCursor": None}
        ).encode()
        files["/v1/packs/third-pack/versions/2.3"] = json.dumps(
            {"packName": "third-pack", "version": "2.3", "state": "active"}
        ).encode()
    if problem and (problem.startswith("published-") or problem == "check-published"):
        for pack in packs:
            record = {
                "packName": pack.pack,
                "version": pack.version,
                "publishedBy": {"principalId": "dinkster"},
                "artifact": {
                    "digest": pack.artifact_digest,
                    "size": (
                        sources / f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl"
                    )
                    .stat()
                    .st_size,
                },
                "installLocks": {},
            }
            for variant in variants:
                raw = json.dumps(
                    expected_locks[pack.pack, variant], sort_keys=True, separators=(",", ":")
                ).encode("ascii")
                record["installLocks"][variant] = {
                    "baseId": "sha256:" + supported[variant],
                    "artifactDigest": pack.artifact_digest,
                    "digest": "blake3:" + blake3(raw).hexdigest(),
                    "size": len(raw),
                    "url": "/ignored/download/url",
                }
            if pack.pack == "second-pack":
                if problem == "published-digest":
                    record["artifact"]["digest"] = "blake3:" + "f" * 64
                elif problem == "published-size":
                    record["artifact"]["size"] += 1
                elif problem == "published-lock":
                    record["installLocks"]["win-nvidia"]["digest"] = "blake3:" + "e" * 64
                elif problem == "published-lock-size":
                    record["installLocks"]["win-nvidia"]["size"] += 1
                elif problem == "published-base":
                    record["installLocks"]["win-nvidia"]["baseId"] = "sha256:" + "e" * 64
                elif problem == "published-coverage":
                    del record["installLocks"]["win-nvidia"]
                elif problem == "published-publisher":
                    record["publishedBy"]["principalId"] = "other"
            files[f"/v1/packs/{pack.pack}/versions/{pack.version}"] = json.dumps(record).encode()
    server = _serve_artifacts(files, requests)
    respond = vars(server.RequestHandlerClass)["respond"]
    submitted = []
    clock = [0.0]
    polls = [0]
    activation_polls: dict[str, int] = {}
    if problem and problem.startswith(("dependency-", "submitted-")):
        monkeypatch.setenv("PUBLICATION_TEST_TOKEN", "test-token")
        monkeypatch.setattr("scripts.publish_official_packs.time.monotonic", lambda: clock[0])

        def advance(seconds):
            clock[0] += seconds

        monkeypatch.setattr("scripts.publish_official_packs.time.sleep", advance)

    def registry_response(handler):
        if handler.command == "POST":
            import email.parser

            body = handler.rfile.read(int(handler.headers["Content-Length"]))
            message = email.parser.BytesParser().parsebytes(
                f"Content-Type: {handler.headers['Content-Type']}\r\n\r\n".encode() + body
            )
            bindings = []
            for part in message.walk():
                if part.get_param("name", header="content-disposition") == "engine":
                    payload = part.get_payload(decode=True)
                    assert isinstance(payload, bytes)
                    bindings.append(json.loads(payload))
            assert bindings == [
                {
                    "commit": "c" * 40,
                    "manifestDigest": "sha256:"
                    + hashlib.sha256((tmp_path / "release-manifest.json").read_bytes()).hexdigest(),
                }
            ]
            submitted.append(handler.path)
            name, version = handler.path.split("/")[3::2]
            if not (problem and problem.startswith("submitted-")) and not (
                problem
                and problem.startswith("dependency-")
                and problem != "dependency-external"
                and name in {"second-pack", "third-pack"}
            ):
                pack = next(pack for pack in packs if pack.pack == name)
                files[f"/v1/packs/{name}/versions/{version}"] = json.dumps(
                    {
                        "packName": name,
                        "version": version,
                        "state": "active",
                        "artifact": {"digest": pack.artifact_digest},
                    }
                ).encode()
        if (
            problem
            and problem.startswith("submitted-")
            and handler.path.startswith("/v1/publish/")
            and handler.command == "GET"
        ):
            name = handler.path.rsplit("/", 1)[1]
            pack = next(pack for pack in packs if pack.pack == name)
            count = activation_polls.get(name, 0)
            activation_polls[name] = count + 1
            state = ("queued", "scanning", "retrying", "accepted")[min(count, 3)]
            if problem == "submitted-rejected" and name == "second-pack" and count >= 2:
                state = "rejected"
            files[handler.path] = json.dumps(
                {"status": state, "reason": "probe.timeout" if state == "rejected" else None}
            ).encode()
            if count >= 3 and state == "accepted" and problem != "submitted-timeout":
                files[f"/v1/packs/{name}/versions/{pack.version}"] = json.dumps(
                    {
                        "packName": name,
                        "version": pack.version,
                        "state": "active",
                        "artifact": {
                            "digest": "blake3:" + "f" * 64
                            if problem == "submitted-digest" and name == "second-pack"
                            else pack.artifact_digest
                        },
                    }
                ).encode()
        if problem and problem.startswith("dependency-") and submitted:
            if handler.path in {"/v1/publish/second-pack", "/v1/publish/third-pack"}:
                assert handler.headers.get("Authorization") == "Bearer test-token"
                polls[0] += 1
                activation_polls[handler.path] = activation_polls.get(handler.path, 0) + 1
                files[handler.path] = json.dumps(
                    {
                        "status": "rejected" if problem == "dependency-rejected" else "scanning",
                        "reason": "scanner refused" if problem == "dependency-rejected" else None,
                        "findings": [{"code": "manifest.invalid", "message": "missing dependency"}],
                    }
                ).encode()
                if problem == "dependency-active" or (
                    chained
                    and activation_polls[handler.path] == 2
                    and not (
                        problem == "dependency-levels-timeout"
                        and handler.path == "/v1/publish/second-pack"
                    )
                ):
                    name = handler.path.rsplit("/", 1)[1]
                    version = dict(versions)[name]
                    files[f"/v1/packs/{name}/versions/{version}"] = json.dumps(
                        {
                            "packName": name,
                            "version": version,
                            "state": "active",
                            "artifact": {
                                "digest": next(
                                    pack.artifact_digest for pack in packs if pack.pack == name
                                )
                            },
                        }
                    ).encode()
            if handler.command == "POST" and "first-pack" in handler.path:
                assert polls[0] == (
                    4
                    if problem == "dependency-levels"
                    else 0
                    if problem == "dependency-external"
                    else 1
                )
        if handler.path.startswith("/v1/packs/"):
            assert handler.headers.get("Authorization") is None
        if (
            problem == "lookup-failed"
            or problem == "publish-conflict"
            and handler.command == "POST"
        ):
            requests.append((handler.path, handler.headers.get("Authorization")))
            handler.send_response(503 if problem == "lookup-failed" else 409)
            handler.end_headers()
            handler.wfile.write(b"release already has different archive or lock bindings")
        else:
            respond(handler)

    monkeypatch.setattr(server.RequestHandlerClass, "respond", registry_response)
    output = tmp_path / "publication"
    before = {path: path.read_bytes() for path in [*sources.iterdir(), *locks, *bases]}
    try:
        registry = NamedRegistry(
            "", f"http://127.0.0.1:{server.server_port}", token_env="PUBLICATION_TEST_TOKEN"
        )
        lookup_paths = [f"/v1/packs/{pack.pack}/versions/{pack.version}" for pack in packs]
        if problem and problem.startswith("submitted-"):
            if problem == "submitted-progress":
                result = publish_official_packs(
                    tmp_path,
                    sources,
                    bases,
                    locks,
                    output,
                    registry=registry,
                    activation_timeout=20,
                )
                assert [row["state"] for row in result] == ["published", "published"]
                assert set(activation_polls) == {"first-pack", "second-pack"}
            else:
                with pytest.raises(RegistryPublishError) as failure:
                    publish_official_packs(
                        tmp_path,
                        sources,
                        bases,
                        locks,
                        output,
                        registry=registry,
                        activation_timeout=20,
                    )
                message = str(failure.value)
                assert "second-pack@7.4.0rc1" in message
                assert "candidate second-pack" in message
                if problem == "submitted-rejected":
                    assert "probe.timeout" in message
                elif problem == "submitted-timeout":
                    assert "first-pack@1.2.3" in message
                    assert "candidate first-pack" in message
                    assert "timed out" in message
                else:
                    assert "digest" in message
        elif problem and problem.startswith("dependency-"):
            if problem == "dependency-external":
                result = publish_official_packs(
                    tmp_path, sources, bases, locks, output, registry=registry, activation_timeout=5
                )
                assert submitted == [
                    "/v1/publish/first-pack/versions/1.2.3",
                    "/v1/publish/second-pack/versions/7.4.0rc1",
                ]
                assert [path for path, _ in requests][2:5] == [
                    "/v1/packs/third-pack/versions?limit=100",
                    "/v1/packs/third-pack/versions?limit=100&cursor=next",
                    "/v1/packs/third-pack/versions/2.3",
                ]
            elif problem in {"dependency-active", "dependency-levels"}:
                result = publish_official_packs(
                    tmp_path, sources, bases, locks, output, registry=registry, activation_timeout=5
                )
                assert [row["pack"] for row in result] == (
                    ["fourth-pack", "third-pack", "second-pack", "first-pack"]
                    if problem == "dependency-levels"
                    else ["second-pack", "first-pack"]
                )
                assert submitted == [
                    *(
                        [
                            "/v1/publish/fourth-pack/versions/3.0",
                            "/v1/publish/third-pack/versions/2.0",
                        ]
                        if problem == "dependency-levels"
                        else []
                    ),
                    "/v1/publish/second-pack/versions/7.4.0rc1",
                    "/v1/publish/first-pack/versions/1.2.3",
                ]
                if chained:
                    assert clock[0] == 8
            else:
                with pytest.raises(
                    RegistryPublishError,
                    match=(
                        "second-pack@7.4.0rc1.*manifest.invalid.*missing dependency"
                        if problem == "dependency-rejected"
                        else "second-pack@7.4.0rc1.*timed out"
                    ),
                ):
                    publish_official_packs(
                        tmp_path,
                        sources,
                        bases,
                        locks,
                        output,
                        registry=registry,
                        activation_timeout=5,
                    )
                assert submitted == [
                    *(
                        [
                            "/v1/publish/fourth-pack/versions/3.0",
                            "/v1/publish/third-pack/versions/2.0",
                        ]
                        if chained
                        else []
                    ),
                    "/v1/publish/second-pack/versions/7.4.0rc1",
                ]
                if chained:
                    assert clock[0] == 9
                else:
                    assert clock[0] <= 5
        elif problem in {"lookup-failed", "publish-conflict"}:
            with pytest.raises(
                RegistryPublishError,
                match="first-pack@1.2.3.*"
                + (
                    "lookup failed"
                    if problem == "lookup-failed"
                    else "not retried.*bump the pack version"
                ),
            ):
                publish_official_packs(tmp_path, sources, bases, locks, output, registry=registry)
            assert [path for path, _ in requests] == (
                lookup_paths[:1]
                if problem == "lookup-failed"
                else lookup_paths + ["/v1/publish/first-pack/versions/1.2.3"]
            )
        elif problem and problem.startswith("published-") and problem != "published-identical":
            with pytest.raises(
                RegistryPublishError, match="second-pack@7.4.0rc1.*bump the pack version"
            ):
                publish_official_packs(tmp_path, sources, bases, locks, output, registry=registry)
            assert [path for path, _ in requests] == lookup_paths
        elif problem not in {None, "prepare-only", "published-identical", "check-published"}:
            with pytest.raises((ValueError, ArtifactError)):
                publish_official_packs(tmp_path, sources, bases, locks, output, registry=registry)
            assert not output.exists()
            assert requests == []
        else:
            result = publish_official_packs(
                tmp_path,
                sources,
                bases,
                locks,
                output,
                registry=registry if problem in {None, "published-identical"} else None,
                preflight_registry=registry.endpoint if problem == "check-published" else None,
            )
            assert len(result) == 2
            assert [row["version"] for row in result] == ["1.2.3", "7.4.0rc1"]
            for pack, row in zip(packs, result, strict=True):
                lock_map = row["locks"]
                assert isinstance(lock_map, str)
                bindings = json.loads(Path(lock_map).read_bytes())
                assert set(bindings) == set(variants)
                for variant, identity in supported.items():
                    assert bindings[variant] == {
                        "baseId": "sha256:" + identity,
                        "artifactDigest": pack.artifact_digest,
                        "lock": expected_locks[pack.pack, variant],
                    }
                if problem is None:
                    assert row["state"] == "published" and row["candidateId"] == pack.pack
                elif problem == "published-identical":
                    assert row["state"] == "published" and "candidateId" not in row
            assert [path for path, _ in requests] == (
                lookup_paths
                + [f"/v1/publish/{pack.pack}/versions/{pack.version}" for pack in packs]
                + lookup_paths
                if problem is None
                else lookup_paths
                if problem in {"check-published", "published-identical"}
                else []
            )
        assert {path: path.read_bytes() for path in before} == before
        assert not list(tmp_path.glob("official-publication-*"))
    finally:
        server.shutdown()
        server.server_close()

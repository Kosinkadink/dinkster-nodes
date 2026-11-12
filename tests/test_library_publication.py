from __future__ import annotations

import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.mark.parametrize(
    "dependent,engine_dependency", [(False, False), (True, False), (False, True)]
)
def test_probe_index_stages_selected_library_dependency_without_public_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependent: bool,
    engine_dependency: bool,
) -> None:
    from dinkster_registry import LockedPack, Lockfile, artifact_digest

    from scripts.stage_publication_probe_index import stage_index
    from tests.registry_fixtures import _wheel

    monkeypatch.setenv("UV_NO_INDEX", "1")
    sources = tmp_path / "selected %20 sources"
    sources.mkdir()
    helper = _wheel(sources, "test-helper", "0.0.1", "test_helper", "VALUE = 37\n")
    consumer = _wheel(
        sources,
        "test-consumer",
        "0.0.2",
        "test_consumer",
        "import test_helper\n",
        requires=("test-helper==0.0.1",) if dependent else (),
    )
    entries = tuple(
        LockedPack(
            name, version, artifact_digest((sources / wheel.filename).read_bytes()), "dinkster", ()
        )
        for name, version, wheel in (
            ("test-consumer", "0.0.2", consumer),
            ("test-helper", "0.0.1", helper),
        )
    )
    (sources / "official-packs.lock").write_text(Lockfile.of(entries).record_json())
    engine = None
    if engine_dependency:
        engine = tmp_path / "engine"
        engine.mkdir()
        _wheel(
            engine,
            "dinkster-api",
            "0.0.3",
            "dinkster_api",
            "",
            requires=("test-helper==0.0.1",),
        )
    output = tmp_path / "index"
    result = stage_index(sources, output, engine=engine)
    if dependent or engine_dependency:
        assert result["distributions"] == 1
        assert result["wheels"] == [
            {
                "package": "test-helper",
                "filename": helper.filename,
                "sha256": helper.sha256,
                "size": helper.size,
            }
        ]
        assert (output / helper.filename).read_bytes() == (sources / helper.filename).read_bytes()
        assert (
            f"../../{helper.filename}#sha256={helper.sha256}"
            in (output / "simple/test-helper/index.html").read_text()
        )
    else:
        assert result == {"distributions": 0, "wheels": []}
    with ZipFile(sources / consumer.filename, "a") as archive:
        archive.writestr("changed.txt", "source drift")
    with pytest.raises(ValueError, match="probe source digest mismatch: test-consumer"):
        stage_index(sources, tmp_path / "bad-index", engine=engine)


def test_probe_index_refuses_local_dependency_outside_selected_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_registry import Lockfile

    from scripts.stage_publication_probe_index import stage_index

    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "official-packs.lock").write_text(Lockfile.of(()).record_json())
    outside = tmp_path / "outside %20 dependency.whl"
    outside.write_bytes(b"unselected wheel")
    output = tmp_path / "index"

    def resolve(*args, **kwargs):
        (output / "pylock.toml").write_text(
            f'[[packages]]\nname = "outside"\n[[packages.wheels]]\nurl = "{outside.as_uri()}"\n'
        )

    monkeypatch.setattr("scripts.stage_publication_probe_index.subprocess.run", resolve)
    with pytest.raises(ValueError, match="local probe wheel must be a selected source"):
        stage_index(sources, output)
    assert not (output / outside.name).exists()


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "engine",
        "base",
        "base-bytes",
        "dependency",
        "protocol",
        "artifact",
        "source",
        "runtime-base",
        "runtime-variant",
    ],
)
def test_native_publication_locks_bind_published_inputs_and_preserve_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str | None
) -> None:
    from dinkster_supervisor.environment_state import content_digest
    from dinkster_workers.pack_lock import PackLock

    from scripts import build_release_pack_locks as native
    from scripts.build_official_packs import build
    from tests.registry_fixtures import _wheel

    files = {
        "packages/helpers/pyproject.toml": (
            '[project]\nname = "test-helpers"\nversion = "0.0.1"\n'
            'requires-python = ">=3.12"\n'
            'dependencies = ["fixture-dependency==1.0", "fixture-runtime==3.0"]\n'
            '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
            '[tool.hatch.build.targets.wheel]\npackages = ["src/test_helpers"]\n'
            "[tool.dinkster]\nofficial-library = true\n"
        ),
        "packages/helpers/src/test_helpers/__init__.py": "VALUE = 37\n",
    }
    tracked = io.BytesIO()
    with tarfile.open(fileobj=tracked, mode="w") as archive:
        for name, text in files.items():
            data = text.encode()
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    original = subprocess.check_output

    def git(command, **kwargs):
        if command[0] != "git":
            return original(command, **kwargs)
        if "rev-parse" in command:
            return "a" * 40
        if "archive" in command:
            return tracked.getvalue()
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "check_output", git)
    sources = tmp_path / "sources"
    build(tmp_path, sources, packs=["test-helpers"])
    originals = {path.name: path.read_bytes() for path in sources.iterdir()}
    system = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}[platform.system()]
    base_wheel = _wheel(tmp_path, "fixture-base", "2.0", "fixture_base", "VALUE = 43\n")
    runtime_wheel = _wheel(tmp_path, "fixture-runtime", "3.0", "fixture_runtime", "VALUE = 47\n")
    runtime_row = {
        "name": "fixture-runtime",
        "version": "3.0",
        "filename": runtime_wheel.filename,
        "sha256": runtime_wheel.sha256,
        "size": runtime_wheel.size,
        "index": "https://pypi.org/simple",
        "url": f"https://example.org/runtime/{runtime_wheel.filename}",
    }
    base_row = {
        "name": "fixture-base",
        "version": "2.0",
        "filename": base_wheel.filename,
        "sha256": base_wheel.sha256,
        "size": base_wheel.size,
        "index": "https://pypi.org/simple",
        "url": f"https://example.org/{base_wheel.filename}",
    }
    base = {
        "format": "dinkster.base/1",
        "variant": "fixture-cpu",
        "platform": {"os": system, "arch": "arm64" if system == "macos" else "x86_64"},
        "python": platform.python_version(),
        "interpreter": {
            "executable": "python/python.exe" if system == "windows" else "python/bin/python3"
        },
        "wheels": [base_row],
        "replaces": ["fixture-base"],
        "workerProtocol": 11,
        "nvidia": None,
        "archive": {"format": "tar+zstd"},
    }
    base["id"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(base, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
    )
    base["archive"]["parts"] = [{"filename": "base.tar.zst.001", "size": 1, "sha256": "b" * 64}]
    manifest = tmp_path / "base.json"
    manifest.write_text(json.dumps(base))
    if problem == "base-bytes":
        manifest.write_text(json.dumps({**base, "workerProtocol": 12}))
    release = tmp_path / "release"
    release.mkdir()
    wheel = _wheel(release, "fixture-dependency", "1.0", "fixture_dependency", "VALUE = 41\n")
    (release / "constraints.txt").write_bytes(b"")
    engine = {
        "format": "dinkster.engine/1",
        "repository": "Kosinkadink/Dinkster",
        "version": "0.0.4.dev123",
        "commit": "c" * 40,
        "tag": "main-" + "c" * 40,
        "workerProtocol": 12 if problem == "protocol" else 11,
        "dependencies": {
            "fixture-cpu": {
                "base": "sha256:" + "d" * 64 if problem == "base" else base["id"],
                "wheels": [
                    {**base_row, "sha256": "f" * 64} if problem == "dependency" else base_row,
                    runtime_row,
                ],
            }
        },
        "artifacts": [
            {"name": wheel.filename, "size": wheel.size, "sha256": wheel.sha256},
            {"name": "constraints.txt", "size": 0, "sha256": hashlib.sha256(b"").hexdigest()},
        ],
    }
    metadata = release / "release-manifest.json"
    metadata.write_text(json.dumps(engine))
    identity = content_digest(metadata)
    if problem == "engine":
        metadata.write_text(json.dumps({**engine, "commit": "e" * 40}))
    if problem == "artifact":
        (release / wheel.filename).write_bytes(b"changed release wheel")
    if problem == "source":
        (sources / "test_helpers-0.0.1-py3-none-any.whl").write_bytes(b"changed source wheel")

    runtime = tmp_path / "runtime"

    def read_runtime(directory, selected_identity):
        # The generation validator has separate coverage; resolution still uses real wheels.
        from dinkster_supervisor.environment_artifacts import engine_constraints
        from dinkster_supervisor.environment_constraints import effective_constraints

        assert directory == runtime
        assert selected_identity == identity
        (runtime / "wheelhouse").mkdir(parents=True)
        for source in (
            tmp_path / base_wheel.filename,
            tmp_path / runtime_wheel.filename,
            release / wheel.filename,
        ):
            shutil.copyfile(source, runtime / "wheelhouse" / source.name)
        constraints = runtime / "constraints.txt"
        constraints.write_text(
            effective_constraints(engine_constraints(engine, "fixture-cpu"), base, strict=True)
        )
        python = runtime / base["interpreter"]["executable"]
        python.parent.mkdir(parents=True)
        if system == "windows":
            for path in Path(sys.base_prefix).iterdir():
                if path.suffix in {".exe", ".dll"}:
                    shutil.copy2(path, python.parent / path.name)
            shutil.copytree(Path(sys.base_prefix) / "Lib", python.parent / "Lib")
            shutil.copytree(Path(sys.base_prefix) / "DLLs", python.parent / "DLLs")
        else:
            python.symlink_to(sys.executable)
        return (
            python,
            "sha256:" + "e" * 64 if problem == "runtime-base" else base["id"],
            constraints,
            "other-cpu" if problem == "runtime-variant" else "fixture-cpu",
        )

    monkeypatch.setattr(native, "read_runtime", read_runtime)
    output = tmp_path / "locks"
    if problem:
        messages = {
            "engine": "engine manifest digest mismatch",
            "base": "selects a different base",
            "base-bytes": "base manifest digest mismatch",
            "dependency": "base wheel differs",
            "protocol": "worker protocols differ",
            "artifact": "release artifact differs",
            "source": "changed source artifact",
            "runtime-base": "runtime differs from the selected base",
            "runtime-variant": "runtime differs from the selected base",
        }
        with pytest.raises(ValueError, match=messages[problem]):
            native.build_locks(
                tmp_path,
                release,
                manifest,
                output,
                sources=sources,
                engine_identity=identity,
                runtime=runtime,
            )
        assert not output.exists()
    else:
        paths = native.build_locks(
            tmp_path,
            release,
            manifest,
            output,
            sources=sources,
            engine_identity=identity,
            runtime=runtime,
        )
        assert [path.name for path in paths] == ["test-helpers-0.0.1-fixture-cpu.pack-lock.json"]
        assert json.loads((output / "engine.json").read_text()) == {
            "engineVersion": "0.0.4.dev123",
            "engineRelease": identity,
            "engineCommit": "c" * 40,
            "packSourceCommit": "a" * 40,
            "baseId": base["id"],
            "variant": "fixture-cpu",
        }
        lock = PackLock.read(paths[0])
        locked = next(package for package in lock.packages if package.name == "fixture-dependency")
        assert [(row.filename, row.sha256, row.size, row.url) for row in locked.wheels] == [
            (
                wheel.filename,
                wheel.sha256,
                wheel.size,
                "https://github.com/Kosinkadink/Dinkster/releases/download/"
                f"main-{'c' * 40}/{wheel.filename}",
            )
        ]
        runtime_locked = next(
            package for package in lock.packages if package.name == "fixture-runtime"
        )
        assert [(row.filename, row.sha256, row.size, row.url) for row in runtime_locked.wheels] == [
            (
                runtime_wheel.filename,
                runtime_wheel.sha256,
                runtime_wheel.size,
                f"https://example.org/runtime/{runtime_wheel.filename}",
            )
        ]
        assert {path.name: path.read_bytes() for path in sources.iterdir()} == originals


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "labelled",
        "engine-bytes",
        "pin",
        "wheel",
        "base-bytes",
        "base-lock",
        "dependency",
        "protocol",
        "compatibility",
    ],
)
def test_publication_preparation_uses_pinned_engine_and_its_exact_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str | None
) -> None:
    from urllib.parse import quote

    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from dinkster_workers.pack_wheel import build_pack_archive

    from scripts import prepare_official_publication as preparation
    from tests.registry_fixtures import _serve_artifacts, _wheel

    commit = "c" * 40
    version = f"0.0.4.dev123+g{commit}"
    tag = f"main-{commit}"
    wheel = _wheel(tmp_path, "dinkster", version, "dinkster", "VALUE = 41\n")
    base_wheel = _wheel(tmp_path, "fixture-base", "2.0", "fixture_base", "VALUE = 43\n")
    base_row = {
        "name": "fixture-base",
        "version": "2.0",
        "filename": base_wheel.filename,
        "size": base_wheel.size,
        "sha256": base_wheel.sha256,
        "index": "https://pypi.org/simple",
        "url": f"https://example.org/{base_wheel.filename}",
    }
    base = {
        "format": "dinkster.base/1",
        "variant": "fixture-cpu",
        "platform": {"os": "linux", "arch": "x86_64"},
        "python": "3.12.9",
        "interpreter": {"executable": "python/bin/python3"},
        "wheels": [base_row],
        "replaces": ["fixture-base"],
        "workerProtocol": 12 if problem == "protocol" else 11,
        "nvidia": None,
        "archive": {"format": "tar+zstd"},
    }
    base["id"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(base, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
    )
    base["archive"]["parts"] = [{"filename": "base.tar.zst.001", "size": 1, "sha256": "b" * 64}]
    engine = {
        "format": "dinkster.engine/1",
        "repository": "Kosinkadink/Dinkster",
        "commit": commit,
        "version": version,
        "tag": tag,
        "workerProtocol": 11,
        "dependencies": {
            "fixture-cpu": {
                "base": "sha256:" + "d" * 64 if problem == "base-lock" else base["id"],
                "wheels": [
                    {**base_row, "sha256": "f" * 64} if problem == "dependency" else base_row
                ],
            }
        },
        "artifacts": [
            {"name": wheel.filename, "size": wheel.size, "sha256": wheel.sha256},
            {"name": "constraints.txt", "size": 0, "sha256": hashlib.sha256(b"").hexdigest()},
        ],
    }
    if problem == "labelled":
        engine["label"] = "2026-11-11-0.0.4.dev123"
        base["label"] = "2026-11-11-py3.12.9-torch2.9.0-" + base["id"][7:15]
    data = json.dumps(engine).encode()
    identity = hashlib.sha256(data).hexdigest()
    engine_name = f"{engine['label']}-{identity[:8]}" if problem == "labelled" else identity
    if problem == "labelled":
        tag = f"main-{engine_name}"
    selected_base = engine["dependencies"]["fixture-cpu"]["base"]
    base_name = base.get("label", selected_base[7:])
    metadata = {
        "engine/index.json": json.dumps(
            [
                {
                    "commit": commit,
                    "version": version,
                    **({"label": engine["label"]} if problem == "labelled" else {}),
                    "sha256": identity,
                    "path": f"engine/{engine_name}",
                }
            ]
        ).encode(),
        f"engine/{engine_name}/release-manifest.json": (
            json.dumps({**engine, "workerProtocol": 12}).encode()
            if problem == "engine-bytes"
            else data
        ),
        "base/fixture-cpu/index.json": json.dumps(
            [
                {
                    "id": selected_base,
                    "path": f"base/fixture-cpu/{base_name}",
                    **({"label": base_name} if problem == "labelled" else {}),
                }
            ]
        ).encode(),
        f"base/fixture-cpu/{base_name}/manifest.json": json.dumps(
            {**base, "python": "3.12.8"} if problem == "base-bytes" else base
        ).encode(),
    }
    pinned_tag = "main-" + "a" * 40 if problem == "pin" else tag
    (tmp_path / "pyproject.toml").write_text(
        "[tool.uv.sources]\ndinkster = { url = "
        f'"https://github.com/Kosinkadink/Dinkster/releases/download/{pinned_tag}/{wheel.filename}"'
        " }\n"
    )
    monkeypatch.setattr(preparation, "read_metadata", lambda root, name: metadata[name])
    fetch = preparation.fetch_artifact
    requests = []
    server = _serve_artifacts(
        {
            f"/{quote(wheel.filename, safe='')}": b"broken"
            if problem == "wheel"
            else (tmp_path / wheel.filename).read_bytes(),
            "/constraints.txt": b"",
        },
        requests,
    )
    mirror = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(
        preparation,
        "fetch_artifact",
        lambda mirrors, *args: fetch((mirror,), *args),
    )

    def sources(root, output, *, packs):
        output.mkdir()
        candidate = root / "candidate"
        (candidate / "src/test_helper").mkdir(parents=True)
        (candidate / "src/test_helper/__init__.py").write_text("NODES = []\n")
        (candidate / "dinkster-pack.toml").write_text(
            '[pack]\nname = "test-helper"\nversion = "0.0.2"\n'
            'publisher = "dinkster"\n'
            f'engine = "{">=0.1" if problem == "compatibility" else ">=0.0.2,<0.1"}"\n'
            '[pack.entry]\nnodes = "test_helper:NODES"\n'
        )
        archive = output / "test_helper-0.0.2-py3-none-any.whl"
        build_pack_archive(candidate, archive)
        (output / "official-packs.lock").write_text(
            Lockfile.of(
                (
                    LockedPack(
                        "test-helper",
                        "0.0.2",
                        artifact_digest(archive.read_bytes()),
                        "dinkster",
                        (),
                    ),
                )
            ).record_json()
        )

    monkeypatch.setattr(preparation, "build", sources)
    output = tmp_path / "prepared"
    try:
        if problem is not None and problem != "labelled":
            messages = {
                "engine-bytes": "engine manifest digest mismatch",
                "pin": "differs from the workspace pin",
                "wheel": "artifact",
                "base-bytes": "base manifest digest mismatch",
                "base-lock": "base manifest digest mismatch",
                "dependency": "base wheel differs",
                "protocol": "worker protocols differ",
                "compatibility": "engine",
            }
            with pytest.raises(ValueError, match=messages[problem]):
                preparation.prepare_inputs(tmp_path, output, ["test-helper"])
            assert not output.exists()
        else:
            assert preparation.prepare_inputs(tmp_path, output, ["test-helper"]) == [
                {
                    "variant": "fixture-cpu",
                    "runner": "ubuntu-latest",
                    "engineIdentity": "sha256:" + identity,
                }
            ]
            assert (output / "engine/release-manifest.json").read_bytes() == data
            assert (output / "engine" / wheel.filename).read_bytes() == (
                tmp_path / wheel.filename
            ).read_bytes()
            assert json.loads((output / "bases/fixture-cpu/manifest.json").read_bytes()) == base
            catalog = Lockfile.from_record_json(
                (output / "sources/official-packs.lock").read_text()
            )
            assert [(pack.pack, pack.version) for pack in catalog.packs] == [
                ("test-helper", "0.0.2")
            ]

            from dinkster_supervisor.environment_state import interpreter
            from dinkster_workers.pack_lock import GenerationPackages

            from scripts import prepare_admission_engine as admission

            def acquire(catalog, variant, store):
                constraints = store / "constraints.txt"
                constraints.write_text("")
                return constraints

            def provision(destination, *, store, base, constraints):
                destination.mkdir()
                shutil.copyfile(constraints, destination / "constraints.txt")
                return interpreter(destination)

            facts = GenerationPackages.inspect(Path(sys.executable))
            monkeypatch.setattr(admission, "acquire", acquire)
            monkeypatch.setattr(admission, "prepare_engine_interpreter", provision)
            monkeypatch.setattr(GenerationPackages, "inspect", lambda python: facts)
            runtime = tmp_path / "runtime"
            selected_identity = "sha256:" + identity
            prepared_wheel = output / "engine" / wheel.filename
            original_wheel = prepared_wheel.read_bytes()
            prepared_wheel.write_bytes(original_wheel + b"changed")
            with pytest.raises(ValueError, match="release artifact differs"):
                admission.prepare_engine(
                    output / "engine",
                    runtime,
                    tmp_path / "store",
                    output / "bases/fixture-cpu/manifest.json",
                    engine_identity=selected_identity,
                )
            prepared_wheel.write_bytes(original_wheel)
            python = admission.prepare_engine(
                output / "engine",
                runtime,
                tmp_path / "store",
                output / "bases/fixture-cpu/manifest.json",
                engine_identity=selected_identity,
            )
            constraints = runtime / "generation/constraints.txt"
            assert admission.read_runtime(runtime, selected_identity) == (
                python,
                base["id"],
                constraints,
                "fixture-cpu",
            )
            record = json.loads((runtime / "runtime.json").read_bytes())
            assert record["commit"] == commit
            assert record["engineVersion"] == version
            assert (runtime / "engine/release-manifest.json").read_bytes() == data
            assert (runtime / "wheelhouse" / wheel.filename).read_bytes() == (
                tmp_path / wheel.filename
            ).read_bytes()
            with pytest.raises(ValueError, match="selected engine"):
                admission.read_runtime(runtime, "sha256:" + "e" * 64)
            constraints.write_text("changed constraints")
            with pytest.raises(ValueError, match="constraints changed"):
                admission.read_runtime(runtime, selected_identity)
            constraints.write_text("")
            (runtime / "runtime.json").write_text(
                json.dumps({**record, "packages": {"unexpected": "1.0"}})
            )
            with pytest.raises(ValueError, match="installed packages changed"):
                admission.read_runtime(runtime, selected_identity)
    finally:
        server.shutdown()
        server.server_close()

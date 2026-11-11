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


@pytest.mark.parametrize("dependent", [False, True])
def test_probe_index_stages_selected_library_dependency_without_public_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dependent: bool
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
    output = tmp_path / "index"
    result = stage_index(sources, output)
    if dependent:
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
        stage_index(sources, tmp_path / "bad-index")


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
    [None, "engine", "base", "base-bytes", "dependency", "protocol", "artifact", "source"],
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
            'requires-python = ">=3.12"\ndependencies = ["fixture-dependency==1.0"]\n'
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
                    {**base_row, "sha256": "f" * 64} if problem == "dependency" else base_row
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

    def extract(parts, tree, **kwargs):
        # Resolution uses the test interpreter; published-base extraction is a supervisor contract.
        (tree / "wheels").mkdir(parents=True)
        shutil.copyfile(tmp_path / base_wheel.filename, tree / "wheels" / base_wheel.filename)
        python = tree / base["interpreter"]["executable"]
        python.parent.mkdir(parents=True)
        if system == "windows":
            for path in Path(sys.base_prefix).iterdir():
                if path.suffix in {".exe", ".dll"}:
                    shutil.copy2(path, python.parent / path.name)
            shutil.copytree(Path(sys.base_prefix) / "Lib", python.parent / "Lib")
            shutil.copytree(Path(sys.base_prefix) / "DLLs", python.parent / "DLLs")
        else:
            python.symlink_to(sys.executable)

    monkeypatch.setattr(native, "extract_base", extract)
    monkeypatch.setattr(native, "fetch_artifact", lambda mirrors, name, path, *args: path)
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
        }
        with pytest.raises(ValueError, match=messages[problem]):
            native.build_locks(
                tmp_path, release, manifest, output, sources=sources, engine_identity=identity
            )
        assert not output.exists()
    else:
        paths = native.build_locks(
            tmp_path, release, manifest, output, sources=sources, engine_identity=identity
        )
        assert [path.name for path in paths] == ["test-helpers-0.0.1-fixture-cpu.pack-lock.json"]
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
        assert {path.name: path.read_bytes() for path in sources.iterdir()} == originals

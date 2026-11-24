from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.mark.parametrize("version", ["0.0.7.dev1053", "0.0.7", "0.0.7.dev4+g" + "a" * 40])
def test_published_engine_uses_indexed_provenance_not_a_local_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    from scripts.prepare_official_publication import published_engine

    payload = b"verified engine constraints\n"
    manifest = {
        "format": "dinkster.engine/1",
        "repository": "Kosinkadink/Dinkster",
        "version": version,
        "label": f"2026-11-15-{version.split('+')[0]}",
        "commit": "a" * 40,
        "workerProtocol": 12,
        "artifacts": [
            {
                "name": "constraints.txt",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    data = json.dumps(manifest).encode()
    sha = hashlib.sha256(data).hexdigest()
    name = f"{manifest['label']}-{sha[:8]}"
    tag = "main-" + name if ".dev" in version else "v" + version
    row = {**manifest, "sha256": sha, "path": "engine/" + name}

    def metadata(mirror, path):
        return json.dumps([row]).encode() if path == "engine/index.json" else data

    def artifact(mirrors, filename, destination, digest, size):
        assert hashlib.sha256(payload).hexdigest() == digest
        assert len(payload) == size
        destination.write_bytes(payload)

    monkeypatch.setattr("scripts.prepare_official_publication.read_metadata", metadata)
    monkeypatch.setattr("scripts.prepare_official_publication.fetch_artifact", artifact)
    assert published_engine(tmp_path, tag) == manifest
    assert (tmp_path / "engine/constraints.txt").read_bytes() == payload
    assert (tmp_path / "engine/release-manifest.json").read_bytes() == data
    row["commit"] = "b" * 40
    with pytest.raises(ValueError, match="published engine identity mismatch"):
        published_engine(tmp_path, tag)


def _engine_and_base(tmp_path: Path) -> tuple[Path, Path]:
    from tests.registry_fixtures import _wheel

    base = {
        "format": "dinkster.base/1",
        "variant": "linux-cpu",
        "platform": {"os": "linux", "arch": "x86_64"},
        "python": "3.12.9",
        "interpreter": {},
        "wheels": [],
        "replaces": [],
        "workerProtocol": 12,
        "nvidia": None,
        "archive": {"format": "tar+zstd"},
    }
    identity = hashlib.sha256(
        json.dumps(base, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    base["id"] = "sha256:" + identity
    base["archive"]["parts"] = []
    bases = tmp_path / "bases/linux-cpu"
    bases.mkdir(parents=True)
    (bases / "manifest.json").write_text(json.dumps(base))
    engine = tmp_path / "engine"
    engine.mkdir()
    api = _wheel(engine, "dinkster-api", "0.0.4", "dinkster_api", "")
    (engine / "release-manifest.json").write_text(
        json.dumps(
            {
                "format": "dinkster.engine/1",
                "repository": "Kosinkadink/Dinkster",
                "version": "0.0.4.dev123",
                "commit": "c" * 40,
                "workerProtocol": 12,
                "artifacts": [
                    {"name": api.filename, "size": api.size, "sha256": api.sha256},
                    {
                        "name": "constraints.txt",
                        "size": 0,
                        "sha256": hashlib.sha256(b"").hexdigest(),
                    },
                ],
                "dependencies": {"linux-cpu": {"base": base["id"], "wheels": []}},
            }
        )
    )
    return engine, bases.parent


def test_incompatible_publication_control_reaches_publisher_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.check_publication_controls import CONSUMER, HELPER, prepare_fixture
    from scripts.publish_official_packs import publish_official_packs

    monkeypatch.setenv("UV_NO_INDEX", "1")
    engine, bases = _engine_and_base(tmp_path)
    output = tmp_path / "broken"
    prepare_fixture(bases, output, incompatible=True, engine=engine)
    with pytest.raises(ValueError, match=f"{CONSUMER}: selected {HELPER} excludes >=2"):
        publish_official_packs(
            output / "engine",
            output / "sources",
            list((output / "bases").glob("*/manifest.json")),
            list((output / "locks").glob("*.pack-lock.json")),
            output / "publication",
        )


def test_preflight_accepts_versions_published_by_the_bound_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blake3 import blake3
    from dinkster_registry import artifact_digest

    from dinkster.registries import RegistryPublishError
    from scripts.check_publication_controls import prepare_fixture
    from scripts.publish_official_packs import publish_official_packs
    from tests.registry_fixtures import _serve_artifacts

    monkeypatch.setenv("UV_NO_INDEX", "1")
    engine, bases = _engine_and_base(tmp_path)
    fixture = tmp_path / "fixture"
    prepare_fixture(bases, fixture)
    inputs = (
        engine,
        fixture / "sources",
        list((fixture / "bases").glob("*/manifest.json")),
        list((fixture / "locks").glob("*.pack-lock.json")),
    )
    publish_official_packs(*inputs, tmp_path / "prepared")
    records: dict[str, dict] = {}
    files: dict[str, bytes | Callable[[], bytes]] = {}
    for prepared in (tmp_path / "prepared").glob("dinkster-*.json"):
        pack, version = prepared.stem.rsplit("-", 1)
        path = f"/v1/packs/{pack}/versions/{version}"
        archive = fixture / "sources" / f"{pack.replace('-', '_')}-{version}-py3-none-any.whl"
        locks = {}
        for variant, binding in json.loads(prepared.read_bytes()).items():
            payload = json.dumps(binding["lock"], sort_keys=True, separators=(",", ":")).encode()
            files[f"{path}/locks/{variant}?baseId={binding['baseId'].replace(':', '%3A')}"] = (
                payload
            )
            locks[variant] = {
                "baseId": binding["baseId"],
                "artifactDigest": binding["artifactDigest"],
                "digest": "blake3:" + blake3(payload).hexdigest(),
                "size": len(payload),
                "url": f"{path}/locks/{variant}",
            }
        records[path] = {
            "state": "active",
            "packName": pack,
            "version": version,
            "artifact": {
                "digest": artifact_digest(archive.read_bytes()),
                "size": archive.stat().st_size,
            },
            "installLocks": locks,
        }
        files[path] = lambda path=path: json.dumps(records[path]).encode()
    server = _serve_artifacts(files, [])
    endpoint = f"http://127.0.0.1:{server.server_port}"

    def preflight(name: str, principal: str) -> None:
        for record in records.values():
            record["publishedBy"] = {"kind": "interactive", "principalId": principal}
        publish_official_packs(*inputs, tmp_path / name, preflight_registry=endpoint)

    try:
        preflight("owner", "u_01a192bb-a823-7cc0-9a96-4d49ecfd7d5a")
        preflight("operator", "dinkster")
        with pytest.raises(RegistryPublishError, match="published by 'u_other', not by publisher"):
            preflight("other", "u_other")
        next(iter(records.values()))["artifact"]["digest"] = "blake3:" + "0" * 64
        with pytest.raises(RegistryPublishError, match="bump the pack version"):
            preflight("changed", "u_01a192bb-a823-7cc0-9a96-4d49ecfd7d5a")
    finally:
        server.shutdown()
        server.server_close()


def test_scanner_runtime_stages_unpublished_locked_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from dinkster_workers.pack_lock import LockedDistribution, PackLock
    from dinkster_workers.wheel_download import fetch_wheel

    from scripts.stage_publication_probe_index import stage_runtime
    from tests.registry_fixtures import _serve_artifacts, _wheel

    sources = tmp_path / "sources"
    sources.mkdir()
    wheel = _wheel(sources, "dinkster-pack-helpers", "0.0.7", "helper", "VALUE = 37\n")
    external = _wheel(tmp_path, "external", "1.0", "external", "VALUE = 91\n")
    requests: list[tuple[str, str | None]] = []
    server = _serve_artifacts({}, requests)
    wheel = replace(wheel, url=f"http://127.0.0.1:{server.server_port}/{wheel.filename}")
    engine = tmp_path / "engine"
    engine.mkdir()
    (engine / "release-manifest.json").write_text(
        json.dumps(
            {
                "format": "dinkster.engine/1",
                "version": "0.0.7",
                "commit": "a" * 40,
                "workerProtocol": 12,
                "artifacts": [
                    {
                        "name": "constraints.txt",
                        "size": 0,
                        "sha256": hashlib.sha256(b"").hexdigest(),
                    }
                ],
            }
        )
    )
    base = {
        "format": "dinkster.base/1",
        "variant": "linux-cpu",
        "platform": {"os": "linux", "arch": "x86_64"},
        "python": "3.12.9",
        "interpreter": {},
        "wheels": [],
        "replaces": [],
        "workerProtocol": 12,
        "nvidia": None,
        "archive": {"format": "tar+zstd"},
    }
    base["id"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(base, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
    )
    base["archive"]["parts"] = []
    bases = tmp_path / "bases/linux-cpu"
    bases.mkdir(parents=True)
    (bases / "manifest.json").write_text(json.dumps(base))
    locks = tmp_path / "locks"
    locks.mkdir()
    PackLock(
        "consumer",
        "linux-cpu",
        ">=3.12",
        ("dinkster-pack-helpers>=0.0.2,<1",),
        (),
        (
            LockedDistribution("dinkster-pack-helpers", "0.0.7", (wheel,)),
            LockedDistribution("external", "1.0", (external,)),
        ),
    ).write(locks / "consumer-linux-cpu.pack-lock.json")
    monkeypatch.setattr("dinkster_supervisor.environment_artifacts.acquire", lambda *args: None)
    try:
        output = tmp_path / "runtime"
        stage_runtime(engine, bases.parent, locks, output, sources=sources)
        cached = fetch_wheel(wheel, output / "store/wheels", offline=True)
        assert cached.read_bytes() == (sources / wheel.filename).read_bytes()
        assert (
            fetch_wheel(external, output / "store/wheels", offline=True).read_bytes()
            == (tmp_path / external.filename).read_bytes()
        )
        assert requests == []
        candidate = sources / wheel.filename
        data = candidate.read_bytes()
        candidate.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        with pytest.raises(ValueError, match="candidate wheel hash or size mismatch"):
            stage_runtime(engine, bases.parent, locks, tmp_path / "corrupt", sources=sources)
    finally:
        server.shutdown()
        server.server_close()


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

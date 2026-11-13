from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def test_incompatible_publication_control_reaches_publisher_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.check_publication_controls import CONSUMER, HELPER, prepare_fixture
    from scripts.publish_official_packs import publish_official_packs
    from tests.registry_fixtures import _wheel

    monkeypatch.setenv("UV_NO_INDEX", "1")
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
    output = tmp_path / "broken"
    prepare_fixture(bases.parent, output, incompatible=True, engine=engine)
    with pytest.raises(ValueError, match=f"{CONSUMER}: selected {HELPER} excludes >=2"):
        publish_official_packs(
            output / "engine",
            output / "sources",
            list((output / "bases").glob("*/manifest.json")),
            list((output / "locks").glob("*.pack-lock.json")),
            output / "publication",
        )


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

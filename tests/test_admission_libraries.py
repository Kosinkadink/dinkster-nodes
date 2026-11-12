from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "dependency", ["consumers", "transitive", "extras", "missing-extra", "fixed"]
)
def test_admission_library_closure_intersects_ranges_and_activates_extras(
    tmp_path: Path, dependency: str, reverse: bool
) -> None:
    from dataclasses import replace

    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from dinkster_store import pack_archive_path
    from dinkster_workers.pack_wheel import build_pack_archive

    from scripts.check_default_pack_admission import NamedRegistry, prepare_library_sources
    from tests.registry_fixtures import _serve_artifacts, _wheel

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    rows = []
    files = {}
    library_entries = {}
    for name, versions in (("library-a", ("1.5", "2.0")), ("library-z", ("1.0",))):
        for version in versions:
            requires = ()
            if name == "library-a" and dependency in {"extras", "missing-extra"}:
                requires = ('library-z>=1; extra == "helpers"',)
            if name == "library-z" and dependency == "transitive":
                requires = ("library-a<2",)
            wheel = _wheel(wheels, name, version, name.replace("-", "_"), "", requires=requires)
            data = (wheels / wheel.filename).read_bytes()
            entry = LockedPack(name, version, artifact_digest(data), "dinkster", ())
            library_entries[name, version] = entry
            if name == "library-z" and dependency == "missing-extra":
                continue
            sha256 = hashlib.sha256(data).hexdigest()
            rows.append(
                {
                    "name": name,
                    "version": version,
                    "publisher": "dinkster",
                    "library": True,
                    "requires": [],
                    "engine": ">=0.0.2.dev0,<1",
                    "installLocks": {},
                    "wheels": [
                        {
                            "filename": wheel.filename,
                            "sha256": sha256,
                            "url": f"wheels/{sha256}/{wheel.filename}",
                            "size": len(data),
                            "digest": entry.artifact_digest,
                            "tags": ["py3-none-any"],
                        }
                    ],
                }
            )
            files[f"/static/wheels/{wheel.sha256}/{wheel.filename}"] = data
    files["/static/packs/index.json"] = json.dumps(
        {
            "schemaVersion": 2,
            "items": [
                {
                    "name": name,
                    "latestVersion": version,
                    "publisher": "dinkster",
                    "library": True,
                    "url": f"packs/{name}/index.json",
                }
                for name, version in (("library-a", "2.0"), ("library-z", "1.0"))
            ],
        }
    ).encode()
    for name in ("library-a", "library-z"):
        files[f"/static/packs/{name}/index.json"] = json.dumps(
            {
                "schemaVersion": 2,
                "name": name,
                "items": [row for row in rows if row["name"] == name],
            }
        ).encode()
    requirements = ["library-a>=1", "library-a<2"]
    if dependency in {"extras", "missing-extra"}:
        requirements = ["library-a[helpers]>=1", "library-a>=1"]
    elif dependency == "transitive":
        requirements = ["library-a>=1", "library-z>=1"]
    elif dependency == "fixed":
        requirements = ["library-a<2", "library-a>=1"]
    if reverse:
        requirements.reverse()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    entries = []
    for number, requirement in enumerate(requirements):
        name = f"consumer-{number}"
        source = tmp_path / name
        source.mkdir()
        module = name.replace("-", "_")
        (source / module).mkdir()
        (source / module / "__init__.py").write_text("NODES = []\n")
        (source / "dinkster-pack.toml").write_text(
            f'[pack]\nname = "{name}"\nversion = "1.0"\npublisher = "dinkster"\n'
            f'engine = ">=0.0.2,<0.1"\nrequires = ["{requirement}"]\n'
            f'[pack.entry]\nnodes = "{module}:NODES"\n'
            '[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n'
        )
        archive = wheels / f"{name.replace('-', '_')}-1.0-py3-none-any.whl"
        build_pack_archive(source, archive)
        entry = LockedPack(name, "1.0", artifact_digest(archive.read_bytes()), "dinkster", ())
        pack_archive_path(artifacts, entry.artifact_digest).write_bytes(archive.read_bytes())
        entries.append(entry)
    if dependency == "fixed":
        entry = replace(library_entries["library-a", "2.0"], source="git:fixed-publication")
        pack_archive_path(artifacts, entry.artifact_digest).write_bytes(
            (wheels / "library_a-2.0-py3-none-any.whl").read_bytes()
        )
        entries.append(entry)
    server = _serve_artifacts(files, [])
    try:
        registry = NamedRegistry("", f"http://127.0.0.1:{server.server_port}")
        if dependency in {"missing-extra", "fixed"}:
            with pytest.raises(ValueError, match=r"library-(z.*>=1|a.*<2)"):
                prepare_library_sources(registry, Lockfile.of(entries), artifacts, tmp_path)
        else:
            selected = prepare_library_sources(registry, Lockfile.of(entries), artifacts, tmp_path)
            versions = {pack.pack: pack.version for pack in selected.packs}
            assert versions["library-a"] == ("2.0" if dependency == "extras" else "1.5")
            assert versions.get("library-z") == (
                "1.0" if dependency in {"extras", "transitive"} else None
            )
    finally:
        server.shutdown()
        server.server_close()

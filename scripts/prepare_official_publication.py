"""Prepare pack sources against the workspace's pinned published engine and bases."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import tomllib
import zipfile
from pathlib import Path
from urllib.parse import quote, unquote

from dinkster_registry import Lockfile, artifact_digest, unpack_artifact
from dinkster_supervisor.environment_artifacts import validate_dependency_base, validate_manifest
from dinkster_supervisor.environment_channels import engine_publication_names
from dinkster_supervisor.environment_downloads import fetch_artifact, read_metadata
from dinkster_supervisor.environment_state import content_digest, digest, relative_file
from dinkster_workers import load_manifest
from dinkster_workers.pack_wheel import pack_manifest_path

from scripts.build_official_packs import build, library_metadata


def prepare_inputs(root: Path, output: Path, packs: list[str]) -> list[dict[str, str]]:
    """Bind engine selection to the existing development pin, never the pack Git head."""
    project = tomllib.loads((root / "pyproject.toml").read_text())
    pin = unquote(project["tool"]["uv"]["sources"]["dinkster"]["url"])
    match = re.fullmatch(
        r"https://github.com/Kosinkadink/Dinkster/releases/download/([^/]+)/"
        r"dinkster-[^/]+\+g([0-9a-f]{40})-py3-none-any.whl",
        pin,
    )
    if match is None:
        raise ValueError("publication requires a commit-bound published engine pin")
    tag, commit = match.groups()
    if output.exists():
        raise ValueError("publication output must be a new directory")
    with tempfile.TemporaryDirectory(prefix="pack-publication-", dir=output.parent) as temporary:
        work = Path(temporary) / "prepared"
        work.mkdir()
        engine = published_engine(work, commit, tag)
        if pin.rsplit("/", 1)[1] not in {row["name"] for row in engine["artifacts"]}:
            raise ValueError("pinned engine wheel is absent from the published manifest")
        identity = content_digest(work / "engine/release-manifest.json")
        selections = engine.get("dependencies")
        if not isinstance(selections, dict) or not selections:
            raise ValueError("published engine must select dependency bases")
        rows = []
        for variant, selection in sorted(selections.items()):
            if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", variant) is None:
                raise ValueError("engine dependency variant must be canonical")
            base_id = digest(selection["base"])
            index = json.loads(
                read_metadata("https://releases.dinkster.org", f"base/{variant}/index.json")
            )
            matches = [row for row in index if row.get("id") == base_id]
            labelled = [row for row in matches if row.get("label") is not None]
            if len(labelled or matches) != 1:
                raise ValueError("base index must identify one publication for the engine base")
            selected = (labelled or matches)[0]
            name = relative_file(selected.get("label", base_id[7:]))
            if selected.get("path") != f"base/{variant}/{name}":
                raise ValueError("base publication path differs from its identity")
            data = read_metadata(
                "https://releases.dinkster.org", f"base/{variant}/{name}/manifest.json"
            )
            base: dict = validate_manifest(data, base_id, variant)
            validate_dependency_base(selection, base)
            if base["workerProtocol"] != engine["workerProtocol"]:
                raise ValueError("engine and selected base worker protocols differ")
            if base.get("label", base_id[7:]) != name:
                raise ValueError("base publication label differs from its manifest")
            destination = work / "bases" / variant
            destination.mkdir(parents=True)
            (destination / "manifest.json").write_bytes(data)
            runner = {
                "linux": "ubuntu-latest",
                "windows": "windows-latest",
                "macos": "macos-latest",
            }[base["platform"]["os"]]
            rows.append({"variant": variant, "runner": runner, "engineIdentity": identity})
        sources = work / "sources"
        build(root, sources, packs=packs)
        catalog = Lockfile.from_record_json((sources / "official-packs.lock").read_text())
        for pack in catalog.packs:
            archive = sources / (
                f"{pack.pack.replace('-', '_').replace('.', '_')}-{pack.version}-py3-none-any.whl"
            )
            if artifact_digest(archive.read_bytes()) != pack.artifact_digest:
                raise ValueError(f"publication source digest mismatch: {pack.pack}")
            with zipfile.ZipFile(archive) as wheel:
                library = not any(
                    Path(name).name == "dinkster-pack.toml" for name in wheel.namelist()
                )
            if library:
                name, version, _ = library_metadata(archive)
                if (name, version) != (pack.pack, pack.version) or pack.claims:
                    raise ValueError(f"publication library identity mismatch: {pack.pack}")
                continue
            tree = unpack_artifact(archive, work / "admission" / pack.pack, pack.artifact_digest)
            manifest = load_manifest(pack_manifest_path(tree))
            if manifest.compatibility is None:
                raise ValueError(f"pack {pack.pack} lacks engine compatibility")
            manifest.compatibility.require_engine(pack.pack, engine["version"])
        work.rename(output)
    return rows


def published_engine(work: Path, commit: str, tag: str) -> dict:
    """Verify indexed manifest bytes and artifacts against the pinned release identity."""
    index = json.loads(read_metadata("https://releases.dinkster.org", "engine/index.json"))
    if not isinstance(index, list) or any(not isinstance(row, dict) for row in index):
        raise ValueError("engine index must contain release objects")
    matches = [row for row in index if row.get("commit") == commit]
    if len(matches) != 1:
        raise ValueError("engine index must identify one published release for the pinned commit")
    selected = matches[0]
    identity = digest(f"sha256:{selected.get('sha256')}")
    name, expected_tag = identity[7:], f"main-{commit}"
    if selected.get("label") is not None:
        name, expected_tag = engine_publication_names(selected, identity, "main")
    if expected_tag != tag or selected.get("path") != f"engine/{name}":
        raise ValueError("engine publication differs from the workspace pin")
    data = read_metadata("https://releases.dinkster.org", f"engine/{name}/release-manifest.json")
    engine: dict = validate_manifest(data, identity, None)
    if (
        engine.get("repository") != "Kosinkadink/Dinkster"
        or engine["commit"] != commit
        or engine["version"] != selected.get("version")
        or engine_publication_names(engine, identity, "main") != (name, tag)
    ):
        raise ValueError("published engine identity mismatch")
    release = work / "engine"
    release.mkdir()
    (release / "release-manifest.json").write_bytes(data)
    mirror = "https://github.com/Kosinkadink/Dinkster/releases/download/" + quote(tag, safe="")
    for row in engine["artifacts"]:
        fetch_artifact((mirror,), row["name"], release / row["name"], row["sha256"], row["size"])
    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pack", action="append", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_inputs(Path(__file__).resolve().parents[1], args.output.resolve(), args.pack)
        )
    )

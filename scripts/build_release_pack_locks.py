"""Resolve official pack dependencies on the selected base's native host."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote, urlsplit

from dinkster_registry import Lockfile
from dinkster_supervisor.environment_archives import extract_base
from dinkster_supervisor.environment_artifacts import (
    engine_constraints,
    validate_dependency_base,
    validate_manifest,
)
from dinkster_supervisor.environment_channels import engine_publication_names
from dinkster_supervisor.environment_constraints import effective_constraints
from dinkster_supervisor.environment_downloads import fetch_artifact
from dinkster_supervisor.environment_state import content_digest
from dinkster_workers.pack_lock import PackLock

from scripts.build_official_packs import build


def build_locks(
    root: Path,
    release: Path,
    manifest: Path,
    output: Path,
    *,
    sources: Path,
    engine_identity: str,
) -> tuple[Path, ...]:
    """Keep source payloads unchanged while adding native per-variant lock bytes."""
    base_data = manifest.read_bytes()
    raw = json.loads(base_data)
    variant = raw["variant"]
    base_tag = f"base-{variant}-{raw.get('label', raw['id'][7:])}"
    base: dict = validate_manifest(base_data, raw["id"], variant)
    engine: dict = validate_manifest(
        (release / "release-manifest.json").read_bytes(), engine_identity, None
    )
    if engine["repository"] != "Kosinkadink/Dinkster":
        raise ValueError("pack locks require a published Dinkster engine")
    validate_dependency_base(engine.get("dependencies", {}).get(variant), base)
    system = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}[platform.system()]
    if base["platform"]["os"] != system:
        raise ValueError("pack locks must be resolved on the selected base's native host")
    if output.exists():
        raise ValueError("pack lock output must be a new directory")
    catalog = Lockfile.from_record_json((sources / "official-packs.lock").read_text())
    if not catalog.packs:
        raise ValueError("independent pack publication requires selected sources")
    _, release_tag = engine_publication_names(engine, engine_identity, "main")
    for row in engine["artifacts"]:
        path = release / row["name"]
        if path.stat().st_size != row["size"] or content_digest(path) != "sha256:" + row["sha256"]:
            raise ValueError(f"release artifact differs from manifest: {path.name}")
    if engine["workerProtocol"] != base["workerProtocol"]:
        raise ValueError("engine and selected base worker protocols differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="release-pack-locks-", dir=output.parent) as directory:
        work = Path(directory)
        archive = base["archive"]
        mirror = (
            "https://github.com/Kosinkadink/Dinkster-Environments/releases/download/" + base_tag
        )
        parts = tuple(
            fetch_artifact(
                (mirror,),
                row["filename"],
                work / row["filename"],
                row["sha256"],
                row["size"],
            )
            for row in archive["parts"]
        )
        tree = work / "base"
        extract_base(
            parts,
            tree,
            platform=system,
            executable=base["interpreter"]["executable"],
            expected_parts=tuple(archive["parts"]),
        )
        constraints = work / "constraints.txt"
        constraints.write_text(
            effective_constraints(engine_constraints(engine, variant), base, strict=True),
            encoding="ascii",
        )
        wheel_urls = {
            (row["filename"], row["size"], row["sha256"]): row["url"] for row in base["wheels"]
        }
        for row in engine["artifacts"]:
            if not row["name"].endswith(".whl"):
                continue
            key = (row["name"], row["size"], row["sha256"])
            wheel_urls[key] = (
                "https://github.com/Kosinkadink/Dinkster/releases/download/"
                f"{quote(release_tag, safe='')}/{quote(row['name'], safe='')}"
            )
        built = build(
            root,
            work / "packs",
            packs=tuple(pack.pack for pack in catalog.packs),
            variant=variant,
            python=tree / base["interpreter"]["executable"],
            constraints=constraints,
            # Native base wheels come from find-links; backend indexes can shadow PyPI pins.
            indexes=("https://pypi.org/simple",),
            find_links=(release, tree / "wheels"),
            accelerator="cuda" if base["nvidia"] is not None else "cpu",
        )
        locks = [path for path in built if path.name.endswith(f"-{variant}.pack-lock.json")]
        staged = work / "output"
        staged.mkdir()
        for path in built:
            if path in locks:
                lock = PackLock.read(path)
                packages = []
                for package in lock.packages:
                    wheels = []
                    for wheel in package.wheels:
                        if urlsplit(wheel.url).scheme == "file":
                            key = (wheel.filename, wheel.size, wheel.sha256)
                            if key not in wheel_urls:
                                raise ValueError(
                                    "local wheel has no verified portable URL provenance: "
                                    f"{wheel.filename}"
                                )
                            wheel = replace(wheel, url=wheel_urls[key])
                        wheels.append(wheel)
                    packages.append(replace(package, wheels=tuple(wheels)))
                replace(
                    lock,
                    packages=tuple(packages),
                ).write(staged / path.name)
            else:
                original = sources / path.name
                if not original.is_file() or path.read_bytes() != original.read_bytes():
                    raise ValueError(f"native pack build changed source artifact: {path.name}")
        if not locks:
            raise ValueError("native pack build produced no dependency locks")
        staged.rename(output)
    return tuple(sorted(output.iterdir()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--engine-identity", required=True)
    args = parser.parse_args()
    for path in build_locks(
        Path(__file__).resolve().parents[1],
        args.release.resolve(),
        args.base_manifest.resolve(),
        args.output.resolve(),
        sources=args.sources.resolve(),
        engine_identity=args.engine_identity,
    ):
        print(path)

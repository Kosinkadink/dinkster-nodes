"""Acquire verified Linux scanner dependencies before the isolated publication job."""

from __future__ import annotations

import argparse
import email.parser
import json
import shutil
import subprocess
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import Request, url2pathname, urlopen
from zipfile import ZipFile

from dinkster_registry import Lockfile, artifact_digest, unpack_artifact
from dinkster_supervisor.environment_downloads import fetch_artifact
from dinkster_supervisor.environment_state import content_digest
from dinkster_workers import load_manifest
from dinkster_workers.pack_wheel import pack_manifest_path

from scripts.build_official_packs import library_metadata


def stage_index(sources: Path, output: Path, *, engine: Path | None = None) -> dict:
    """Resolve CPU probe requirements independently of the native install locks."""
    output.mkdir(parents=True, exist_ok=True)
    catalog = Lockfile.from_record_json((sources / "official-packs.lock").read_text())
    requirements: set[str] = set()
    if engine is not None:
        apis = list(engine.glob("dinkster_api-*.whl"))
        if len(apis) != 1:
            raise ValueError("probe index requires exactly one selected API wheel")
        with ZipFile(apis[0]) as wheel:
            metadata = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError("selected API wheel requires one distribution")
            fields = email.parser.BytesParser().parsebytes(wheel.read(metadata[0]))
            requirements.update(fields.get_all("Requires-Dist", []))
    with tempfile.TemporaryDirectory(prefix="publication-probe-") as directory:
        work = Path(directory)
        for pack in catalog.packs:
            archive = sources / f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl"
            if artifact_digest(archive.read_bytes()) != pack.artifact_digest:
                raise ValueError(f"probe source digest mismatch: {pack.pack}")
            with ZipFile(archive) as wheel:
                is_pack = any(Path(name).name == "dinkster-pack.toml" for name in wheel.namelist())
            if is_pack:
                extracted = unpack_artifact(archive, work / pack.pack, pack.artifact_digest)
                requirements.update(
                    load_manifest(pack_manifest_path(extracted)).requires_for("cpu")
                )
            else:
                requirements.update(library_metadata(archive)[2])
        inputs = output / "requirements.in"
        inputs.write_text("\n".join(sorted(requirements)) + "\n", encoding="ascii")
        lock = output / "pylock.toml"
        subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                "--python-version",
                "3.12",
                "--python-platform",
                "x86_64-manylinux_2_28",
                "--only-binary",
                ":all:",
                "--torch-backend",
                "cpu",
                "--find-links",
                str(sources.resolve()),
                "--format",
                "pylock.toml",
                "--output-file",
                str(lock),
                str(inputs),
            ],
            check=True,
            timeout=180,
        )
    resolution = tomllib.loads(lock.read_text())

    def acquire(item: tuple[str, dict]) -> dict:
        package, wheel = item
        parts = urlsplit(wheel["url"])
        name = unquote(parts.path.rsplit("/", 1)[-1])
        destination = output / name
        if parts.scheme == "file":
            source = Path(url2pathname(parts.path)).resolve()
            if source.parent != sources.resolve():
                raise ValueError("local probe wheel must be a selected source")
            digest, size = content_digest(source).removeprefix("sha256:"), source.stat().st_size
            shutil.copyfile(source, destination)
        else:
            if parts.scheme != "https":
                raise ValueError("remote probe wheels require HTTPS")
            digest = wheel["hashes"]["sha256"]
            if "size" in wheel:
                fetch_artifact(
                    (wheel["url"].rsplit("/", 1)[0],), name, destination, digest, wheel["size"]
                )
            elif not destination.exists() or content_digest(destination) != "sha256:" + digest:
                # Some CPU wheel indexes provide hashes without file sizes.
                request = Request(wheel["url"], headers={"User-Agent": "dinkster-env"})
                with urlopen(request, timeout=120) as response, destination.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            if content_digest(destination) != "sha256:" + digest:
                raise ValueError(f"probe dependency digest mismatch: {name}")
            size = destination.stat().st_size
        return {"package": package, "filename": name, "sha256": digest, "size": size}

    items = [(p["name"], w) for p in resolution.get("packages", []) for w in p["wheels"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(acquire, items))
    pages: dict[str, list[str]] = {}
    for row in rows:
        filename = escape(row["filename"], quote=True)
        pages.setdefault(row["package"], []).append(
            f'<a href="../../{filename}#sha256={row["sha256"]}">{filename}</a>\n'
        )
    for package, links in pages.items():
        directory = output / "simple" / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.html").write_text("".join(links), encoding="ascii")
    receipt = {"distributions": len(pages), "wheels": rows}
    (output / "inventory.json").write_text(json.dumps(receipt, indent=2), encoding="ascii")
    return receipt


def stage_runtime(engine: Path, bases: Path, locks: Path, output: Path) -> None:
    """Acquire the submitted engine and CPU base before scanner network isolation."""
    from dinkster_supervisor.environment_artifacts import (
        EnvironmentCatalog,
        acquire,
        validate_manifest,
    )
    from dinkster_supervisor.environment_state import Channel, writer_lock

    from scripts.base_manifest import read_base_manifest

    manifest = engine / "release-manifest.json"
    identity = content_digest(manifest)
    data = manifest.read_bytes()
    release = validate_manifest(data, identity, None)
    mirror = output / "mirror"
    engine_path = f"engine/{identity[7:]}"
    shutil.copytree(engine, mirror / engine_path)
    (mirror / "engine/index.json").write_text(
        json.dumps([{**release, "sha256": identity[7:], "path": engine_path}]), encoding="ascii"
    )
    selected = {}
    locations = {}
    for path in sorted(bases.glob("*/manifest.json")):
        base = read_base_manifest(path)
        variant = base["variant"]
        base_path = f"base/{variant}/{base['id'][7:]}"
        destination = mirror / base_path
        destination.mkdir(parents=True)
        shutil.copyfile(path, destination / "manifest.json")
        (destination.parent / "index.json").write_text(
            json.dumps([{"id": base["id"], "path": base_path}]), encoding="ascii"
        )
        selected[variant] = base
        locations[variant] = {
            "id": base["id"],
            "locations": [{"type": "mirror", "path": f"{base_path}/manifest.json"}],
        }
    document = {
        "format": "dinkster.channel/1",
        "minimumDesktopVersion": "0.0.0",
        "engine": {
            "version": release["version"],
            "release": identity,
            "locations": [{"type": "mirror", "path": f"{engine_path}/release-manifest.json"}],
        },
        "base": locations,
    }
    variant = "linux-cpu"
    base = selected[variant]
    catalog = EnvironmentCatalog(
        "main",
        document,
        Channel.parse(document),
        release,
        (str(engine),),
        selected,
        {
            variant: (
                f"https://releases.dinkster.org/base/{variant}/{base.get('label', base['id'][7:])}",
            )
        },
        data,
    )
    store = output / "store"
    store.mkdir(exist_ok=True)
    with writer_lock(store):
        acquire(catalog, variant, store)
        from dinkster_workers.pack_lock import PackLock
        from dinkster_workers.wheel_download import fetch_wheel

        for path in sorted(locks.glob("*-linux-cpu.pack-lock.json")):
            for package in PackLock.read(path).packages:
                for wheel in package.wheels:
                    fetch_wheel(wheel, store / "wheels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            stage_index(
                args.sources.resolve(),
                args.output.resolve(),
                engine=args.engine.resolve() if args.engine else None,
            )
        )
    )

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from dinkster_workers import load_manifest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version


@pytest.mark.parametrize(
    "path",
    sorted(Path("packages").glob("dinkster-nodes-*/dinkster-pack.toml"))
    + sorted(Path("packages/dinkster-nodes-vision").glob("*_pack/dinkster-pack.toml")),
)
def test_official_pack_declares_its_version_and_engine_range(path: Path) -> None:
    from dinkster import __version__

    manifest = load_manifest(path)
    assert manifest.compatibility is not None, manifest.name
    manifest.compatibility.require_engine(manifest.name, __version__)


@pytest.mark.parametrize("path", sorted(Path("packages").glob("*/*.toml")))
def test_official_runtime_requirements_do_not_pin_engine_builds(path: Path) -> None:
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    pack = document.get("pack", {})
    project = document.get("project", {})
    library = document.get("tool", {}).get("dinkster", {})
    if not pack and not (path.parent / "dinkster-pack.toml").exists():
        if library.get("official-library") is not True:
            return
    requirements = [*pack.get("requires", []), *project.get("dependencies", [])]
    ranges = [value for value in (pack.get("engine"), library.get("engine")) if value]
    for value in requirements:
        requirement = Requirement(value)
        if canonicalize_name(requirement.name) in {
            "dinkster",
            "dinkster-api",
            "dinkster-engine",
            "dinkster-inference",
        }:
            assert requirement.url is None, f"{path}: source-pinned runtime: {value}"
            ranges.append(str(requirement.specifier))
    for value in ranges:
        for specifier in SpecifierSet(value):
            assert specifier.operator not in {"==", "==="}, f"{path}: exact runtime pin: {value}"
            assert not Version(specifier.version.rstrip(".*")).is_devrelease, (
                f"{path}: development runtime bound: {value}"
            )

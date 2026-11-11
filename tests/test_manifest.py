from __future__ import annotations

from pathlib import Path

import pytest
from dinkster_workers import load_manifest


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

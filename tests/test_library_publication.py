from __future__ import annotations

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

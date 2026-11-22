"""Pack publication requires a public, immutable inference minimum with provenance."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from tools import upstream_node_prerequisites as prerequisites


def test_stable_tag_provenance_is_accepted_without_relaxing_its_schema() -> None:
    identity = {
        "version": "0.1.3",
        "sourceCommit": "a" * 40,
        "upstreamCommit": "b" * 40,
        "upstreamTag": "v0.45.1",
        "upstreamTagCommit": "b" * 40,
        "upstreamMergeBase": "c" * 40,
        "mainSourceCommit": "d" * 40,
        "releaseSource": "upstream-tag-branch",
        "adaptedTestSources": {"tests/example.py": "tests/example.py"},
        "excludedAdaptedTests": {},
    }

    assert prerequisites.tag_identity(json.dumps(identity), "0.1.3") == identity

    with pytest.raises(ValueError, match="invalid provenance"):
        prerequisites.tag_identity(json.dumps({**identity, "unexpected": True}), "0.1.3")
    with pytest.raises(ValueError, match="invalid provenance"):
        prerequisites.tag_identity(json.dumps({**identity, "version": "0.1.2"}), "0.1.3")


@pytest.mark.parametrize("fault", [None, "yanked", "digest", "tag", "source", "minimum"])
def test_pack_prerequisite_is_verified_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    identity = {
        "version": "0.1.3.dev7",
        "sourceCommit": "a" * 40,
        "upstreamCommit": "b" * 40,
    }
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(
            "dinkster_inference-0.1.3.dev7.dist-info/METADATA",
            "Name: dinkster-inference\nVersion: 0.1.3.dev7\n",
        )
        archive.writestr("dinkster_inference/_release.json", json.dumps(identity))
    payload = data.getvalue()
    record = {
        **identity,
        "url": "https://files.pythonhosted.org/inference.whl",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "tagObject": "c" * 40,
    }
    document = {
        "urls": [
            {
                "url": record["url"],
                "size": record["size"],
                "digests": {"sha256": record["sha256"]},
                "packagetype": "bdist_wheel",
                "yanked": fault == "yanked",
            }
        ]
    }
    if fault == "digest":
        payload += b"changed"
    if fault == "source":
        record["sourceCommit"] = "d" * 40
    refs = (
        ("e" * 40 if fault == "tag" else record["tagObject"])
        + "\trefs/tags/v0.1.3.dev7\n"
        + record["sourceCommit"]
        + "\trefs/tags/v0.1.3.dev7^{}\n"
    )
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: refs)
    monkeypatch.setattr(
        prerequisites,
        "read_public",
        lambda url, *args: payload if url == record["url"] else json.dumps(document).encode(),
    )
    (tmp_path / "upstream-node-prerequisites.json").write_text(
        json.dumps(
            {
                "nodes": {
                    "ExampleNode": {"upstream": identity["upstreamCommit"], "inference": record}
                }
            }
        )
    )
    requirement = "dinkster-inference>=0.1.3.dev7,<0.2"
    if fault == "minimum":
        requirement = "dinkster-inference>=0.1.2,<0.2"
    if fault is None:
        prerequisites.verify_pack_prerequisites(tmp_path, (requirement,))
    else:
        with pytest.raises(ValueError):
            prerequisites.verify_pack_prerequisites(tmp_path, (requirement,))

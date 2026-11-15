"""Node arrival timing uses verified public contents and survives upstream advances."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import zipfile
from datetime import datetime
from email.message import Message
from pathlib import Path

import pytest
from blake3 import blake3

from tools import upstream_node_publications as publications


@pytest.mark.parametrize("missing_history", [False, True])
def test_first_node_publication_uses_time_registered_contents_and_verified_bytes(
    monkeypatch: pytest.MonkeyPatch, missing_history: bool
) -> None:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(
            "demo/nodes.py",
            "class Native:\n"
            "    def define_schema(cls):\n"
            '        schema = NodeSchema(node_type="demo.native")\n'
            '        return with_identity_replacement(schema, "UpstreamNode")\n',
        )
    payload = data.getvalue()
    sha256 = hashlib.sha256(payload).hexdigest()
    digest = "blake3:" + blake3(payload).hexdigest()
    releases = [
        {
            "version": version,
            "state": "active",
            "publishedAt": timestamp,
            "artifactDigest": digest,
            "artifactSize": len(payload),
            "nodeTypes": registered,
        }
        for version, timestamp, registered in [
            ("9.0", "2026-11-14T04:00:00Z", ["demo.native"]),
            ("1.0", "2026-11-13T03:00:00Z", ["demo.other"]),
            ("3.0", "2026-11-13T23:00:00-08:00", ["demo.native"]),
        ]
    ]
    documents = {
        "https://registry.example/v1/packs/demo/versions?limit=100&includeYanked=true": {
            "items": releases[:1],
            "nextCursor": "page2",
        },
        (
            "https://registry.example/v1/packs/demo/versions?limit=100&includeYanked=true"
            "&cursor=page2"
        ): {
            "items": releases[1:],
            "nextCursor": None,
        },
    }

    def read(url: str, limit: int = 0) -> bytes:
        if url == "https://registry.example/v1/artifacts/" + digest:
            if missing_history:
                raise urllib.error.HTTPError(url, 404, "missing", Message(), None)
            return payload
        return json.dumps(documents[url]).encode()

    monkeypatch.setattr(publications, "read_public", read)
    result, gap = publications.published_nodes("https://registry.example", "demo", {"UpstreamNode"})
    if missing_history:
        assert result == {} and gap is not None
        return
    assert result["UpstreamNode"]["version"] == "9.0"
    assert result["UpstreamNode"]["publishedAt"] == "2026-11-14T04:00:00Z"
    assert result["UpstreamNode"]["sha256"] == sha256
    payload = b"changed" + payload[7:]
    with pytest.raises(ValueError, match="digest differs"):
        publications.published_nodes("https://registry.example", "demo", {"UpstreamNode"})


@pytest.mark.parametrize("history_gap", [False, True])
def test_first_publication_follows_a_node_across_packs(
    monkeypatch: pytest.MonkeyPatch, history_gap: bool
) -> None:
    packs = {
        "items": [
            {"name": name, "owner": {"slug": "dinkster"}} for name in ("new-pack", "old-pack")
        ],
        "nextCursor": None,
    }
    monkeypatch.setattr(publications, "read_public", lambda *args: json.dumps(packs).encode())
    older = {"pack": "old-pack", "version": "9.0", "publishedAt": "2026-11-11T01:00:00Z"}
    newer = {"pack": "new-pack", "version": "1.0", "publishedAt": "2026-11-12T01:00:00Z"}
    gap = datetime.fromisoformat("2026-11-10T01:00:00Z") if history_gap else None
    monkeypatch.setattr(
        publications,
        "published_nodes",
        lambda registry, pack, wanted: (
            {"UpstreamNode": older if pack == "old-pack" else newer},
            gap,
        ),
    )
    result = publications.first_publications("https://registry.example", {"UpstreamNode"})
    assert result == ({} if history_gap else {"UpstreamNode": older})


def test_pending_node_publication_is_rechecked_without_new_upstream_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = {
        "name": "UpstreamNode",
        "pack": "demo",
        "introduction": {"commit": "a" * 40, "committedAt": "2026-11-13T01:00:00-08:00"},
        "firstPublished": None,
    }
    comments = [{"body": "<!-- node-publications-state\n" + json.dumps([pending]) + "\n-->"}]
    publication = {
        "pack": "demo",
        "version": "2.0",
        "publishedAt": "2026-11-13T10:30:00Z",
        "url": "https://registry.example/v1/packs/demo/versions/2.0",
    }
    monkeypatch.setattr(
        publications, "first_publications", lambda *args: {"UpstreamNode": publication}
    )
    result = publications.publication_records(
        Path("unused"), {"upstream": "b" * 40, "nodes": []}, comments, "https://registry.example"
    )
    assert result == [{**pending, "firstPublished": {**publication, "latencySeconds": 5400.0}}]
    rendered = publications.publication_report(result)
    assert "2026-11-13 02:30:00 PST" in rendered
    assert "1.50 hours" in rendered

"""Node arrival timing uses verified public contents and survives upstream advances."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

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
    wheel = {
        "digest": "blake3:" + "a" * 64,
        "sha256": sha256,
        "filename": "demo.whl",
        "url": f"wheels/{sha256}/demo.whl",
        "size": len(payload),
    }
    releases = [
        {
            "version": version,
            "state": "active",
            "publishedAt": timestamp,
            "artifactDigest": wheel["digest"],
            "nodeTypes": registered,
        }
        for version, timestamp, registered in [
            ("9.0", "2026-11-14T04:00:00Z", ["demo.native"]),
            ("1.0", "2026-11-13T03:00:00Z", ["demo.other"]),
            ("3.0", "2026-11-13T23:00:00-08:00", ["demo.native"]),
        ]
    ]
    static = {
        "items": [
            {"version": row["version"], "wheels": [wheel]}
            for row in releases
            if not missing_history or row["version"] != "1.0"
        ]
    }
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
        "https://registry.example/static/packs/demo/index.json": static,
    }

    def read(url: str, limit: int = 0) -> bytes:
        if url == "https://registry.example/static/" + wheel["url"]:
            return payload
        return json.dumps(documents[url]).encode()

    monkeypatch.setattr(publications, "read_public", read)
    result = publications.published_nodes("https://registry.example", "demo", {"UpstreamNode"})
    if missing_history:
        assert result == {}
        return
    assert result["UpstreamNode"]["version"] == "9.0"
    assert result["UpstreamNode"]["publishedAt"] == "2026-11-14T04:00:00Z"
    assert result["UpstreamNode"]["sha256"] == sha256
    wheel["sha256"] = "b" * 64
    wheel["url"] = "wheels/" + "b" * 64 + "/demo.whl"
    monkeypatch.setattr(
        publications,
        "read_public",
        lambda url, limit=0: payload if url.endswith("demo.whl") else read(url, limit),
    )
    with pytest.raises(ValueError, match="SHA256 differs"):
        publications.published_nodes("https://registry.example", "demo", {"UpstreamNode"})


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
        publications, "published_nodes", lambda *args: {"UpstreamNode": publication}
    )
    result = publications.publication_records(
        Path("unused"), {"upstream": "b" * 40, "nodes": []}, comments, "https://registry.example"
    )
    assert result == [{**pending, "firstPublished": {**publication, "latencySeconds": 5400.0}}]
    rendered = publications.publication_report(result)
    assert "2026-11-13 02:30:00 PST" in rendered
    assert "1.50 hours" in rendered

"""Read the canonical base identity used by official pack publication."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def read_base_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "dinkster.base/1":
        raise ValueError(f"unsupported base manifest format: {path}")
    archive = manifest.get("archive")
    if (
        "parts" in manifest
        or not isinstance(archive, dict)
        or set(archive) != {"format", "parts"}
        or archive["format"] != "tar+zstd"
        or not isinstance(archive["parts"], list)
    ):
        raise ValueError(f"unsupported base archive contract: {path}")
    for number, part in enumerate(archive["parts"], 1):
        if (
            not isinstance(part, dict)
            or set(part) != {"filename", "size", "sha256"}
            or part["filename"] != f"base.tar.zst.{number:03d}"
            or type(part["size"]) is not int
            or not 0 < part["size"] < 2_000_000_000
            or not isinstance(part["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", part["sha256"]) is None
        ):
            raise ValueError(f"invalid base archive part identity: {path}")
    identity_fields = (
        "format",
        "variant",
        "platform",
        "python",
        "interpreter",
        "wheels",
        "replaces",
        "workerProtocol",
        "nvidia",
    )
    payload = {key: manifest[key] for key in identity_fields}
    payload["archive"] = {"format": archive["format"]}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    if manifest.get("id") != "sha256:" + hashlib.sha256(canonical).hexdigest():
        raise ValueError(f"base manifest id does not match its payload: {path}")
    return manifest

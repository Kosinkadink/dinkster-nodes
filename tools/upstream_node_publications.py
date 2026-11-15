"""Measure upstream node introduction to first verified public pack publication."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlencode
from zoneinfo import ZoneInfo

from blake3 import blake3

from tools.upstream_node_intake import git, snapshot

PUBLICATION_STATE = re.compile(r"<!-- node-publications-state\n(.*?)\n-->", re.DOTALL)


def introductions(root: Path, target: str, names: set[str]) -> dict[str, dict[str, str]]:
    """Find the first registered occurrence in the reachable upstream Git history."""
    if not names:
        return {}
    pattern = "|".join(re.escape(name) for name in sorted(names))
    commits = git(
        root,
        "log",
        "--reverse",
        "--topo-order",
        "--format=%H",
        "-G" + pattern,
        target,
        "--",
        "nodes.py",
        "comfy_extras",
        "comfy_api_nodes",
    ).splitlines()
    origins = {}
    for commit in commits:
        remaining = names - origins.keys()
        if not remaining:
            break
        changed = git(
            root, "diff-tree", "--root", "-m", "--no-commit-id", "--name-only", "-r", commit
        ).splitlines()
        present = set(git(root, "ls-tree", "-r", "--name-only", commit).splitlines())
        paths = sorted(set(changed) & present)
        found = snapshot(root, commit, paths=paths, names=remaining)
        timestamp = git(root, "show", "-s", "--format=%cI", commit).strip()
        for name in found:
            origins[name] = {"commit": commit, "committedAt": timestamp}
    if missing := names - origins.keys():
        raise ValueError(f"upstream introduction could not be established: {sorted(missing)}")
    return origins


def identity_replacements(source: str, registered: set[str]) -> set[str]:
    """Require a published registered node type and its class's explicit replacement ID."""
    result = set()
    for cls in ast.parse(source).body:
        if not isinstance(cls, ast.ClassDef):
            continue
        calls = [item for item in ast.walk(cls) if isinstance(item, ast.Call)]
        types = {
            field.value.value
            for call in calls
            for field in call.keywords
            if field.arg == "node_type"
            and isinstance(field.value, ast.Constant)
            and isinstance(field.value.value, str)
        }
        if not types & registered:
            continue
        result.update(
            call.args[1].value
            for call in calls
            if isinstance(call.func, ast.Name)
            and call.func.id == "with_identity_replacement"
            and len(call.args) >= 2
            and isinstance(call.args[1], ast.Constant)
            and isinstance(call.args[1].value, str)
        )
    return result


def read_public(url: str, limit: int = 16 * 1024 * 1024) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "dinkster-node-arrivals"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("publication evidence exceeds its size limit")
    return payload


def published_nodes(
    registry: str, pack: str, wanted: set[str]
) -> tuple[dict[str, dict], datetime | None]:
    """Select by publication timestamp, not pack version or the time of observation."""
    registry = registry.rstrip("/")
    releases = []
    cursor = ""
    seen = set()
    while True:
        query = urlencode(
            {"limit": 100, "includeYanked": "true", **({"cursor": cursor} if cursor else {})}
        )
        document = json.loads(
            read_public(f"{registry}/v1/packs/{quote(pack, safe='')}/versions?{query}")
        )
        releases.extend(document["items"])
        cursor = document.get("nextCursor")
        if not cursor:
            break
        if cursor in seen:
            raise ValueError("registry publication pagination did not advance")
        seen.add(cursor)
    releases = [
        row for row in releases if row.get("state") in {"active", "yanked"} and row["nodeTypes"]
    ]
    releases.sort(key=lambda row: (datetime.fromisoformat(row["publishedAt"]), row["version"]))
    result = {}
    archives: dict[tuple[str, tuple[str, ...]], tuple[set[str], str]] = {}
    for release in releases:
        if not wanted - result.keys():
            break
        version = release["version"]
        digest = release["artifactDigest"]
        if re.fullmatch(r"blake3:[0-9a-f]{64}", digest) is None:
            raise ValueError("publication artifact must have a content-addressed digest")
        key = digest, tuple(sorted(release["nodeTypes"]))
        if key not in archives:
            try:
                payload = read_public(f"{registry}/v1/artifacts/{digest}", 256 * 1024 * 1024)
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                # Missing historical bytes prevent a claim that a later version was first.
                return result, datetime.fromisoformat(release["publishedAt"])
            if (
                len(payload) != release["artifactSize"]
                or "blake3:" + blake3(payload).hexdigest() != digest
            ):
                raise ValueError(f"{pack}@{version}: published archive size or digest differs")
            replacements = set()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                if sum(info.file_size for info in archive.infolist()) > 256 * 1024 * 1024:
                    raise ValueError("publication wheel expands beyond the size limit")
                for name in archive.namelist():
                    if name.endswith(".py"):
                        replacements.update(
                            identity_replacements(
                                archive.read(name).decode("utf-8"), set(release["nodeTypes"])
                            )
                        )
            archives[key] = replacements, hashlib.sha256(payload).hexdigest()
        replacements, sha256 = archives[key]
        for name in (wanted - result.keys()) & replacements:
            result[name] = {
                "pack": pack,
                "version": version,
                "publishedAt": release["publishedAt"],
                "artifactDigest": release["artifactDigest"],
                "sha256": sha256,
                "url": (
                    f"{registry}/v1/packs/{quote(pack, safe='')}/versions/{quote(version, safe='')}"
                ),
            }
    return result, None


def first_publications(registry: str, wanted: set[str]) -> dict[str, dict]:
    """Check every public first-party pack, including releases before a node moved packs."""
    if not wanted:
        return {}
    packs = set()
    cursor = ""
    seen = set()
    while True:
        query = urlencode({"limit": 100, **({"cursor": cursor} if cursor else {})})
        document = json.loads(read_public(f"{registry.rstrip('/')}/v1/packs?{query}"))
        packs.update(row["name"] for row in document["items"] if row["owner"]["slug"] == "dinkster")
        cursor = document.get("nextCursor")
        if not cursor:
            break
        if cursor in seen:
            raise ValueError("registry pack pagination did not advance")
        seen.add(cursor)
    result = {}
    gaps = []
    for pack in sorted(packs):
        releases, gap = published_nodes(registry, pack, wanted)
        if gap is not None:
            gaps.append(gap)
        for name, release in releases.items():
            if name not in result or datetime.fromisoformat(
                release["publishedAt"]
            ) < datetime.fromisoformat(result[name]["publishedAt"]):
                result[name] = release
    return {
        name: release
        for name, release in result.items()
        if not gaps or datetime.fromisoformat(release["publishedAt"]) < min(gaps)
    }


def publication_records(
    inference: Path, result: dict, comments: list[dict], registry: str
) -> list[dict]:
    """Keep pending nodes across inference advances and recheck until they are published."""
    records = {}
    for comment in comments:
        if match := PUBLICATION_STATE.search(comment["body"]):
            for row in json.loads(match[1]):
                previous = records.get(row["name"])
                if previous is None or previous.get("firstPublished") is None:
                    records[row["name"]] = row
    arrivals = {row["name"]: row for row in result["nodes"] if row["name"] not in records}
    origins = introductions(inference, result["upstream"], set(arrivals))
    for name, row in arrivals.items():
        records[name] = {
            "name": name,
            "pack": row["pack"],
            "introduction": origins[name],
            "firstPublished": None,
        }
    pending = {name for name, row in records.items() if row["firstPublished"] is None}
    for name, publication in first_publications(registry, pending).items():
        origin = datetime.fromisoformat(records[name]["introduction"]["committedAt"])
        published = datetime.fromisoformat(publication["publishedAt"])
        records[name]["firstPublished"] = {
            **publication,
            "latencySeconds": (published - origin).total_seconds(),
        }
    return [records[name] for name in sorted(records)]


def publication_report(records: list[dict]) -> str:
    lines = [
        "## Node publication latency",
        "",
        "Time from the introducing ComfyUI commit to the first published pack "
        "declaring its replacement ID.",
        "Publication means registry publication, not a merged PR or an unpublished source version.",
        "",
        "| Node | Introducing commit | First containing pack | "
        "Publication time (Pacific) | Latency |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in records:
        origin = row["introduction"]
        commit = origin["commit"]
        introduction = f"[{commit[:12]}](https://github.com/Comfy-Org/ComfyUI/commit/{commit})"
        if publication := row["firstPublished"]:
            pack = f"[{publication['pack']}@{publication['version']}]({publication['url']})"
            timestamp = (
                datetime.fromisoformat(publication["publishedAt"])
                .astimezone(ZoneInfo("America/Los_Angeles"))
                .strftime("%Y-%m-%d %H:%M:%S %Z")
            )
            latency = f"{publication['latencySeconds'] / 3600:.2f} hours"
        else:
            pack, timestamp, latency = row["pack"], "not yet verified", "pending"
        name = row["name"].replace("|", "\\|")
        lines.append(f"| {name} | {introduction} | {pack} | {timestamp} | {latency} |")
    return "\n".join(lines).encode("ascii", "backslashreplace").decode() + "\n"

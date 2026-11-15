"""Bind node proposals to published packages with verifiable source provenance."""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import unquote

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from tools.upstream_node_intake import git, inference_upstream
from tools.upstream_node_publications import read_public


def wheel_identity(payload: bytes, row: dict, name: str, version: str) -> zipfile.ZipFile:
    if len(payload) != row["size"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError(f"{name}: published wheel size or SHA256 differs")
    archive = zipfile.ZipFile(io.BytesIO(payload))
    if sum(item.file_size for item in archive.infolist()) > 256 * 1024 * 1024:
        archive.close()
        raise ValueError(f"{name}: published wheel expands beyond its size limit")
    metadata = [item for item in archive.namelist() if item.endswith(".dist-info/METADATA")]
    if len(metadata) != 1:
        raise ValueError(f"{name}: published wheel must identify one distribution")
    fields = BytesParser().parsebytes(archive.read(metadata[0]))
    if fields["Name"] != name or fields["Version"] != version:
        raise ValueError(f"{name}: published wheel metadata identity differs")
    return archive


def published_inference(inference: Path, head: str, upstream: str) -> dict | None:
    """Find the minimum published version containing the target's landed upstream code."""
    index = json.loads(read_public("https://pypi.org/pypi/dinkster-inference/json"))
    tags = {
        line.split()[1]: line.split()[0]
        for line in git(inference, "ls-remote", "--tags", "origin", "refs/tags/v*").splitlines()
    }
    for version, files in sorted(index["releases"].items(), key=lambda item: Version(item[0])):
        ref = "refs/tags/v" + version
        # Legacy releases without annotated provenance cannot establish an upstream minimum.
        if ref + "^{}" not in tags:
            continue
        wheels = [row for row in files if row["packagetype"] == "bdist_wheel" and not row["yanked"]]
        if not wheels:
            continue
        git(inference, "fetch", "--no-tags", "origin", ref)
        tag = git(inference, "rev-parse", "FETCH_HEAD").strip()
        if tag != tags[ref] or git(inference, "cat-file", "-t", tag).strip() != "tag":
            raise ValueError("inference publication tag changed during inspection")
        annotation = git(inference, "cat-file", "-p", tag).split("\n\n", 1)[1]
        identity = json.loads(annotation)
        if (
            set(identity) != {"version", "sourceCommit", "upstreamCommit"}
            or identity["version"] != version
            or any(
                re.fullmatch("[0-9a-f]{40}", identity[key]) is None
                for key in ("sourceCommit", "upstreamCommit")
            )
            or git(inference, "rev-parse", tag + "^{commit}").strip() != identity["sourceCommit"]
            or inference_upstream(inference, identity["sourceCommit"]) != identity["upstreamCommit"]
        ):
            raise ValueError("inference publication tag and landed ledger provenance differ")
        if any(
            subprocess.run(
                ["git", "-C", str(inference), "merge-base", "--is-ancestor", before, after],
                check=False,
                capture_output=True,
            ).returncode
            for before, after in (
                (identity["sourceCommit"], head),
                (upstream, identity["upstreamCommit"]),
            )
        ):
            continue
        wheel = next((row for row in wheels if row["filename"].endswith("-py3-none-any.whl")), None)
        if wheel is None:
            continue
        row = {**wheel, "sha256": wheel["digests"]["sha256"]}
        payload = read_public(row["url"], 256 * 1024 * 1024)
        with wheel_identity(payload, row, "dinkster-inference", version) as archive:
            if json.loads(archive.read("dinkster_inference/_release.json")) != identity:
                raise ValueError("inference wheel and annotated tag provenance differ")
        return {
            **identity,
            "tagObject": tag,
            "url": row["url"],
            "sha256": row["sha256"],
            "size": row["size"],
            "publishedAt": row["upload_time_iso_8601"],
        }
    return None


def verify_inference_record(record: dict) -> None:
    """Recheck publication before pack admission, including a subsequently yanked release."""
    version = record["version"]
    document = json.loads(read_public(f"https://pypi.org/pypi/dinkster-inference/{version}/json"))
    wheels = [
        row
        for row in document["urls"]
        if row["packagetype"] == "bdist_wheel"
        and not row["yanked"]
        and row["url"] == record["url"]
        and row["digests"]["sha256"] == record["sha256"]
        and row["size"] == record["size"]
    ]
    if len(wheels) != 1:
        raise ValueError("inference prerequisite is missing, yanked or has different bytes")
    refs = subprocess.check_output(
        [
            "git",
            "ls-remote",
            "https://github.com/Kosinkadink/dinkster-inference.git",
            "refs/tags/v" + version,
            "refs/tags/v" + version + "^{}",
        ],
        text=True,
    )
    tags = {line.split()[1]: line.split()[0] for line in refs.splitlines()}
    if (
        tags.get("refs/tags/v" + version) != record["tagObject"]
        or tags.get("refs/tags/v" + version + "^{}") != record["sourceCommit"]
    ):
        raise ValueError("inference prerequisite tag or landed source changed")
    with wheel_identity(
        read_public(wheels[0]["url"], 256 * 1024 * 1024), record, "dinkster-inference", version
    ) as archive:
        identity = {key: record[key] for key in ("version", "sourceCommit", "upstreamCommit")}
        if json.loads(archive.read("dinkster_inference/_release.json")) != identity:
            raise ValueError("inference prerequisite wheel provenance differs")


def verify_pack_prerequisites(root: Path, requirements: tuple[str, ...]) -> None:
    """Refuse pack publication before its declared inference prerequisite is public."""
    path = root / "upstream-node-prerequisites.json"
    if not path.is_file():
        return
    state = json.loads(path.read_text())
    inference = [
        Requirement(value)
        for value in requirements
        if canonicalize_name(Requirement(value).name) == "dinkster-inference"
    ]
    seen = set()
    for node in state["nodes"].values():
        record = node["inference"]
        minimum = Version(record["version"])
        if node["upstream"] != record["upstreamCommit"] or not inference:
            raise ValueError("node prerequisite differs from its published upstream provenance")
        for requirement in inference:
            if requirement.url:
                raise ValueError("pack inference prerequisite must use a version range")
            if not any(
                spec.operator in {">=", ">", "==", "==="}
                and not spec.version.endswith(".*")
                and Version(spec.version) >= minimum
                for spec in requirement.specifier
            ):
                raise ValueError("pack requirement does not enforce its inference minimum")
        key = record["sha256"], record["tagObject"]
        if key not in seen:
            verify_inference_record(record)
            seen.add(key)


def published_api(root: Path) -> dict:
    """Use the existing engine pin, not a guessed newer engine or planning-host version."""
    project = tomllib.loads((root / "pyproject.toml").read_text())
    pin = unquote(project["tool"]["uv"]["sources"]["dinkster-api"]["url"])
    match = re.fullmatch(
        r"https://github.com/Kosinkadink/Dinkster/releases/download/([^/]+)/"
        r"dinkster_api-([^/]+)\+g([0-9a-f]{40})-py3-none-any.whl",
        pin,
    )
    if match is None:
        raise ValueError("node proposals require a commit-bound published API pin")
    _, version, commit = match.groups()
    version += "+g" + commit
    index = json.loads(read_public("https://releases.dinkster.org/engine/index.json"))
    matches = [row for row in index if row["commit"] == commit]
    if len(matches) != 1:
        raise ValueError("pinned API requires one indexed engine publication")
    selected = matches[0]
    data = read_public(f"https://releases.dinkster.org/{selected['path']}/release-manifest.json")
    if hashlib.sha256(data).hexdigest() != selected["sha256"]:
        raise ValueError("published engine manifest SHA256 differs")
    manifest = json.loads(data)
    if manifest["commit"] != commit or manifest["repository"] != "Kosinkadink/Dinkster":
        raise ValueError("published engine source identity differs")
    artifacts = [row for row in manifest["artifacts"] if row["name"] == pin.rsplit("/", 1)[1]]
    if len(artifacts) != 1:
        raise ValueError("pinned API wheel is absent from the published engine")
    row = artifacts[0]
    with wheel_identity(read_public(pin), row, "dinkster-api", version):
        pass
    return {"version": version, "url": pin, "sha256": row["sha256"], "commit": commit}

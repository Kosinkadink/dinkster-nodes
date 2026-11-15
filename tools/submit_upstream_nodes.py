"""Publish node-intake proposals as draft pack PRs and an assigned arrival digest.

Without --submit, writes only a local report. GitHub commits use the token's
own identity; existing branches advance without force and keep human edits.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from tools.upstream_node_intake import git, inspect_changes
from tools.upstream_node_prerequisites import published_api, published_inference
from tools.upstream_node_proposals import slug
from tools.upstream_node_publications import publication_records, publication_report

REPOSITORY = "Kosinkadink/dinkster-nodes"
LABEL = "upstream-node-intake"
STATE = re.compile(r"<!-- upstream-node-intake: ([0-9a-f]{40}) -->")


def gh(*args: str, data: object = None) -> object:
    command = ["gh", *args]
    if data is not None:
        command.extend(["--input", "-"])
    result = subprocess.check_output(
        command, input=json.dumps(data) if data is not None else None, text=True
    )
    return json.loads(result) if result.strip() else None


def api(path: str, data: object) -> dict:
    return gh("api", "--method", "POST", f"repos/{REPOSITORY}/{path}", data=data)


def report(result: dict, links: dict[str, str] | None = None) -> str:
    links = links or {}
    lines = [
        "# ComfyUI node arrivals",
        "",
        f"Inference source: https://github.com/Kosinkadink/dinkster-inference/commit/{result['inference']}",
        f"Upstream point: https://github.com/Comfy-Org/ComfyUI/commit/{result['upstream']}",
        "",
        "| Node | Change | Pack | Mechanical | Reason and consolidation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in result["nodes"]:
        name = row["name"].replace("|", "\\|")
        destination = links.get(row["pack"], "")
        pack = f"[{row['pack']}]({destination})" if destination else row["pack"]
        explanation = " ".join(filter(None, (row["reason"], row["consolidation"])))
        explanation = explanation.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {name} | {row['change']} | {pack} | "
            f"{'yes' if row['mechanical'] else 'no - draft'} | {explanation} |"
        )
    if prerequisites := result.get("prerequisites"):
        api_release = prerequisites["api"]
        lines.extend(
            [
                "",
                f"Published target API: [{api_release['version']}]({api_release['url']}). "
                "The existing pack.engine API range is preserved and checked, not guessed.",
            ]
        )
        if inference := prerequisites["inference"]:
            lines.append(
                f"Published inference minimum: [{inference['version']}]({inference['url']}), "
                f"SHA256 `{inference['sha256']}`; annotated tag and wheel provenance agree. "
                "Inference-dependent additions declare this minimum and publication rechecks it."
            )
        else:
            lines.append(
                "Inference publication is pending: "
                "inference-dependent additions stay blocked drafts."
            )
    if result["removed"]:
        lines.extend(
            [
                "",
                "Removed upstream IDs (no automatic native deletion): "
                + ", ".join(result["removed"]),
            ]
        )
    lines.extend(
        [
            "",
            "Every proposal stays draft until its node load/value tests, compatible published "
            "engine and inference prerequisites, normal hosted checks and independent review pass.",
            "Proposals containing non-mechanical nodes carry the parked label for semantic review; "
            "each node's reason and consolidation path are listed above.",
        ]
    )
    return "\n".join(lines).encode("ascii", "backslashreplace").decode() + "\n"


def read_file(root: Path, revision: str, path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{revision}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def changed_files(
    root: Path, revision: str, rows: list[dict], upstream: str, prerequisites: dict
) -> dict[str, str]:
    """Prepare native additions and document the concrete unsupported translations."""
    package = rows[0]["package"]
    manifest_path = f"{package}/dinkster-pack.toml"
    manifest_text = read_file(root, revision, manifest_path)
    manifest = tomllib.loads(manifest_text)["pack"]
    module = manifest.get("entry", {}).get("nodes", "").split(":")[0]
    module_path = f"{package}/src/{module.replace('.', '/')}"
    init_path = f"{module_path}/__init__.py"
    init = read_file(root, revision, init_path)
    changes: dict[str, str] = {}
    added = []
    needs_inference = any(
        canonicalize_name(Requirement(value).name) == "dinkster-inference"
        for value in manifest.get("requires", [])
    )
    blocker = ""
    if not SpecifierSet(manifest["engine"]).contains(
        prerequisites["api"]["version"], prereleases=True
    ):
        blocker = (
            "Pack API range excludes the published target engine; engine capability needs review"
        )
    elif needs_inference and prerequisites["inference"] is None:
        blocker = "No verified published inference release contains this upstream point"
    for row in rows:
        if blocker and row["mechanical"]:
            row["mechanical"] = False
            row["reason"] = blocker
        if not row["mechanical"]:
            continue
        filename = f"{module_path}/upstream_{slug(row['name'])}.py"
        test_path = f"{package}/tests/test_upstream_{slug(row['name'])}.py"
        tree = ast.parse(init)
        registration = next(
            (
                item
                for item in tree.body
                if isinstance(item, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "NODES" for target in item.targets
                )
                and isinstance(item.value, ast.Tuple)
            ),
            None,
        )
        existing = read_file(root, revision, filename)
        existing_test = read_file(root, revision, test_path)
        if existing or existing_test:
            if (
                existing
                and existing_test
                and ast.dump(ast.parse(existing)) == ast.dump(ast.parse(row["native_source"]))
                and ast.dump(ast.parse(existing_test)) == ast.dump(ast.parse(row["load_test"]))
                and registration is not None
                and any(
                    isinstance(item, ast.Name) and item.id == row["class_name"]
                    for item in registration.value.elts
                )
                and any(
                    isinstance(item, ast.ImportFrom)
                    and item.level == 1
                    and item.module == f"upstream_{slug(row['name'])}"
                    and any(
                        alias.name == row["class_name"] and alias.asname is None
                        for alias in item.names
                    )
                    for item in tree.body
                )
            ):
                continue
            row["mechanical"] = False
            row["reason"] = (
                "A native proposal already exists; preserve it for review rather than overwrite it"
            )
            continue
        bound_names = {item.id for item in ast.walk(tree) if isinstance(item, ast.Name)} | {
            alias.asname or alias.name
            for item in ast.walk(tree)
            if isinstance(item, (ast.Import, ast.ImportFrom))
            for alias in item.names
        }
        if filename in changes or row["class_name"] in bound_names:
            row["mechanical"] = False
            row["reason"] = "Generated module or class name collides with an existing registration"
            continue
        if registration is None:
            row["mechanical"] = False
            row["reason"] = (
                "Pack entry point builds its node list dynamically; registration needs review"
            )
            continue
        original = ast.get_source_segment(init, registration.value)
        replacement = original[:-1] + f"\n    {row['class_name']},\n)"
        init = init.replace(original, replacement, 1)
        offset = sum(
            len(line) for line in init.splitlines(keepends=True)[: registration.lineno - 1]
        )
        init = (
            init[:offset]
            + f"from .upstream_{slug(row['name'])} import {row['class_name']}\n\n"
            + init[offset:]
        )
        changes[filename] = row["native_source"]
        changes[f"{package}/tests/test_upstream_{slug(row['name'])}.py"] = row["load_test"]
        added.append(row["name"])
    if added:
        changes[init_path] = init
        version = manifest["version"]
        parts = version.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError(f"{manifest['name']}: version needs an explicit release choice")
        next_version = ".".join([*parts[:2], str(int(parts[2]) + 1)])
        changes[manifest_path] = manifest_text.replace(
            f'version = "{version}"', f'version = "{next_version}"', 1
        )
        project_path = f"{package}/pyproject.toml"
        project = read_file(root, revision, project_path)
        changes[project_path] = project.replace(
            f'version = "{version}"', f'version = "{next_version}"', 1
        )
        lock = read_file(root, revision, "uv.lock")
        lock, count = re.subn(
            r'(\[\[package\]\]\nname = "'
            + re.escape(manifest["name"])
            + r'"\nversion = ")'
            + re.escape(version)
            + '"',
            lambda match: match[1] + next_version + '"',
            lock,
        )
        if count != 1:
            raise ValueError(f"{manifest['name']}: local lock identity needs regeneration")
        changes["uv.lock"] = lock
        if needs_inference:
            record = prerequisites["inference"]
            requires = manifest["requires"]
            for value in requires:
                requirement = Requirement(value)
                if canonicalize_name(requirement.name) != "dinkster-inference":
                    continue
                if requirement.url or not requirement.specifier.contains(
                    record["version"], prereleases=True
                ):
                    raise ValueError("published inference excludes the pack's existing requirement")
                minimum = str(requirement)
                if requirement.marker:
                    minimum = minimum.split(";", 1)[0]
                minimum += ("," if str(requirement.specifier) else "") + ">=" + record["version"]
                if requirement.marker:
                    minimum += "; " + str(requirement.marker)
                changes[manifest_path] = changes[manifest_path].replace(
                    json.dumps(value), json.dumps(minimum)
                )
                # The runtime manifest owns requirements; keep development metadata synchronized.
                declaration = tomllib.loads(project)["project"]
                for dependency in declaration.get("dependencies", []):
                    if canonicalize_name(Requirement(dependency).name) == "dinkster-inference":
                        changes[project_path] = changes[project_path].replace(
                            json.dumps(dependency), json.dumps(minimum)
                        )
            provenance_path = f"{package}/upstream-node-prerequisites.json"
            previous = read_file(root, revision, provenance_path)
            state = json.loads(previous) if previous else {"nodes": {}}
            for name in added:
                state["nodes"][name] = {"upstream": upstream, "inference": record}
            changes[provenance_path] = json.dumps(state, indent=2, sort_keys=True) + "\n"
            with tempfile.TemporaryDirectory(prefix="node-proposal-") as directory:
                work = Path(directory)
                payload = subprocess.check_output(["git", "-C", str(root), "archive", revision])
                with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
                    archive.extractall(work, filter="data")
                for path, text in changes.items():
                    destination = work / path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(text)
                subprocess.run(
                    [
                        "uv",
                        "lock",
                        "--directory",
                        str(work),
                        "--upgrade-package",
                        "dinkster-inference",
                    ],
                    check=True,
                )
                changes["uv.lock"] = (work / "uv.lock").read_text()
    drafts = [row for row in rows if not row["mechanical"]]
    if drafts:
        readme_path = f"{package}/README.md"
        readme = read_file(root, revision, readme_path)
        if "\n## Upstream compatibility\n" not in readme:
            readme += "\n## Upstream compatibility\n\n"
        for row in drafts:
            prefix = f"- `{row['name']}`: "
            note = prefix + row["reason"] + ". " + row["consolidation"]
            note += f" [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/{upstream}/{row['path']})."
            readme = "\n".join(line for line in readme.splitlines() if not line.startswith(prefix))
            readme += "\n" + note + "\n"
        changes[readme_path] = readme
    for path, text in changes.items():
        if path.endswith(".py"):
            text = subprocess.check_output(
                ["ruff", "format", "--stdin-filename", path, "-"],
                input=text,
                text=True,
            )
            text = subprocess.check_output(
                ["ruff", "check", "--fix", "--stdin-filename", path, "-"],
                input=text,
                text=True,
            )
            changes[path] = subprocess.check_output(
                ["ruff", "format", "--stdin-filename", path, "-"],
                input=text,
                text=True,
            )
    return {
        path: text.encode("ascii", "backslashreplace").decode() for path, text in changes.items()
    }


def draft_proposal(number: int, head: str) -> dict:
    """Recheck ownership and review state before advancing a generated proposal."""
    detail = gh(
        "pr",
        "view",
        str(number),
        "--repo",
        REPOSITORY,
        "--json",
        "isDraft,labels,headRefOid,body,state",
    )
    if (
        detail["state"] != "OPEN"
        or not detail["isDraft"]
        or detail["headRefOid"] != head
        or "<!-- upstream-node-intake -->" not in detail["body"]
        or any(label["name"] == "ready-to-land" for label in detail["labels"])
    ):
        raise ValueError("proposal head or review state changed; preserve it for review")
    return detail


def submit_pack(root: Path, pack: str, rows: list[dict], result: dict) -> str:
    branch = f"upstream-nodes-{pack}-{result['upstream'][:12]}"
    commit_message = (
        f"Propose upstream node changes for {pack}\n\nUpstream-node-intake: {result['upstream']}\n"
    )
    existing = ""
    pulls = gh(
        "pr",
        "list",
        "--repo",
        REPOSITORY,
        "--state",
        "all",
        "--head",
        branch,
        "--json",
        "number,url,headRefOid,body,state",
    )
    if pulls:
        # The workflow owns only its marked draft PRs, never an accepted head.
        if len(pulls) != 1:
            raise ValueError(f"{pack}: multiple proposals for the same upstream point need review")
        pull = pulls[0]
        if "<!-- upstream-node-intake -->" not in pull["body"]:
            raise ValueError(f"{pack}: branch is not an intake-owned proposal")
        if pull["state"] == "MERGED":
            return pull["url"]
        if pull["state"] != "OPEN":
            raise ValueError(f"{pack}: proposal was closed; preserve the review decision")
        draft_proposal(pull["number"], pull["headRefOid"])
        subprocess.run(["git", "-C", str(root), "fetch", "--no-tags", "origin", branch], check=True)
        parent = pull["headRefOid"]
    else:
        existing = git(root, "ls-remote", "origin", f"refs/heads/{branch}").strip()
        if existing:
            subprocess.run(
                ["git", "-C", str(root), "fetch", "--no-tags", "origin", branch], check=True
            )
            parent = existing.split()[0]
            if git(root, "log", "-1", "--format=%B", parent).strip() != commit_message.strip():
                raise ValueError(f"{pack}: existing branch is not an intake-owned proposal")
        else:
            parent = git(root, "rev-parse", "origin/main").strip()
    changes = changed_files(root, parent, rows, result["upstream"], result["prerequisites"])
    body = "Refs Kosinkadink/comfy-vibe-station#903\n<!-- upstream-node-intake -->\n\n"
    body += report({**result, "nodes": rows, "removed": []})
    base_tree = git(root, "rev-parse", parent + "^{tree}").strip()
    tree = api(
        "git/trees",
        {
            "base_tree": base_tree,
            "tree": [
                {"path": path, "mode": "100644", "type": "blob", "content": text}
                for path, text in changes.items()
            ],
        },
    )
    if tree["sha"] != base_tree:
        commit = api(
            "git/commits",
            {
                "message": commit_message,
                "tree": tree["sha"],
                "parents": [parent],
            },
        )
        if pulls:
            draft_proposal(pull["number"], parent)
        if pulls or existing:
            gh(
                "api",
                "--method",
                "PATCH",
                f"repos/{REPOSITORY}/git/refs/heads/{branch}",
                data={"sha": commit["sha"], "force": False},
            )
        else:
            api("git/refs", {"ref": f"refs/heads/{branch}", "sha": commit["sha"]})
    if pulls:
        # Preserve any human notes outside the generated intake report.
        current = draft_proposal(
            pull["number"], commit["sha"] if tree["sha"] != base_tree else parent
        )
        previous = current["body"]
        marker = "\n<!-- intake-report-end -->"
        notes = previous.split(marker, 1)[1] if marker in previous else ""
        gh(
            "api",
            "--method",
            "PATCH",
            f"repos/{REPOSITORY}/pulls/{pull['number']}",
            data={"body": body + marker + notes},
        )
        number, url = pull["number"], pull["url"]
    else:
        created = api(
            "pulls",
            {
                "title": f"Add upstream ComfyUI nodes to {pack}",
                "head": branch,
                "base": "main",
                "draft": True,
                "body": body + "\n<!-- intake-report-end -->",
            },
        )
        number, url = created["number"], created["html_url"]
    if any(not row["mechanical"] for row in rows):
        api(f"issues/{number}/labels", {"labels": ["parked"]})
    return url


def save_publications(number: int, records: list[dict], comments: list[dict]) -> None:
    """Keep bounded readable latency records durable even when upstream is unchanged."""
    for offset in range(0, len(records), 60):
        marker = f"<!-- node-publications: {offset} -->"
        rows = records[offset : offset + 60]
        body = marker + "\n" + publication_report(rows)
        state = json.dumps(rows, ensure_ascii=True).replace("<", "\\u003c")
        body += "\n<!-- node-publications-state\n" + state + "\n-->"
        end = "\n<!-- node-publications-end -->"
        previous = next((item for item in comments if item["body"].startswith(marker + "\n")), None)
        if previous:
            current = gh("api", f"repos/{REPOSITORY}/issues/comments/{previous['id']}")
            notes = current["body"].split(end, 1)[1] if end in current["body"] else ""
            body += end + notes
            if len(body.encode("utf-8")) > 60_000:
                raise ValueError("publication report exceeds the GitHub comment limit")
            if body != current["body"]:
                gh(
                    "api",
                    "--method",
                    "PATCH",
                    f"repos/{REPOSITORY}/issues/comments/{previous['id']}",
                    data={"body": body},
                )
        else:
            if len((body + end).encode("utf-8")) > 60_000:
                raise ValueError("publication report exceeds the GitHub comment limit")
            api(f"issues/{number}/comments", {"body": body + end})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--registry", default="https://registry.dinkster.org")
    args = parser.parse_args()
    reference = json.loads((args.root / "tools/data/comfy_core_reference.json").read_text())
    baseline = reference["source_commit"]
    issue = None
    if args.submit:
        for name, description in (
            (LABEL, "Automatic upstream node arrivals"),
            ("parked", "Awaiting review of non-mechanical upstream node changes"),
        ):
            labels = gh("label", "list", "--repo", REPOSITORY, "--search", name, "--json", "name")
            if not any(label["name"] == name for label in labels or []):
                api("labels", {"name": name, "description": description, "color": "1d76db"})
        issues = gh(
            "issue",
            "list",
            "--repo",
            REPOSITORY,
            "--state",
            "open",
            "--label",
            LABEL,
            "--json",
            "number,body",
        )
        if issues and len(issues) > 1:
            raise ValueError("multiple node-intake state issues need review")
        issue = issues[0] if issues else None
        if issue:
            match = STATE.search(issue["body"])
            if match is None:
                raise ValueError("node-intake issue is missing its processed upstream point")
            baseline = match[1]
    head = git(args.inference, "rev-parse", "origin/main^{commit}").strip()
    result = inspect_changes(args.root, args.inference, baseline, head)
    result["prerequisites"] = {
        "api": published_api(args.root),
        "inference": published_inference(args.inference, head, result["upstream"]),
    }
    comments = []
    if issue:
        pages = gh(
            "api",
            "--paginate",
            "--slurp",
            f"repos/{REPOSITORY}/issues/{issue['number']}/comments?per_page=100",
        )
        comments = [comment for page in pages for comment in page]
    records = publication_records(args.inference, result, comments, args.registry)
    args.output.write_text(report(result) + "\n" + publication_report(records))
    if not args.submit or result["upstream"] == baseline:
        if args.submit and issue:
            save_publications(issue["number"], records, comments)
        print(f"{len(result['nodes'])} node changes; upstream point {result['upstream']}")
        return 0
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in result["nodes"]:
        groups[row["pack"]].append(row)
    links = {
        pack: submit_pack(args.root, pack, rows, result) for pack, rows in sorted(groups.items())
    }
    digest = report(result, links) + "\n" + publication_report(records)
    args.output.write_text(digest)
    # GitHub issue bodies have a size limit; the full list lives in bounded comments.
    summary = "# ComfyUI node arrivals\n\n" + "\n".join(
        f"- [{pack}]({links[pack]}): {len(rows)} added or changed nodes"
        for pack, rows in sorted(groups.items())
    )
    summary += (
        "\n\nThe comments below contain every node and its translation/consolidation reason.\n"
    )
    end_marker = "\n<!-- intake-report-end -->"
    body = f"<!-- upstream-node-intake: {result['upstream']} -->\n\n" + summary + end_marker
    if issue is None:
        issue = api(
            "issues",
            {
                "title": "ComfyUI node arrivals",
                "labels": [LABEL],
                "assignees": ["Kosinkadink"],
                "body": f"<!-- upstream-node-intake: {baseline} -->\n\nPreparing node arrivals.",
            },
        )
    pages = gh(
        "api",
        "--paginate",
        "--slurp",
        f"repos/{REPOSITORY}/issues/{issue['number']}/comments?per_page=100",
    )
    comments = [comment for page in pages for comment in page]
    save_publications(issue["number"], records, comments)
    previous = {comment["body"].split("\n", 1)[0] for comment in comments}
    for offset in range(0, max(1, len(result["nodes"])), 60):
        marker = f"<!-- node-arrivals: {result['upstream']} {offset} -->"
        if marker in previous:
            continue
        part = report({**result, "nodes": result["nodes"][offset : offset + 60]}, links)
        if len(part) > 60_000:
            raise ValueError("arrival report exceeds the GitHub comment limit")
        api(f"issues/{issue['number']}/comments", {"body": marker + "\n" + part})
    current = gh("api", f"repos/{REPOSITORY}/issues/{issue['number']}")
    state = STATE.search(current["body"])
    if state is None or state[1] != baseline or current["state"] != "open":
        raise ValueError("intake state changed during submission; preserve it for review")
    notes = current["body"].split(end_marker, 1)[1] if end_marker in current["body"] else ""
    gh(
        "api",
        "--method",
        "PATCH",
        f"repos/{REPOSITORY}/issues/{issue['number']}",
        data={"body": body + notes},
    )
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as stream:
            stream.write(digest)
    print(f"Node arrivals: https://github.com/{REPOSITORY}/issues/{issue['number']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

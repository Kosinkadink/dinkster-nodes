"""Publish node-intake proposals as draft pack PRs and an assigned arrival digest.

Without --submit, writes only a local report. GitHub commits use the token's
own identity; existing branches advance without force and keep human edits.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import tomllib
from collections import defaultdict
from pathlib import Path

from tools.upstream_node_intake import git, inspect_changes
from tools.upstream_node_proposals import slug

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


def changed_files(root: Path, revision: str, rows: list[dict], upstream: str) -> dict[str, str]:
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
    for row in rows:
        if not row["mechanical"]:
            continue
        filename = f"{module_path}/upstream_{slug(row['name'])}.py"
        if read_file(root, revision, filename):
            row["mechanical"] = False
            row["reason"] = (
                "A native proposal already exists; preserve it for review rather than overwrite it"
            )
            continue
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


def submit_pack(root: Path, pack: str, rows: list[dict], result: dict) -> str:
    branch = f"upstream-nodes-{pack}-{result['upstream'][:12]}"
    existing = ""
    pulls = gh(
        "pr",
        "list",
        "--repo",
        REPOSITORY,
        "--state",
        "open",
        "--head",
        branch,
        "--json",
        "number,url,headRefOid,body",
    )
    if pulls:
        # The workflow owns only its marked draft PRs, never an accepted head.
        pull = pulls[0]
        detail = gh(
            "pr", "view", str(pull["number"]), "--repo", REPOSITORY, "--json", "isDraft,labels"
        )
        if not detail["isDraft"] or any(
            label["name"] == "ready-to-land" for label in detail["labels"]
        ):
            raise ValueError(
                f"{pack}: proposal is under review; wait for its landing before updating it"
            )
        if "<!-- upstream-node-intake -->" not in pull["body"]:
            raise ValueError(f"{pack}: branch is not an intake-owned proposal")
        subprocess.run(["git", "-C", str(root), "fetch", "--no-tags", "origin", branch], check=True)
        parent = pull["headRefOid"]
    else:
        existing = git(root, "ls-remote", "origin", f"refs/heads/{branch}").strip()
        if existing:
            subprocess.run(
                ["git", "-C", str(root), "fetch", "--no-tags", "origin", branch], check=True
            )
            parent = existing.split()[0]
            if git(root, "log", "-1", "--format=%s", parent).strip() != (
                f"Propose upstream node changes for {pack}"
            ):
                raise ValueError(f"{pack}: existing branch is not an intake-owned proposal")
        else:
            parent = git(root, "rev-parse", "origin/main").strip()
    changes = changed_files(root, parent, rows, result["upstream"])
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
                "message": f"Propose upstream node changes for {pack}",
                "tree": tree["sha"],
                "parents": [parent],
            },
        )
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
        previous = pull["body"]
        marker = "\n<!-- intake-report-end -->"
        notes = previous.split(marker, 1)[1] if marker in previous else ""
        gh(
            "api",
            "--method",
            "PATCH",
            f"repos/{REPOSITORY}/pulls/{pull['number']}",
            data={"body": body + marker + notes},
        )
        return pull["url"]
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
    return created["html_url"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    reference = json.loads((args.root / "tools/data/comfy_core_reference.json").read_text())
    baseline = reference["source_commit"]
    issue = None
    if args.submit:
        labels = gh("label", "list", "--repo", REPOSITORY, "--search", LABEL, "--json", "name")
        if not any(label["name"] == LABEL for label in labels):
            api(
                "labels",
                {
                    "name": LABEL,
                    "description": "Automatic upstream node arrivals",
                    "color": "1d76db",
                },
            )
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
        if len(issues) > 1:
            raise ValueError("multiple node-intake state issues need review")
        issue = issues[0] if issues else None
        if issue:
            match = STATE.search(issue["body"])
            if match is None:
                raise ValueError("node-intake issue is missing its processed upstream point")
            baseline = match[1]
    head = git(args.inference, "rev-parse", "origin/main^{commit}").strip()
    result = inspect_changes(args.root, args.inference, baseline, head)
    args.output.write_text(report(result))
    if not args.submit or result["upstream"] == baseline:
        print(f"{len(result['nodes'])} node changes; upstream point {result['upstream']}")
        return 0
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in result["nodes"]:
        groups[row["pack"]].append(row)
    links = {
        pack: submit_pack(args.root, pack, rows, result) for pack, rows in sorted(groups.items())
    }
    digest = report(result, links)
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
    notes = issue["body"].split(end_marker, 1)[1] if issue and end_marker in issue["body"] else ""
    body = f"<!-- upstream-node-intake: {result['upstream']} -->\n\n" + summary + end_marker + notes
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
    previous = {comment["body"].split("\n", 1)[0] for page in pages for comment in page}
    for offset in range(0, max(1, len(result["nodes"])), 60):
        marker = f"<!-- node-arrivals: {result['upstream']} {offset} -->"
        if marker in previous:
            continue
        part = report({**result, "nodes": result["nodes"][offset : offset + 60]}, links)
        if len(part) > 60_000:
            raise ValueError("arrival report exceeds the GitHub comment limit")
        api(f"issues/{issue['number']}/comments", {"body": marker + "\n" + part})
    gh(
        "api",
        "--method",
        "PATCH",
        f"repos/{REPOSITORY}/issues/{issue['number']}",
        data={"body": body},
    )
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as stream:
            stream.write(digest)
    print(f"Node arrivals: https://github.com/{REPOSITORY}/issues/{issue['number']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Inspect ComfyUI node changes at the upstream point reached by inference.

Reads Git objects and Python syntax, never imports or executes upstream nodes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from tools.upstream_node_proposals import translate


@dataclass(frozen=True)
class UpstreamNode:
    name: str
    class_name: str
    path: str
    digest: str
    source: str
    api: bool
    blocker: str = ""


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def inference_upstream(root: Path, revision: str) -> str:
    ledger = git(root, "show", f"{revision}:docs/ledger.md")
    match = re.search(r"Upstream fork point:.*?`([0-9a-f]{40})`", ledger, re.DOTALL)
    if match is None:
        raise ValueError("inference ledger has no full upstream commit")
    return match[1]


def snapshot(
    root: Path,
    revision: str,
    *,
    paths: list[str] | None = None,
    names: set[str] | None = None,
) -> dict[str, UpstreamNode]:
    """Discover V1 mappings and V3 node IDs without loading application code."""
    result: dict[str, UpstreamNode] = {}
    if paths is None:
        paths = git(root, "ls-tree", "-r", "--name-only", revision).splitlines()
    for path in paths:
        if not (
            path == "nodes.py"
            or path.startswith(("comfy_extras/", "comfy_api_nodes/"))
            and path.endswith(".py")
        ):
            continue
        source = git(root, "show", f"{revision}:{path}").lstrip("\ufeff")
        tree = ast.parse(source, filename=path)
        classes = {item.name: item for item in tree.body if isinstance(item, ast.ClassDef)}
        registrations: dict[str, str] = {}
        unresolved: set[str] = set()
        for item in ast.walk(tree):
            if isinstance(item, ast.Assign) and isinstance(item.value, ast.Dict):
                if not any(
                    isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS"
                    for target in item.targets
                ):
                    continue
                for key, value in zip(item.value.keys, item.value.values, strict=True):
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        if names is not None and key.value not in names:
                            continue
                        if isinstance(value, ast.Name):
                            registrations[key.value] = value.id
                        else:
                            raise ValueError(f"{path}: nonlocal V1 registration {key.value}")
            if isinstance(item, ast.ClassDef):
                for assignment in ast.walk(item):
                    if not isinstance(assignment, ast.Assign):
                        continue
                    if not isinstance(assignment.value, ast.Constant) or not isinstance(
                        assignment.value.value, str
                    ):
                        continue
                    if any(
                        isinstance(target, ast.Name)
                        and target.id in {"NodeId", "node_id"}
                        or isinstance(target, ast.Attribute)
                        and target.attr == "node_id"
                        for target in assignment.targets
                    ):
                        registrations[assignment.value.value] = item.name
                for call in ast.walk(item):
                    if not isinstance(call, ast.Call):
                        continue
                    for keyword in call.keywords:
                        if keyword.arg == "node_id":
                            value = keyword.value
                            if isinstance(value, ast.Attribute) and isinstance(
                                value.value, ast.Name
                            ):
                                if value.value.id == "cls":
                                    for assignment in item.body:
                                        if isinstance(assignment, ast.Assign) and any(
                                            isinstance(target, ast.Name) and target.id == value.attr
                                            for target in assignment.targets
                                        ):
                                            value = assignment.value
                                            break
                            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                                registrations[value.value] = item.name
                            else:
                                unresolved.add(item.name)
        exported: set[str] = set()
        for item in ast.walk(tree):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name != "get_node_list":
                    continue
                for statement in ast.walk(item):
                    if isinstance(statement, ast.Return) and isinstance(
                        statement.value, (ast.List, ast.Tuple)
                    ):
                        for entry in statement.value.elts:
                            if isinstance(entry, ast.Name):
                                exported.add(entry.id)
                                if entry.id not in registrations.values():
                                    unresolved.add(entry.id)
        unresolved.difference_update(registrations.values())
        if exported:
            registrations = {name: cls for name, cls in registrations.items() if cls in exported}
            unresolved.intersection_update(exported)
        for class_name in unresolved:
            registrations.setdefault(class_name, class_name)
        # Helper changes affect every consumer, but adding an unrelated node does not.
        context = []
        for item in tree.body:
            if isinstance(item, ast.ClassDef) and (
                item.name in registrations.values()
                or any(
                    isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and method.name == "get_node_list"
                    for method in item.body
                )
            ):
                continue
            if isinstance(item, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS"
                for target in item.targets
            ):
                continue
            context.append(ast.dump(item))
        for name, class_name in registrations.items():
            if names is not None and name not in names:
                continue
            if class_name not in classes:
                raise ValueError(f"{path}: registered class {class_name} is not defined locally")
            dependencies: set[str] = set()
            pending = [class_name]
            while pending:
                dependency = pending.pop()
                if dependency in dependencies or dependency not in classes:
                    continue
                dependencies.add(dependency)
                pending.extend(
                    entry.id
                    for entry in ast.walk(classes[dependency])
                    if isinstance(entry, ast.Name) and isinstance(entry.ctx, ast.Load)
                )
            node = UpstreamNode(
                name,
                class_name,
                path,
                hashlib.sha256(
                    "\n".join(
                        [*context, *(ast.dump(classes[key]) for key in sorted(dependencies))]
                    ).encode()
                ).hexdigest(),
                ast.get_source_segment(source, classes[class_name]) or "",
                path.startswith("comfy_api_nodes/"),
                "Dynamic or inherited schema requires review of the runtime node ID"
                if class_name in unresolved
                else "",
            )
            if name in result:
                raise ValueError(
                    f"duplicate upstream node ID {name}: {path} and {result[name].path}"
                )
            result[name] = node
    if not result and names is None:
        raise ValueError("upstream snapshot has no registered nodes")
    return result


def pack_owners(
    root: Path,
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, list[dict[str, str]]]]:
    """Index original replacement IDs and upstream filenames already used by packs."""
    owners: dict[str, set[str]] = {}
    sources: dict[str, str] = {}
    targets: dict[str, list[dict[str, str]]] = {}
    for manifest in sorted((root / "packages").glob("*/dinkster-pack.toml")):
        name = tomllib.loads(manifest.read_text())["pack"]["name"]
        sources[name] = str(manifest.parent.relative_to(root))
        for path in (manifest.parent / "src").rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                if isinstance(call.func, ast.Name) and call.func.id == "with_identity_replacement":
                    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
                        source = call.args[1].value
                        if isinstance(source, str):
                            owners.setdefault(source, set()).add(name)
                            targets.setdefault(source, []).append(
                                {
                                    "pack": name,
                                    "path": str(path.relative_to(root)),
                                }
                            )
    return owners, sources, targets


def route(node: UpstreamNode, owners: dict[str, set[str]], before: dict[str, UpstreamNode]) -> str:
    existing = owners.get(node.name, set())
    if len(existing) == 1:
        return next(iter(existing))
    if node.api:
        return "dinkster-nodes-remote"
    for family in (
        "minimax",
        "hunyuan",
        "hidream",
        "seedvr",
        "cosmos",
        "qwen",
        "flux",
        "ltx",
        "wan",
        "void",
    ):
        if node.class_name.lower().startswith(family):
            return f"dinkster-{family}"
    siblings = {
        owner
        for name, previous in before.items()
        if previous.path == node.path
        for owner in owners.get(name, set())
    }
    if len(siblings) == 1:
        return next(iter(siblings))
    # Model-family names take precedence over a general modality in the filename.
    family = re.search(
        r"(?:^|_)(minimax|wan|flux|qwen|ltx|hunyuan|cosmos|hidream|seedvr|void)(?:_|\.|$)",
        Path(node.path).name,
    )
    if family:
        return f"dinkster-{family[1]}"
    for token, pack in (
        ("audio", "audio"),
        ("video", "video"),
        ("image", "image"),
        ("mask", "image"),
        ("latent", "latent"),
        ("conditioning", "conditioning"),
        ("sampl", "sampling"),
        ("upscale", "upscale"),
        ("text", "text"),
        ("3d", "3d"),
        ("mesh", "3d"),
        ("splat", "3d"),
    ):
        if token in node.path.lower():
            return f"dinkster-{pack}"
    return "dinkster-utilities"


def inspect_changes(root: Path, inference: Path, base: str, head: str) -> dict[str, object]:
    before = snapshot(inference, base)
    target = inference_upstream(inference, head)
    if subprocess.run(
        ["git", "-C", str(inference), "merge-base", "--is-ancestor", base, target],
        check=False,
        capture_output=True,
    ).returncode:
        raise ValueError("intake baseline is not an ancestor of the inference upstream point")
    after = snapshot(inference, target)
    owners, sources, targets = pack_owners(root)
    rows = []
    for name, node in sorted(after.items()):
        old = before.get(name)
        if old and old.digest == node.digest and old.class_name == node.class_name:
            continue
        pack = route(node, owners, before)
        if pack not in sources:
            raise ValueError(f"{name}: destination pack {pack} is absent")
        existing = sorted(owners.get(name, set()))
        manifest = tomllib.loads((root / sources[pack] / "dinkster-pack.toml").read_text())["pack"]
        entry = manifest.get("entry", {}).get("nodes", "")
        reason = node.blocker
        if node.api:
            reason = "Partner API node needs the remote gateway catalog and credential contract"
        elif node.path in {"comfy_extras/nodes_train.py", "comfy_extras/nodes_dataset.py"}:
            reason = "Training-origin node is excluded by the supported engine contract"
        elif existing:
            reason = "Update the existing native replacement; do not register a duplicate node"
        elif not entry:
            reason = "Destination pack needs a native static node entry point"
        proposal = None
        if not reason:
            proposal = translate(
                name, node.class_name, node.source, manifest["namespaces"][0], entry.split(":")[0]
            )
            reason = proposal.reason
        rows.append(
            {
                **asdict(node),
                "change": "changed" if old else "added",
                "pack": pack,
                "package": sources[pack],
                "mechanical": not reason,
                "reason": reason,
                "native_source": proposal.source if proposal and not reason else "",
                "load_test": proposal.test if proposal and not reason else "",
                "replacements": targets.get(name, []),
                "consolidation": (
                    "Update the existing native schema and execute method in "
                    + ", ".join(sorted({entry["path"] for entry in targets[name]}))
                    + "; preserve its node ID and replacement mapping."
                    if existing
                    else "No identity replacement exists; add the node to this pack. "
                    "Review related schemas before folding it into a consolidated node."
                ),
            }
        )
    return {
        "baseline": base,
        "upstream": target,
        "inference": head,
        "nodes": rows,
        "removed": sorted(before.keys() - after.keys()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--inference-ref", default="origin/main")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    head = git(args.inference, "rev-parse", "--verify", args.inference_ref + "^{commit}").strip()
    result = inspect_changes(args.root, args.inference, args.baseline, head)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n")
    print(f"{len(result['nodes'])} added or changed nodes at {result['upstream']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read every first-party declaration with today's engine and say how each
old construct becomes a row of the node body tree.

Every finding names the row of the plan's table "Each old construct"
(section 13, "The one-time rewrite of the first-party nodes") that covers
it, and whether a script rewrites it or a person does. A field the table
does not name is a stop: the rewrite asks the design thread before it
writes that node.

    python -m tools.rewrite_declarations.read OUT_JSON

Runs with the engine the old declarations were written for, because the
new engine no longer imports them.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.node_interface import ROOT, load_nodes
from tools.rewrite_declarations import sites

ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Fields that keep their name and meaning in the new declaration.
INPUT_UNCHANGED = {
    "id",
    "type",
    "required",
    "default",
    "on_absent",
    "widget",
    "hidden",
    "lazy",
    "source_filename",
    "accepts_storage",
    "accepts_stream",
    "accepts_list",
    "alpha_policy",
    "mask_polarity",
    "mask_semantic",
}
INPUT_RENAMED = {"doc": "tooltip", "display_name": "label", "force_input": "pin_first"}
OUTPUT_UNCHANGED = {
    "id",
    "type",
    "optional",
    "preview",
    "represents",
    "known_value",
    "alpha_policy",
    "mask_polarity",
    "mask_semantic",
}
OUTPUT_RENAMED = {"doc": "tooltip", "display_name": "label"}
COMBO_FIELDS = {"id", "options", "default", "required", "hidden", "doc", "display_name"}
FAMILY_FIELDS = {
    "id",
    "template",
    "min_members",
    "max_members",
    "doc",
    "display_name",
    "required",
    "member_prefix",
    "member_names",
    "independent_types",
    "optional_after_minimum",
}
# NodeSchema fields that the body replaces, and fields with no row in the table.
SCHEMA_BODY = {"inputs", "input_families", "combos", "input_order"}
SCHEMA_STOP = {"output_families", "slots", "output_descriptors", "widget_groups", "aliases"}

SAVE_VIDEO = "dinkster.video.save_video"


@dataclasses.dataclass
class Finding:
    path: str
    construct: str
    becomes: str
    by: str  # "script", "hand" or "stop"


def non_default(spec: Any) -> dict[str, Any]:
    return {
        f.name: getattr(spec, f.name)
        for f in dataclasses.fields(spec)
        if getattr(spec, f.name) != f.default
    }


def _widget_findings(path: str, widget: Any, out: list[Finding]) -> None:
    if widget is None:
        return
    name = type(widget).__name__
    if name == "WidgetRepresentations":
        out.append(Finding(path, "widget representations", "", "stop"))
        return
    if name == "CustomWidgetDescriptor":
        out.append(
            Finding(path, "custom widget", "per input, table of inputs that are not inputs", "hand")
        )
        return
    if name in ("CompositorWidget", "CurveWidget"):
        out.append(Finding(path, name, "per input, table of inputs that are not inputs", "hand"))
        return
    fields = non_default(widget)
    if fields.get("control_after_generate") is not None:
        out.append(Finding(path, "control_after_generate", "Input.after_run", "script"))
    if fields.get("remote_route"):
        out.append(Finding(path, "remote_route", "options=NamedList(...)", "script"))
    if fields.get("option_source") is not None:
        out.append(Finding(path, "option_source", "", "stop"))


ROW_SITES: dict[str, str] = {}


def read_entries(entries: Any, prefix: str, scope: str, out: list[Finding]) -> None:
    for entry in entries:
        name = type(entry).__name__
        path = prefix + entry.id
        ROW_SITES.setdefault(path, sites.site(entry))
        if not ID.fullmatch(entry.id) and not (name == "InputSpec" and "." in entry.id):
            out.append(Finding(path, f"id {entry.id!r}", "", "stop"))
        if name == "InputSpec":
            read_input(entry, path, scope, out)
        elif name == "InputFamilySpec":
            read_family(entry, path, out)
        elif name == "DynamicComboSpec":
            read_combo(entry, path, out)
        else:
            out.append(Finding(path, name, "", "stop"))


def read_input(spec: Any, path: str, scope: str, out: list[Finding]) -> None:
    fields = non_default(spec)
    if "." in spec.id:
        out.append(Finding(path, "id with dots", "one Group(value=...) with Named keys", "hand"))
    for field, value in fields.items():
        if field in INPUT_UNCHANGED:
            continue
        if field in INPUT_RENAMED:
            out.append(Finding(path, field, INPUT_RENAMED[field], "script"))
        elif field == "advanced":
            where = "the node" if scope == "body" else "that option"
            becomes = (
                f'Section("advanced", label="Advanced", collapsed=True) as the last row of {where}'
            )
            out.append(Finding(path, f"advanced in {scope}", becomes, "script"))
        elif field == "adjacent_to":
            if spec.id == "mount" and value == "filename_prefix":
                out.append(
                    Finding(
                        path,
                        "adjacent_to",
                        "Row(filename_prefix, mount), mount pin=False",
                        "script",
                    )
                )
            else:
                out.append(Finding(path, f"adjacent_to {value!r} on {spec.id!r}", "", "stop"))
        elif field == "socketless":
            out.append(
                Finding(
                    path, "socketless", "per input, table of inputs that are not inputs", "hand"
                )
            )
        else:
            out.append(Finding(path, f"input field {field}", "", "stop"))
    _widget_findings(path, spec.widget, out)


def read_family(family: Any, path: str, out: list[Finding]) -> None:
    fields = non_default(family)
    for field in fields:
        if field not in FAMILY_FIELDS:
            out.append(Finding(path, f"family field {field}", "", "stop"))
    template = family.template
    if len(template) == 1 and type(template[0]).__name__ == "InputSpec":
        out.append(Finding(path, "InputFamilySpec, one row", "Group(value=...)", "script"))
    else:
        out.append(Finding(path, "InputFamilySpec, several rows", "Group(member=...)", "script"))
    if family.member_names:
        out.append(Finding(path, "family naming: names", "Named((...))", "script"))
    elif family.member_prefix:
        out.append(
            Finding(path, "family naming: prefix", f"Numbered({family.member_prefix!r})", "script")
        )
    else:
        out.append(Finding(path, "family naming: bare numbers", 'Numbered("", start=0)', "script"))
    if "required" in fields:
        out.append(Finding(path, "family required=False", "removed (Gone)", "script"))
    if "display_name" in fields or "doc" in fields:
        out.append(Finding(path, "family label or tooltip", "label, tooltip", "script"))
    read_entries(template, path + ".<member>.", "member", out)


def read_combo(combo: Any, path: str, out: list[Finding]) -> None:
    fields = non_default(combo)
    for field in fields:
        if field not in COMBO_FIELDS:
            out.append(Finding(path, f"choice field {field}", "", "stop"))
    out.append(Finding(path, "DynamicComboSpec", "Choice", "script"))
    if combo.hidden:
        if path == "codec":
            out.append(
                Finding(
                    path,
                    "Save Video's hidden second codec",
                    "deleted with its execute branch",
                    "hand",
                )
            )
        else:
            out.append(Finding(path, "hidden choice", "", "stop"))
    if not combo.required:
        out.append(Finding(path, "choice required=False", 'Choice with "none"', "script"))
    by_id: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for option in combo.options:
        for entry in option.inputs:
            by_id[entry.id].append((option.key, entry))
        read_entries(option.inputs, f"{path}.", "option", out)
    for row_id, copies in by_id.items():
        if len(copies) < 2:
            continue
        shared = f"{path}.{row_id}"
        first = copies[0][1]
        if all(c == first for _, c in copies):
            out.append(
                Finding(shared, "row under several options, identical", "shared and Use", "script")
            )
            continue
        kinds = {type(c).__name__ for _, c in copies}
        if kinds != {"InputSpec"}:
            # A shared choice whose options differ by parent option.
            out.append(
                Finding(
                    shared,
                    "shared choice that differs by parent option",
                    "shared choice and Use(only=...)",
                    "hand",
                )
            )
            continue
        differing = {
            f.name
            for f in dataclasses.fields(first)
            if any(getattr(c, f.name) != getattr(first, f.name) for _, c in copies)
        }
        # Use may override label, tooltip, widget and default, and a row that is
        # advanced in some options only is placed inside or outside the section.
        # The plan's table leaves every differing copy to a person.
        overridable = {"display_name", "doc", "widget", "default", "advanced"}
        by = "hand" if differing <= overridable else "stop"
        detail = ", ".join(sorted(differing))
        out.append(
            Finding(
                shared,
                f"row under several options, differs in {detail}",
                "shared and Use with overrides",
                by,
            )
        )


def read_schema(schema: Any) -> list[Finding]:
    out: list[Finding] = []
    ROW_SITES.clear()
    fields = non_default(schema)
    for field in SCHEMA_STOP & set(fields):
        out.append(Finding("", f"schema field {field}", "", "stop"))
    if "input_order" in fields:
        out.append(Finding("", "input_order", "order as written", "script"))
    if schema.editor_role:
        out.append(Finding("", "editor_role", "unchanged", "script"))
    read_entries(schema.input_entries, "", "body", out)
    for output in schema.outputs:
        ROW_SITES.setdefault(f"output {output.id}", sites.site(output))
        for field in non_default(output):
            if field in OUTPUT_RENAMED:
                out.append(
                    Finding(
                        f"output {output.id}", f"output {field}", OUTPUT_RENAMED[field], "script"
                    )
                )
            elif field not in OUTPUT_UNCHANGED:
                out.append(Finding(f"output {output.id}", f"output field {field}", "", "stop"))
    return out


def unregistered(nodes: dict[str, tuple[str, type]]) -> list[type]:
    """Classes in pack source with define_schema that no pack registers."""
    registered = set()
    for _, cls in nodes.values():
        for base in cls.__mro__:
            registered.add((base.__module__, base.__name__))
    found = []
    for path in sorted(ROOT.glob("packages/*/src/**/*.py")):
        parts = path.relative_to(ROOT).parts
        module = ".".join(parts[parts.index("src") + 1 :]).removesuffix(".py")
        module = module.removesuffix(".__init__")
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef) or (module, node.name) in registered:
                continue
            if any(isinstance(b, ast.FunctionDef) and b.name == "define_schema" for b in node.body):
                cls = getattr(importlib.import_module(module), node.name, None)
                if inspect.isclass(cls):
                    found.append(cls)
                else:
                    print(f"not importable at module level: {module}.{node.name}")
    return found


def main(argv: list[str]) -> int:
    sites.record()
    nodes = load_nodes()
    for cls in unregistered(nodes):
        nodes[f"unregistered {cls.__module__}.{cls.__qualname__}"] = ("", cls)
    report: dict[str, Any] = {}
    construct_nodes: dict[str, set[str]] = defaultdict(set)
    construct_count: Counter[str] = Counter()
    for node_type, (pack, cls) in sorted(nodes.items()):
        findings = read_schema(cls.schema())
        report[node_type] = {
            "pack": pack,
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "findings": [dataclasses.asdict(f) for f in findings],
            "rows": dict(ROW_SITES),
            "by": sorted({f.by for f in findings}),
        }
        for f in findings:
            key = f"[{f.by}] {f.construct}"
            construct_nodes[key].add(node_type)
            construct_count[key] += 1
    Path(argv[0]).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"node types read: {sum(not t.startswith('unregistered ') for t in report)}")
    print(f"unregistered classes read: {sum(t.startswith('unregistered ') for t in report)}")
    for by in ("script", "hand", "stop"):
        print(f"node types with a {by} finding: {sum(by in r['by'] for r in report.values())}")
    print()
    for key in sorted(construct_nodes):
        print(f"{key}: {len(construct_nodes[key])} node types, {construct_count[key]} instances")
    print()
    for key in sorted(k for k in construct_nodes if not k.startswith("[script]")):
        print(key)
        for node_type in sorted(construct_nodes[key]):
            paths = [
                f["path"]
                for f in report[node_type]["findings"]
                if f"[{f['by']}] {f['construct']}" == key
            ]
            print(f"   {node_type}: {', '.join(paths)[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

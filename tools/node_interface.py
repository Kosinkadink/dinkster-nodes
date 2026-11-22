"""Snapshot every first-party node's interface and compare two snapshots.

The rewrite of the first-party nodes to the node body tree must keep what
each node takes, gives and shows. A snapshot records, for each registered
node type:

- ``assignments``: for every choice assignment (each option of each choice,
  nested ones included, other required choices on their default or first
  option) the flattened input ids with type, required mark and default, and
  the flattened outputs. Groups get two members, or ``min_members``.
- ``rows``: the input rows in the order the editor shows them (rows marked
  advanced after the others, a row placed beside another right after it),
  with every layout field: label, tooltip, widget, pin, hidden, advanced,
  beside. Choices and groups list their own rows the same way.
- ``outputs``: each output's id, type and layout fields.
- ``node``: the remaining node-level fields (title, category, search terms,
  replacement rules and the rest). A replacement case's input mappings are
  listed by input id, because their order means nothing.

The order of the flattened inputs is not compared: the body tree flattens
each scope in one order by id, so the order a person sees is the order of
``rows``.

    python tools/node_interface.py dump OUT_JSON
    python tools/node_interface.py compare BEFORE_JSON AFTER_JSON DIFFERENCES_JSON

``compare`` prints each difference not on the written list, and each listed
difference that did not happen, and exits 1 when there is either. The list
(``tools/data/node_interface_differences.json``) gives each difference its
reason; its ``notes`` hold intended differences a snapshot cannot see, such as
ids the editor makes up or text it formats, and are not compared.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
import tomllib
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# NodeSchema fields that the input and output listings already cover.
_LISTED = {
    "inputs",
    "outputs",
    "input_families",
    "output_families",
    "combos",
    "slots",
    "slot_choices",
    "input_order",
}


def load_nodes(root: Path = ROOT) -> dict[str, tuple[str, type]]:
    """Every registered node class by node type, with its pack name."""
    found: dict[str, tuple[str, type]] = {}
    for manifest in sorted(root.glob("packages/*/dinkster-pack.toml")):
        pack = tomllib.loads(manifest.read_text())["pack"]
        module_name, _, attr = pack["entry"]["nodes"].partition(":")
        for cls in getattr(importlib.import_module(module_name), attr):
            found[cls.schema().node_type] = (pack["name"], cls)
    return found


def type_text(expr: Any) -> str:
    """A short text for a type that does not depend on its Python repr."""
    kind = expr.kind
    if kind == "concrete":
        return expr.types[0]
    if kind == "union":
        return "|".join(expr.types)
    if kind == "variable":
        allowed = f"<{'|'.join(expr.types)}>" if expr.types else ""
        return f"${expr.template_id}{allowed}"
    parts = [kind]
    if expr.types:
        parts.append("|".join(expr.types))
    if expr.template_id:
        parts.append("$" + expr.template_id)
    if expr.element is not None:
        parts.append(type_text(expr.element))
    if expr.elements:
        parts.append(", ".join(type_text(item) for item in expr.elements))
    if expr.input_id:
        parts.append("@" + expr.input_id)
    if expr.index is not None:
        parts.append(f"#{expr.index}")
    return f"{parts[0]}[{'; '.join(parts[1:])}]"


def plain(value: Any) -> Any:
    """JSON data for a schema value: dataclasses by class name and fields."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if type(value).__name__ == "TypeExpr":
            return type_text(value)
        fields = {
            f.name: plain(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if getattr(value, f.name) != f.default
        }
        return {"class": type(value).__name__, **fields}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, frozenset, set)):
        items = [plain(item) for item in value]
        return sorted(items, key=json.dumps) if isinstance(value, (set, frozenset)) else items
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def shown_order(entries: Iterable[Any]) -> list[Any]:
    """Rows in the order the editor shows them today."""
    entries = list(entries)
    beside: dict[str, list[Any]] = {}
    for entry in entries:
        anchor = getattr(entry, "adjacent_to", None)
        if anchor:
            beside.setdefault(anchor, []).append(entry)

    def place(group: list[Any]) -> list[Any]:
        out: list[Any] = []
        for entry in group:
            if getattr(entry, "adjacent_to", None):
                continue
            out.append(entry)
            out.extend(beside.get(entry.id, ()))
        return out

    plain_rows = [e for e in entries if not getattr(e, "advanced", False)]
    advanced = [e for e in entries if getattr(e, "advanced", False)]
    return place(plain_rows) + place(advanced)


def row(entry: Any) -> dict[str, Any]:
    name = type(entry).__name__
    if name == "InputSpec":
        fields = plain(entry)
        fields.pop("class")
        fields["type"] = type_text(entry.type)
        return {"row": "input", **fields}
    if name == "InputFamilySpec":
        fields = {
            f.name: plain(getattr(entry, f.name))
            for f in dataclasses.fields(entry)
            # A group's required mark is not part of the flattened interface.
            if f.name not in ("template", "required") and getattr(entry, f.name) != f.default
        }
        return {"row": "group", **fields, "rows": [row(e) for e in shown_order(entry.template)]}
    if name == "DynamicComboSpec":
        fields = {
            f.name: plain(getattr(entry, f.name))
            for f in dataclasses.fields(entry)
            if f.name != "options" and getattr(entry, f.name) != f.default
        }
        options = [
            {"key": o.key, "rows": [row(e) for e in shown_order(o.inputs)]} for o in entry.options
        ]
        return {"row": "choice", **fields, "options": options}
    if name == "DynamicSlotSpec":
        fields = {
            f.name: plain(getattr(entry, f.name))
            for f in dataclasses.fields(entry)
            if f.name not in ("inputs", "variants") and getattr(entry, f.name) != f.default
        }
        variants = [
            {"key": v.key, "rows": [row(e) for e in shown_order(v.inputs)]}
            for v in entry.variants or ()
        ]
        return {
            "row": "type_choice",
            **fields,
            "rows": [row(e) for e in shown_order(entry.inputs)],
            "variants": variants,
        }
    raise TypeError(f"no listing for {name}")


def _chains(entries: Iterable[Any], prefix: str, chosen: dict[str, str]) -> Iterator[dict]:
    entries = list(entries)
    for combo in (e for e in entries if type(e).__name__ == "DynamicComboSpec"):
        path = prefix + combo.id
        for option in combo.options:
            here = {**chosen, path: option.key}
            yield here
            yield from _chains(option.inputs, path + ".", here)
    for slot in (e for e in entries if type(e).__name__ == "DynamicSlotSpec"):
        for variant in slot.variants or ():
            yield {**chosen, prefix + slot.id: variant.key}


def _complete(entries: Iterable[Any], prefix: str, chosen: dict[str, str]) -> dict[str, str]:
    for entry in entries:
        name = type(entry).__name__
        if name == "DynamicSlotSpec" and entry.variants and entry.required:
            chosen.setdefault(prefix + entry.id, entry.variants[0].key)
        if name != "DynamicComboSpec":
            continue
        path = prefix + entry.id
        if path not in chosen:
            if not entry.required:
                continue
            chosen[path] = entry.default or entry.options[0].key
        _complete(entry.option(chosen[path]).inputs, path + ".", chosen)
    return chosen


def _members(schema: Any) -> list[str]:
    stored = []
    for family in schema.input_families:
        count = max(2, family.min_members)
        if family.max_members is not None:
            count = min(count, family.max_members)
        for index in range(count):
            if family.member_names:
                key = family.member_names[index]
            else:
                key = f"{family.member_prefix or ''}{index}"
            if len(family.template) == 1 and type(family.template[0]).__name__ == "InputSpec":
                stored.append(f"{family.id}.{key}")
            else:
                stored.extend(f"{family.id}.{key}.{r.id}" for r in family.template)
    return stored


def _sorted_predicate(predicate: Any) -> Any:
    """A predicate with the operands of each "all" and "any" in one order."""
    if not isinstance(predicate, dict) or "of" not in predicate:
        return predicate
    operands = [_sorted_predicate(item) for item in predicate["of"]]
    if predicate.get("kind") in ("all", "any"):
        operands.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return {**predicate, "of": operands}


def sort_cases(node: dict[str, Any]) -> None:
    """List replacement mappings in one order.

    A case's input mappings and chosen variants are mappings, so their order
    means nothing, and neither does the order of an "all" or "any" predicate's
    operands. An identity rule makes one case per combination of choices,
    so its conditional cases are disjoint and are compared as a set."""
    for rule in node.get("replacements", ()):
        for case in rule.get("cases", ()):
            for key in ("inputs", "slot_variants"):
                if key in case:
                    case[key] = sorted(case[key], key=json.dumps)
            if "when" in case:
                case["when"] = _sorted_predicate(case["when"])
        cases = rule.get("cases", [])
        # Conditional cases before one closing fallback cannot overlap in an
        # identity rule; the fallback stays last, where its position matters.
        if rule.get("migration") is None and cases and "when" not in cases[-1]:
            if all("when" in case for case in cases[:-1]):
                ordered = sorted(cases[:-1], key=lambda case: json.dumps(case, sort_keys=True))
                rule["cases"] = [*ordered, cases[-1]]


def snapshot_node(schema: Any) -> dict[str, Any]:
    from dinkster_schema import elaborate

    if getattr(schema, "body", ()):
        from tools.node_interface_body import snapshot_body

        return snapshot_body(schema)

    stored: dict[str, object] = {i.id: i.default for i in schema.inputs if i.default is not None}
    stored.update({key: None for key in _members(schema)})
    assignments = {}
    for chosen in [{}, *_chains(schema.input_entries, "", {})]:
        full = _complete(schema.input_entries, "", dict(chosen))
        key = json.dumps(full, sort_keys=True)
        try:
            effective = elaborate(schema, stored, None, full or None)
        except Exception as exc:  # noqa: BLE001 - a refusal is part of the interface
            assignments[key] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        assignments[key] = {
            "inputs": [
                [i.id, type_text(i.type), i.required, repr(i.default)] for i in effective.inputs
            ],
            "outputs": [[o.id, type_text(o.type)] for o in effective.outputs],
        }
    node = {
        f.name: plain(getattr(schema, f.name))
        for f in dataclasses.fields(schema)
        if f.name not in _LISTED and getattr(schema, f.name) != f.default
    }
    sort_cases(node)
    outputs = []
    for output in schema.outputs:
        fields = plain(output)
        fields.pop("class")
        fields["type"] = type_text(output.type)
        outputs.append(fields)
    return {
        "assignments": assignments,
        "rows": [row(e) for e in shown_order(schema.input_entries)],
        "outputs": outputs,
        "output_families": plain(schema.output_families),
        "node": node,
    }


def dump(root: Path = ROOT) -> dict[str, Any]:
    nodes = load_nodes(root)
    return {
        "nodes": {
            node_type: {"pack": pack, **snapshot_node(cls.schema())}
            for node_type, (pack, cls) in sorted(nodes.items())
        }
    }


def differences(before: Mapping[str, Any], after: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Each place the two snapshots differ, as node type, part, path and both values."""
    old, new = before["nodes"], after["nodes"]
    for node_type in sorted(set(old) | set(new)):
        if node_type not in new or node_type not in old:
            yield {
                "node_type": node_type,
                "part": "node",
                "path": "",
                "before": node_type in old,
                "after": node_type in new,
            }
            continue
        a, b = old[node_type], new[node_type]
        for key in sorted(set(a["assignments"]) | set(b["assignments"])):
            left, right = a["assignments"].get(key), b["assignments"].get(key)
            if left is None or right is None or "error" in left or "error" in right:
                if left != right:
                    yield {
                        "node_type": node_type,
                        "part": "assignment",
                        "path": key,
                        "before": left,
                        "after": right,
                    }
                continue
            for side in ("inputs", "outputs"):
                lrows = {r[0]: r[1:] for r in left[side]}
                rrows = {r[0]: r[1:] for r in right[side]}
                for ident in sorted(set(lrows) | set(rrows)):
                    if lrows.get(ident) != rrows.get(ident):
                        yield {
                            "node_type": node_type,
                            "part": side,
                            "path": f"{key} {ident}",
                            "before": lrows.get(ident),
                            "after": rrows.get(ident),
                        }
        for part in ("rows", "outputs", "output_families", "node"):
            yield from _walk(node_type, part, "", a[part], b[part])


def _walk(node_type: str, part: str, path: str, a: Any, b: Any) -> Iterator[dict[str, Any]]:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            yield from _walk(node_type, part, f"{path}/{key}", a.get(key), b.get(key))
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            label = x.get("id", x.get("key", index)) if isinstance(x, dict) else index
            yield from _walk(node_type, part, f"{path}/{label}", x, y)
    elif a != b:
        yield {"node_type": node_type, "part": part, "path": path, "before": a, "after": b}


def compare(before: Mapping, after: Mapping, listed: Iterable[Mapping]) -> list[str]:
    """Problems: differences not on the written list, and listed ones that did not happen."""
    wanted = {(d["node_type"], d["part"], d["path"]): d for d in listed}
    seen: set[tuple[str, str, str]] = set()
    problems = []
    for diff in differences(before, after):
        key = (diff["node_type"], diff["part"], diff["path"])
        expected = wanted.get(key)
        if expected is None or expected.get("after", diff["after"]) != diff["after"]:
            problems.append("not listed: " + json.dumps(diff, sort_keys=True))
        seen.add(key)
    for key in sorted(set(wanted) - seen):
        problems.append("listed but not found: " + json.dumps(wanted[key], sort_keys=True))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    dump_parser = sub.add_parser("dump")
    dump_parser.add_argument("out", type=Path)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("before", type=Path)
    compare_parser.add_argument("after", type=Path)
    compare_parser.add_argument("differences", type=Path)
    args = parser.parse_args(argv)
    if args.command == "dump":
        args.out.write_text(json.dumps(dump(), indent=1, sort_keys=True) + "\n")
        return 0
    listed = json.loads(args.differences.read_text())["differences"]
    problems = compare(
        json.loads(args.before.read_text()), json.loads(args.after.read_text()), listed
    )
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s)", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The interface snapshot of a node declared as a body tree, in the shape
``node_interface.py`` writes for the old declaration, so the two compare.

Each row is listed as the old declaration would have said it: ``tooltip``
as ``doc``, ``label`` as ``display_name``, ``pin_first`` as ``force_input``,
no pin as ``socketless``, ``after_run`` as the widget's
``control_after_generate``, a row in the "Advanced" section as
``advanced``, the second field of a ``Row`` as ``adjacent_to`` the first,
and a ``Use`` as the shared row with its overrides. What the old
declaration had no word for (``State``, ``Note``) is listed by its kind.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from typing import Any

from tools.node_interface import plain, sort_cases, type_text

INPUT_BACK = {"tooltip": "doc", "label": "display_name", "pin_first": "force_input"}
_SKIP_INPUT = {"pin", "after_run", "on_edit"}


def _fields(obj: Any, skip: set[str]) -> dict[str, Any]:
    return {
        f.name: plain(getattr(obj, f.name))
        for f in dataclasses.fields(obj)
        if f.name not in skip and getattr(obj, f.name) != f.default
    }


def _input(spec: Any, advanced: bool, beside: str | None) -> dict[str, Any]:
    fields = _fields(spec, _SKIP_INPUT)
    out: dict[str, Any] = {INPUT_BACK.get(k, k): v for k, v in fields.items()}
    out["type"] = type_text(spec.type)
    if not spec.pin:
        out["socketless"] = True
    if spec.after_run is not None and isinstance(out.get("widget"), dict):
        out["widget"]["control_after_generate"] = spec.after_run
    if advanced:
        out["advanced"] = True
    if beside:
        out["adjacent_to"] = beside
    return {"row": "input", **out}


def _apply_use(row: Any, use: Any) -> Any:
    changes = {}
    for name in ("label", "tooltip", "widget", "default", "required"):
        value = getattr(use, name)
        if type(value).__name__ != "_Unset":
            changes[name] = value
    if type(row).__name__ == "Choice":
        if use.only is not None:
            changes["options"] = tuple(o for o in row.options if o.key in use.only)
        if use.inside:
            inner = {u.id: u for u in use.inside}
            changes["shared"] = tuple(
                _apply_use(r, inner[r.id]) if getattr(r, "id", None) in inner else r
                for r in row.shared
            )
            changes["options"] = tuple(
                dataclasses.replace(
                    o,
                    rows=tuple(
                        _apply_use(r, inner[r.id])
                        if type(r).__name__ == "Input" and r.id in inner
                        else r
                        for r in o.rows
                    ),
                )
                for o in changes.get("options", row.options)
            )
    return dataclasses.replace(row, **changes) if changes else row


def _walk(rows: tuple, shared: dict[str, Any], advanced: bool = False) -> Iterator[dict]:
    """Rows in the order the editor shows them, the advanced section's last."""
    later: list[dict] = []
    for row in rows:
        kind = type(row).__name__
        if kind == "Section":
            target = later if row.id == "advanced" else None
            found = list(_walk(row.rows, shared, advanced or row.id == "advanced"))
            if target is None:
                yield from found
            else:
                later.extend(found)
        elif kind == "Row":
            first = None
            for item in row.rows:
                if type(item).__name__ == "Input":
                    yield _input(item, advanced, first)
                    first = first or item.id
                else:
                    yield {"row": type(item).__name__.lower(), "id": item.id}
        elif kind == "Use":
            yield from _walk((_apply_use(shared[row.id], row),), shared, advanced)
        else:
            yield _row(row, shared, advanced)
    yield from later


def _row(row: Any, shared: dict[str, Any], advanced: bool) -> dict[str, Any]:
    kind = type(row).__name__
    if kind == "Input":
        return _input(row, advanced, None)
    if kind == "Group":
        fields = _fields(row, {"member", "value", "keys", "grows"})
        out = {INPUT_BACK.get(k, k): v for k, v in fields.items()}
        keys = row.keys
        if type(keys).__name__ == "Named":
            out["member_names"] = list(keys.names)
            out.setdefault("max_members", len(keys.names))
        elif keys is not None and keys.prefix:
            out["member_prefix"] = keys.prefix
        rows = (row.value,) if row.value is not None else row.member
        if advanced:
            out["advanced_group"] = True
        return {"row": "group", **out, "rows": list(_walk(rows, {}))}
    if kind == "Choice":
        fields = _fields(row, {"options", "shared", "leader", "on_edit"})
        out = {INPUT_BACK.get(k, k): v for k, v in fields.items()}
        inner = {**shared, **{r.id: r for r in row.shared if hasattr(r, "id")}}
        options = [{"key": o.key, "rows": list(_walk(o.rows, inner))} for o in row.options]
        if advanced:
            out["advanced_choice"] = True
        return {"row": "choice", **out, "options": options}
    if kind == "TypeChoice":
        fields = _fields(row, {"rows", "variants"})
        variants = [{"key": v.key, "rows": list(_walk(v.rows, {}))} for v in row.variants or ()]
        return {
            "row": "type_choice",
            **fields,
            "rows": list(_walk(row.rows, {})),
            "variants": variants,
        }
    return {"row": kind.lower(), **_fields(row, set())}


def _choices(rows: tuple, prefix: str, chosen: dict[str, str]) -> Iterator[dict]:
    from dinkster_schema.body import scope_rows

    for row in scope_rows(rows):
        kind = type(row).__name__
        if kind == "Choice":
            path = prefix + row.id
            for option in row.options:
                here = {**chosen, path: option.key}
                yield here
                yield from _choices(_option_rows(row, option), path + ".", here)
        elif kind == "TypeChoice":
            for variant in row.variants or ():
                yield {**chosen, prefix + row.id: variant.key}


def _option_rows(choice: Any, option: Any) -> tuple:
    shared = {r.id: r for r in choice.shared if hasattr(r, "id")}
    rows = []
    for row in option.rows:
        rows.append(_apply_use(shared[row.id], row) if type(row).__name__ == "Use" else row)
    return tuple(rows)


def _complete(rows: tuple, prefix: str, chosen: dict[str, str]) -> dict[str, str]:
    from dinkster_schema.body import scope_rows

    for row in scope_rows(rows):
        kind = type(row).__name__
        if kind == "TypeChoice" and row.variants and row.required:
            chosen.setdefault(prefix + row.id, row.variants[0].key)
        if kind != "Choice":
            continue
        path = prefix + row.id
        if path not in chosen:
            if not row.required:
                continue
            chosen[path] = row.default or row.options[0].key
        option = next(o for o in row.options if o.key == chosen[path])
        _complete(_option_rows(row, option), path + ".", chosen)
    return chosen


def _members(body: tuple) -> list[str]:
    from dinkster_schema.body import scope_rows

    stored = []
    for group in (r for r in scope_rows(body) if type(r).__name__ == "Group"):
        count = max(2, group.min_members)
        keys = group.keys
        if type(keys).__name__ == "Named":
            count = min(count, len(keys.names))
        elif group.max_members is not None:
            count = min(count, group.max_members)
        for index in range(count):
            key = keys.names[index] if type(keys).__name__ == "Named" else keys.key(index)
            if group.value is not None:
                stored.append(f"{group.id}.{key}")
            else:
                stored += [
                    f"{group.id}.{key}.{r.id}"
                    for r in scope_rows(group.member)
                    if type(r).__name__ == "Input"
                ]
    return stored


def snapshot_body(schema: Any) -> dict[str, Any]:
    from dinkster_schema import elaborate
    from dinkster_schema.body import scope_rows

    top = [r for r in scope_rows(schema.body) if type(r).__name__ == "Input"]
    stored: dict[str, object] = {i.id: i.default for i in top if i.default is not None}
    stored.update({key: None for key in _members(schema.body)})
    assignments = {}
    for chosen in [{}, *_choices(schema.body, "", {})]:
        full = _complete(schema.body, "", dict(chosen))
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
    listed = {"inputs", "outputs", "body", "tables", "derived", "validate", "title_suffix", "menu"}
    node = {
        f.name: plain(getattr(schema, f.name))
        for f in dataclasses.fields(schema)
        if f.name not in listed and getattr(schema, f.name) != f.default
    }
    sort_cases(node)
    body_inputs = [r for r in schema.body if type(r).__name__ != "Output"]
    outputs = []
    for output in (r for r in scope_rows(schema.body) if type(r).__name__ == "Output"):
        fields = _fields(output, set())
        out = {INPUT_BACK.get(k, k): v for k, v in fields.items()}
        out["type"] = type_text(output.type)
        outputs.append(out)
    return {
        "assignments": assignments,
        "rows": list(_walk(tuple(body_inputs), {})),
        "outputs": outputs,
        "output_families": [],
        "node": node,
    }

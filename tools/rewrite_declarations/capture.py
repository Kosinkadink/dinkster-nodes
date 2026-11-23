"""Record, for every ``NodeSchema(...)`` call in pack source, which old rows
each element of its row lists builds.

Runs with the engine the old declarations were written for. Each call is
found by the exact source position of the call expression, and each element
of its ``inputs``, ``input_families``, ``combos`` and ``outputs`` tuples is
evaluated again in the frame that made the call, so the rewrite knows which
rows every element stands for without guessing from counts.

    python -m tools.rewrite_declarations.capture OUT_JSON
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LISTS = ("inputs", "input_families", "combos", "outputs")
_CALLS: dict[tuple[str, int, int], ast.Call] = {}
_ANY_CALLS: dict[str, dict[tuple[int, int], ast.Call]] = {}
_SEEN: dict[str, dict[str, Any]] = {}
_FAMILIES: dict[str, dict[str, Any]] = {}


def _calls(path: Path) -> None:
    rel = str(path.relative_to(ROOT))
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "NodeSchema":
                _CALLS[(rel, node.lineno, node.col_offset)] = node


def _entry(entry: Any) -> dict[str, Any]:
    return {
        "id": entry.id,
        "kind": type(entry).__name__,
        "advanced": bool(getattr(entry, "advanced", False)),
        "adjacent_to": getattr(entry, "adjacent_to", None),
    }


def _elements(value: ast.expr) -> list[tuple[ast.expr, bool]]:
    if isinstance(value, (ast.Tuple, ast.List)):
        return [(e.value, True) if isinstance(e, ast.Starred) else (e, False) for e in value.elts]
    return [(value, True)]


def _evaluate(expr: ast.expr, frame: Any, path: str) -> Any:
    code = compile(ast.Expression(expr), path, "eval")
    # One namespace, so generator expressions in the element see the locals too.
    return eval(code, {**frame.f_globals, **frame.f_locals})  # noqa: S307 - pack source


def _call_at(frame: Any) -> tuple[str, ast.Call] | None:
    path = Path(frame.f_code.co_filename)
    if not path.is_relative_to(ROOT):
        return None
    rel = str(path.relative_to(ROOT))
    if rel not in _ANY_CALLS:
        _ANY_CALLS[rel] = {
            (n.lineno, n.col_offset): n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Call)
        }
    positions = list(frame.f_code.co_positions())[frame.f_lasti // 2]
    call = _ANY_CALLS[rel].get((positions[0], positions[2]))
    return None if call is None else (f"{rel}:{positions[0]}:{positions[2]}", call)


def _from_caller(frame: Any, name: str) -> dict[str, Any] | None:
    """The caller's argument for parameter ``name`` of the frame's function."""
    code = frame.f_code
    count = code.co_argcount + code.co_kwonlyargcount
    params = code.co_varnames[:count]
    if name not in params or frame.f_back is None:
        return None
    found = _call_at(frame.f_back)
    if found is None:
        return None
    site, call = found
    index = params.index(name)
    argument: ast.expr | None = None
    locator: str | int = name
    for keyword in call.keywords:
        if keyword.arg == name:
            argument = keyword.value
    if argument is None and index < code.co_argcount and index < len(call.args):
        argument, locator = call.args[index], index
    if argument is None or any(isinstance(a, ast.Starred) for a in call.args):
        return None
    path = site.rsplit(":", 2)[0]
    elements = []
    for expr, starred in _elements(argument):
        value = _evaluate(expr, frame.f_back, path)
        entries = list(value) if starred else [value]
        elements.append({"starred": starred, "entries": [_entry(e) for e in entries]})
    return {"site": site, "argument": locator, "elements": elements}


def _record(frame: Any, kwargs: dict[str, Any]) -> None:
    path = Path(frame.f_code.co_filename)
    if not path.is_relative_to(ROOT) or path.relative_to(ROOT).parts[0] == "tools":
        return
    positions = list(frame.f_code.co_positions())[frame.f_lasti // 2]
    key = (str(path.relative_to(ROOT)), positions[0], positions[2])
    if not _CALLS:
        for source in ROOT.glob("packages/*/src/**/*.py"):
            _calls(source)
    call = _CALLS.get(key)
    if call is None:
        return
    site = f"{key[0]}:{key[1]}:{key[2]}"
    lists: dict[str, list[dict[str, Any]]] = {}
    for keyword in call.keywords:
        if keyword.arg not in LISTS:
            continue
        found = []
        for expr, starred in _elements(keyword.value):
            value = _evaluate(expr, frame, key[0])
            entries = list(value) if starred else [value]
            element: dict[str, Any] = {
                "starred": starred,
                "entries": [_entry(e) for e in entries],
            }
            if isinstance(expr, ast.Name):
                caller = _from_caller(frame, expr.id)
                if caller is not None:
                    element["caller"] = caller
            found.append(element)
        lists[keyword.arg] = found
    observation = {
        "node_type": kwargs.get("node_type"),
        "lists": lists,
        "input_order": list(kwargs.get("input_order", ())),
        "keywords": sorted(k.arg for k in call.keywords if k.arg),
        "positional": len(call.args),
    }
    _SEEN.setdefault(site, {"observations": []})["observations"].append(observation)


def install() -> None:
    from dinkster_schema import model

    original = model.NodeSchema.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        _record(sys._getframe(1), kwargs)

    model.NodeSchema.__init__ = init  # type: ignore[method-assign]

    family_init = model.InputFamilySpec.__init__

    def family(self: Any, id: str, template: Any, *args: Any, **kwargs: Any) -> None:
        family_init(self, id, template, *args, **kwargs)
        found = _call_at(sys._getframe(1))
        if found is not None:
            _FAMILIES[found[0]] = {
                "bare_type": isinstance(template, model.TypeExpr),
                "rows": [type(row).__name__ for row in self.template],
            }

    model.InputFamilySpec.__init__ = family  # type: ignore[method-assign]


def main(argv: list[str]) -> int:
    install()
    from tools.node_interface import load_nodes
    from tools.rewrite_declarations.read import unregistered

    nodes = load_nodes()
    for cls in unregistered(nodes):
        cls.schema()
    missing = sorted(f"{k[0]}:{k[1]}" for k in _CALLS if f"{k[0]}:{k[1]}:{k[2]}" not in _SEEN)
    Path(argv[0]).write_text(
        json.dumps(
            {"sites": _SEEN, "families": _FAMILIES, "never_called": missing},
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"NodeSchema calls in pack source: {len(_CALLS)}; called: {len(_SEEN)}")
    print(f"never called while loading: {len(missing)}")
    for item in missing:
        print("   ", item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

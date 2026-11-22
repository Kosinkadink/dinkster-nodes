"""Rewrite the first-party declarations to the node body tree, in place.

Reads the capture (``capture.py``) made with the old engine, then edits the
pack sources and the tests with libcst:

- the old row classes become the new ones wherever they are built, with
  their renamed fields (``doc`` -> ``tooltip``, ``display_name`` -> ``label``,
  ``force_input`` -> ``pin_first``, ``socketless`` -> ``pin=False``,
  a widget's ``control_after_generate`` -> the input's ``after_run``);
  ``advanced`` and ``adjacent_to`` are dropped where they are written,
  because the body says them;
- each ``NodeSchema(...)`` gets one ``body=`` in the planned order: plain
  rows, ``filename_prefix`` and ``mount`` on one ``Row``, an "Advanced"
  section last, then the outputs;
- a helper that passes its caller's rows on gets them arranged at the caller.

Sites the plan stops on are listed and left for a person.

    python -m tools.rewrite_declarations.write CAPTURE_JSON
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from tools.rewrite_declarations.plan import Arrangement, Placed, Stop, plan_site

ROOT = Path(__file__).resolve().parents[2]
RENAMED = {
    "InputSpec": "Input",
    "OutputSpec": "Output",
    "DynamicComboSpec": "Choice",
    "DynamicComboOption": "Option",
    "InputFamilySpec": "Group",
    "ControlAfterGenerate": "AfterRunMode",
}
INPUT_FIELDS = {"doc": "tooltip", "display_name": "label", "force_input": "pin_first"}
OUTPUT_FIELDS = {"doc": "tooltip", "display_name": "label"}
CHOICE_POSITIONAL = ("id", "options", "default", "required", "doc", "display_name", "hidden")
FAMILY_POSITIONAL = (
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
)
DROPPED = {"advanced", "adjacent_to"}
LISTS = ("inputs", "input_families", "combos", "input_order", "outputs")
NEW_NAMES = ("Section", "Row", "Numbered", "Named", "Input", "Output", "Choice", "Option", "Group")


def _name(func: cst.BaseExpression) -> str | None:
    if isinstance(func, cst.Name):
        return func.value
    if isinstance(func, cst.Attribute):
        return func.attr.value
    return None


def _kw(name: str, value: cst.BaseExpression) -> cst.Arg:
    return cst.Arg(
        value=value,
        keyword=cst.Name(name),
        equal=cst.AssignEqual(cst.SimpleWhitespace(""), cst.SimpleWhitespace("")),
    )


def _positional_to_keywords(call: cst.Call, names: tuple[str, ...], keep: int) -> list[cst.Arg]:
    args: list[cst.Arg] = []
    for index, arg in enumerate(call.args):
        if arg.keyword is None and arg.star == "" and index >= keep:
            args.append(_kw(names[index], arg.value))
        else:
            args.append(arg)
    return args


def _rename_keywords(args: list[cst.Arg], names: dict[str, str]) -> list[cst.Arg]:
    out = []
    for arg in args:
        if arg.keyword is not None and arg.keyword.value in names:
            arg = arg.with_changes(keyword=cst.Name(names[arg.keyword.value]))
        out.append(arg)
    return out


def _tuple(elements: list[cst.BaseElement]) -> cst.Tuple:
    fixed = [
        e.with_changes(comma=cst.MaybeSentinel.DEFAULT) if i < len(elements) - 1 else e
        for i, e in enumerate(elements)
    ]
    if len(fixed) == 1 and isinstance(fixed[0], cst.Element):
        fixed[0] = fixed[0].with_changes(comma=cst.Comma())
    return cst.Tuple(elements=fixed)


def _elements(value: cst.BaseExpression) -> list[cst.BaseElement]:
    if isinstance(value, (cst.Tuple, cst.List)):
        return list(value.elements)
    return [cst.StarredElement(value=value)]


class Rewriter(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        path: str,
        sites: dict[str, tuple[Arrangement, int]],
        fragments: dict[str, list[Any]],
        families: dict[str, dict[str, Any]],
        stopped: dict[str, str],
    ) -> None:
        super().__init__()
        self.path = path
        self.pack_source = "/src/" in path
        self.sites = sites
        self.fragments = fragments
        self.families = families
        self.stopped = stopped
        self.used: set[str] = set()
        self.problems: list[str] = []

    def _site(self, node: cst.CSTNode) -> str:
        start = self.get_metadata(PositionProvider, node).start
        return f"{self.path}:{start.line}:{start.column}"

    # Names and imports -------------------------------------------------

    def leave_Name(self, original: cst.Name, updated: cst.Name) -> cst.Name:
        if updated.value in RENAMED:
            self.used.add(RENAMED[updated.value])
            return updated.with_changes(value=RENAMED[updated.value])
        return updated

    def leave_FunctionDef(
        self, original: cst.FunctionDef, updated: cst.FunctionDef
    ) -> cst.FunctionDef:
        if not self.pack_source:
            return updated
        params = updated.params
        kept = [p for p in params.kwonly_params if p.name.value not in DROPPED]
        plain = [p for p in params.params if p.name.value not in DROPPED]
        if len(kept) == len(params.kwonly_params) and len(plain) == len(params.params):
            return updated
        star = params.star_arg
        if not kept and isinstance(star, cst.ParamStar):
            star = cst.MaybeSentinel.DEFAULT
        return updated.with_changes(
            params=params.with_changes(params=plain, kwonly_params=kept, star_arg=star)
        )

    # Calls -------------------------------------------------------------

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.BaseExpression:
        name = _name(original.func)
        site = self._site(original)
        if self.pack_source or name == "InputSpec":
            args = [a for a in updated.args if not (a.keyword and a.keyword.value in DROPPED)]
        else:
            args = list(updated.args)
        if name == "replace" and len(args) == 1 and len(updated.args) > 1:
            return args[0].value
        updated = updated.with_changes(args=args)
        if site in self.fragments:
            updated = self._fragment(site, updated)
        if name == "InputSpec":
            return self._input(updated)
        if name == "OutputSpec":
            return updated.with_changes(args=_rename_keywords(list(updated.args), OUTPUT_FIELDS))
        if name == "DynamicComboSpec":
            args = _positional_to_keywords(updated, CHOICE_POSITIONAL, 1)
            return updated.with_changes(args=_rename_keywords(args, INPUT_FIELDS))
        if name == "DynamicComboOption":
            args = _positional_to_keywords(updated, ("key", "rows"), 1)
            return updated.with_changes(args=_rename_keywords(args, {"inputs": "rows"}))
        if name == "InputFamilySpec":
            return self._family(site, updated)
        if name == "NodeSchema":
            return self._schema(site, updated)
        return updated

    def _input(self, call: cst.Call) -> cst.Call:
        args = _rename_keywords(list(call.args), INPUT_FIELDS)
        out: list[cst.Arg] = []
        after_run: cst.BaseExpression | None = None
        for arg in args:
            if arg.keyword and arg.keyword.value == "socketless":
                if not (isinstance(arg.value, cst.Name) and arg.value.value == "True"):
                    self.problems.append(f"{self.path}: socketless that is not True")
                out.append(_kw("pin", cst.Name("False")))
                continue
            if arg.keyword and arg.keyword.value == "widget" and isinstance(arg.value, cst.Call):
                widget = arg.value
                inner = []
                for w in widget.args:
                    if w.keyword and w.keyword.value == "control_after_generate":
                        after_run = w.value
                    else:
                        inner.append(w)
                if after_run is not None:
                    if inner:
                        inner[-1] = inner[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
                    arg = arg.with_changes(value=widget.with_changes(args=inner))
            out.append(arg)
        if after_run is not None:
            out.append(_kw("after_run", after_run))
        return call.with_changes(args=out)

    def _family(self, site: str, call: cst.Call) -> cst.Call:
        info = self.families.get(site)
        args = _positional_to_keywords(call, FAMILY_POSITIONAL, 1)
        out: list[cst.Arg] = []
        keys: cst.BaseExpression | None = None
        for arg in args:
            key = arg.keyword.value if arg.keyword else None
            if key == "template":
                if info is None:
                    self.problems.append(f"{site}: family not seen while loading")
                    out.append(arg.with_changes(keyword=cst.Name("member")))
                elif info["bare_type"]:
                    row = cst.parse_expression('Input("value", T)')
                    assert isinstance(row, cst.Call)
                    row = row.with_changes(
                        args=[row.args[0], row.args[1].with_changes(value=arg.value)]
                    )
                    self.used.add("Input")
                    out.append(_kw("value", row))
                elif len(info["rows"]) == 1:
                    elements = _elements(arg.value)
                    if len(elements) == 1 and isinstance(elements[0], cst.Element):
                        out.append(_kw("value", elements[0].value))
                    else:
                        out.append(
                            _kw(
                                "value",
                                cst.Subscript(
                                    value=arg.value,
                                    slice=[cst.SubscriptElement(cst.Index(cst.Integer("0")))],
                                ),
                            )
                        )
                else:
                    out.append(_kw("member", arg.value))
            elif key == "member_names":
                keys = cst.Call(func=cst.Name("Named"), args=[cst.Arg(arg.value)])
                self.used.add("Named")
            elif key == "member_prefix":
                keys = cst.parse_expression("Numbered(P, start=0)")
                assert isinstance(keys, cst.Call)
                keys = keys.with_changes(
                    args=[
                        cst.Arg(
                            arg.value, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
                        ),
                        keys.args[1],
                    ]
                )
                self.used.add("Numbered")
            elif key == "required":
                continue
            else:
                out.append(arg)
        if keys is None:
            keys = cst.parse_expression('Numbered("", start=0)')
            self.used.add("Numbered")
        out.append(_kw("keys", keys))
        return call.with_changes(args=_rename_keywords(out, INPUT_FIELDS))

    def _arranged(
        self, arrangement: Arrangement, element: dict[tuple[str, int], cst.BaseElement]
    ) -> list[cst.BaseElement]:
        def one(placed: Placed) -> cst.BaseElement:
            first = element[placed.element]
            if placed.beside is None:
                return first
            second = element[placed.beside]
            assert isinstance(first, cst.Element) and isinstance(second, cst.Element)
            self.used.add("Row")
            return cst.Element(
                cst.Call(func=cst.Name("Row"), args=[_kw("rows", _tuple([first, second]))])
            )

        rows = [one(p) for p in arrangement.plain]
        if arrangement.advanced:
            section = cst.parse_expression(
                'Section("advanced", label="Advanced", collapsed=True, rows=())'
            )
            assert isinstance(section, cst.Call)
            inner = _tuple([one(p) for p in arrangement.advanced])
            args = [*section.args[:-1], section.args[-1].with_changes(value=inner)]
            rows.append(cst.Element(section.with_changes(args=args)))
            self.used.add("Section")
        return rows

    def _fragment(self, site: str, call: cst.Call) -> cst.Call:
        args = list(call.args)
        for fragment in self.fragments[site]:
            for index, arg in enumerate(args):
                hit = (arg.keyword is not None and arg.keyword.value == fragment.argument) or (
                    arg.keyword is None and index == fragment.argument
                )
                if not hit:
                    continue
                if not isinstance(arg.value, (cst.Tuple, cst.List)):
                    if fragment.arrangement.advanced or any(
                        p.beside for p in fragment.arrangement.plain
                    ):
                        self.problems.append(f"{site}: argument to arrange is not a literal")
                    continue
                elements = {("arg", i): e for i, e in enumerate(arg.value.elements)}
                rows = self._arranged(fragment.arrangement, elements)
                args[index] = arg.with_changes(value=_tuple(rows))
        return call.with_changes(args=args)

    def _schema(self, site: str, call: cst.Call) -> cst.Call:
        given = {a.keyword.value: a for a in call.args if a.keyword and a.keyword.value in LISTS}
        if not given:
            return call
        if site in self.stopped:
            self.problems.append(f"{site}: {self.stopped[site]}")
            return call
        planned = self.sites.get(site)
        if planned is None:
            if set(given) - {"inputs", "outputs"}:
                self.problems.append(f"{site}: not seen while loading, and not inputs/outputs only")
                return call
            rows = [
                *(_elements(given["inputs"].value) if "inputs" in given else []),
                *(_elements(given["outputs"].value) if "outputs" in given else []),
            ]
        else:
            arrangement, outputs = planned
            element: dict[tuple[str, int], cst.BaseElement] = {}
            for name in ("inputs", "input_families", "combos"):
                if name in given:
                    for index, e in enumerate(_elements(given[name].value)):
                        element[(name, index)] = e
            rows = self._arranged(arrangement, element)
            if "outputs" in given:
                found = _elements(given["outputs"].value)
                if len(found) != outputs:
                    self.problems.append(f"{site}: outputs changed since the capture")
                rows += found
        out: list[cst.Arg] = []
        placed = False
        for arg in call.args:
            if arg.keyword and arg.keyword.value in LISTS:
                if not placed:
                    out.append(arg.with_changes(keyword=cst.Name("body"), value=_tuple(rows)))
                    placed = True
                continue
            out.append(arg)
        return call.with_changes(args=out)


def _imports(module: cst.Module, used: set[str]) -> cst.Module:
    """Import every new name the file now uses from the pack API."""

    class Add(cst.CSTTransformer):
        done = False

        def leave_ImportFrom(
            self, original: cst.ImportFrom, updated: cst.ImportFrom
        ) -> cst.ImportFrom:
            module_name = cst.Module([]).code_for_node(updated.module) if updated.module else ""
            if module_name != "dinkster_api.v1" or isinstance(updated.names, cst.ImportStar):
                return updated
            have = {a.name.value for a in updated.names if isinstance(a.name, cst.Name)}
            missing = sorted(n for n in used if n in NEW_NAMES and n not in have)
            if not missing and len(have) == len(updated.names):
                return updated
            self.done = True
            names = list(updated.names) + [cst.ImportAlias(cst.Name(n)) for n in missing]
            names = [n.with_changes(comma=cst.Comma()) for n in names]
            seen: set[str] = set()
            unique = []
            for alias in names:
                key = alias.name.value if isinstance(alias.name, cst.Name) else ""
                if key not in seen:
                    seen.add(key)
                    unique.append(alias)
            return updated.with_changes(
                names=unique,
                lpar=updated.lpar if isinstance(updated.lpar, cst.LeftParen) else cst.LeftParen(),
                rpar=updated.rpar if isinstance(updated.rpar, cst.RightParen) else cst.RightParen(),
            )

    adder = Add()
    return module.visit(adder)


def main(argv: list[str]) -> int:
    capture = json.loads(Path(argv[0]).read_text())
    sites: dict[str, tuple[Arrangement, int]] = {}
    fragments: dict[str, list[Any]] = defaultdict(list)
    stopped: dict[str, str] = {}
    for site, data in capture["sites"].items():
        try:
            arrangement, frags, outputs = plan_site(data)
        except Stop as stop:
            stopped[site] = str(stop)
            continue
        sites[site] = (arrangement, outputs)
        for fragment in frags:
            fragments[fragment.site].append(fragment)
    files = sorted(
        {
            *ROOT.glob("packages/*/src/**/*.py"),
            *ROOT.glob("packages/*/tests/**/*.py"),
            *ROOT.glob("tests/**/*.py"),
        }
    )
    problems: list[str] = []
    changed = 0
    for path in files:
        text = path.read_text()
        if not any(old in text for old in (*RENAMED, "NodeSchema", "advanced", "adjacent_to")):
            continue
        rel = str(path.relative_to(ROOT))
        wrapper = MetadataWrapper(cst.parse_module(text))
        rewriter = Rewriter(rel, sites, fragments, capture["families"], stopped)
        module = wrapper.visit(rewriter)
        module = _imports(module, rewriter.used)
        problems += rewriter.problems
        if module.code != text:
            path.write_text(module.code)
            changed += 1
    print(f"files changed: {changed}")
    print(f"sites planned: {len(sites)}, stopped: {len(stopped)}")
    for problem in problems:
        print("left for a person:", problem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

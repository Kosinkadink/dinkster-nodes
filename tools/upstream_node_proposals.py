"""Translate literal scalar nodes, leaving non-mechanical ports as explicit drafts."""

from __future__ import annotations

import ast
import keyword
import re
from dataclasses import dataclass

SCALARS = {"INT": "int", "FLOAT": "float", "STRING": "str", "BOOLEAN": "bool"}


@dataclass(frozen=True)
class Proposal:
    source: str = ""
    test: str = ""
    reason: str = ""


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()).strip("_")


def literal(value: ast.AST) -> object:
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError) as error:
        raise ValueError("schema depends on runtime values or an engine capability") from error


def call_name(value: ast.AST) -> str:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return f"{call_name(value.value)}.{value.attr}"
    return ""


def scalar_input(name: str, kind: str, options: dict[str, object]) -> str:
    if not name.isascii() or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"input {name!r} needs a native parameter-name mapping")
    if kind not in SCALARS:
        raise ValueError(f"{kind} needs a native socket type/codec or widget translation")
    allowed = {"default", "min", "max", "step", "round", "multiline", "tooltip", "forceInput"}
    unknown = options.keys() - allowed
    if unknown:
        raise ValueError(f"input {name} needs widget metadata: {', '.join(sorted(unknown))}")
    fields = [repr(name), f"TypeExpr.concrete(CORE_{kind})"]
    if "default" in options:
        fields.append(f"default={options['default']!r}")
    if "tooltip" in options:
        fields.append(f"doc={options['tooltip']!r}")
    if "forceInput" in options:
        fields.append(f"force_input={options['forceInput']!r}")
    numeric = {
        key: value for key, value in options.items() if key in {"min", "max", "step", "round"}
    }
    if numeric:
        fields.append(
            "widget=NumberWidget(" + ", ".join(f"{k}={v!r}" for k, v in numeric.items()) + ")"
        )
    if "multiline" in options:
        if numeric:
            raise ValueError("input mixes numeric and string widget metadata")
        fields.append(f"widget=StringWidget(multiline={options['multiline']!r})")
    return "InputSpec(" + ", ".join(fields) + ")"


def translate(name: str, class_name: str, source: str, namespace: str, module: str) -> Proposal:
    """Only pure scalar expressions with fully literal schemas can be copied safely."""
    try:
        cls = ast.parse(source).body[0]
        if not isinstance(cls, ast.ClassDef):
            raise ValueError("source is not a local node class")
        if not class_name.isidentifier() or keyword.iskeyword(class_name):
            raise ValueError("node class needs an explicit Python identifier mapping")
        methods = {item.name: item for item in cls.body if isinstance(item, ast.FunctionDef)}
        attributes = {
            target.id: assignment.value
            for assignment in cls.body
            if isinstance(assignment, ast.Assign)
            for target in assignment.targets
            if isinstance(target, ast.Name)
        }
        inputs: list[tuple[str, str, dict[str, object]]] = []
        output_kinds: list[str] = []
        output_names: list[str] = []
        category = "utilities"
        display_name = name
        if "INPUT_TYPES" in methods:
            schema_method = methods["INPUT_TYPES"]
            if len(schema_method.body) != 1 or not isinstance(schema_method.body[0], ast.Return):
                raise ValueError("V1 schema uses helpers or runtime choices")
            schema = literal(schema_method.body[0].value)
            if not isinstance(schema, dict) or set(schema) != {"required"}:
                raise ValueError(
                    "optional, hidden or dynamic inputs need native execution semantics"
                )
            for input_name, spec in schema["required"].items():
                if not isinstance(spec, (tuple, list)) or len(spec) not in (1, 2):
                    raise ValueError("input declaration needs native schema translation")
                inputs.append((input_name, spec[0], spec[1] if len(spec) == 2 else {}))
            output_kinds = list(literal(attributes["RETURN_TYPES"]))
            output_names = list(literal(attributes.get("RETURN_NAMES", attributes["RETURN_TYPES"])))
            category = str(literal(attributes.get("CATEGORY", ast.Constant("utilities"))))
            function_name = literal(attributes["FUNCTION"])
            if not isinstance(function_name, str) or function_name not in methods:
                raise ValueError("execution function is not a local method")
            execute = methods[function_name]
            if any(key in attributes for key in ("OUTPUT_NODE", "INPUT_IS_LIST", "OUTPUT_IS_LIST")):
                raise ValueError("output-node or list semantics need a native implementation")
        elif "define_schema" in methods:
            schema_method = methods["define_schema"]
            if len(schema_method.body) != 1 or not isinstance(schema_method.body[0], ast.Return):
                raise ValueError("V3 schema is inherited or assembled dynamically")
            schema = schema_method.body[0].value
            if not isinstance(schema, ast.Call) or not call_name(schema.func).endswith(".Schema"):
                raise ValueError("V3 schema requires a custom engine capability")
            fields = {item.arg: item.value for item in schema.keywords}
            unknown = fields.keys() - {"node_id", "display_name", "category", "inputs", "outputs"}
            if unknown or schema.args:
                raise ValueError(
                    "V3 schema has execution or presentation metadata requiring review"
                )
            if literal(fields["node_id"]) != name:
                raise ValueError("runtime node identity differs from the registered identity")
            category = str(literal(fields.get("category", ast.Constant("utilities"))))
            display_name = str(literal(fields.get("display_name", ast.Constant(name))))
            for item in fields["inputs"].elts:
                if not isinstance(item, ast.Call) or len(item.args) != 1:
                    raise ValueError("dynamic V3 input family needs a native implementation")
                kind = call_name(item.func).split(".")[-2].upper()
                kind = {"BOOL": "BOOLEAN"}.get(kind, kind)
                options = {field.arg: literal(field.value) for field in item.keywords}
                if None in options:
                    raise ValueError("expanded V3 widget options need review")
                inputs.append((str(literal(item.args[0])), kind, options))
            for index, item in enumerate(fields["outputs"].elts):
                if not isinstance(item, ast.Call) or item.args:
                    raise ValueError("dynamic V3 output family needs a native implementation")
                kind = call_name(item.func).split(".")[-2].upper()
                output_kinds.append({"BOOL": "BOOLEAN"}.get(kind, kind))
                options = {field.arg: literal(field.value) for field in item.keywords}
                if options.keys() - {"display_name"}:
                    raise ValueError("V3 output semantics need native translation")
                output_names.append(str(options.get("display_name", output_kinds[index])))
            execute = methods["execute"]
        else:
            raise ValueError(
                "node needs an inherited schema, runtime widget or custom native implementation"
            )
        if not output_kinds or any(kind not in SCALARS for kind in output_kinds):
            raise ValueError("outputs need native value codecs or engine capabilities")
        if len(output_names) != len(output_kinds):
            raise ValueError("output names and types do not have matching positions")
        input_specs = [scalar_input(*item) for item in inputs]
        body = [
            item
            for item in execute.body
            if not (
                isinstance(item, ast.Expr)
                and isinstance(item.value, ast.Constant)
                and isinstance(item.value.value, str)
            )
        ]
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            raise ValueError(
                "execution uses helpers, state or control flow requiring native translation"
            )
        value = body[0].value
        if isinstance(value, ast.Call) and call_name(value.func).endswith(".NodeOutput"):
            if value.keywords:
                raise ValueError("UI results or execution metadata need native translation")
            expressions = value.args
        elif isinstance(value, ast.Tuple):
            expressions = value.elts
        else:
            raise ValueError("execution return needs a native output mapping")
        if len(expressions) != len(output_kinds):
            raise ValueError("execution output arity differs from the schema")
        names = {item[0] for item in inputs}
        expected_args = [item.arg for item in execute.args.args]
        if expected_args and expected_args[0] in {"self", "cls"}:
            expected_args.pop(0)
        expected_args += [item.arg for item in execute.args.kwonlyargs]
        if set(expected_args) != names or execute.args.vararg or execute.args.kwarg:
            raise ValueError("execution parameters do not match the declared scalar inputs")
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.BoolOp,
            ast.Compare,
            ast.IfExp,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.USub,
            ast.UAdd,
            ast.Not,
            ast.And,
            ast.Or,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
        )
        for expression in expressions:
            nodes = list(ast.walk(expression))
            if len(nodes) > 64:
                raise ValueError("scalar expression is too large for bounded automatic validation")
            if any(
                isinstance(item, ast.Constant)
                and (
                    not isinstance(item.value, (str, int, float, bool, type(None)))
                    or isinstance(item.value, str)
                    and len(item.value) > 256
                    or isinstance(item.value, (int, float))
                    and abs(item.value) > 1_000_000
                )
                for item in nodes
            ):
                raise ValueError("scalar constants exceed automatic validation bounds")
            if (
                any(kind == "STRING" for _, kind, _ in inputs)
                or any(
                    isinstance(item, ast.Constant) and isinstance(item.value, str) for item in nodes
                )
            ) and any(isinstance(item, ast.BinOp) for item in nodes):
                raise ValueError("string arithmetic needs bounded native behavior checks")
            if any(not isinstance(item, allowed) for item in nodes):
                raise ValueError(
                    "execution calls an API, helper or value method requiring native translation"
                )
            if any(
                isinstance(item, ast.Name) and item.id not in names for item in ast.walk(expression)
            ):
                raise ValueError("execution depends on module globals or an engine API")
        if not slug(name):
            raise ValueError("node identity needs a native name mapping")
        # Expected values come from upstream expressions, not generated native code.
        vectors = []
        for offset in (0, 1):
            arguments = {}
            for index, (input_name, kind, _options) in enumerate(inputs):
                arguments[input_name] = {
                    "INT": -7 + index * 10 + offset,
                    "FLOAT": -2.5 + index * 4.25 + offset,
                    "STRING": f"input-{index}-{offset}",
                    "BOOLEAN": bool((index + offset) % 2),
                }[kind]
            results = [
                eval(
                    compile(ast.Expression(item), "<upstream scalar>", "eval"),
                    {"__builtins__": {}},
                    arguments,
                )
                for item in expressions
            ]
            vectors.append(
                (
                    arguments,
                    dict(zip((f"output_{i}" for i in range(len(results))), results, strict=True)),
                )
            )
        parameters = ", ".join(f"{key}: {SCALARS[kind]}" for key, kind, _ in inputs)
        result_code = ", ".join(
            f"output_{i}={ast.unparse(item)}" for i, item in enumerate(expressions)
        )
        outputs = [
            f"OutputSpec('output_{i}', TypeExpr.concrete(CORE_{kind}), display_name={label!r})"
            for i, (kind, label) in enumerate(zip(output_kinds, output_names, strict=True))
        ]
        imports = {
            "Node",
            "NodeSchema",
            "OutputSpec",
            "TypeExpr",
            "with_identity_replacement",
        }
        if inputs:
            imports.add("InputSpec")
        imports.update(f"CORE_{kind}" for _, kind, _ in inputs)
        imports.update(f"CORE_{kind}" for kind in output_kinds)
        imports.update(
            widget
            for widget in ("NumberWidget", "StringWidget")
            if any(widget in item for item in input_specs)
        )
        node_type = f"{namespace}.{slug(name)}"
        generated = f"""{ascii(f"Native scalar implementation of {name}.")}
from collections.abc import Mapping
from dinkster_api.v1 import {", ".join(sorted(imports))}

class {class_name}(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type={node_type!r}, display_name={display_name!r}, category={category!r},
            inputs=({", ".join(input_specs)}{"," if input_specs else ""}),
            outputs=({", ".join(outputs)},),
        )
        return with_identity_replacement(schema, {name!r})

    @classmethod
    def execute(cls{", *, " + parameters if parameters else ""}) -> Mapping[str, object]:
        return cls.outputs({result_code})
"""
        test = f'''"""Upstream scalar behavior remains available from the native pack."""
import pytest
from dinkster_schema import build_schemas
from {module}.upstream_{slug(name)} import {class_name}

@pytest.mark.parametrize(("arguments", "expected"), {vectors!r})
def test_{slug(name)}_loads_and_preserves_values(arguments, expected):
    schemas = build_schemas([{class_name}])
    assert {node_type!r} in schemas
    assert {class_name}.execute(**arguments) == expected
'''
        return Proposal(generated, test)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, ArithmeticError) as error:
        return Proposal(reason=str(error))

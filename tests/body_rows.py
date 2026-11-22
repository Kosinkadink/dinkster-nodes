"""Read a node body's rows the way the old schema lists named them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_api.v1 import Choice, Group, NodeSchema, Option, Row, Section, TypeChoice, Use


def scope(rows: Sequence[object], shared: Sequence[object] = ()) -> list[object]:
    """One scope's rows through sections and rows, each Use replaced by its shared row."""
    named = {getattr(row, "id", ""): row for row in scope(shared)} if shared else {}
    found: list[object] = []
    for row in rows:
        if isinstance(row, (Section, Row)):
            found.extend(scope(row.rows, shared))
        elif isinstance(row, Use):
            found.append(named[row.id])
        else:
            found.append(row)
    return found


def option_rows(choice: Choice, option: Option) -> list[object]:
    return scope(option.rows, choice.shared)


def option_ids(choice: Choice) -> dict[str, tuple[str, ...]]:
    """Each option's key and the ids of the rows it holds."""
    return {
        option.key: tuple(getattr(row, "id", "") for row in option_rows(choice, option))
        for option in choice.options
    }


def choices(schema: NodeSchema) -> list[Choice]:
    return [row for row in scope(schema.body) if isinstance(row, Choice)]


def type_choices(schema: NodeSchema) -> list[TypeChoice]:
    return [row for row in scope(schema.body) if isinstance(row, TypeChoice)]


def groups(schema: NodeSchema) -> list[Group]:
    return [row for row in scope(schema.body) if isinstance(row, Group)]


def wire_entry(wire: Mapping[str, object], entry_id: str) -> dict[str, Any]:
    """The first body wire entry with ``entry_id``, at any depth."""

    def find(entries: object) -> dict[str, Any] | None:
        for entry in cast("list[dict[str, Any]]", entries or []):
            if entry.get("id") == entry_id:
                return entry
            for nested in (
                entry.get("rows"),
                entry.get("shared"),
                entry.get("member"),
                [entry["value"]] if isinstance(entry.get("value"), dict) else None,
                *(option.get("rows") for option in entry.get("options", [])),
                *(variant.get("rows") for variant in entry.get("variants") or []),
            ):
                found = find(nested)
                if found is not None:
                    return found
        return None

    found = find(wire["body"])
    if found is None:
        raise AssertionError(f"missing {entry_id}")
    return found

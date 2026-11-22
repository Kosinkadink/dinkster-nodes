"""Turn the captured row lists of each ``NodeSchema(...)`` call into the order
of its body: which element goes where, which go into the "Advanced" section,
and which pairs go on one line.

An element that is a parameter of a helper which builds the schema for its
callers (``_schema(node_type, inputs)``) is planned at each caller instead:
the caller's argument is arranged the same way, and the helper places the
parameter as one run of rows.

A call site is planned only when every observation of it (one per node type
it builds) gives the same plan. Otherwise, and for any element whose rows do
not fit one place, the site is a stop with its reason.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

INPUT_LISTS = ("inputs", "input_families", "combos")


@dataclass
class Placed:
    """One element of the body, ``(list name, element index)``, with the
    element placed beside it on one line, if any."""

    element: tuple[str, int]
    beside: tuple[str, int] | None = None


@dataclass
class Arrangement:
    plain: list[Placed] = field(default_factory=list)
    advanced: list[Placed] = field(default_factory=list)


@dataclass
class Fragment:
    """A caller's argument that the helper passes on as rows."""

    site: str
    argument: str | int
    arrangement: Arrangement


class Stop(Exception):
    pass


def arrange(
    elements: dict[tuple[str, int], list[dict[str, Any]]],
    order: list[str],
    passed: set[tuple[str, int]] = frozenset(),  # type: ignore[assignment]
) -> Arrangement:
    """Arrange elements whose rows are known. ``passed`` elements are placed
    as one run of plain rows whatever they hold."""
    owner: dict[str, tuple[str, int]] = {}
    info: dict[str, dict[str, Any]] = {}
    for element, entries in elements.items():
        for entry in entries:
            if entry["id"] in owner:
                raise Stop(f"row {entry['id']} is built by two elements")
            owner[entry["id"]] = element
            info[entry["id"]] = entry
    if sorted(order) != sorted(owner):
        raise Stop("input_order does not name every row once")
    sequence: list[tuple[str, int]] = []
    for row_id in order:
        element = owner[row_id]
        if sequence and sequence[-1] == element:
            continue
        if element in sequence:
            raise Stop(f"the rows of element {element} are not next to each other in the order")
        sequence.append(element)
    flag: dict[tuple[str, int], bool] = {}
    for row_id in order:
        element = owner[row_id]
        advanced = False if element in passed else info[row_id]["advanced"]
        if flag.setdefault(element, advanced) != advanced:
            raise Stop(f"element {element} holds advanced and plain rows")
    beside: dict[tuple[str, int], tuple[str, int]] = {}
    for row_id, entry in info.items():
        anchor = entry["adjacent_to"]
        if anchor is None or owner[row_id] in passed:
            continue
        if row_id != "mount" or anchor != "filename_prefix" or anchor not in owner:
            raise Stop(f"{row_id} is placed beside {anchor}")
        a, b = owner[anchor], owner[row_id]
        if any(len(elements[e]) != 1 for e in (a, b)) or flag[a] != flag[b] or a in passed:
            raise Stop("filename_prefix and mount do not each stand alone in one place")
        beside[a] = b
    placed = Arrangement()
    paired = set(beside.values())
    for element in sequence:
        if element in paired:
            continue
        target = placed.advanced if flag[element] else placed.plain
        target.append(Placed(element, beside.get(element)))
    return placed


def plan_observation(obs: dict[str, Any]) -> tuple[Arrangement, list[Fragment], int]:
    lists = obs["lists"]
    elements: dict[tuple[str, int], list[dict[str, Any]]] = {}
    passed: set[tuple[str, int]] = set()
    fragments: list[Fragment] = []
    natural: list[str] = []
    for name in INPUT_LISTS:
        for index, element in enumerate(lists.get(name, ())):
            elements[(name, index)] = element["entries"]
            natural += [e["id"] for e in element["entries"]]
            caller = element.get("caller")
            if caller is None:
                continue
            passed.add((name, index))
            if obs["input_order"]:
                raise Stop("a helper's rows are reordered by input_order")
            inner = {("arg", i): e["entries"] for i, e in enumerate(caller["elements"])}
            order = [e["id"] for i in range(len(inner)) for e in inner[("arg", i)]]
            fragments.append(Fragment(caller["site"], caller["argument"], arrange(inner, order)))
    order = obs["input_order"] or natural
    arrangement = arrange(elements, order, passed)
    if any(f.arrangement.advanced for f in fragments):
        last = arrangement.plain[-1].element if arrangement.plain else None
        if arrangement.advanced or last not in passed:
            raise Stop("a helper's advanced rows would not come last")
    return arrangement, fragments, len(lists.get("outputs", ()))


def plan_site(site: dict[str, Any]) -> tuple[Arrangement, list[Fragment], int]:
    plans = [plan_observation(obs) for obs in site["observations"]]
    first = json.dumps([asdict(plans[0][0]), plans[0][2]])
    for other in plans[1:]:
        if json.dumps([asdict(other[0]), other[2]]) != first:
            raise Stop("the node types this call builds need different bodies")
    fragments = [f for plan in plans for f in plan[1]]
    return plans[0][0], fragments, plans[0][2]

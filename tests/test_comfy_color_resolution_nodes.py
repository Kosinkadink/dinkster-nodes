"""Public color and resolution contracts without a Torch dependency."""

from __future__ import annotations

from typing import cast

import pytest
from dinkster_api.v1 import ComboWidget, Formula, Input, Note
from dinkster_schema.formula import UNKNOWN, FormulaScope, FormulaValues, parse_formula
from dinkster_utilities.color_resolution import ColorToRGBInt, ResolutionSelector

from tests.body_rows import scope


def test_color_values_preserve_case_alpha_and_reference_rejection() -> None:
    assert ColorToRGBInt.execute(color="#aB12CD80") == {
        "rgb_int": 11211469,
        "hex": "#aB12CD",
        "alpha": 128 / 255,
    }
    assert ColorToRGBInt.execute(color="#01020f") == {
        "rgb_int": 66063,
        "hex": "#01020f",
        "alpha": 1.0,
    }
    assert ColorToRGBInt.execute(color="#12_345") == {
        "rgb_int": 16777215,
        "hex": "#12_345",
        "alpha": 1.0,
    }
    assert ColorToRGBInt.execute(color="#00000000")["alpha"] == 0.0
    for invalid in ("", "red", "#abc", "001122", "#12345g", "#1122334455"):
        with pytest.raises(ValueError, match="Color must be in format #RRGGBB or #RRGGBBAA"):
            ColorToRGBInt.execute(color=invalid)


def test_resolution_rounds_to_requested_multiple() -> None:
    assert ResolutionSelector.execute(aspect_ratio="1:1 (Square)", megapixels=1, multiple=20) == {
        "width": 1020,
        "height": 1020,
    }
    assert ResolutionSelector.execute(aspect_ratio="1:1 (Square)", megapixels=0.1, multiple=8) == {
        "width": 320,
        "height": 320,
    }
    assert ResolutionSelector.execute(
        aspect_ratio="21:9 (Ultrawide)", megapixels=1, multiple=8
    ) == {"width": 1568, "height": 672}
    with pytest.raises(KeyError):
        ResolutionSelector.execute(aspect_ratio="unrecognized", megapixels=1, multiple=8)


def test_resolution_note_shows_the_size_execute_gives() -> None:
    schema = ResolutionSelector.schema()
    rows = scope(schema.body)
    note = next(row for row in rows if isinstance(row, Note))
    assert (note.id, note.unknown) == ("preview", "Not known until a run")
    assert isinstance(note.text, Formula)
    text = parse_formula(
        note.text.text,
        FormulaScope(
            names={"aspect_ratio": "choice", "megapixels": "float", "multiple": "int"},
            inputs=frozenset({"aspect_ratio", "megapixels", "multiple"}),
            tables=dict(schema.tables),
            derived={name: formula.text for name, formula in schema.derived.items()},
        ),
    )
    ratio_input = next(row for row in rows if isinstance(row, Input) and row.id == "aspect_ratio")
    ratios = [option.value for option in cast(ComboWidget, ratio_input.widget).options]
    assert len(ratios) == 8
    cases = [
        (ratio, step / 10, multiple)
        for ratio in ratios
        for step in range(1, 161)
        for multiple in range(8, 129, 4)
    ]
    # 508 / 8 and 516 / 8 land on 63.5 and 64.5, which round to the even 64.
    cases += [("1:1 (Square)", (127 / 256) ** 2, 8), ("1:1 (Square)", (129 / 256) ** 2, 8)]
    for ratio, megapixels, multiple in cases:
        size = ResolutionSelector.execute(
            aspect_ratio=ratio, megapixels=megapixels, multiple=multiple
        )
        shown = text.evaluate(
            FormulaValues(
                values={"aspect_ratio": ratio, "megapixels": megapixels, "multiple": multiple}
            )
        )
        assert shown == f"{size['width']} x {size['height']} pixels", (ratio, megapixels, multiple)
    assert ResolutionSelector.execute(
        aspect_ratio="1:1 (Square)", megapixels=(129 / 256) ** 2, multiple=8
    ) == {"width": 512, "height": 512}
    assert (
        text.evaluate(
            FormulaValues(values={"aspect_ratio": "1:1 (Square)", "megapixels": 1.0, "multiple": 8})
        )
        == "1024 x 1024 pixels"
    )
    assert (
        text.evaluate(FormulaValues(values={"aspect_ratio": "1:1 (Square)", "multiple": 8}))
        is UNKNOWN
    )

"""Curve editing identity and evenly spaced samples."""

import pytest
from dinkster_api.v1 import Curve
from dinkster_schema import use_reporter
from dinkster_utilities.curve_editor import CurveEditor, SampleCurve


def test_curve_editor_returns_original_curve_and_reports_no_histogram() -> None:
    curve = Curve(((0, 2), (1, 4)), "monotone_cubic")
    events = []
    with use_reporter(lambda name, data, blob: events.append((name, data, blob))):
        assert CurveEditor.execute(curve=curve, histogram=list(range(256)))["curve"] is curve
    # The engine, not the node, sends the histogram to the curve's editor.
    assert events == []


@pytest.mark.parametrize(
    ("interpolation", "expected"),
    [
        ("linear", [1, 3, 5, 4.5, 4, 3.5, 3]),
        # Hermite slopes are 2, 0, -0.5 across the unequal two- and four-unit intervals.
        ("monotone_cubic", [1, 3.5, 5, 4.78125, 4.25, 3.59375, 3]),
        ("step", [1, 1, 5, 5, 5, 5, 3]),
    ],
)
def test_sample_curve_spans_the_full_range(interpolation: str, expected: list[float]) -> None:
    from typing import Literal, cast

    curve = Curve(
        ((8, 3), (2, 1), (4, 5)),
        cast("Literal['linear', 'monotone_cubic', 'step']", interpolation),
    )
    assert SampleCurve.execute(curve=curve, count=7)["values"] == pytest.approx(expected)
    assert SampleCurve.execute(curve=curve, count=1)["values"] == [1]
    assert SampleCurve.execute(curve=curve, count=2)["values"] == [1, 3]


def test_sample_curve_degenerate_and_extreme_ranges() -> None:
    assert SampleCurve.execute(curve=Curve(()), count=3)["values"] == [0, 0, 0]
    assert SampleCurve.execute(curve=Curve(((7, -2),)), count=3)["values"] == [-2, -2, -2]
    curve = Curve(((-1e308, 2), (1e308, 6)), "linear")
    assert SampleCurve.execute(curve=curve, count=3)["values"] == [2, 4, 6]
    with pytest.raises(ValueError, match="at least 1"):
        SampleCurve.execute(curve=curve, count=0)

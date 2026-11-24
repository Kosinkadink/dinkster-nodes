"""Curve editor identity and the histogram its curve names."""

from dinkster_api.v1 import Curve
from dinkster_schema import use_reporter
from dinkster_utilities.curve_editor import CurveEditor


def test_curve_editor_returns_original_curve_and_reports_no_histogram() -> None:
    curve = Curve(((0, 2), (1, 4)), "monotone_cubic")
    events = []
    with use_reporter(lambda name, data, blob: events.append((name, data, blob))):
        assert CurveEditor.execute(curve=curve, histogram=list(range(256)))["curve"] is curve
    # The engine, not the node, sends the histogram to the curve's editor.
    assert events == []

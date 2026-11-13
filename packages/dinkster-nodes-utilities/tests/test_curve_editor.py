"""Curve editor identity and histogram event behavior through ordinary hosts."""

from dinkster_api.v1 import Curve
from dinkster_schema import use_reporter
from dinkster_utilities.curve_editor import CurveEditor


def test_curve_editor_returns_original_curve_and_reports_histogram() -> None:
    curve = Curve(((0, 2), (1, 4)), "monotone_cubic")
    histogram = list(range(256))
    events = []
    with use_reporter(lambda name, data, blob: events.append((name, data, blob))):
        assert CurveEditor.execute(curve=curve, histogram=histogram)["curve"] is curve
    assert events == [("dinkster.curve.histogram", {"histogram": histogram}, None)]

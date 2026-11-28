"""Native editing and sampling of shared curve values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    CURVE_TYPE,
    HISTOGRAM_TYPE,
    Curve,
    CurveWidget,
    Input,
    Node,
    NodeSchema,
    NumberWidget,
    Output,
    TypeExpr,
    with_identity_replacement,
)


class CurveEditor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.curve_editor",
            display_name="Curve Editor",
            category="utilities",
            editor_role="curve",
            body=(
                Input(
                    "curve",
                    TypeExpr.concrete(CURVE_TYPE),
                    default={
                        "points": [[0.0, 0.0], [1.0, 1.0]],
                        "interpolation": "monotone_cubic",
                    },
                    widget=CurveWidget(histogram="histogram"),
                ),
                Input("histogram", TypeExpr.concrete(HISTOGRAM_TYPE), required=False),
                Output("curve", TypeExpr.concrete(CURVE_TYPE)),
            ),
            emits_previews=True,
        )
        return with_identity_replacement(schema, "CurveEditor")

    @classmethod
    def execute(
        cls, *, curve: Curve, histogram: Sequence[int] | None = None
    ) -> Mapping[str, object]:
        # The engine sends the histogram to the curve's editor; the node passes the curve on.
        return cls.outputs(curve=curve)


class SampleCurve(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.utilities.sample_curve",
            display_name="Sample curve",
            category="utilities",
            body=(
                Input(
                    "curve",
                    TypeExpr.concrete(CURVE_TYPE),
                    default={
                        "points": [[0.0, 0.0], [1.0, 1.0]],
                        "interpolation": "monotone_cubic",
                    },
                    widget=CurveWidget(),
                ),
                Input(
                    "count",
                    TypeExpr.concrete(CORE_INT),
                    default=10,
                    widget=NumberWidget(min=1, step=1),
                ),
                Output("values", TypeExpr.list_of(TypeExpr.concrete(CORE_FLOAT))),
            ),
        )

    @classmethod
    def execute(cls, *, curve: Curve, count: int) -> Mapping[str, object]:
        if count < 1:
            raise ValueError("sample count must be at least 1")
        if count == 1:
            return cls.outputs(values=[curve.evaluate(curve.start)])
        values: list[float] = []
        for index in range(count):
            amount = index / (count - 1)
            # Weighted endpoints also handle a finite range whose subtraction overflows.
            position = (1.0 - amount) * curve.start + amount * curve.end
            values.append(curve.evaluate(position))
        return cls.outputs(values=values)

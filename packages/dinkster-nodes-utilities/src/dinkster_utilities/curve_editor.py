"""Native curve editor over the shared curve value, with a histogram beside its curve."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dinkster_api.v1 import (
    CURVE_TYPE,
    HISTOGRAM_TYPE,
    Curve,
    CurveWidget,
    Input,
    Node,
    NodeSchema,
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

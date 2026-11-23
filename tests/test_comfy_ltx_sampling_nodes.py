"""Torch-free native LTXV sampling schema contracts."""

from dinkster_api.v1 import NumberWidget, schema_from_wire, schema_to_wire
from dinkster_ltx import NODES
from dinkster_ltx.ltx_sampling import LTXVScheduler, ModelSamplingLTXV

from tests.body_rows import advanced_ids


def test_native_ltx_sampling_schemas() -> None:
    expected = (
        (
            "dinkster.ltx.model_sampling_ltxv",
            ("model", "max_shift", "base_shift", "latent"),
            (),
            "MODEL",
        ),
        (
            "dinkster.ltx.ltxv_scheduler",
            ("steps", "max_shift", "base_shift", "latent"),
            ("stretch", "terminal"),
            "SIGMAS",
        ),
    )
    for node, (name, inputs, advanced, output) in zip(
        (ModelSamplingLTXV, LTXVScheduler), expected, strict=True
    ):
        schema = node.schema()
        assert schema.node_type == name
        assert tuple(item.id for item in schema.inputs) == (*inputs, *advanced)
        assert advanced_ids(schema.body) == advanced
        assert schema.outputs[0].type.types == (output,)
        assert [item.id for item in schema.inputs if not item.required] == ["latent"]
        assert schema_from_wire(schema_to_wire(schema)) == schema
        assert node in NODES
    terminal = LTXVScheduler.schema().input("terminal")
    assert terminal is not None
    widget = terminal.widget
    assert isinstance(widget, NumberWidget)
    assert widget.min == 0.0
    assert widget.max == 0.99

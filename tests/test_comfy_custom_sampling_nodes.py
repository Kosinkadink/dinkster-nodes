"""Public custom-sampling contracts remain safe without Torch."""

from dinkster_api.v1 import NumberWidget, schema_from_wire, schema_to_wire
from dinkster_sampling import NODES
from dinkster_sampling.custom_sampling import SamplerCustom, SamplerCustomAdvanced

from tests.body_rows import advanced_ids


def test_custom_sampling_schema_contracts() -> None:
    expected = (
        (
            (
                "model",
                "noise_seed",
                "cfg",
                "positive",
                "negative",
                "sampler",
                "sigmas",
                "latent_image",
            ),
            ("add_noise",),
        ),
        (("noise", "guider", "sampler", "sigmas", "latent_image"), ()),
    )
    for node, node_type, (names, advanced) in zip(
        (SamplerCustom, SamplerCustomAdvanced),
        ("dinkster.sampling.sampler_custom", "dinkster.sampling.sampler_custom_advanced"),
        expected,
        strict=True,
    ):
        assert node in NODES
        schema = node.schema()
        assert schema.node_type == node_type
        assert tuple(item.id for item in schema.inputs) == (*names, *advanced)
        assert advanced_ids(schema.body) == advanced
        assert tuple(item.id for item in schema.outputs) == ("output", "denoised_output")
        assert schema_from_wire(schema_to_wire(schema)) == schema
    inputs = {item.id: item for item in SamplerCustom.schema().inputs}
    assert inputs["add_noise"].default is True
    assert inputs["noise_seed"].widget == NumberWidget(min=0, max=0xFFFFFFFFFFFFFFFF)
    assert inputs["noise_seed"].after_run == "fixed"
    assert inputs["cfg"].default == 8.0
    assert inputs["cfg"].widget == NumberWidget(min=0.0, max=100.0, step=0.1, round=0.01)

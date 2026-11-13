"""Core image transform schemas preserve the ComfyUI workflow contract."""

from dinkster_image.ported.image_transform import (
    ImageAddNoise,
    ImageCrop,
    ImageFlip,
    ImageFromBatch,
    ImageRotate,
    ImageScaleToMaxDimension,
    ImageStitch,
    PadImageNode,
    RepeatImageBatch,
    ResizeAndPadImage,
)
from dinkster_schema import schema_from_wire, schema_to_wire, validate_replacement_references
from dinkster_schema.replace import rule_to_wire

IMAGE_TRANSFORM_NODES = (
    ImageCrop,
    RepeatImageBatch,
    ImageFromBatch,
    ImageAddNoise,
    ImageStitch,
    ResizeAndPadImage,
    ImageRotate,
    ImageFlip,
    ImageScaleToMaxDimension,
    PadImageNode,
)


def test_deprecated_longer_edge_resize_publishes_its_successor_mapping() -> None:
    schemas = {node.schema().node_type: node.schema() for node in IMAGE_TRANSFORM_NODES}
    schema = schemas["dinkster.image.image_scale_to_max_dimension"]
    wire = schema_to_wire(schema)
    assert [
        rule_to_wire(rule)
        for rule in schema.replacements
        if rule.from_type == "ResizeImagesByLongerEdge"
    ] == [
        {
            "from": "ResizeImagesByLongerEdge",
            "cases": [
                {
                    "to": "dinkster.image.image_scale_to_max_dimension",
                    "inputs": {
                        "image": {"kind": "copy", "input": "images"},
                        "largest_size": {"kind": "copy", "input": "longer_edge"},
                        "upscale_method": {"kind": "constant", "value": "lanczos"},
                    },
                    "outputs": {"IMAGE": "images"},
                }
            ],
        }
    ]
    assert schema_to_wire(schema_from_wire(wire)) == wire
    assert "ResizeImagesByLongerEdge" not in schemas
    assert validate_replacement_references(schemas) == ()

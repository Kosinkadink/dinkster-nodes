"""Schema contracts for native tile and mask preview nodes."""

from typing import Any, cast

from dinkster_api.v1 import IMAGE_TYPE, MASK_TYPE, schema_to_wire
from dinkster_image.ported.image_tiles import ImageMergeTileList, SplitImageToTileList
from dinkster_image.ported.mask_preview import MaskPreview


def test_tile_node_schemas_preserve_types_defaults_and_aliases() -> None:
    split = cast("dict[str, Any]", schema_to_wire(SplitImageToTileList.schema()))
    merge = cast("dict[str, Any]", schema_to_wire(ImageMergeTileList.schema()))
    assert split["searchTerms"] == [
        "split image",
        "tile image",
        "slice image",
        "SplitImageToTileList",
    ]
    assert merge["searchTerms"] == [
        "split image",
        "tile image",
        "slice image",
        "ImageMergeTileList",
    ]
    assert split["body"][0]["type"]["types"] == [IMAGE_TYPE]
    assert split["body"][-1]["type"] == {
        "kind": "list",
        "element": {"kind": "concrete", "types": [IMAGE_TYPE]},
    }
    assert [item["default"] for item in split["body"][1:4]] == [1024, 1024, 128]
    assert [item["default"] for item in merge["body"][1:4]] == [1024, 1024, 128]


def test_mask_preview_schema_declares_preview_output() -> None:
    wire = cast("dict[str, Any]", schema_to_wire(MaskPreview.schema()))
    assert wire["nodeType"] == "dinkster.image.mask_preview"
    assert wire["emitsPreviews"] is True
    assert wire["body"] == [
        {
            "id": "mask",
            "required": True,
            "kind": "input",
            "type": {"kind": "concrete", "types": [MASK_TYPE]},
        },
        {
            "id": "mask",
            "preview": True,
            "kind": "output",
            "type": {"kind": "concrete", "types": [MASK_TYPE]},
        },
    ]

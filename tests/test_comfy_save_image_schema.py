"""Schema contract for the native ComfyUI SaveImage node."""

from typing import Any, cast

from dinkster_api.v1 import (
    ASSET_TYPE,
    Choice,
    ComboWidget,
    Input,
    MountWidget,
    NumberWidget,
    Output,
    Row,
    Section,
    StringWidget,
    TypeExpr,
)
from dinkster_image.ported.image_file_tools import (
    LoadImage,
    LoadImageMask,
    LoadImageOutput,
    PreviewImage,
    SaveAnimatedPNG,
    SaveAnimatedWEBP,
    SaveImage,
)
from dinkster_image.ported.save_image_advanced import SaveImageAdvanced
from dinkster_schema import schema_to_wire


def test_save_image_preserves_fixed_inputs_and_appends_mount_widget() -> None:
    schema = SaveImage.schema()

    assert schema.node_type == "dinkster.image.save_image"
    assert schema.display_name == "Save Image"
    assert schema.category == "image"
    images, line, output = schema.body
    assert isinstance(images, Input) and isinstance(line, Row) and isinstance(output, Output)
    filename_prefix, mount = line.rows
    assert isinstance(filename_prefix, Input) and isinstance(mount, Input)
    assert (images.id, filename_prefix.id, mount.id, output.id) == (
        "images",
        "filename_prefix",
        "mount",
        "images",
    )
    assert filename_prefix.required
    assert filename_prefix.default == "ComfyUI"
    assert filename_prefix.widget is None
    assert images.tooltip == "The images to save."
    assert filename_prefix.tooltip.startswith("The prefix for the file to save.")
    assert not mount.required
    assert mount.default is None
    assert mount.widget == MountWidget()
    assert schema.output_node
    assert not schema.idempotent
    assert schema.search_terms == (
        "save",
        "save image",
        "export image",
        "output image",
        "write image",
        "download",
        "SaveImage",
    )

    wire = schema_to_wire(schema)
    body = cast(list[dict[str, Any]], wire["body"])
    assert body[1]["kind"] == "row"
    filename_wire, mount_wire = body[1]["rows"]
    assert filename_wire["id"] == "filename_prefix" and "widget" not in filename_wire
    assert mount_wire["id"] == "mount" and mount_wire["widget"] == {"type": "MOUNT"}


def test_preview_image_uses_the_public_image_preview_output() -> None:
    schema = PreviewImage.schema()

    assert tuple(spec.id for spec in schema.inputs) == ("images",)
    assert tuple(spec.id for spec in schema.outputs) == ("images",)
    assert schema.outputs[0].preview
    assert schema.output_node
    assert schema.search_terms == (
        "preview",
        "preview image",
        "show image",
        "view image",
        "display image",
        "image viewer",
        "PreviewImage",
    )


def test_load_image_output_preserves_fixed_description() -> None:
    assert LoadImageOutput.schema().description == (
        "Load an image from the output folder. When the refresh button is clicked, the node will "
        "update the image list and automatically select the first image, allowing for easy "
        "iteration."
    )


def test_image_file_inputs_use_the_bound_asset_contract() -> None:
    for node in (LoadImage, LoadImageMask, LoadImageOutput):
        image = node.schema().inputs[0]
        assert image.type == TypeExpr.concrete(ASSET_TYPE)
        assert image.source_filename is not None


def test_animated_savers_preserve_source_controls_and_append_mount() -> None:
    for node, label, controls in (
        (SaveAnimatedPNG, "Save Animated PNG", ("compress_level",)),
        (SaveAnimatedWEBP, "Save Animated WEBP", ("lossless", "quality", "method")),
    ):
        schema = node.schema()
        assert schema.display_name == label
        images, line, fps, *rest, output = schema.body
        assert isinstance(line, Row)
        filename_prefix, mount = line.rows
        fields = {
            row.id: row
            for row in (images, filename_prefix, mount, fps, *rest)
            if isinstance(row, Input)
        }
        advanced = [row for row in rest if isinstance(row, Section)]
        for section in advanced:
            assert section.id == "advanced" and section.collapsed
            fields.update((row.id, row) for row in section.rows if isinstance(row, Input))
        assert tuple(fields) == ("images", "filename_prefix", "mount", "fps", *controls)
        assert fields["filename_prefix"].default == "ComfyUI"
        assert fields["filename_prefix"].widget == StringWidget(multiline=False)
        assert fields["fps"].default == 6.0
        assert fields["fps"].widget == NumberWidget(min=0.01, max=1000.0, step=0.01)
        assert fields["mount"].widget == MountWidget()
        assert not fields["mount"].required and fields["mount"].default is None
        assert schema.output_node and not schema.idempotent
        assert isinstance(output, Output) and output.id == "images"
        if node is SaveAnimatedPNG:
            assert [row.id for section in advanced for row in section.rows] == ["compress_level"]
            assert fields["compress_level"].default == 4
            assert fields["compress_level"].widget == NumberWidget(min=0, max=9)
        else:
            assert not advanced
            assert fields["lossless"].default is True
            assert fields["quality"].default == 80
            assert fields["quality"].widget == NumberWidget(min=0, max=100)
            assert fields["method"].widget == ComboWidget(options=("default", "fastest", "slowest"))


def test_advanced_image_save_retains_format_branches_and_adjacent_mount() -> None:
    schema = SaveImageAdvanced.schema()
    images, line, formats, output = schema.body
    assert isinstance(images, Input) and images.id == "images"
    assert isinstance(line, Row)
    filename_prefix, mount = line.rows
    assert isinstance(filename_prefix, Input) and isinstance(mount, Input)
    assert filename_prefix.widget == StringWidget(multiline=False)
    assert mount.widget == MountWidget()
    assert not mount.required and mount.default is None
    assert schema.output_node and not schema.idempotent
    assert isinstance(output, Output) and output.id == "images"
    assert isinstance(formats, Choice) and formats.id == "format"
    assert tuple(option.key for option in formats.options) == ("png", "exr", "avif")
    assert schema_to_wire(schema)["nodeType"] == "dinkster.image.save_image_advanced"

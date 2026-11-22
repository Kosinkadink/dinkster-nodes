"""Post-processing nodes keep their source schemas and CPU behavior."""

from __future__ import annotations

from typing import Any, cast

import pytest
from dinkster_api.v1 import (
    Record,
    schema_to_wire,
)
from dinkster_image import NODES as IMAGE_NODES
from dinkster_image.ported.post_processing import (
    BatchImagesNode,
    BatchMasksNode,
    ColorTransfer,
    ResizeImageMaskNode,
)
from dinkster_latent import NODES as LATENT_NODES
from dinkster_latent.post_processing import BatchLatentsNode


def _wire(node: type[Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", schema_to_wire(node.schema()))


def test_post_processing_schemas_use_exact_core_ids_and_dynamic_interfaces() -> None:
    nodes = (ResizeImageMaskNode, BatchImagesNode, BatchMasksNode, BatchLatentsNode, ColorTransfer)
    assert all(
        node in (LATENT_NODES if node is BatchLatentsNode else IMAGE_NODES) for node in nodes
    )
    schemas = {node.schema().node_type: _wire(node) for node in nodes}
    assert tuple(schemas) == (
        "dinkster.image.resize_image_mask_node",
        "dinkster.image.batch_images_node",
        "dinkster.image.batch_masks_node",
        "dinkster.latent.batch_latents_node",
        "dinkster.image.color_transfer",
    )
    resize = schemas["dinkster.image.resize_image_mask_node"]
    assert [item["id"] for item in resize["body"]] == [
        "input",
        "resize_type",
        "scale_method",
        "resized",
    ]
    assert resize["body"][0]["type"] == {
        "kind": "variable",
        "templateId": "input_type",
        "allowed": ["IMAGE", "MASK"],
    }
    assert [option["key"] for option in resize["body"][1]["options"]] == [
        "scale dimensions",
        "scale by multiplier",
        "scale longer dimension",
        "scale shorter dimension",
        "scale width",
        "scale height",
        "scale total pixels",
        "match size",
        "scale to multiple",
    ]
    for node_type, family, prefix, template_name, value_type in (
        ("dinkster.image.batch_images_node", "images", "image", "image", "IMAGE"),
        ("dinkster.image.batch_masks_node", "masks", "mask", "mask", "MASK"),
        ("dinkster.latent.batch_latents_node", "latents", "latent", "latent", "LATENT"),
    ):
        body = schemas[node_type]["body"]
        assert body[0]["kind"] == "group" and body[0]["id"] == family
        assert body[0]["keys"] == {"numbered": {"prefix": prefix, "start": 0, "pad": 0}}
        assert body[0]["minMembers"] == 1 and body[0]["maxMembers"] == 50
        assert body[0]["value"]["id"] == template_name
        assert body[0]["value"]["type"]["types"] == [value_type]
        assert body[1]["type"]["types"] == [value_type]
    color = schemas["dinkster.image.color_transfer"]
    assert [item["id"] for item in color["body"]] == [
        "image_target",
        "image_ref",
        "method",
        "source_stats",
        "strength",
        "image",
    ]
    assert [option["key"] for option in color["body"][3]["options"]] == [
        "per_frame",
        "uniform",
        "target_frame",
    ]


def test_color_transfer_identity_strength_does_not_materialize_the_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = object()
    monkeypatch.setattr(
        "dinkster_image.ported.post_processing.materialize_image", lambda value: value
    )
    assert (
        ColorTransfer.execute(
            image_target=target,
            image_ref=None,
            method="histogram",
            source_stats=Record("per_frame", {}),
            strength=0.0,
        )["image"]
        is target
    )

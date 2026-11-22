"""Torch-free MiniMax H3 guide schema and discovery."""

from typing import cast

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    Input,
    Numbered,
    NumberWidget,
    Output,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_minimax import NODES
from dinkster_minimax.minimax_h3_latent import MiniMaxH3AddGuide
from dinkster_minimax.minimax_h3_reference import MiniMaxH3ReferenceToVideo

from tests.body_rows import groups


def test_minimax_h3_add_guide_schema_and_cold_discovery() -> None:
    schema = MiniMaxH3AddGuide.schema()
    assert NODES.count(MiniMaxH3AddGuide) == 1
    assert (schema.node_type, schema.display_name, schema.category) == (
        "dinkster.minimax.mini_max_h3_add_guide",
        "Add Guide for MiniMax H3",
        "model/conditioning/minimax",
    )
    assert schema.inputs == (
        Input("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
        Input(
            "vae",
            TypeExpr.concrete(VAE_TYPE),
            required=False,
            tooltip="Video VAE, needed when an image is connected.",
        ),
        Input(
            "audio_vae",
            TypeExpr.concrete(VAE_TYPE),
            required=False,
            tooltip="Audio VAE, needed when an audio is connected.",
        ),
        Input("latent", TypeExpr.concrete(LATENT_TYPE)),
        Input(
            "image",
            TypeExpr.concrete(IMAGE_TYPE),
            required=False,
            tooltip=(
                "Image or video frames to anchor. Multi-frame batches are anchored as a clip and "
                "cropped down to the model's valid clip lengths: 5, 22, 39... (17k + 5) frames. "
                "Batches shorter than 5 frames use only the first image."
            ),
        ),
        Input(
            "audio",
            TypeExpr.concrete(AUDIO_TYPE),
            required=False,
            tooltip=(
                "Soundtrack to anchor starting at the same frame index, cropped to the video's "
                "remaining duration."
            ),
        ),
        Input(
            "frame_idx",
            TypeExpr.concrete(CORE_INT),
            default=0,
            widget=NumberWidget(min=-9999, max=9999),
            tooltip=(
                "Frame index to anchor the image or the clip's first frame at. Negative values are "
                "counted from the end of the video."
            ),
        ),
    )
    assert schema.outputs == (Output("positive", TypeExpr.concrete(CONDITIONING_TYPE)),)
    assert schema_from_wire(schema_to_wire(schema)) == schema


def test_minimax_h3_reference_members_are_optional_and_roundtrip() -> None:
    schema = MiniMaxH3ReferenceToVideo.schema()
    assert NODES.count(MiniMaxH3ReferenceToVideo) == 1
    assert schema.node_type == "dinkster.minimax.mini_max_h3_reference_to_video"
    assert schema.inputs[0] == Input("clip", TypeExpr.concrete(CLIP_TYPE))
    assert schema.inputs[0].required

    expected_families = (
        ("ref_images", "ref_image_", IMAGE_TYPE, 9),
        ("ref_videos", "ref_video_", IMAGE_TYPE, 3),
        ("ref_video_audios", "ref_video_audio_", AUDIO_TYPE, 3),
        ("ref_audios", "ref_audio_", AUDIO_TYPE, 3),
    )
    families = groups(schema)
    assert len(families) == len(expected_families)
    assert tuple(family.id for family in families) == tuple(
        family[0] for family in expected_families
    )
    for family, (family_id, prefix, kind, limit) in zip(families, expected_families, strict=True):
        assert (
            family.id,
            cast(Numbered, family.keys).prefix,
            family.min_members,
            family.max_members,
        ) == (
            family_id,
            prefix,
            0,
            limit,
        )
        assert family.value == Input(prefix.rstrip("_"), TypeExpr.concrete(kind), required=False)

    assert schema_from_wire(schema_to_wire(schema)) == schema

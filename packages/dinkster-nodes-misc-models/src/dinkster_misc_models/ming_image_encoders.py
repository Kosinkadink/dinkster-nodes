"""Ming Image edit conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    VAE_TYPE,
    Group,
    Input,
    Item,
    Members,
    Named,
    Node,
    NodeSchema,
    Output,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    with_identity_replacement,
)
from dinkster_pack_helpers.conditioning import conditioning_set_values

_CLIP = TypeExpr.concrete(CLIP_TYPE)

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)

_STRING = TypeExpr.concrete(CORE_STRING)

_VAE = TypeExpr.concrete(VAE_TYPE)


class TextEncodeMingImageEdit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.misc.models.text_encode_ming_image_edit",
            display_name="Text Encode Ming Image Edit",
            category="model/conditioning/ming image",
            body=(
                Input("clip", _CLIP),
                Input("vae", _VAE, required=False),
                Input("prompt", _STRING, widget=StringWidget(multiline=True, dynamic_prompts=True)),
                Group(
                    "images",
                    value=Input("image", _IMAGE, required=False),
                    min_members=0,
                    keys=Named(tuple(f"image_{index}" for index in range(1, 9))),
                ),
                Output("CONDITIONING", _CONDITIONING),
            ),
        )
        return with_identity_replacement(schema, "TextEncodeMingImageEdit")

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        prompt: str,
        images: Members[Item] = Members[Item](),
        vae: Any = None,
    ) -> Mapping[str, object]:
        references = [
            materialize_image(member.value) for member in images if member.value is not None
        ]
        conditioning: Any = clip.encode_from_tokens_scheduled(
            clip.tokenize(prompt, images=[image[..., :3] for image in references])
        )
        if vae is None or not references:
            return cls.outputs(CONDITIONING=conditioning)

        initialize_inference_process()
        height, width = references[0].shape[1:3]
        common_upscale = importlib.import_module("dinkster_inference.utils").common_upscale
        latents = [
            vae.encode(
                common_upscale(image.movedim(-1, 1), width, height, "bilinear", "disabled").movedim(
                    1, -1
                )
            )
            for image in references
        ]
        return cls.outputs(
            CONDITIONING=cast(
                "Any",
                conditioning_set_values(
                    conditioning,
                    {"reference_latents": latents},
                    append=True,
                ),
            )
        )

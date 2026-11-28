"""Model-family-specific block merge schemas."""

from __future__ import annotations

from .model_merging import ModelMergeBlocks

_CATEGORY = "model/merging/model specific"


def _numbered(prefix: str, count: int, *, suffix: str = ".") -> tuple[str, ...]:
    return tuple(f"{prefix}{index}{suffix}" for index in range(count))


_SD1_BLOCK_NAMES = (
    "time_embed.",
    "label_emb.",
    *_numbered("input_blocks.", 12),
    *_numbered("middle_block.", 3),
    *_numbered("output_blocks.", 12),
    "out.",
)


class ModelMergeSD1(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_sd1"
    SOURCE = "ModelMergeSD1"
    CATEGORY = _CATEGORY
    BLOCKS = _SD1_BLOCK_NAMES


class ModelMergeSD2(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_sd2"
    SOURCE = "ModelMergeSD2"
    CATEGORY = _CATEGORY
    BLOCKS = _SD1_BLOCK_NAMES


class ModelMergeSDXL(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_sdxl"
    SOURCE = "ModelMergeSDXL"
    CATEGORY = _CATEGORY
    BLOCKS = (
        "time_embed.",
        "label_emb.",
        *_numbered("input_blocks.", 9, suffix=""),
        *_numbered("middle_block.", 3, suffix=""),
        *_numbered("output_blocks.", 9, suffix=""),
        "out.",
    )


class ModelMergeSD3_2B(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_sd3_2_b"
    SOURCE = "ModelMergeSD3_2B"
    CATEGORY = _CATEGORY
    BLOCKS = (
        "pos_embed.",
        "x_embedder.",
        "context_embedder.",
        "y_embedder.",
        "t_embedder.",
        *_numbered("joint_blocks.", 24),
        "final_layer.",
    )


class ModelMergeAuraflow(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_auraflow"
    SOURCE = "ModelMergeAuraflow"
    CATEGORY = _CATEGORY
    BLOCKS = (
        "init_x_linear.",
        "positional_encoding",
        "cond_seq_linear.",
        "register_tokens",
        "t_embedder.",
        *_numbered("double_layers.", 4),
        *_numbered("single_layers.", 32),
        "modF.",
        "final_linear.",
    )


class ModelMergeFlux1(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_flux1"
    SOURCE = "ModelMergeFlux1"
    CATEGORY = _CATEGORY
    BLOCKS = (
        "img_in.",
        "time_in.",
        "guidance_in",
        "vector_in.",
        "txt_in.",
        *_numbered("double_blocks.", 19),
        *_numbered("single_blocks.", 38),
        "final_layer.",
    )


class ModelMergeSD35_Large(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_sd35_large"
    SOURCE = "ModelMergeSD35_Large"
    CATEGORY = _CATEGORY
    BLOCKS = (
        "pos_embed.",
        "x_embedder.",
        "context_embedder.",
        "y_embedder.",
        "t_embedder.",
        *_numbered("joint_blocks.", 38),
        "final_layer.",
    )


class ModelMergeMochiPreview(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_mochi_preview"
    SOURCE = "ModelMergeMochiPreview"
    CATEGORY = _CATEGORY
    BLOCKS = (
        "pos_frequencies.",
        "t_embedder.",
        "t5_y_embedder.",
        "t5_yproj.",
        *_numbered("blocks.", 48),
        "final_layer.",
    )


class ModelMergeLTXV(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_ltxv"
    SOURCE = "ModelMergeLTXV"
    CATEGORY = _CATEGORY
    BLOCKS = (
        "patchify_proj.",
        "adaln_single.",
        "caption_projection.",
        *_numbered("transformer_blocks.", 28),
        "scale_shift_table",
        "proj_out.",
    )


class ModelMergeCosmos7B(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_cosmos7_b"
    SOURCE = "ModelMergeCosmos7B"
    CATEGORY = _CATEGORY
    BLOCKS = (
        "pos_embedder.",
        "extra_pos_embedder.",
        "x_embedder.",
        "t_embedder.",
        "affline_norm.",
        *_numbered("blocks.block", 28),
        "final_layer.",
    )


class ModelMergeCosmos14B(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_cosmos14_b"
    SOURCE = "ModelMergeCosmos14B"
    CATEGORY = _CATEGORY
    BLOCKS = (
        "pos_embedder.",
        "extra_pos_embedder.",
        "x_embedder.",
        "t_embedder.",
        "affline_norm.",
        *_numbered("blocks.block", 36),
        "final_layer.",
    )


class ModelMergeWAN2_1(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_wan2_1"
    SOURCE = "ModelMergeWAN2_1"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "1.3B model has 30 blocks, 14B model has 40 blocks. "
        "Image to video model has the extra img_emb."
    )
    BLOCKS = (
        "patch_embedding.",
        "time_embedding.",
        "time_projection.",
        "text_embedding.",
        "img_emb.",
        *_numbered("blocks.", 40),
        "head.",
    )


class ModelMergeCosmosPredict2_2B(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_cosmos_predict2_2_b"
    SOURCE = "ModelMergeCosmosPredict2_2B"
    CATEGORY = _CATEGORY
    BLOCKS = (
        "pos_embedder.",
        "x_embedder.",
        "t_embedder.",
        "t_embedding_norm.",
        *_numbered("blocks.", 28),
        "final_layer.",
    )


class ModelMergeCosmosPredict2_14B(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_cosmos_predict2_14_b"
    SOURCE = "ModelMergeCosmosPredict2_14B"
    CATEGORY = _CATEGORY
    BLOCKS = (
        "pos_embedder.",
        "x_embedder.",
        "t_embedder.",
        "t_embedding_norm.",
        *_numbered("blocks.", 36),
        "final_layer.",
    )


class ModelMergeQwenImage(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_qwen_image"
    SOURCE = "ModelMergeQwenImage"
    CATEGORY = _CATEGORY
    BLOCKS = (
        "pos_embeds.",
        "img_in.",
        "txt_norm.",
        "txt_in.",
        "time_text_embed.",
        *_numbered("transformer_blocks.", 60),
        "proj_out.",
    )


class ModelMergeKrea2(ModelMergeBlocks):
    NODE_TYPE = "dinkster.models.model_merge_krea2"
    SOURCE = "ModelMergeKrea2"
    CATEGORY = _CATEGORY
    BLOCKS = (
        "first.",
        "tmlp.",
        "txtmlp.",
        "tproj.",
        *_numbered("txtfusion.layerwise_blocks.", 2),
        "txtfusion.projector.",
        *_numbered("txtfusion.refiner_blocks.", 2),
        *_numbered("blocks.", 28),
        "last.",
    )

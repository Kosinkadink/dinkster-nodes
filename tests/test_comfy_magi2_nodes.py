"""Torch-free MAGI-2 pack schema, wire roundtrip, and cold discovery tests."""

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CLIP_TYPE,
    CONDITIONING_TYPE,
    GUIDER_TYPE,
    LATENT_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    OutputSpec,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_magi2 import NODES
from dinkster_magi2.magi2_nodes import (
    EmptyMagi2LatentAV,
    Magi2AudioDecode,
    Magi2Conditioning,
    Magi2DualCFGGuider,
    Magi2FlowSchedule,
    Magi2FlowUniPCSampler,
    Magi2RefinerUpscaleLatent,
    Magi2TextEncoderLoader,
    resolve_magi2_lengths,
)


def test_magi2_lengths_match_upstream_10s_and_5s_grids() -> None:
    # 10.0s @ 25 fps -> (250 - 1) // 8 + 1 = 32 video latent frames, 250 audio latent frames
    assert resolve_magi2_lengths(10.0) == (32, 250)
    # 5.0s @ 25 fps -> (125 - 1) // 8 + 1 = 16 video latent frames, 125 audio latent frames
    assert resolve_magi2_lengths(5.0) == (16, 125)


def test_magi2_pack_schemas_and_cold_discovery() -> None:
    expected = (
        (EmptyMagi2LatentAV, "dinkster.magi2.empty_magi2_latent_av", "model/latent/magi2"),
        (Magi2Conditioning, "dinkster.magi2.magi2_conditioning", "model/conditioning/magi2"),
        (Magi2DualCFGGuider, "dinkster.magi2.magi2_dual_cfg_guider", "model/sampling/guiders"),
        (
            Magi2RefinerUpscaleLatent,
            "dinkster.magi2.magi2_refiner_upscale_latent",
            "model/latent/magi2",
        ),
        (Magi2AudioDecode, "dinkster.magi2.magi2_audio_decode", "model/latent/audio"),
        (
            Magi2TextEncoderLoader,
            "dinkster.magi2.magi2_text_encoder_loader",
            "model/loaders",
        ),
        (
            Magi2FlowUniPCSampler,
            "dinkster.magi2.magi2_flow_unipc_sampler",
            "model/sampling/samplers",
        ),
        (
            Magi2FlowSchedule,
            "dinkster.magi2.magi2_flow_schedule",
            "model/sampling/schedulers",
        ),
    )
    assert len(NODES) == len(expected)
    for cls, node_type, category in expected:
        assert NODES.count(cls) == 1
        schema = cls.schema()
        assert schema.node_type == node_type
        assert schema.category == category
        assert schema_from_wire(schema_to_wire(schema)) == schema

    assert Magi2Conditioning.schema().outputs == (
        OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
        OutputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
    )
    assert Magi2DualCFGGuider.schema().inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
    assert Magi2DualCFGGuider.schema().outputs == (
        OutputSpec("GUIDER", TypeExpr.concrete(GUIDER_TYPE)),
    )
    assert Magi2AudioDecode.schema().inputs[1].type == TypeExpr.concrete(VAE_TYPE)
    assert Magi2AudioDecode.schema().outputs == (
        OutputSpec("AUDIO", TypeExpr.concrete(AUDIO_TYPE)),
    )
    assert len(Magi2RefinerUpscaleLatent.schema().outputs) == 2
    assert Magi2RefinerUpscaleLatent.schema().outputs[0].type == TypeExpr.concrete(LATENT_TYPE)
    assert EmptyMagi2LatentAV.schema().inputs[0].default == 896
    assert EmptyMagi2LatentAV.schema().inputs[1].default == 512
    assert Magi2Conditioning.schema().inputs[0].type == TypeExpr.concrete(CLIP_TYPE)
    assert Magi2TextEncoderLoader.schema().outputs == (
        OutputSpec("CLIP", TypeExpr.concrete(CLIP_TYPE)),
    )

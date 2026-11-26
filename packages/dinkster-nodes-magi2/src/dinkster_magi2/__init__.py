"""First-party MAGI-2 audio-video model nodes."""

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CLIP_TYPE,
    CONDITIONING_TYPE,
    GUIDER_TYPE,
    MODEL_TYPE,
    SAMPLER_TYPE,
    SIGMAS_TYPE,
    VAE_TYPE,
    TypeRegistry,
    comfy_resident_meta,
    default_pool,
    inference_nodes,
    register_audio_value_type,
    register_media_types,
    register_resident_type,
    resolver_from_env,
)
from dinkster_inference_wire.conditioning_wire import register_conditioning_type
from dinkster_inference_wire.hook_values import normalize_clip_hook_owners
from dinkster_inference_wire.sampling_wire import register_sampling_type

from .magi2_nodes import (
    EmptyMagi2LatentAV,
    Magi2AddAVGuide,
    Magi2AudioDecode,
    Magi2Conditioning,
    Magi2DualCFGGuider,
    Magi2FlowSchedule,
    Magi2FlowUniPCSampler,
    Magi2RefinerUpscaleLatent,
    Magi2TextEncoderLoader,
)

NODES = (
    EmptyMagi2LatentAV,
    Magi2AddAVGuide,
    Magi2AudioDecode,
    Magi2Conditioning,
    Magi2DualCFGGuider,
    Magi2FlowSchedule,
    Magi2FlowUniPCSampler,
    Magi2RefinerUpscaleLatent,
    Magi2TextEncoderLoader,
)


def register_types(registry: TypeRegistry) -> None:
    """Install consumed canonical codecs in this pack's isolated worker."""
    register_media_types(registry)
    if AUDIO_TYPE not in registry:
        register_audio_value_type(registry, AUDIO_TYPE, resolver_from_env())
    if CONDITIONING_TYPE not in registry:
        register_conditioning_type(registry, resident_table=default_pool())
    for sampling_id in (SAMPLER_TYPE, SIGMAS_TYPE):
        if sampling_id not in registry:
            register_sampling_type(registry, sampling_id)
    for type_id in (CLIP_TYPE, GUIDER_TYPE, MODEL_TYPE, VAE_TYPE):
        if type_id not in registry:
            register_resident_type(
                registry,
                type_id,
                table=default_pool(),
                meta=comfy_resident_meta,
                coerce=normalize_clip_hook_owners if type_id == CLIP_TYPE else None,
            )


__all__ = ["NODES", "register_types"]

inference_nodes(NODES)

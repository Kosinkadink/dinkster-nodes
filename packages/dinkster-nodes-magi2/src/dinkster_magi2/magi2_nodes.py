from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    GUIDER_TYPE,
    IMAGE_TYPE,
    LATENT_TYPE,
    MODEL_TYPE,
    SAMPLER_TYPE,
    SIGMAS_TYPE,
    VAE_TYPE,
    InputSpec,
    LatentStream,
    MultiStreamLatent,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_latent,
    with_identity_replacement,
)
from dinkster_pack_helpers.conditioning import conditioning_set_values
from dinkster_pack_helpers.model_file_loaders import (
    asset_input as _asset,
)
from dinkster_pack_helpers.model_file_loaders import (
    asset_path as _path,
)
from dinkster_pack_helpers.model_file_loaders import (
    clip_options as _clip_options,
)
from dinkster_pack_helpers.model_file_loaders import (
    combo_input as _combo,
)
from dinkster_pack_helpers.model_file_loaders import (
    embedding_roots as _embedding_roots,
)

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in "
    "the background, walking backwards, noise, background noise, hiss, hum, buzz, crackle, "
    "compression artifacts, digital clipping, distortion, muffled, muddy, unclear, echo, "
    "emotionless, flat delivery, robotic, mechanical, monotone, synthetic, text-to-speech"
)


def _extract_av_streams(value: Any) -> tuple[Any, Any | None]:
    if type(value) is MultiStreamLatent:
        streams = cast("MultiStreamLatent[Any]", value)
        video = streams.by_role("video")
        audio = streams.by_role("audio") if "audio" in streams.roles else None
        return video, audio
    if getattr(value, "is_nested", False):
        parts = value.unbind()
        return parts[0], (parts[1] if len(parts) > 1 else None)
    return value, None


def resolve_magi2_lengths(seconds: float) -> tuple[int, int]:
    dur = max(0.32, float(seconds))
    video_frames_25fps = max(1, round(dur * 25.0))
    video_latent_t = (video_frames_25fps - 1) // 8 + 1
    audio_latent_t = max(1, round(dur * 25.0))
    return video_latent_t, audio_latent_t


class Magi2TextEncoderLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.magi2.magi2_text_encoder_loader",
            display_name="Load MAGI-2 Text Encoder",
            category="model/loaders",
            description=(
                "Load Qwen3.5-27B configured for MAGI-2 conditioning "
                "(layer 62 tap, isolated CJK split)."
            ),
            inputs=(
                _asset("text_encoder", "model/text-encoder"),
                _combo("device", ("default", "cpu"), "default", advanced=True),
            ),
            outputs=(OutputSpec("CLIP", TypeExpr.concrete(CLIP_TYPE)),),
        )
        return with_identity_replacement(schema, "Magi2TextEncoderLoader")

    @classmethod
    def execute(cls, *, text_encoder: object, device: str = "default") -> Mapping[str, object]:
        path = _path(text_encoder, "text_encoder")
        initialize_inference_process()
        sd = importlib.import_module("dinkster_inference.sd")
        clip = sd.load_clip(
            ckpt_paths=[path],
            embedding_directory=_embedding_roots(),
            clip_type=sd.CLIPType.MAGI2,
            model_options=_clip_options("cpu") if device == "cpu" else {},
        )
        return cls.outputs(CLIP=clip)


class EmptyMagi2LatentAV(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        integer = TypeExpr.concrete(CORE_INT)
        schema = NodeSchema(
            node_type="dinkster.magi2.empty_magi2_latent_av",
            display_name="Empty MAGI-2 AV Latent",
            category="model/latent/magi2",
            description=(
                "Joint Wan 2.2 48-channel video + Oobleck 64-channel audio latent "
                "for MAGI-2 Preview."
            ),
            inputs=(
                InputSpec(
                    "width",
                    integer,
                    default=896,
                    widget=NumberWidget(min=64, max=4096, step=16),
                ),
                InputSpec(
                    "height",
                    integer,
                    default=512,
                    widget=NumberWidget(min=64, max=4096, step=16),
                ),
                InputSpec(
                    "seconds",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=10.0,
                    widget=NumberWidget(min=0.5, max=30.0, step=0.5, round=0.01),
                    doc=(
                        "Clip duration in seconds "
                        "(10.0s = 32 video latent frames + 250 audio tokens)."
                    ),
                ),
            ),
            outputs=(OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),),
        )
        return with_identity_replacement(schema, "EmptyMagi2LatentAV")

    @classmethod
    def execute(cls, *, width: int, height: int, seconds: float) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        video_t, audio_t = resolve_magi2_lengths(seconds)
        device = management.intermediate_device()
        video = torch.zeros((1, 48, video_t, height // 16, width // 16), device=device)
        audio = torch.zeros((1, 64, audio_t), device=device)
        return cls.outputs(
            LATENT={
                "samples": MultiStreamLatent(
                    (LatentStream("video", video), LatentStream("audio", audio))
                )
            }
        )


class Magi2Conditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        integer = TypeExpr.concrete(CORE_INT)
        schema = NodeSchema(
            node_type="dinkster.magi2.magi2_conditioning",
            display_name="MAGI-2 Text and Image Conditioning",
            category="model/conditioning/magi2",
            description=(
                "Encode positive and negative Qwen3.5-27B conditioning for MAGI-2, "
                "with optional `<Figure 1>` letterboxed Wan 2.2 VAE reference image conditioning."
            ),
            inputs=(
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec(
                    "prompt",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec(
                    "negative_prompt",
                    TypeExpr.concrete(CORE_STRING),
                    default=DEFAULT_NEGATIVE_PROMPT,
                    widget=StringWidget(multiline=True),
                ),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE), required=False),
                InputSpec("reference_image", TypeExpr.concrete(IMAGE_TYPE), required=False),
                InputSpec(
                    "width",
                    integer,
                    default=896,
                    widget=NumberWidget(min=64, max=4096, step=16),
                ),
                InputSpec(
                    "height",
                    integer,
                    default=512,
                    widget=NumberWidget(min=64, max=4096, step=16),
                ),
                InputSpec(
                    "use_ref_for_uncond",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    default=True,
                    doc=(
                        "Keep the reference image on the unconditional branch "
                        "so CFG steers text alone."
                    ),
                ),
            ),
            outputs=(
                OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
        )
        return with_identity_replacement(schema, "Magi2Conditioning")

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        vae: Any = None,
        reference_image: object = None,
        width: int = 896,
        height: int = 512,
        use_ref_for_uncond: bool = True,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        has_ref = reference_image is not None
        fig_ids = [1] if has_ref else None

        pos_tokens = clip.tokenize(prompt, magi2_figure_ids=fig_ids)
        pos_cond = clip.encode_from_tokens_scheduled(pos_tokens)
        neg_tokens = clip.tokenize(negative_prompt)
        neg_cond = clip.encode_from_tokens_scheduled(neg_tokens)

        if has_ref:
            if vae is None:
                raise ValueError(
                    "Magi2Conditioning requires `vae` (Wan 2.2 VAE) "
                    "when `reference_image` is connected."
                )
            sampling_mod = importlib.import_module("dinkster_inference.ldm.magi2.sampling")
            pixels = materialize_image(reference_image)
            letterboxed = sampling_mod.letterbox_reference_image(
                pixels, target_height=height, target_width=width, vae_scale_factor=32
            )
            ref_latent = vae.encode(letterboxed.float())
            if ref_latent.ndim == 4:
                ref_latent = ref_latent.unsqueeze(2)

            fig_tokens = pos_cond[0][1].get("magi2_figure_tokens", {})
            special_tok = fig_tokens.get(1)
            pos_ref = [{"latent": ref_latent, "special_token": special_tok}]
            neg_ref = [
                {
                    "latent": ref_latent if use_ref_for_uncond else torch.zeros_like(ref_latent),
                    "special_token": special_tok,
                }
            ]
            pos_cond = conditioning_set_values(pos_cond, {"magi2_ref_images": pos_ref})
            neg_cond = conditioning_set_values(neg_cond, {"magi2_ref_images": neg_ref})

        return cls.outputs(positive=pos_cond, negative=neg_cond)


class Magi2DualCFGGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        conditioning = TypeExpr.concrete(CONDITIONING_TYPE)
        flt = TypeExpr.concrete(CORE_FLOAT)
        schema = NodeSchema(
            node_type="dinkster.magi2.magi2_dual_cfg_guider",
            display_name="MAGI-2 Dual CFG Guider",
            category="model/sampling/guiders",
            description=(
                "Separate video and audio CFG guidance scales for MAGI-2 joint AV sampling."
            ),
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("positive", conditioning),
                InputSpec("negative", conditioning),
                InputSpec(
                    "video_cfg",
                    flt,
                    default=5.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1, round=0.01),
                ),
                InputSpec(
                    "audio_cfg",
                    flt,
                    default=7.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1, round=0.01),
                ),
            ),
            outputs=(OutputSpec("GUIDER", TypeExpr.concrete(GUIDER_TYPE)),),
        )
        return with_identity_replacement(schema, "Magi2DualCFGGuider")

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        positive: Any,
        negative: Any,
        video_cfg: float,
        audio_cfg: float,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        samplers = importlib.import_module("dinkster_inference.samplers")

        class GuiderMagi2DualCFG(samplers.CFGGuider):
            def set_dual_cfg(self, v_cfg: float, a_cfg: float) -> None:
                self.video_cfg = v_cfg
                self.audio_cfg = a_cfg
                self.cfg = max(v_cfg, a_cfg)

            def sample(self, noise: Any, latent_image: Any, *args: Any, **kwargs: Any) -> Any:
                self._v_numel: int | None = None
                if getattr(latent_image, "is_nested", False):
                    parts = latent_image.unbind()
                    if len(parts) >= 2:
                        self._v_numel = math.prod(parts[0].shape[1:])
                parent: Any = super()
                return parent.sample(noise, latent_image, *args, **kwargs)

            def predict_noise(
                self,
                x: Any,
                timestep: Any,
                model_options: dict[str, Any] | None = None,
                seed: Any = None,
            ) -> Any:
                opts = {} if model_options is None else model_options
                split = getattr(self, "_v_numel", None)
                parent: Any = super()
                if split is None or math.isclose(self.video_cfg, self.audio_cfg):
                    self.cfg = self.video_cfg
                    return parent.predict_noise(x, timestep, opts, seed)

                v_scale, a_scale = self.video_cfg, self.audio_cfg

                def dual_cfg(args: dict[str, Any]) -> Any:
                    cond, uncond = args["cond"], args["uncond"]
                    out = uncond + (cond - uncond) * v_scale
                    out[..., split:] = (
                        uncond[..., split:] + (cond[..., split:] - uncond[..., split:]) * a_scale
                    )
                    return out

                merged_opts = {
                    **opts,
                    "sampler_cfg_function": dual_cfg,
                    "disable_cfg1_optimization": True,
                }
                return parent.predict_noise(x, timestep, merged_opts, seed)

        guider = GuiderMagi2DualCFG(model)
        guider.set_conds(positive, negative)
        guider.set_dual_cfg(video_cfg, audio_cfg)
        return cls.outputs(GUIDER=guider)


class Magi2RefinerUpscaleLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        integer = TypeExpr.concrete(CORE_INT)
        latent_t = TypeExpr.concrete(LATENT_TYPE)
        schema = NodeSchema(
            node_type="dinkster.magi2.magi2_refiner_upscale_latent",
            display_name="MAGI-2 Refiner Upscale Latent",
            category="model/latent/magi2",
            description=(
                "Extract the MAGI-2 Preview video latent, trilinearly upsample in normalized "
                "Wan 2.2 space to `(2*T - 1, height // 16, width // 16)`, and split out the "
                "preserved audio latent."
            ),
            inputs=(
                InputSpec("av_latent", latent_t),
                InputSpec(
                    "width",
                    integer,
                    default=1920,
                    widget=NumberWidget(min=64, max=4096, step=16),
                ),
                InputSpec(
                    "height",
                    integer,
                    default=1088,
                    widget=NumberWidget(min=64, max=4096, step=16),
                ),
            ),
            outputs=(
                OutputSpec("VIDEO_LATENT", latent_t, display_name="refiner_video_latent"),
                OutputSpec("AUDIO_LATENT", latent_t, display_name="audio_latent"),
            ),
        )
        return with_identity_replacement(schema, "Magi2RefinerUpscaleLatent")

    @classmethod
    def execute(
        cls, *, av_latent: object, width: int = 1920, height: int = 1088
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch_nn_f = importlib.import_module("torch.nn.functional")
        latent_formats = importlib.import_module("dinkster_inference.latent_formats")

        mat = cast("Mapping[str, Any]", materialize_latent(av_latent))
        video_raw, audio_raw = _extract_av_streams(mat["samples"])
        if video_raw.ndim != 5 or video_raw.shape[1] != 48:
            raise ValueError(
                f"Expected MAGI-2 48-channel 5D video latent, got shape {tuple(video_raw.shape)}"
            )

        fmt = latent_formats.Magi2AV()
        v_norm = fmt.process_video_in(video_raw.float())
        target_t = max(1, v_norm.shape[2] * 2 - 1)
        target_h = max(1, height // 16)
        target_w = max(1, width // 16)
        v_up_norm = torch_nn_f.interpolate(
            v_norm, size=(target_t, target_h, target_w), mode="trilinear", align_corners=True
        )
        v_up_raw = fmt.process_video_out(v_up_norm)

        if audio_raw is None:
            torch = importlib.import_module("torch")
            audio_raw = torch.zeros((video_raw.shape[0], 64, 1), device=video_raw.device)

        return cls.outputs(
            VIDEO_LATENT={"samples": v_up_raw},
            AUDIO_LATENT={"samples": audio_raw},
        )


class Magi2AudioDecode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.magi2.magi2_audio_decode",
            display_name="MAGI-2 Audio VAE Decode",
            category="model/latent/audio",
            description=(
                "Decode MAGI-2 Oobleck audio latents and sinc-resample by 441/512 so 25 Hz "
                "audio tokens match exact video duration at 44.1 kHz."
            ),
            inputs=(
                InputSpec("samples", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("audio_vae", TypeExpr.concrete(VAE_TYPE)),
            ),
            outputs=(OutputSpec("AUDIO", TypeExpr.concrete(AUDIO_TYPE)),),
        )
        return with_identity_replacement(schema, "Magi2AudioDecode")

    @classmethod
    def execute(cls, *, samples: object, audio_vae: Any) -> Mapping[str, object]:
        initialize_inference_process()
        sampling_mod = importlib.import_module("dinkster_inference.ldm.magi2.sampling")
        mat = cast("Mapping[str, Any]", materialize_latent(samples))
        raw: object = mat["samples"]
        if type(raw) is MultiStreamLatent or getattr(raw, "is_nested", False):
            _, audio_lat = _extract_av_streams(raw)
        else:
            audio_lat = raw
        if audio_lat is None:
            raise ValueError("No audio latent stream found in input.")

        decoded = audio_vae.decode(audio_lat).movedim(-1, 1)  # [B, C, samples]
        std = decoded.std(dim=(1, 2), keepdim=True) * 5.0
        std = std.clamp(min=1.0)
        decoded = decoded / std
        resampled = sampling_mod.resample_magi2_audio(decoded, ratio=441.0 / 512.0)
        sample_rate = int(getattr(audio_vae, "audio_sample_rate", 44100))
        return cls.outputs(AUDIO={"waveform": resampled, "sample_rate": sample_rate})


class Magi2FlowUniPCSampler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.magi2.magi2_flow_unipc_sampler",
            display_name="MAGI-2 Flow UniPC Sampler",
            category="model/sampling/samplers",
            description=(
                "Rectified-Flow UniPC multistep predictor-corrector (`alpha_t = 1 - sigma_t`, "
                "`bh2`, order 2) matching MAGI-2's `FlowUniPCMultistepScheduler`."
            ),
            inputs=(
                InputSpec(
                    "solver_order",
                    TypeExpr.concrete(CORE_INT),
                    default=2,
                    widget=NumberWidget(min=1, max=3, step=1),
                ),
                _combo("solver_type", ("bh2", "bh1"), "bh2"),
            ),
            outputs=(OutputSpec("SAMPLER", TypeExpr.concrete(SAMPLER_TYPE)),),
        )
        return with_identity_replacement(schema, "Magi2FlowUniPCSampler")

    @classmethod
    def execute(cls, *, solver_order: int = 2, solver_type: str = "bh2") -> Mapping[str, object]:
        initialize_inference_process()
        samplers = importlib.import_module("dinkster_inference.samplers")
        sampling_mod = importlib.import_module("dinkster_inference.ldm.magi2.sampling")
        sampler = samplers.KSAMPLER(
            sampling_mod.sample_flow_unipc,
            extra_options={"solver_order": int(solver_order), "solver_type": solver_type},
        )
        return cls.outputs(SAMPLER=sampler)


class Magi2FlowSchedule(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.magi2.magi2_flow_schedule",
            display_name="MAGI-2 Flow Sigma Schedule",
            category="model/sampling/schedulers",
            description=(
                "Shifted linear flow schedule (`shift=7.0` for Preview, `shift=5.0` for Refiner) "
                "matching MAGI-2's `FlowUniPCMultistepScheduler.set_timesteps`."
            ),
            inputs=(
                InputSpec(
                    "steps",
                    TypeExpr.concrete(CORE_INT),
                    default=100,
                    widget=NumberWidget(min=1, max=500, step=1),
                ),
                InputSpec(
                    "shift",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=7.0,
                    widget=NumberWidget(min=0.1, max=50.0, step=0.1, round=0.01),
                ),
            ),
            outputs=(OutputSpec("SIGMAS", TypeExpr.concrete(SIGMAS_TYPE)),),
        )
        return with_identity_replacement(schema, "Magi2FlowSchedule")

    @classmethod
    def execute(cls, *, steps: int = 100, shift: float = 7.0) -> Mapping[str, object]:
        initialize_inference_process()
        sampling_mod = importlib.import_module("dinkster_inference.ldm.magi2.sampling")
        sigmas = sampling_mod.flow_unipc_sigmas(steps=int(steps), shift=float(shift))
        return cls.outputs(SIGMAS=sigmas)

"""CPU execution tests for MAGI-2 Dinkster nodes."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from dinkster_api.v1 import MultiStreamLatent, initialize_inference_process
from dinkster_magi2.magi2_nodes import (
    EmptyMagi2LatentAV,
    Magi2AddAVGuide,
    Magi2AudioDecode,
    Magi2Conditioning,
    Magi2DualCFGGuider,
    Magi2RefinerUpscaleLatent,
)

initialize_inference_process()
NestedTensor = importlib.import_module("dinkster_inference.nested_tensor").NestedTensor


def test_empty_magi2_av_latent_and_refiner_upscale() -> None:
    out = EmptyMagi2LatentAV.execute(width=896, height=512, seconds=10.0)["LATENT"]
    streams = out["samples"]
    assert type(streams) is MultiStreamLatent
    video = streams.by_role("video")
    audio = streams.by_role("audio")
    assert video.shape == (1, 48, 32, 32, 56)
    assert audio.shape == (1, 64, 250)

    up = Magi2RefinerUpscaleLatent.execute(av_latent=out, width=1920, height=1088)
    v_up = up["VIDEO_LATENT"]["samples"]
    a_up = up["AUDIO_LATENT"]["samples"]
    assert v_up.shape == (1, 48, 63, 68, 120)
    assert a_up.shape == (1, 64, 250)


def test_magi2_conditioning_reference_image_and_uncond_toggle() -> None:
    fig_vec = torch.arange(16, dtype=torch.float32)

    class StubClip:
        def tokenize(self, text: str, magi2_figure_ids=None):
            return {"text": text, "figs": magi2_figure_ids}

        def encode_from_tokens_scheduled(self, tokens):
            extra = {"magi2_txt_len": 5}
            if tokens.get("figs"):
                extra["magi2_figure_tokens"] = {1: fig_vec.clone()}
            return [[torch.ones(1, 5, 16), extra]]

    class StubVAE:
        def encode(self, pixels: torch.Tensor) -> torch.Tensor:
            assert pixels.dtype == torch.float32
            assert pixels.shape == (1, 576, 896, 3)
            return torch.full((1, 48, 1, 36, 56), 2.5)

    img = torch.rand(1, 200, 300, 3)
    res_keep = Magi2Conditioning.execute(
        clip=StubClip(),
        prompt="hello",
        negative_prompt="bad",
        vae=StubVAE(),
        reference_image=img,
        width=896,
        height=512,
        use_ref_for_uncond=True,
    )
    pos_refs = res_keep["positive"][0][1]["magi2_ref_images"]
    neg_refs = res_keep["negative"][0][1]["magi2_ref_images"]
    assert torch.equal(pos_refs[0]["latent"], neg_refs[0]["latent"])
    assert torch.equal(pos_refs[0]["special_token"], fig_vec)

    res_zero = Magi2Conditioning.execute(
        clip=StubClip(),
        prompt="hello",
        negative_prompt="bad",
        vae=StubVAE(),
        reference_image=img,
        width=896,
        height=512,
        use_ref_for_uncond=False,
    )
    neg_zero_entry = res_zero["negative"][0][1]["magi2_ref_images"][0]
    assert neg_zero_entry.get("normalized") is True
    assert float(neg_zero_entry["latent"].abs().max().item()) == 0.0


@pytest.mark.parametrize("video_cfg,audio_cfg", [(5.0, 7.0), (2.0, 0.0)])
def test_magi2_dual_cfg_guider_splits_streams(video_cfg: float, audio_cfg: float) -> None:
    model = SimpleNamespace(model_options={}, is_dynamic=lambda: False)
    pos = [[torch.ones(1, 2, 4), {}]]
    neg = [[torch.zeros(1, 2, 4), {}]]
    guider = Magi2DualCFGGuider.execute(
        model=model, positive=pos, negative=neg, video_cfg=video_cfg, audio_cfg=audio_cfg
    )["GUIDER"]
    guider.inner_model = None
    guider.conds = guider.original_conds

    latent = NestedTensor((torch.zeros(1, 2, 1, 2, 2), torch.zeros(1, 3, 2)))
    assert guider.sample(None, latent, None, torch.tensor([])) is latent
    cond = torch.full((1, 1, 14), 3.0)
    uncond = torch.full((1, 1, 14), 1.0)
    out = guider.predict_noise(
        torch.zeros(1, 1, 14),
        torch.tensor([0.5]),
        {"sampler_calc_cond_batch_function": lambda _a: (cond, uncond)},
        seed=7,
    )
    expected = torch.tensor([[[1.0 + 2.0 * video_cfg] * 8 + [1.0 + 2.0 * audio_cfg] * 6]])
    assert torch.allclose(out, expected)


def test_magi2_audio_decode_resamples_250_tokens_to_exact_10s() -> None:
    class StubAudioVAE:
        audio_sample_rate = 44100

        def decode(self, lat: torch.Tensor) -> torch.Tensor:
            # AudioOobleckVAE.decode returns [B, samples, C] in sd.VAE.decode!
            b, _, t_aud = lat.shape
            return (
                torch.sin(torch.linspace(0.0, 20.0, t_aud * 2048)).view(1, -1, 1).expand(b, -1, 2)
            )

    audio_lat = {"samples": torch.zeros(1, 64, 250)}
    audio = Magi2AudioDecode.execute(samples=audio_lat, audio_vae=StubAudioVAE())["AUDIO"]
    assert audio["sample_rate"] == 44100
    # 250 * 2048 = 512,000 samples * (441 / 512) = 441,000 samples = 10.000s @ 44.1 kHz
    assert audio["waveform"].shape == (1, 2, 441000)


def test_magi2_conditioning_multi_figure_family_and_square_resize() -> None:
    seen_figs: list[list[int] | None] = []
    encoded_shapes: list[tuple[int, ...]] = []

    class StubClip:
        def tokenize(self, text: str, magi2_figure_ids=None):
            seen_figs.append(magi2_figure_ids)
            return {"text": text, "figs": magi2_figure_ids}

        def encode_from_tokens_scheduled(self, tokens):
            extra: dict[str, object] = {"magi2_txt_len": 8}
            figs = tokens.get("figs") or []
            if figs:
                extra["magi2_figure_tokens"] = {
                    int(fid): torch.full((16,), float(fid)) for fid in figs
                }
            return [[torch.ones(1, 8, 16), extra]]

    class StubVAE:
        def encode(self, pixels: torch.Tensor) -> torch.Tensor:
            encoded_shapes.append(tuple(pixels.shape))
            return torch.ones(1, 48, 1, pixels.shape[1] // 16, pixels.shape[2] // 16)

    img1 = torch.rand(1, 200, 300, 3)
    img2 = torch.rand(1, 300, 200, 3)
    img3 = torch.rand(1, 180, 180, 3)
    res = Magi2Conditioning.execute(
        clip=StubClip(),
        prompt="<Figure 1> meets <Figure 2> inside <Figure 3>.",
        vae=StubVAE(),
        reference_image=img1,
        extra_reference_images={"ref_image_2": img3, "ref_image_1": img2},
        resize_type="square",
        width=896,
        height=512,
        use_ref_for_uncond=True,
    )
    assert seen_figs[0] == [1, 2, 3]
    assert encoded_shapes == [(1, 320, 320, 3)] * 3
    pos_refs = res["positive"][0][1]["magi2_ref_images"]
    assert len(pos_refs) == 3
    for idx, entry in enumerate(pos_refs, start=1):
        assert entry["latent"].shape == (1, 48, 1, 20, 20)
        assert torch.allclose(entry["special_token"], torch.full((16,), float(idx)))


def test_magi2_add_av_guide_chains_keyframes_and_25hz_audio() -> None:
    class StubVideoVAE:
        def encode(self, pixels: torch.Tensor) -> torch.Tensor:
            t_lat = (pixels.shape[0] - 1) // 4 + 1
            return torch.full((1, 48, t_lat, pixels.shape[1] // 16, pixels.shape[2] // 16), 3.25)

    class StubAudioVAE:
        audio_sample_rate = 44100

        def encode(self, wav_btc: torch.Tensor) -> torch.Tensor:
            # AudioOobleckVAE hop length is 2048 samples -> [B, 64, samples // 2048]
            t_aud = max(1, wav_btc.shape[1] // 2048)
            return torch.full((wav_btc.shape[0], 64, t_aud), -1.5)

    base = EmptyMagi2LatentAV.execute(width=512, height=320, seconds=4.0)["LATENT"]
    # Anchor first frame (t=0) + 2.0s of 44.1 kHz stereo audio (88,200 samples -> 50 MAGI-2 tokens!)
    first_img = torch.rand(1, 320, 512, 3)
    wav_2s = {"waveform": torch.zeros(1, 2, 88200), "sample_rate": 44100}
    guided1 = Magi2AddAVGuide.execute(
        av_latent=base,
        vae=StubVideoVAE(),
        audio_vae=StubAudioVAE(),
        image=first_img,
        audio=wav_2s,
        frame_idx=0,
        strength=1.0,
    )["LATENT"]

    # Chain a second guide anchoring the last video frame (frame_idx=-1 -> latent idx 12)
    guided2 = Magi2AddAVGuide.execute(
        av_latent=guided1,
        vae=StubVideoVAE(),
        image=first_img,
        frame_idx=-1,
        strength=1.0,
    )["LATENT"]

    samples = guided2["samples"]
    masks = guided2["noise_mask"]
    assert type(samples) is MultiStreamLatent
    assert type(masks) is MultiStreamLatent
    v_lat = samples.by_role("video")
    a_lat = samples.by_role("audio")
    v_mask = masks.by_role("video")
    a_mask = masks.by_role("audio")

    assert v_lat.shape == (1, 48, 13, 20, 32)
    assert a_lat.shape == (1, 64, 100)
    # Frame 0 and frame 12 anchored (mask == 0), middle frames 1..11 unmasked (mask == 1)
    assert float(v_mask[:, :, 0].max().item()) == 0.0
    assert float(v_mask[:, :, -1].max().item()) == 0.0
    assert torch.allclose(v_mask[:, :, 1:-1], torch.ones_like(v_mask[:, :, 1:-1]))
    assert torch.allclose(v_lat[:, :, 0], torch.full_like(v_lat[:, :, 0], 3.25))
    assert torch.allclose(v_lat[:, :, -1], torch.full_like(v_lat[:, :, -1], 3.25))

    # 2.0s audio -> 88200 * (512 / 441) = 102400 samples // 2048 = exact 50 tokens!
    assert torch.allclose(a_mask[:, :, :50], torch.zeros_like(a_mask[:, :, :50]))
    assert torch.allclose(a_mask[:, :, 50:], torch.ones_like(a_mask[:, :, 50:]))
    assert torch.allclose(a_lat[:, :, :50], torch.full_like(a_lat[:, :, :50], -1.5))

"""CPU behavior for native ComfyUI noise nodes."""

import torch
from dinkster_sampling.noise import DisableNoise, RandomNoise


def test_random_noise_uses_seeded_batch_indices_and_disable_noise_preserves_layout() -> None:
    latent = {
        "samples": torch.zeros((2, 1, 2, 2), dtype=torch.float32),
        "batch_index": [7, 3],
    }
    noise = RandomNoise.execute(noise_seed=42)["NOISE"].generate_noise(latent)
    # ComfyUI Noise_Random, F bc5f880d: seed=42, float32 zeros (2,1,2,2),
    # batch_index=[7,3]; generate_noise delegates to comfy.sample.prepare_noise.
    assert noise.flatten().tolist() == [
        -0.049903348088264465,
        0.5263369679450989,
        -0.008498823270201683,
        0.7290605902671814,
        1.110290288925171,
        -1.6897989511489868,
        -0.9889599084854126,
        0.9579718112945557,
    ]
    disabled = DisableNoise.execute()["NOISE"].generate_noise(latent)
    assert disabled.dtype == latent["samples"].dtype
    assert disabled.device == latent["samples"].device
    assert torch.equal(disabled, torch.zeros_like(latent["samples"]))

"""CPU Hunyuan image latent behavior and ordinary manifest hosting."""

from __future__ import annotations

import torch
from dinkster_hunyuan.hunyuan_latent import EmptyHunyuanImageLatent, HunyuanRefinerLatent


def test_hunyuan_image_latents_preserve_source_values_and_allocate_on_cpu() -> None:
    empty = EmptyHunyuanImageLatent.execute(width=195, height=99, batch_size=2)["LATENT"]
    empty_samples = empty["samples"]
    assert tuple(empty_samples.shape) == (2, 64, 3, 6)
    assert empty_samples.dtype == torch.float32
    assert empty_samples.device.type == "cpu"
    assert not empty_samples.count_nonzero()

    samples = torch.arange(2 * 16 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 16, 3, 5, 7)
    source = {"samples": samples, "batch_index": [3, 1], "unrelated": "retained"}
    positive = [[torch.ones((1, 2, 4)), {"positive": True}]]
    negative = [[torch.full((1, 2, 4), 2.0), {"negative": True}]]
    result = HunyuanRefinerLatent.execute(
        positive=positive, negative=negative, latent=source, noise_augmentation=0.35
    )
    refined = result["latent"]
    refined_samples = refined["samples"]
    assert tuple(refined_samples.shape) == (2, 32, 3, 5, 7)
    assert refined_samples.dtype == torch.float32
    assert refined_samples.device.type == "cpu"
    assert not refined_samples.count_nonzero()
    assert set(refined) == {"samples"}
    assert source["samples"] is samples
    assert source["batch_index"] == [3, 1]
    assert source["unrelated"] == "retained"
    assert positive[0][1] == {"positive": True}
    assert negative[0][1] == {"negative": True}
    for conditioning, metadata_key in (
        (result["positive"], "positive"),
        (result["negative"], "negative"),
    ):
        assert conditioning[0][0] is (positive if metadata_key == "positive" else negative)[0][0]
        assert conditioning[0][1][metadata_key] is True
        assert conditioning[0][1]["concat_latent_image"] is samples
        assert conditioning[0][1]["noise_augmentation"] == 0.35

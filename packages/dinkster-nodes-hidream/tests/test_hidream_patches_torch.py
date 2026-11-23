from __future__ import annotations

import torch
from dinkster_api.v1 import Item, Members
from dinkster_hidream.hidream_patches import (
    HiDreamO1ReferenceImages,
)


def test_reference_position_order_append_and_shallow_identity() -> None:
    token, negative_token, marker = object(), object(), object()
    prior, negative_prior = torch.tensor([-7.0]), torch.tensor([-13.0])
    positive = [[token, {"reference_latents": [prior], "marker": marker}]]
    negative = [[negative_token, {"reference_latents": [negative_prior], "prompt_type": "old"}]]
    # Members arrive in position order; their keys say nothing about order.
    images = Members(
        Item(key, None if value is None else torch.full((1, 2, 3, 3), float(value)))
        for key, value in (("image_10", 10), ("image_2", 2), ("image_1", None), ("image_100", 100))
    )
    result = HiDreamO1ReferenceImages.execute(positive=positive, negative=negative, images=images)
    for output, original, previous in (
        ("positive", positive, prior),
        ("negative", negative, negative_prior),
    ):
        entry = result[output][0]
        assert entry[0] is original[0][0] and entry[1] is not original[0][1]
        refs = entry[1]["reference_latents"]
        assert refs[0] is previous
        assert refs[1:] == [member.value for member in images]
        assert [float(ref[0, 0, 0, 0]) for ref in refs[1:] if ref is not None] == [10, 2, 100]
        assert len(original[0][1]["reference_latents"]) == 1
    assert result["positive"][0][1]["marker"] is marker
    assert result["negative"][0][1]["prompt_type"] == "negative"
    assert negative[0][1]["prompt_type"] == "old"
    for empty in ({}, {"images": Members()}):
        unchanged = HiDreamO1ReferenceImages.execute(positive=positive, negative=negative, **empty)
        assert unchanged["positive"] is positive and unchanged["negative"] is negative

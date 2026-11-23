"""MiniMax H3 reference prompt items from the reference groups."""

from __future__ import annotations

from typing import Any

import torch
from dinkster_api.v1 import Item, Members
from dinkster_minimax.minimax_h3_reference import MiniMaxH3ReferenceToVideo


class _Clip:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def tokenize(self, prompt: str, *, minimax_ref_items: list[dict[str, Any]]) -> str:
        self.items = minimax_ref_items
        return prompt

    def encode_from_tokens_scheduled(self, tokens: str) -> list[list[object]]:
        return [[tokens, {}]]


def test_reference_videos_pair_with_soundtracks_by_position() -> None:
    clip = _Clip()
    soundtrack = {"waveform": torch.zeros(1, 1, 8), "sample_rate": 8}
    MiniMaxH3ReferenceToVideo.execute(
        clip=clip,
        prompt="p",
        width=64,
        height=64,
        length=5,
        ref_images=Members(()),
        ref_videos=Members(
            (
                Item("ref_video_1", torch.rand(5, 32, 32, 3)),
                Item("ref_video_0", torch.rand(5, 32, 64, 3)),
            )
        ),
        # The soundtrack's key names the second video, but its position is the first.
        ref_video_audios=Members((Item("ref_video_audio_0", soundtrack),)),
        ref_audios=Members(()),
    )
    assert [item["type"] for item in clip.items] == ["audio", "video", "video"]
    assert tuple(clip.items[1]["data"].shape[1:3]) == (32, 32)

"""Face drawing pixels, source preservation and ordinary-host cache behavior."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from dinkster_api.v1 import (
    materialize_image,
    materialize_latent,
)
from dinkster_vision_mediapipe.mediapipe import MediaPipeFaceMask, MediaPipeFaceMeshVisualize


def _packet() -> dict[str, Any]:
    def ring(indices: list[int]) -> frozenset[tuple[int, int]]:
        return frozenset(zip(indices, indices[1:] + indices[:1], strict=True))

    landmarks = np.array(
        [[1, 1], [6, 1], [6, 4], [1, 4], [2, 2], [3, 2], [3, 3], [2, 3]], dtype=np.float32
    )
    return {
        "frames": [
            [{"landmarks_xy": landmarks}, {"landmarks_xy": landmarks + [4, 0]}],
            [],
            [{"landmarks_xy": landmarks}],
        ],
        "image_size": (7, 11),
        "connection_sets": {
            "face_oval": ring([0, 1, 2, 3]),
            **{part: ring([4, 5, 6, 7]) for part in ("lips", "left_eye", "right_eye", "irises")},
            "left_eyebrow": frozenset({(4, 5)}),
            "right_eyebrow": frozenset({(7, 6)}),
            "nose": frozenset({(0, 6)}),
            "tesselation": frozenset({(0, 2), (1, 3), (0, 99)}),
        },
    }


def _visualize(packet: Any, **options: Any) -> Any:
    return MediaPipeFaceMeshVisualize.execute(
        face_landmarks=packet,
        **{
            "connections": {"connections": "fill"},
            "color": "#d02070",
            "thickness": 2,
            "point_size": 3,
            **options,
        },
    )["IMAGE"]


def _assert_pixels(image: Any, mask: Any) -> None:
    image, mask = materialize_image(image), torch.as_tensor(materialize_latent(mask))
    expected = torch.zeros(3, 7, 11)
    expected[0, 1:5, 1:11] = 1
    expected[2, 1:5, 1:7] = 1
    assert torch.equal(mask, expected)
    assert image.dtype == mask.dtype == torch.float32
    assert image.device.type == mask.device.type == "cpu"
    assert torch.equal(image, expected.unsqueeze(-1) * torch.tensor([208, 32, 112]) / 255)


def test_region_union_fill_and_source_preservation() -> None:
    packet = _packet()
    before = copy.deepcopy(packet)
    mask = MediaPipeFaceMask.execute(face_landmarks=packet, regions={"regions": "all"})["MASK"]
    _assert_pixels(_visualize(packet), mask)
    lips = MediaPipeFaceMask.execute(
        face_landmarks=packet, regions={"regions": "custom", "lips": True}
    )["MASK"]
    expected = torch.zeros(3, 7, 11)
    expected[0, 2:4, 2:4] = expected[0, 2:4, 6:8] = expected[2, 2:4, 2:4] = 1
    assert torch.equal(lips, expected)
    assert torch.equal(
        MediaPipeFaceMask.execute(
            face_landmarks=packet, regions="custom", **{"regions.lips": True}
        )["MASK"],
        expected,
    )
    assert (
        torch.count_nonzero(
            MediaPipeFaceMask.execute(face_landmarks=packet, regions={"regions": "custom"})["MASK"]
        )
        == 0
    )
    for original, actual in zip(before["frames"], packet["frames"], strict=True):
        for a, b in zip(original, actual, strict=True):
            np.testing.assert_array_equal(a["landmarks_xy"], b["landmarks_xy"])


def test_edges_points_background_and_empty_batches() -> None:
    packet = _packet()
    background = torch.linspace(-0.2, 1.2, 4 * 7 * 11 * 4).reshape(4, 7, 11, 4)
    before = background.clone()
    output = _visualize(
        packet,
        image=background,
        thickness=0,
        point_size=0,
        connections={"connections": "all"},
        color="invalid-color",
    )
    expected = before[..., :3].mul(255).add(0.5).clamp(0, 255).to(torch.uint8).float().div(255)
    assert torch.equal(output, expected) and torch.equal(background, before)
    output = _visualize(
        packet,
        image=background,
        thickness=1,
        point_size=0,
        connections={"connections": "custom", "left_eyebrow": True},
        color="invalid-color",
    )
    expected[0, 2, 2:4] = expected[0, 2, 6:8] = expected[2, 2, 2:4] = torch.tensor([0.0, 1.0, 0.0])
    assert torch.equal(output, expected) and torch.equal(background, before)
    for size in (0, 1, 3):
        output = _visualize(
            packet, thickness=0, point_size=size, connections={"connections": "custom"}
        )
        assert torch.count_nonzero(output[1]) == 0
        if size:
            assert torch.equal(output[2, 1, 1], torch.tensor([208, 32, 112]) / 255)
        else:
            assert torch.count_nonzero(output) == 0
    # The out-of-range tessellation edge is ignored, not indexed.
    assert _visualize(packet, connections={"connections": "custom", "tesselation": True}).shape == (
        3,
        7,
        11,
        3,
    )
    empty = {**packet, "frames": []}
    assert MediaPipeFaceMask.execute(face_landmarks=empty, regions={"regions": "all"})[
        "MASK"
    ].shape == (0, 7, 11)
    assert _visualize(empty).shape == (0, 7, 11, 3)

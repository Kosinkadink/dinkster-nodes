"""Color fields give the same node result whether a node receives the stored color
text or the color the engine fits to the field's declared opacity."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_3d.mesh_render import RenderMesh
from dinkster_3d.splat_render import RenderSplat
from dinkster_api.v1 import ColorWidget, Input, Node, Record, TriangleMeshBatch
from dinkster_image.drawing import DrawMask, DrawRegion, DrawText, MakeImage
from dinkster_image.mask import ImageToMask, MaskToImage, TextMask
from dinkster_image.ported.image_crop import ImageCropToMask
from dinkster_image.ported.image_file_tools import Painter
from dinkster_image.ported.image_transform import PadImageNode
from dinkster_image.ported.text_overlay import TextOverlay
from dinkster_image.types import Region
from dinkster_utilities.color_resolution import ColorToRGBInt
from dinkster_values.colors import fit_color
from dinkster_vision_mediapipe.mediapipe import MediaPipeFaceMeshVisualize

VALUES = ("#3366cc80", "#3366cc")
# A color unlike both values, to show the field's color reaches the output.
CONTRAST = "#cc6633"


def _inputs(rows: Iterable[object]) -> Iterator[Input]:
    """Every Input in a body, through sections, rows, choice options and shared rows."""
    for row in rows:
        if isinstance(row, Input):
            yield row
        yield from _inputs(getattr(row, "rows", ()))
        yield from _inputs(getattr(row, "shared", ()))
        for option in getattr(row, "options", ()):
            yield from _inputs(getattr(option, "rows", ()))


def _declared_widget(node: type[Node], field: str) -> ColorWidget:
    found = [row for row in _inputs(node.define_schema().body) if row.id == field]
    assert len(found) == 1, f"{node.__name__}.{field} declared {len(found)} times"
    widget = found[0].widget
    assert isinstance(widget, ColorWidget)
    return widget


def _same(first: object, second: object) -> bool:
    if isinstance(first, Mapping):
        if not isinstance(second, Mapping) or first.keys() != second.keys():
            return False
        left, right = dict[Any, Any](first), dict[Any, Any](second)
        return all(_same(left[key], right[key]) for key in left)
    if isinstance(first, torch.Tensor):
        return (
            isinstance(second, torch.Tensor)
            and first.dtype == second.dtype
            and torch.equal(first, second)
        )
    if isinstance(first, np.ndarray):
        return (
            isinstance(second, np.ndarray)
            and first.dtype == second.dtype
            and np.array_equal(first, second)
        )
    return first == second


def _canvas() -> np.ndarray:
    return np.full((1, 12, 16, 3), 0.5, dtype=np.float32)


def _region_mask() -> np.ndarray:
    mask = np.zeros((1, 12, 16), dtype=np.float32)
    mask[:, 2:8, 3:10] = 1.0
    return mask


def _splat() -> SimpleNamespace:
    # The second item has no splats, so its frame is the background alone.
    return SimpleNamespace(
        positions=torch.tensor(
            [[[0, 0, 0], [0, 0, 1], [0, 0, 0]], [[0, 0, 0]] * 3], dtype=torch.float64
        ),
        scales=torch.tensor([[[0.18, 0.07, 0.03]] * 3] * 2, dtype=torch.float64),
        rotations=torch.tensor([[[1, 0, 0, 0]] * 3] * 2, dtype=torch.float64),
        opacities=torch.tensor([[[0.6], [0.3], [1]], [[1], [1], [1]]], dtype=torch.float64),
        sh=torch.zeros(2, 3, 1, 3, dtype=torch.float64),
        counts=torch.tensor([2, 0], dtype=torch.int32),
    )


def _mesh() -> TriangleMeshBatch:
    # A unit quad that covers the center of the view; the corners show the background.
    return TriangleMeshBatch(
        torch.tensor([[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]]),
        torch.tensor([[[0, 1, 2], [0, 2, 3]]]),
    )


_CAMERA = {
    "position": {"x": 0.0, "y": 0.0, "z": 3.0},
    "target": {"x": 0.0, "y": 0.0, "z": 0.0},
    "quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    "zoom": 1.0,
    "fov": 40.0,
    "cameraType": "perspective",
}


def _face_packet() -> dict[str, Any]:
    oval = [0, 1, 2, 3]
    return {
        "frames": [
            [{"landmarks_xy": np.array([[1, 1], [6, 1], [6, 4], [1, 4]], dtype=np.float32)}]
        ],
        "image_size": (7, 11),
        "connection_sets": {"face_oval": frozenset(zip(oval, oval[1:] + oval[:1], strict=True))},
    }


@dataclass(frozen=True)
class Case:
    node: type[Node]
    field: str
    alpha: bool
    run: Callable[[str], object]
    # Before the engine fitted colors, an eight-digit color made this node fall
    # back to black (comfy-vibe-station#1233 comment 7092441223).
    eight_digits_gave_black: bool = False


CASES = {
    "MakeImage.color_a": Case(
        MakeImage,
        "color_a",
        True,
        lambda color: MakeImage.execute(
            operation="solid", width=4, height=3, channels="rgba", color_a=color
        ),
    ),
    "MakeImage.color_b": Case(
        MakeImage,
        "color_b",
        True,
        lambda color: MakeImage.execute(
            operation="linear_gradient", width=4, height=3, channels="rgba", color_b=color
        ),
    ),
    "DrawText.color": Case(
        DrawText,
        "color",
        True,
        lambda color: DrawText.execute(
            image=_canvas(), text="Hi", x=1, y=0, font_size=10, color=color
        ),
    ),
    "DrawRegion.color": Case(
        DrawRegion,
        "color",
        True,
        lambda color: DrawRegion.execute(
            image=_canvas(),
            region=Region(x=2, y=2, width=8, height=6),
            mode="fill",
            color=color,
        ),
    ),
    "DrawMask.color": Case(
        DrawMask,
        "color",
        True,
        lambda color: DrawMask.execute(
            image=_canvas(), mask=_region_mask(), opacity=1.0, color=color
        ),
    ),
    "TextOverlay.color": Case(
        TextOverlay,
        "color",
        True,
        lambda color: TextOverlay.execute(
            images=torch.zeros(1, 24, 32, 3),
            text="Hi",
            font_size=40.0,
            color=color,
            position="top",
            align="left",
            outline=False,
        ),
    ),
    # Listed exception: for #3366cc80 the hex output was #3366cc before this
    # pack change and is #3366cc80 now; it keeps the opacity digits when opacity
    # is not full (comfy-vibe-station#1233 comment 7092470567). Both runs here
    # use the current node, so tests/test_comfy_color_resolution_nodes.py pins
    # the hex rule itself.
    "ColorToRGBInt.color": Case(
        ColorToRGBInt, "color", True, lambda color: ColorToRGBInt.execute(color=color)
    ),
    "ImageToMask.color": Case(
        ImageToMask,
        "color",
        True,
        # Pixels: the color at half opacity, at full opacity, and another color.
        lambda color: ImageToMask.execute(
            image=np.array(
                [[[[51, 102, 204, 128], [51, 102, 204, 255], [204, 102, 51, 255]]]],
                dtype=np.float32,
            )
            / 255,
            policy="tolerance_color",
            color_source="hex",
            color=color,
            metric="euclidean_rgba",
            tolerance=0.01,
        ),
    ),
    "TextMask.color": Case(
        TextMask,
        "color",
        False,
        lambda color: TextMask.execute(
            text="Hi", width=16, height=12, font_size=10, mask_value="color_red", color=color
        ),
    ),
    "MaskToImage.color": Case(
        MaskToImage,
        "color",
        False,
        lambda color: MaskToImage.execute(
            mask=np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32), channels="rgba", color=color
        ),
    ),
    "RenderMesh.background": Case(
        RenderMesh,
        "background",
        False,
        lambda color: RenderMesh.execute(
            mesh=_mesh(), mode="solid", width=20, height=16, background=color, camera_info=_CAMERA
        ),
    ),
    "RenderSplat.background": Case(
        RenderSplat,
        "background",
        False,
        lambda color: RenderSplat.execute(
            splat=_splat(),
            width=20,
            height=16,
            frames=1,
            splat_scale=1.0,
            sharpen=2.0,
            headlight_shading=0.0,
            opacity_threshold=0.0,
            background=color,
            render_style="color",
            camera_info=_CAMERA,
        ),
        eight_digits_gave_black=True,
    ),
    "PadImageNode.color": Case(
        PadImageNode,
        "color",
        False,
        lambda color: PadImageNode.execute(
            image=torch.full((1, 2, 2, 3), 0.5),
            padding_type=Record("pad dimensions", {"left": 1, "top": 1, "right": 0, "bottom": 0}),
            padding_style=Record("color", {"color": color, "feathering": 0}),
        ),
    ),
    "ImageCropToMask.background": Case(
        ImageCropToMask,
        "background",
        False,
        # The padded crop around the masked block shows the background.
        lambda color: ImageCropToMask.execute(
            images=torch.tensor([0.9, 0.1, 0.5]).expand(1, 8, 12, 3).clone(),
            masks=torch.from_numpy(_region_mask()[:, :8, :12]),
            width=64,
            height=64,
            pad_factor=1.5,
            grow_mask=0,
            background=color,
        ),
        eight_digits_gave_black=True,
    ),
    "Painter.bg_color": Case(
        Painter,
        "bg_color",
        False,
        lambda color: Painter.execute(width=64, height=64, bg_color=color),
        eight_digits_gave_black=True,
    ),
    "MediaPipeFaceMeshVisualize.color": Case(
        MediaPipeFaceMeshVisualize,
        "color",
        False,
        lambda color: MediaPipeFaceMeshVisualize.execute(
            face_landmarks=_face_packet(),
            connections=Record("fill", {}),
            color=color,
            thickness=1,
            point_size=0,
        ),
    ),
}


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_color_field_result_is_unchanged_by_fitting(case: Case, value: str) -> None:
    widget = _declared_widget(case.node, case.field)
    assert widget.alpha is case.alpha
    # Identical code paths on identical parsed channels: results must be bit-identical.
    before = case.run(value)
    after = case.run(fit_color(value, alpha=widget.alpha))
    if case.eight_digits_gave_black and len(value) == 9:
        assert not _same(before, after)
        assert _same(after, case.run(value[:7]))
    else:
        assert _same(before, after)
    assert not _same(after, case.run(CONTRAST))

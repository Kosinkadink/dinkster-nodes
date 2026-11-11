from __future__ import annotations

from typing import Any, cast

import pytest
from dinkster_image.compositor import CreateLayeredImage
from dinkster_image.glsl import GlslShader
from dinkster_image.layer_document import EditLayers, FlattenLayers
from dinkster_image.layer_io import LoadLayers
from dinkster_schema import Node
from dinkster_utilities.curve_editor import CurveEditor
from dinkster_values import CustomWidgetDescriptor
from dinkster_video.video_edits import VideoCrop, VideoTrim


@pytest.mark.parametrize(
    ("node", "role"),
    [
        (CurveEditor, "curve"),
        (CreateLayeredImage, "compositor"),
        (GlslShader, "glsl"),
        (FlattenLayers, "layers-flatten"),
        (EditLayers, "layers-edit"),
        (LoadLayers, "layers-load"),
        (VideoTrim, "video-trim"),
        (VideoCrop, "video-crop"),
    ],
)
def test_editor_capabilities_are_declared_by_schema(node: type[Node], role: str) -> None:
    assert node.schema().editor_role == role


@pytest.mark.parametrize(
    ("node", "features"),
    [(VideoTrim, ["trim"]), (VideoCrop, ["crop"])],
)
def test_video_edit_features_are_declared_by_widget(node: type[Node], features: list[str]) -> None:
    video_edit = next(item for item in node.schema().inputs if item.id == features[0])
    assert video_edit.widget == CustomWidgetDescriptor(
        "VIDEO_EDIT", cast("dict[str, Any]", {"features": features})
    )

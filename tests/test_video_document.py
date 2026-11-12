"""Timeline title effects and public image drawing share their pixel result."""

from typing import cast

import numpy as np
from dinkster_image.drawing import DrawText
from dinkster_video_runtime.timeline_runtime import CPUKernels


def test_title_effect_matches_draw_text_node() -> None:
    frame = np.zeros((32, 64, 3), np.float32)
    expected = cast(
        np.ndarray,
        DrawText.execute(image=frame[None], text="test", font_size=10, opacity=0.5)["image"],
    )[0]
    actual = CPUKernels().effect(
        "dinkster.image.draw_text", frame, {"text": "test", "font_size": 10, "opacity": 0.5}
    )
    np.testing.assert_array_equal(actual, expected)

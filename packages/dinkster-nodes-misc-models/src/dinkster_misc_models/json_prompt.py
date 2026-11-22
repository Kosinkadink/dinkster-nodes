"""Structured Ideogram prompts using native array and color inputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    ARRAY_TYPE,
    COLORS_TYPE,
    CORE_STRING,
    DICT_TYPE,
    Choice,
    Input,
    Node,
    NodeSchema,
    Option,
    Output,
    StringWidget,
    TypeExpr,
    with_identity_replacement,
)

_STRING = TypeExpr.concrete(CORE_STRING)


class BuildJsonPromptIdeogram(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.misc.models.build_json_prompt_ideogram",
            display_name="Build JSON Prompt (Ideogram)",
            category="text",
            description="Build a JSON prompt for the Ideogram 4 model.",
            body=(
                Input("element", TypeExpr.concrete(ARRAY_TYPE)),
                Input(
                    "high_level_description",
                    _STRING,
                    default="",
                    widget=StringWidget(multiline=True),
                ),
                Input("background", _STRING, default="", widget=StringWidget(multiline=True)),
                Choice(
                    "style",
                    options=(
                        Option("none"),
                        Option(
                            "photo",
                            rows=(
                                Input(
                                    "photo",
                                    _STRING,
                                    default="",
                                    widget=StringWidget(multiline=False),
                                ),
                            ),
                        ),
                        Option(
                            "art_style",
                            rows=(
                                Input(
                                    "art_style",
                                    _STRING,
                                    default="",
                                    widget=StringWidget(multiline=False),
                                ),
                            ),
                        ),
                    ),
                ),
                Input("aesthetics", _STRING, default="", widget=StringWidget(multiline=False)),
                Input("lighting", _STRING, default="", widget=StringWidget(multiline=False)),
                Input("medium", _STRING, default="", widget=StringWidget(multiline=False)),
                Input("color_palette", TypeExpr.concrete(COLORS_TYPE), default=[]),
                Output("prompt", TypeExpr.concrete(DICT_TYPE), label="prompt"),
            ),
        )
        return with_identity_replacement(schema, "BuildJsonPromptIdeogram")

    @classmethod
    def execute(
        cls,
        *,
        element: Any,
        style: str,
        high_level_description: str = "",
        background: str = "",
        aesthetics: str = "",
        lighting: str = "",
        medium: str = "",
        color_palette: Any = None,
        **inputs: Any,
    ) -> Mapping[str, object]:
        colors: Any = color_palette or []
        if isinstance(colors, dict):
            colors = cast("dict[object, Any]", colors).values()
        palette = [
            color.upper()
            for color in cast("Iterable[Any]", colors)
            if isinstance(color, str) and color
        ]
        caption: dict[str, Any] = {}
        if high_level_description.strip():
            caption["high_level_description"] = high_level_description
        if style != "none":
            description: dict[str, Any] = {"aesthetics": aesthetics, "lighting": lighting}
            if style == "photo":
                description["photo"] = inputs.get("style.photo", "")
                description["medium"] = medium
            else:
                description["medium"] = medium
                description["art_style"] = inputs.get("style.art_style", "")
            if palette:
                description["color_palette"] = palette
            caption["style_description"] = description
        caption["compositional_deconstruction"] = {
            "background": background,
            "elements": element if isinstance(element, list) else [],
        }
        return cls.outputs(prompt=caption)

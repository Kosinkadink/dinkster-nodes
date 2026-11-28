from dinkster_api.v1 import Record
from dinkster_misc_models.json_prompt import BuildJsonPromptIdeogram


def test_json_prompt_photo_preserves_text_elements_and_palette_order() -> None:
    elements = [{"description": "subject", "x": 0.25}]
    palette = {"second": "#ab12cd", "first": "#00ef45", "empty": "", "number": 7}
    result = BuildJsonPromptIdeogram.execute(
        element=elements,
        style=Record("photo", {"photo": "35mm"}),
        high_level_description="  portrait  ",
        background="studio",
        aesthetics="muted",
        lighting="side",
        medium="photograph",
        color_palette=palette,
    )["prompt"]
    assert result == {
        "high_level_description": "  portrait  ",
        "style_description": {
            "aesthetics": "muted",
            "lighting": "side",
            "photo": "35mm",
            "medium": "photograph",
            "color_palette": ["#AB12CD", "#00EF45"],
        },
        "compositional_deconstruction": {"background": "studio", "elements": elements},
    }
    assert palette == {"second": "#ab12cd", "first": "#00ef45", "empty": "", "number": 7}
    assert elements == [{"description": "subject", "x": 0.25}]


def test_json_prompt_none_omits_style_and_blank_description() -> None:
    assert BuildJsonPromptIdeogram.execute(
        element="not an array",
        style=Record("none", {}),
        high_level_description=" \n ",
        color_palette=["#abcdef"],
    )["prompt"] == {"compositional_deconstruction": {"background": "", "elements": []}}


def test_json_prompt_art_style_omits_empty_palette_and_photo() -> None:
    assert BuildJsonPromptIdeogram.execute(
        element=[],
        style=Record("art_style", {"art_style": "watercolor"}),
        medium="painting",
    )["prompt"] == {
        "style_description": {
            "aesthetics": "",
            "lighting": "",
            "medium": "painting",
            "art_style": "watercolor",
        },
        "compositional_deconstruction": {"background": "", "elements": []},
    }

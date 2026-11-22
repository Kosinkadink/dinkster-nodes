"""Text generation through the loaded text encoder's native generation API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CLIP_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    Choice,
    ComboWidget,
    Input,
    Node,
    NodeSchema,
    NumberWidget,
    Option,
    Output,
    Record,
    Section,
    StringWidget,
    TypeExpr,
    materialize_audio,
    materialize_image,
    with_identity_replacement,
)

_STRING = TypeExpr.concrete(CORE_STRING)


class TextGenerate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.llm.text_generate",
            display_name="Generate Text",
            category="text",
            search_terms=("LLM", "gemma"),
            body=(
                Input("clip", TypeExpr.concrete(CLIP_TYPE)),
                Input(
                    "prompt",
                    _STRING,
                    default="",
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                Input("image", TypeExpr.concrete(IMAGE_TYPE), required=False),
                Input(
                    "video",
                    TypeExpr.concrete(IMAGE_TYPE),
                    required=False,
                    tooltip=(
                        "Video frames as image batch. Assumed to be 24 FPS; "
                        "subsampled to 1 FPS internally."
                    ),
                ),
                Input("audio", TypeExpr.concrete(AUDIO_TYPE), required=False),
                Input(
                    "max_length",
                    TypeExpr.concrete(CORE_INT),
                    default=512,
                    widget=NumberWidget(min=1, max=32768),
                ),
                Choice(
                    "sampling_mode",
                    options=(
                        Option(
                            "on",
                            rows=(
                                Input(
                                    "temperature",
                                    TypeExpr.concrete(CORE_FLOAT),
                                    default=0.7,
                                    widget=NumberWidget(min=0.01, max=2.0, step=0.000001),
                                ),
                                Input(
                                    "top_k",
                                    TypeExpr.concrete(CORE_INT),
                                    default=64,
                                    widget=NumberWidget(min=0, max=1000),
                                ),
                                Input(
                                    "top_p",
                                    TypeExpr.concrete(CORE_FLOAT),
                                    default=0.95,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                                Input(
                                    "min_p",
                                    TypeExpr.concrete(CORE_FLOAT),
                                    default=0.05,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                                Input(
                                    "repetition_penalty",
                                    TypeExpr.concrete(CORE_FLOAT),
                                    default=1.05,
                                    widget=NumberWidget(min=0.0, max=5.0, step=0.01),
                                ),
                                Input(
                                    "seed",
                                    TypeExpr.concrete(CORE_INT),
                                    default=0,
                                    widget=NumberWidget(min=0, max=0xFFFFFFFFFFFFFFFF),
                                ),
                                Input(
                                    "presence_penalty",
                                    TypeExpr.concrete(CORE_FLOAT),
                                    required=False,
                                    default=0.0,
                                    widget=NumberWidget(min=0.0, max=5.0, step=0.01),
                                ),
                            ),
                        ),
                        Option("off"),
                    ),
                    label="Sampling Mode",
                ),
                Input(
                    "thinking",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    required=False,
                    default=False,
                    tooltip="Operate in thinking mode if the model supports it.",
                ),
                Input(
                    "mtp",
                    TypeExpr.concrete(CORE_COMBO),
                    required=False,
                    default="auto",
                    widget=ComboWidget(
                        (
                            "auto",
                            "off",
                            "2",
                            "3",
                            "4",
                            "5",
                            "tree2s1",
                            "tree2s2",
                            "tree3s1",
                            "tree4s1m8",
                        )
                    ),
                    tooltip=(
                        "Speculative decoding with the checkpoint's multi-token-prediction head. "
                        "No effect without MTP weights. auto adapts the draft depth and breadth "
                        "to observed token matches and execution cost; 2-5 pins a constant draft "
                        "depth; tree modes pin both breadth and depth. Sampled output stays "
                        "correctly distributed but differs from non-MTP output for the same seed."
                    ),
                ),
                Input(
                    "system_prompt",
                    _STRING,
                    required=False,
                    pin_first=True,
                    widget=StringWidget(multiline=False),
                    tooltip=(
                        "Replaces the system prompt in the model's chat template. "
                        "Ignored when the default template is not used."
                    ),
                ),
                Section(
                    "advanced",
                    label="Advanced",
                    collapsed=True,
                    rows=(
                        Input(
                            "use_default_template",
                            TypeExpr.concrete(CORE_BOOLEAN),
                            required=False,
                            default=True,
                            tooltip="Use the built in system prompt/template if the model has one.",
                        ),
                    ),
                ),
                Output("generated_text", _STRING, label="generated_text"),
                Output("thinking", _STRING, label="thinking"),
            ),
        )
        return with_identity_replacement(schema, "TextGenerate")

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        prompt: str,
        max_length: int,
        sampling_mode: Record,
        image: object = None,
        thinking: bool = False,
        use_default_template: bool = True,
        video: object = None,
        audio: object = None,
        mtp: str = "auto",
        system_prompt: str = "",
    ) -> Mapping[str, object]:
        sampling = sampling_mode
        tokens = clip.tokenize(
            prompt,
            image=materialize_image(image) if image is not None else None,
            skip_template=not use_default_template,
            min_length=1,
            thinking=thinking,
            video=materialize_image(video) if video is not None else None,
            audio=materialize_audio(audio) if audio is not None else None,
            system_prompt=system_prompt if use_default_template else "",
        )
        generated_ids = clip.generate(
            tokens,
            do_sample=sampling.key == "on",
            max_length=max_length,
            temperature=sampling.get("temperature", 1.0),
            top_k=sampling.get("top_k", 50),
            top_p=sampling.get("top_p", 1.0),
            min_p=sampling.get("min_p", 0.0),
            repetition_penalty=sampling.get("repetition_penalty", 1.0),
            presence_penalty=sampling.get("presence_penalty", 0.0),
            seed=sampling.get("seed"),
            mtp={"off": False, "auto": True}.get(mtp, mtp),
        )
        generated_text = clip.decode(generated_ids)
        reasoning, separator, text = generated_text.partition("</think>")
        if not separator and not reasoning.lstrip().startswith("<think>"):
            if isinstance(tokens, dict):
                tokens = next(iter(cast("dict[str, Any]", tokens).values()))
            prompt_ids = [
                token[0]
                for token in cast("list[Any]", tokens[-1])
                if isinstance(token, tuple) and isinstance(token[0], int)
            ]
            if not prompt.rstrip().endswith("<think>") and not clip.decode(
                prompt_ids[-4:]
            ).rstrip().endswith("<think>"):
                reasoning, text = "", generated_text
        return cls.outputs(
            generated_text=text.strip(), thinking=reasoning.replace("<think>", "", 1).strip()
        )

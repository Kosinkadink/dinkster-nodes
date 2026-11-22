"""Native video input through indexed, invocation-granted assets."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    ASSET_TYPE,
    VIDEO_TYPE,
    AssetRef,
    AssetWidget,
    Input,
    LocalAssetRef,
    Node,
    NodeSchema,
    Output,
    SourceFilenameSpec,
    TypeExpr,
    admit_video_source,
    with_identity_replacement,
)


class LoadVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.video.load_video",
            display_name="Load Video",
            category="video",
            search_terms=("import video", "open video", "video file"),
            body=(
                Input(
                    "file",
                    TypeExpr.concrete(ASSET_TYPE),
                    widget=AssetWidget(kind="media/video", accept=("video/*",), allow_upload=True),
                    source_filename=SourceFilenameSpec("media/video", "input"),
                ),
                Output("VIDEO", TypeExpr.concrete(VIDEO_TYPE), preview=True),
            ),
        )
        return with_identity_replacement(schema, "LoadVideo")

    @classmethod
    def execute(cls, *, file: object) -> Mapping[str, object]:
        if not isinstance(file, (AssetRef, LocalAssetRef)):
            raise TypeError("file must be an AssetRef or LocalAssetRef")
        return cls.outputs(VIDEO=admit_video_source(file))

"""Load original 3D files and authored viewport captures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_INT,
    CORE_STRING,
    FILE_3D_TYPE,
    IMAGE_TYPE,
    LOAD3D_CAMERA_TYPE,
    LOAD3D_MODEL_INFO_TYPE,
    LOAD_3D_TYPE,
    MASK_TYPE,
    VIDEO_TYPE,
    AbsentOutput,
    AssetRef,
    AssetWidget,
    CustomWidgetDescriptor,
    Input,
    LocalAssetRef,
    Node,
    NodeSchema,
    NumberWidget,
    Output,
    SourceFilenameSpec,
    TypeExpr,
    admit_video_source,
    load_file3d,
    report_log,
    with_identity_replacement,
)
from dinkster_image.ported.image_file_tools import LoadImage

EARLIER_VIEW = (
    "These pictures show an earlier view. Queue the run from the editor to take new ones."
)
NO_MODEL = "No pictures: no model is chosen."
NOT_DRAWN = (
    "No pictures: the view was not drawn before this run. That happens when the model "
    "arrives by a wire, or when the run was not queued from the editor."
)


def _same_model(pictured: object, source: AssetRef | LocalAssetRef | None) -> bool:
    """Whether the model the pictures were drawn from is the one this run receives.

    The app records model_file as it sent it: a selected name, an asset reference,
    or null for no model. Two files may share a name, so a reference is matched by
    its digest, or by its binding when it names a granted local file.
    """
    if source is None:
        return pictured is None
    if isinstance(pictured, str):
        return pictured == source.name
    if isinstance(pictured, Mapping):
        reference = cast("Mapping[str, object]", pictured)
        if isinstance(source, AssetRef):
            return reference.get("digest") == source.digest
        return reference.get("localBinding") == source.binding
    return False


class Load3D(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.3d.load3_d",
            display_name="Load 3D & Animation",
            category="3d",
            body=(
                Input(
                    "model_file",
                    TypeExpr.concrete(ASSET_TYPE),
                    default="none",
                    widget=AssetWidget(kind="media/model3d", allow_upload=True),
                    source_filename=SourceFilenameSpec("media/model3d", "input"),
                ),
                Input(
                    "image",
                    TypeExpr.concrete(LOAD_3D_TYPE),
                    widget=CustomWidgetDescriptor(
                        "LOAD_3D",
                        reads={"model_file": "model_file", "width": "width", "height": "height"},
                    ),
                ),
                *(
                    Input(
                        axis,
                        TypeExpr.concrete(CORE_INT),
                        default=1024,
                        widget=NumberWidget(min=1, max=4096, step=1),
                    )
                    for axis in ("width", "height")
                ),
                *tuple(
                    Output(
                        name,
                        TypeExpr.concrete(type_id),
                        preview=name == "model_3d",
                        optional=name in ("image", "mask", "normal", "recording_video"),
                    )
                    for name, type_id in (
                        ("image", IMAGE_TYPE),
                        ("mask", MASK_TYPE),
                        ("mesh_path", CORE_STRING),
                        ("normal", IMAGE_TYPE),
                        ("camera_info", LOAD3D_CAMERA_TYPE),
                        ("recording_video", VIDEO_TYPE),
                        ("model_3d", FILE_3D_TYPE),
                        ("model_3d_info", LOAD3D_MODEL_INFO_TYPE),
                    )
                ),
            ),
        )
        return with_identity_replacement(schema, "Load3D")

    @classmethod
    def execute(
        cls,
        *,
        model_file: AssetRef | LocalAssetRef | str | None,
        image: Mapping[str, object],
        width: int = 1024,
        height: int = 1024,
    ) -> Mapping[str, object]:
        packet = load_file3d(model_file)
        source = cast("AssetRef | LocalAssetRef", packet["source"]) if packet is not None else None
        pictures = {name: image.get(name) for name in ("image", "mask", "normal")}
        if all(pictures.values()):
            loaded = {
                name: LoadImage.execute(image=cast("AssetRef | LocalAssetRef", ref))
                for name, ref in pictures.items()
            }
            output_image = loaded["image"]["IMAGE"]
            output_mask = loaded["mask"]["MASK"]
            normal = loaded["normal"]["IMAGE"]
            # The app stores the view its pictures show; a value without it is not known.
            pictured = image.get("pictures_view")
            if isinstance(pictured, Mapping):
                view = cast("Mapping[str, object]", pictured)
                current = {
                    "camera_info": image.get("camera_info"),
                    "model_3d_info": image.get("model_3d_info"),
                    "width": width,
                    "height": height,
                }
                if not _same_model(view.get("model_file"), source) or any(
                    view.get(key) != value for key, value in current.items()
                ):
                    report_log("warning", EARLIER_VIEW)
        else:
            reason = NO_MODEL if packet is None else NOT_DRAWN
            report_log("warning", reason)
            output_image = output_mask = normal = AbsentOutput(reason)
        recording = image.get("recording")
        video = (
            admit_video_source(cast("AssetRef | LocalAssetRef", recording))
            if recording
            else AbsentOutput("no recording")
        )
        return cls.outputs(
            image=output_image,
            mask=output_mask,
            mesh_path=source.name if source is not None else "",
            normal=normal,
            camera_info=image.get("camera_info"),
            recording_video=video,
            model_3d=packet,
            model_3d_info=image.get("model_3d_info", []),
        )

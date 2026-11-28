"""Construct animated GLB and BVH files from canonical body pose packets."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    FILE_3D_TYPE,
    KIMODO_POSE_DATA_TYPE,
    MHR_POSE_DATA_TYPE,
    SAM3D_BODY_MODEL_TYPE,
    AssetRef,
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
    TypeExpr,
    Use,
    digest_bytes,
    initialize_inference_process,
    materialize_vision_packet,
    produced_output_scratch,
    with_identity_replacement,
)


class BuildPoseFile(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        floating = TypeExpr.concrete(CORE_FLOAT)
        combo = TypeExpr.concrete(CORE_COMBO)
        bone_vis = Choice(
            "bone_vis",
            options=(
                Option("off"),
                Option(
                    "octahedrons",
                    rows=(
                        Input(
                            "bone_vis_color",
                            combo,
                            default="rainbow_y",
                            widget=ComboWidget(("white", "rainbow_y")),
                        ),
                        Section(
                            "advanced",
                            label="Advanced",
                            collapsed=True,
                            rows=(
                                Input(
                                    "bone_vis_radius_m",
                                    floating,
                                    default=0.02,
                                    widget=NumberWidget(min=0.005, max=0.5, step=0.005),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        tilts = tuple(
            Input(
                f"rainbow_tilt_{axis}",
                floating,
                default=value,
                widget=NumberWidget(min=-90.0, max=90.0, step=0.5),
            )
            for axis, value in (("z", -35.0), ("x", 0.0))
        ) + (
            Input(
                "person_palette_falloff",
                floating,
                default=0.6,
                widget=NumberWidget(min=0.1, max=1.0, step=0.05),
            ),
        )
        include_hands = Input("include_hands", TypeExpr.concrete(CORE_BOOLEAN), default=False)
        face = Input(
            "face_style",
            combo,
            default="disabled",
            widget=ComboWidget(("disabled", "full", "eyes_mouth")),
        )
        schema = NodeSchema(
            "dinkster.vision.sam3d.body.build_pose_file",
            display_name="Create 3D Animation File",
            description="Build an animated GLB from pose data, or save a BVH mocap file.",
            search_terms=(
                "pose animation",
                "mocap",
                "glb",
                "bvh",
                "build pose file",
                "save pose file",
            ),
            category="3d",
            body=(
                Input("pose_data", TypeExpr.union(MHR_POSE_DATA_TYPE, KIMODO_POSE_DATA_TYPE)),
                Choice(
                    "format",
                    options=(
                        Option(
                            "glb",
                            rows=(
                                Choice(
                                    "mesh_style",
                                    options=(
                                        Option(
                                            "body_mesh",
                                            rows=(
                                                Use("bone_vis"),
                                                Choice(
                                                    "shader",
                                                    options=tuple(
                                                        Option(
                                                            name,
                                                            rows=()
                                                            if name == "default"
                                                            else tuple(
                                                                Use(row.id) for row in tilts
                                                            ),
                                                        )
                                                        for name in (
                                                            "default",
                                                            "rainbow",
                                                            "rainbow_face_normal",
                                                            "rainbow_face_semantic",
                                                        )
                                                    ),
                                                    shared=tilts,
                                                ),
                                            ),
                                        ),
                                        Option(
                                            "bones_only",
                                            rows=(Use("bone_vis", only=("octahedrons",)),),
                                        ),
                                        Option(
                                            "openpose",
                                            rows=(
                                                Use("include_hands"),
                                                Use("face_style"),
                                                Section(
                                                    "advanced",
                                                    label="Advanced",
                                                    collapsed=True,
                                                    rows=(
                                                        Use("marker_radius_m"),
                                                        Use("stick_radius_m"),
                                                        Use("hand_marker_radius_m"),
                                                        Use("hand_stick_radius_m"),
                                                        Input(
                                                            "face_marker_radius_m",
                                                            floating,
                                                            default=0.0,
                                                            widget=NumberWidget(
                                                                min=0.0, max=0.05, step=0.0005
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        Option(
                                            "scail",
                                            rows=(
                                                Use("include_hands"),
                                                Use("face_style"),
                                                Section(
                                                    "advanced",
                                                    label="Advanced",
                                                    collapsed=True,
                                                    rows=(
                                                        Use(
                                                            "stick_radius_m",
                                                            default=0.022,
                                                            widget=NumberWidget(
                                                                min=0.002, max=0.1, step=0.001
                                                            ),
                                                        ),
                                                        Use(
                                                            "marker_radius_m",
                                                            default=0.0,
                                                            widget=NumberWidget(
                                                                min=0.0, max=0.1, step=0.001
                                                            ),
                                                        ),
                                                        Input(
                                                            "material_roughness",
                                                            floating,
                                                            default=0.3,
                                                            widget=NumberWidget(
                                                                min=0.0, max=1.0, step=0.05
                                                            ),
                                                        ),
                                                        Use(
                                                            "hand_marker_radius_m",
                                                            widget=NumberWidget(
                                                                min=0.001, max=0.05, step=0.001
                                                            ),
                                                        ),
                                                        Use("hand_stick_radius_m"),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    shared=(
                                        bone_vis,
                                        include_hands,
                                        face,
                                        Input(
                                            "marker_radius_m",
                                            floating,
                                            default=0.010,
                                            widget=NumberWidget(min=0.005, max=0.1, step=0.001),
                                        ),
                                        Input(
                                            "stick_radius_m",
                                            floating,
                                            default=0.008,
                                            widget=NumberWidget(min=0.002, max=0.05, step=0.001),
                                        ),
                                        Input(
                                            "hand_marker_radius_m",
                                            floating,
                                            default=0.005,
                                            widget=NumberWidget(min=0.001, max=0.1, step=0.001),
                                        ),
                                        Input(
                                            "hand_stick_radius_m",
                                            floating,
                                            default=0.003,
                                            widget=NumberWidget(min=0.001, max=0.05, step=0.001),
                                        ),
                                    ),
                                ),
                                Input(
                                    "bone_smooth_window",
                                    TypeExpr.concrete(CORE_INT),
                                    default=0,
                                    widget=NumberWidget(min=0, max=51, step=2),
                                ),
                            ),
                        ),
                        Option(
                            "bvh",
                            rows=(
                                Input(
                                    "units", combo, default="cm", widget=ComboWidget(("cm", "m"))
                                ),
                            ),
                        ),
                    ),
                ),
                Input("sam3d_body_model", TypeExpr.concrete(SAM3D_BODY_MODEL_TYPE), required=False),
                Input(
                    "fps", floating, default=24.0, widget=NumberWidget(min=1.0, max=240.0, step=1.0)
                ),
                Input(
                    "camera_translation",
                    combo,
                    default="off",
                    widget=ComboWidget(("off", "centered", "absolute")),
                ),
                Input(
                    "track_index",
                    TypeExpr.concrete(CORE_INT),
                    default=-1,
                    widget=NumberWidget(min=-1, max=15),
                ),
                Output("model_3d", TypeExpr.concrete(FILE_3D_TYPE)),
            ),
        )
        return with_identity_replacement(schema, "BuildPoseFile")

    @classmethod
    def execute(
        cls,
        *,
        pose_data: Any,
        format: Record,
        sam3d_body_model: Any = None,
        fps: float = 24.0,
        camera_translation: str = "off",
        track_index: int = -1,
    ) -> Mapping[str, object]:
        pose = materialize_vision_packet(pose_data)
        file_format = format.key
        style = cast("Record | None", format.get("mesh_style"))
        mode = style.key if style is not None else ""
        external = "_skeleton_override" in pose
        if file_format == "bvh" or mode in ("body_mesh", "bones_only"):
            if sam3d_body_model is None and not external:
                if file_format == "bvh":
                    raise ValueError(
                        "Create 3D Animation: 'bvh' format needs the `sam3d_body_model` input OR a "
                        "`_skeleton_override` dict in pose_data (e.g. from KimodoSample)."
                    )
                raise ValueError(
                    f"BuildPoseFile: '{mode}' mode needs the `sam3d_body_model` input OR a "
                    "`_skeleton_override` dict in pose_data. Connect the SAM3DBody model "
                    "or feed pose_data from a node that supplies the override (e.g. KimodoSample)."
                )
            if sam3d_body_model is not None and not external:
                initialize_inference_process()
                importlib.import_module("dinkster_inference.model_management").load_model_gpu(
                    sam3d_body_model
                )
        common = dict(
            fps=float(fps), camera_translation=str(camera_translation), track_index=int(track_index)
        )
        if file_format == "bvh":
            serializer = importlib.import_module(".sam3d_bvh", __package__)
            common["track_index"] = max(0, int(track_index))
            data = serializer.build_bvh(
                pose, sam3d_body_model, units=str(format["units"]), **common
            )
        elif file_format == "glb" and style is not None:
            smooth = int(cast("int", format["bone_smooth_window"]))
            if mode in ("body_mesh", "bones_only"):
                serializer = importlib.import_module(".sam3d_skeletal", __package__)
                bone_vis = cast("Record", style["bone_vis"])
                shader = cast("Record | None", style.get("shader"))
                data = serializer.build_glb_skeletal(
                    pose,
                    sam3d_body_model,
                    **common,
                    bone_smooth_window=smooth,
                    bone_vis=bone_vis.key,
                    bone_vis_radius_m=float(cast("float", bone_vis.get("bone_vis_radius_m", 0.04))),
                    bone_vis_color=str(bone_vis.get("bone_vis_color", "white")),
                    include_body_mesh=mode == "body_mesh",
                    shader=shader.key if shader is not None else "default",
                    rainbow_tilt_x_deg=float(
                        cast("float", shader.get("rainbow_tilt_x", 0.0) if shader else 0.0)
                    ),
                    rainbow_tilt_z_deg=float(
                        cast("float", shader.get("rainbow_tilt_z", 0.0) if shader else 0.0)
                    ),
                    person_palette_falloff=float(
                        cast("float", shader.get("person_palette_falloff", 0.6) if shader else 0.6)
                    ),
                )
            elif mode in ("openpose", "scail"):
                serializer = importlib.import_module(".sam3d_openpose_glb", __package__)
                scail = mode == "scail"
                radius = float(
                    cast("float", style.get("stick_radius_m", 0.022 if scail else 0.008))
                )
                marker = float(cast("float", style.get("marker_radius_m", 0.0 if scail else 0.025)))
                if scail and marker <= 0.0:
                    marker = radius
                data = serializer.build_glb_openpose(
                    pose,
                    **common,
                    bone_smooth_window=smooth,
                    marker_radius_m=marker,
                    stick_radius_m=radius,
                    include_hands=bool(style.get("include_hands", False)),
                    hand_marker_radius_m=float(
                        cast("float", style.get("hand_marker_radius_m", 0.005))
                    ),
                    hand_stick_radius_m=float(
                        cast("float", style.get("hand_stick_radius_m", 0.003))
                    ),
                    face_style=str(style.get("face_style", "disabled")),
                    face_marker_radius_m=0.0
                    if scail
                    else float(cast("float", style.get("face_marker_radius_m", 0.0))),
                    palette="scail" if scail else "openpose",
                    shape="capsule" if scail else "ellipsoid",
                    smooth_shade=scail,
                    material_roughness=float(cast("float", style.get("material_roughness", 0.3)))
                    if scail
                    else 0.85,
                    material_double_sided=scail,
                )
            else:
                raise ValueError(f"BuildPoseGLB: unknown mesh_style {mode!r}")
        else:
            raise ValueError(f"Unsupported pose file format: {file_format!r}")
        vault = produced_output_scratch()
        if vault is None:
            raise RuntimeError("BuildPoseFile requires a produced-output store")
        digest = digest_bytes(data)
        with vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        source = AssetRef(
            digest,
            f"pose.{file_format}",
            len(data),
            "model/gltf-binary" if file_format == "glb" else "application/octet-stream",
            resolver=vault,
        )
        return cls.outputs(model_3d={"format": file_format, "source": source})

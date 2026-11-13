"""Production video edits preserve lazy values across worker boundaries."""

from __future__ import annotations

import io
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest
from dinkster_api.v1 import VIDEO_TYPE
from dinkster_values import TypeRegistry, effective_video_facts, encode_video, video_from_source
from dinkster_video.video_edits import VideoCrop, VideoTrim

from tests.async_scenarios import run_scenario


def _source(*, depth: int = 8, b_frames: int = 0, container: str = "mp4", offset: int = 0) -> bytes:
    output = io.BytesIO()
    with av.open(output, "w", format=container) as opened:
        mux = cast(Any, opened)
        stream = mux.add_stream("libx264", rate=10)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p10le" if depth == 10 else "yuv420p"
        stream.codec_context.thread_count = 1
        stream.codec_context.max_b_frames = b_frames
        stream.gop_size = 5
        stream.options = {"crf": "0", "x264-params": "scenecut=0"}
        if depth == 10:
            stream.codec_context.color_primaries = 9
            stream.codec_context.color_trc = 18
            stream.codec_context.colorspace = 9
            stream.codec_context.color_range = 1
        for i in range(20):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 64, 3), (i * 10 + offset) % 256, np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = i, Fraction(1, 10)
            for packet in stream.encode(frame):
                mux.mux(packet)
        for packet in stream.encode():
            mux.mux(packet)
    return output.getvalue()


@pytest.mark.parametrize(
    "widget", [{}, {"crop": {}}, {"trim": {}}, {"trim": {"start_time": -0.5, "duration": 0}}]
)
def test_video_edit_is_read_only_and_sections_are_independent(
    widget: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    value = video_from_source(_source())
    snapshot = json.dumps(widget, sort_keys=True)
    monkeypatch.setattr(av, "open", lambda *a, **kw: pytest.fail("edits/info must not probe"))
    trimmed = VideoTrim.execute(video=value, trim=widget)["video"]
    cropped = VideoCrop.execute(video=trimmed, crop=widget)["video"]
    assert json.dumps(widget, sort_keys=True) == snapshot
    info = effective_video_facts(cast(dict[str, object], cropped))
    assert (info["width"], info["height"]) == (64, 32)
    assert info["duration"] == (0.5 if widget.get("trim") else 2)


def test_native_video_chain_matches_across_worker_shared_memory() -> None:
    from dinkster_protocol import Invocation
    from dinkster_schema import build_node_types
    from dinkster_values import register_core_types
    from dinkster_video import NODES, register_types
    from dinkster_workers import BoundaryDiagnostic, InProcessWorker, IsolatedWorker

    source = video_from_source(_source(depth=10))

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)

        async def run(worker: InProcessWorker | IsolatedWorker) -> tuple[bytes, np.ndarray]:
            trimmed = await worker.invoke(
                Invocation(
                    invocation_id="trim",
                    node_id="trim",
                    node_type="dinkster.video.video_slice",
                    inputs={
                        "video": registry.wrap(VIDEO_TYPE, source),
                        "start_time": registry.wrap("FLOAT", 0.5),
                        "duration": registry.wrap("FLOAT", 1.0),
                    },
                    effective_schema=worker.schemas["dinkster.video.video_slice"],
                )
            )
            assert trimmed.error is None
            assert trimmed.outputs is not None
            frames = await worker.invoke(
                Invocation(
                    invocation_id="frames",
                    node_id="frames",
                    node_type="dinkster.video.get_video_components",
                    inputs={"video": trimmed.outputs["video"]},
                    effective_schema=worker.schemas["dinkster.video.get_video_components"],
                )
            )
            assert frames.error is None
            assert frames.outputs is not None
            assert frames.outputs["bit_depth"].resolve() == 10
            return (
                encode_video(trimmed.outputs["video"].resolve()),
                cast(np.ndarray, frames.outputs["images"].resolve()),
            )

        expected = await run(InProcessWorker(build_node_types(NODES), registry))
        diagnostics: list[BoundaryDiagnostic] = []
        manifest = (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-video/dinkster-pack.toml"
        )
        worker = IsolatedWorker(
            manifest, registry, shm_threshold=64, on_diagnostic=diagnostics.append
        )
        await worker.start()
        try:
            actual = await run(worker)
            assert actual[0] == expected[0]
            np.testing.assert_array_equal(actual[1], expected[1])
            assert [
                crossing.transport
                for diagnostic in diagnostics
                for crossing in diagnostic.inputs
                if crossing.type_id == VIDEO_TYPE
            ] == ["shm", "shm"]
        finally:
            await worker.close()

    run_scenario(scenario())

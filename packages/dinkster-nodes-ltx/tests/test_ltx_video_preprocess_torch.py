"""CPU contracts for native LTXV image preprocessing."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import torch
from dinkster_api.v1 import (
    IMAGE_TYPE,
    TypeExpr,
)
from dinkster_ltx.ltx_video_preprocess import LTXVPreprocess

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)


def _frames(dtype: torch.dtype) -> torch.Tensor:
    frame0 = torch.arange(5 * 7 * 3, dtype=dtype).reshape(5, 7, 3) / 104
    frame1 = torch.flip(frame0, (0, 1))
    return torch.stack((frame0, frame1))


def _expected_h264(image: torch.Tensor, compression: int) -> torch.Tensor:
    import av

    output_frames = []
    for frame in image:
        pixels = (
            (frame[: (frame.shape[0] // 2) * 2, : (frame.shape[1] // 2) * 2] * 255.0)
            .byte()
            .cpu()
            .numpy()
        )
        with BytesIO() as encoded:
            container = av.open(encoded, "w", format="mp4")
            stream = container.add_stream(
                "libx264", rate=1, options={"crf": str(compression), "preset": "veryfast"}
            )
            stream.height, stream.width = pixels.shape[:2]
            av_frame = av.VideoFrame.from_ndarray(pixels, format="rgb24").reformat(format="yuv420p")
            container.mux(stream.encode(av_frame))
            container.mux(stream.encode())
            container.close()
            with BytesIO(encoded.getvalue()) as decoded:
                container = av.open(decoded)
                stream = next(stream for stream in container.streams if stream.type == "video")
                output_frames.append(next(container.decode(stream)).to_ndarray(format="rgb24"))
                container.close()
    return (
        torch.from_numpy(np.stack(output_frames)).to(dtype=image.dtype, device=image.device) / 255.0
    )


def test_compression_zero_preserves_all_frames_identity_and_values() -> None:
    image = _frames(torch.float64)
    output = LTXVPreprocess.execute(image=image, img_compression=0)["IMAGE"]
    assert output.dtype is torch.float64
    assert output.device == image.device
    assert output.shape == image.shape
    assert torch.equal(output, image)


def test_h264_preprocess_trims_odd_dimensions_preserves_dtype_and_processes_each_frame() -> None:
    image = _frames(torch.float32)
    output = LTXVPreprocess.execute(image=image, img_compression=35)["IMAGE"]
    assert output.shape == (2, 4, 6, 3)
    assert output.dtype is torch.float32
    assert output.device == image.device
    assert torch.all((output >= 0) & (output <= 1))
    assert not torch.equal(output[0], output[1])
    torch.testing.assert_close(output, _expected_h264(image, 35), rtol=0, atol=0)
    assert torch.equal(image, _frames(torch.float32))

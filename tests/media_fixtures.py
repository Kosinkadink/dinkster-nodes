"""Media inputs and loopback services for pack integration tests."""

from __future__ import annotations

import asyncio
import io
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import psutil
from aiohttp import web
from dinkster_workers.service import READY_LINE_PREFIX

TOKEN = "remote-secret-token-0123456789ab"
ENDPOINT_TOKEN = "staging-bearer-secret-0123456789"


@dataclass
class FakeTensor:
    array: np.ndarray

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array


@cache
def mp4_bytes() -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as opened:
        output = cast(Any, opened)
        stream = output.add_stream("libx264", rate=Fraction(30000, 1001))
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        for i in range(3):
            array = np.full((32, 64, 4), 64 + i * 32, dtype=np.uint8)
            array[:, :, 3] = 128
            frame = av.VideoFrame.from_ndarray(array, format="rgba")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return buffer.getvalue()


async def start_service(
    manifest: Path,
    tmp_path: Path,
    *extra_args: str,
    pythonpath: tuple[Path, ...] = (),
    port: int = 0,
) -> tuple[asyncio.subprocess.Process, str, int]:
    token_file = tmp_path / "remote-token.txt"
    token_file.write_text(TOKEN, encoding="utf-8")
    search_path = os.pathsep.join(str(entry) for entry in (*pythonpath, Path(__file__).parent))
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "dinkster_workers.service",
        "--listen",
        f"127.0.0.1:{port}",
        "--manifest",
        str(manifest),
        "--token-file",
        str(token_file),
        *extra_args,
        stdout=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": search_path},
    )
    assert proc.stdout is not None
    try:
        line = (await asyncio.wait_for(proc.stdout.readline(), 60.0)).decode()
    except BaseException:
        proc.kill()
        await proc.wait()
        raise
    assert line.startswith(READY_LINE_PREFIX), f"unexpected ready line: {line!r}"
    endpoint = line[len(READY_LINE_PREFIX) :].strip()
    kind, host, port_text = endpoint.split(":")
    assert kind == "tcp"
    return proc, host, int(port_text)


async def stop_service(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        # Windows launchers exit before their service children finish terminating.
        children = (
            psutil.Process(proc.pid).children(recursive=True) if sys.platform == "win32" else []
        )
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 10.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()
        if children:
            _, alive = await asyncio.to_thread(psutil.wait_procs, children, timeout=10.0)
            assert not alive, f"service children did not exit: {[child.pid for child in alive]}"


class AssetHost:
    """Loopback HTTP host whose test supplies the asset response."""

    def __init__(self) -> None:
        self.requests = 0
        self._runner: web.AppRunner | None = None
        self.endpoint = ""

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        raise NotImplementedError

    async def __aenter__(self) -> AssetHost:
        app = web.Application()
        app.router.add_get("/assets/{digest}", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        server = self._runner.addresses[0]
        self.endpoint = f"http://{server[0]}:{server[1]}"
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

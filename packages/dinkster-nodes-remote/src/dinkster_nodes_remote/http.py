from __future__ import annotations

import os
from collections.abc import AsyncIterator, Generator, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, cast

import httpcore
import httpx


@contextmanager
def _transport_errors() -> Generator[None]:
    try:
        yield
    except (
        httpcore.TimeoutException,
        httpcore.NetworkError,
        httpcore.ProtocolError,
        httpcore.ProxyError,
        httpcore.UnsupportedProtocol,
    ) as exc:
        raise httpx.TransportError(str(exc)) from exc


def _request(request: httpx.Request) -> httpcore.Request:
    return httpcore.Request(
        request.method,
        httpcore.URL(
            scheme=request.url.raw_scheme,
            host=request.url.raw_host,
            port=request.url.port,
            target=request.url.raw_path,
        ),
        headers=request.headers.raw,
        content=request.stream,
        extensions=request.extensions,
    )


class _ResponseStream(httpx.SyncByteStream):
    def __init__(self, response: httpcore.Response) -> None:
        self.response = response

    def __iter__(self) -> Iterator[bytes]:
        with _transport_errors():
            yield from self.response.iter_stream()

    def close(self) -> None:
        self.response.close()


class _AsyncResponseStream(httpx.AsyncByteStream):
    def __init__(self, response: httpcore.Response) -> None:
        self.response = response

    async def __aiter__(self) -> AsyncIterator[bytes]:
        with _transport_errors():
            async for chunk in self.response.aiter_stream():
                yield chunk

    async def aclose(self) -> None:
        await self.response.aclose()


class _UnixBackend(httpcore.SyncBackend):
    def __init__(self, path: str) -> None:
        self.path = path

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        return self.connect_unix_socket(self.path, timeout, socket_options)


class _AsyncUnixBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, path: str) -> None:
        self.path = path
        self.backend = cast("httpcore.AsyncNetworkBackend", httpcore.AnyIOBackend())

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        return await self.backend.connect_unix_socket(self.path, timeout, socket_options)


class _UnixProxyTransport(httpx.BaseTransport):
    def __init__(self, path: str) -> None:
        # Proxy pools ignore uds; force their only network connection onto the host socket.
        self.pool = httpcore.HTTPProxy(
            proxy_url="http://localhost",
            network_backend=_UnixBackend(path),
            ssl_context=httpx.create_ssl_context(),
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        with _transport_errors():
            response = self.pool.handle_request(_request(request))
        return httpx.Response(
            response.status,
            headers=response.headers,
            stream=_ResponseStream(response),
            extensions=cast("Any", response).extensions,
        )

    def close(self) -> None:
        self.pool.close()


class _AsyncUnixProxyTransport(httpx.AsyncBaseTransport):
    def __init__(self, path: str) -> None:
        self.pool = httpcore.AsyncHTTPProxy(
            proxy_url="http://localhost",
            network_backend=_AsyncUnixBackend(path),
            ssl_context=httpx.create_ssl_context(),
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        with _transport_errors():
            response = await self.pool.handle_async_request(_request(request))
        return httpx.Response(
            response.status,
            headers=response.headers,
            stream=_AsyncResponseStream(response),
            extensions=cast("Any", response).extensions,
        )

    async def aclose(self) -> None:
        await self.pool.aclose()


def http_client(*, timeout: float) -> httpx.Client:
    proxy = os.environ.get("DINKSTER_EGRESS_PROXY", "")
    if not proxy:
        return httpx.Client(timeout=timeout)
    if proxy.startswith("/"):
        return httpx.Client(timeout=timeout, transport=_UnixProxyTransport(proxy), trust_env=False)
    return httpx.Client(
        timeout=timeout, proxy=proxy, trust_env=False, verify=httpx.create_ssl_context()
    )


def async_http_client(*, timeout: float) -> httpx.AsyncClient:
    proxy = os.environ.get("DINKSTER_EGRESS_PROXY", "")
    if not proxy:
        return httpx.AsyncClient(timeout=timeout)
    if proxy.startswith("/"):
        return httpx.AsyncClient(
            timeout=timeout, transport=_AsyncUnixProxyTransport(proxy), trust_env=False
        )
    return httpx.AsyncClient(
        timeout=timeout, proxy=proxy, trust_env=False, verify=httpx.create_ssl_context()
    )

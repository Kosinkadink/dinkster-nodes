"""Local HTTP artifacts for registry publication tests."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _serve_artifacts(
    files: Mapping[str, bytes | Callable[[], bytes]], requests: list[tuple[str, str | None]]
) -> ThreadingHTTPServer:
    """A registry stand-in: plain HTTP GET of hosted paths, recording
    (path, Authorization header) per request. Caller must shutdown()."""

    class Handler(BaseHTTPRequestHandler):
        def respond(self) -> None:
            requests.append((self.path, self.headers.get("Authorization")))
            blob = files.get(self.path)
            if blob is None:
                self.send_response(404)
                self.end_headers()
                return
            if callable(blob):
                blob = blob()
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def do_GET(self) -> None:
            self.respond()

        def do_POST(self) -> None:
            self.respond()

        def log_message(self, format: str, *args: object) -> None:
            pass  # keep test output clean

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

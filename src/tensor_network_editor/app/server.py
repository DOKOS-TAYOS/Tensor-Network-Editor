from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import routes
from .session import EditorSession

LOGGER = logging.getLogger(__name__)


class EditorServer:
    def __init__(
        self, session: EditorSession, host: str = "127.0.0.1", port: int = 0
    ) -> None:
        self.session = session
        self.host = host
        self.port = port
        self._static_dir = Path(__file__).resolve().parent / "static"
        self._asset_version = str(
            int(
                max(
                    path.stat().st_mtime
                    for path in self._static_dir.rglob("*")
                    if path.is_file()
                )
            )
        )
        self._server = ThreadingHTTPServer((host, port), self._build_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        server_address = self._server.server_address
        host = server_address[0]
        port = server_address[1]
        host_text = host.decode("utf-8") if isinstance(host, bytes) else str(host)
        return f"http://{host_text}:{port}"

    def start(self) -> None:
        self._thread.start()
        LOGGER.info("Editor server started at %s", self.base_url)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        LOGGER.info("Editor server stopped")

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        session = self.session
        static_dir = self._static_dir
        asset_version = self._asset_version

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/api/bootstrap":
                    status, payload = routes.handle_bootstrap(session)
                    self._write_json(status, payload)
                    return

                if parsed.path == "/":
                    self._serve_index(static_dir / "index.html", asset_version)
                    return

                static_path = (static_dir / parsed.path.lstrip("/")).resolve()
                if (
                    not str(static_path).startswith(str(static_dir.resolve()))
                    or not static_path.is_file()
                ):
                    self._write_json(
                        HTTPStatus.NOT_FOUND, {"ok": False, "message": "Not found."}
                    )
                    return

                content_type = "application/octet-stream"
                if static_path.suffix == ".css":
                    content_type = "text/css; charset=utf-8"
                elif static_path.suffix == ".js":
                    content_type = "application/javascript; charset=utf-8"
                self._serve_static(static_path, content_type)

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                try:
                    payload = routes.read_json(body)
                except ValueError as exc:
                    LOGGER.warning(
                        "Rejected malformed JSON request for %s: %s",
                        parsed.path,
                        exc,
                    )
                    self._write_json(
                        HTTPStatus.BAD_REQUEST,
                        {"ok": False, "message": str(exc)},
                    )
                    return
                try:
                    if parsed.path == "/api/validate":
                        status, response = routes.handle_validate(session, payload)
                    elif parsed.path == "/api/generate":
                        status, response = routes.handle_generate(session, payload)
                    elif parsed.path == "/api/complete":
                        status, response = routes.handle_complete(session, payload)
                    elif parsed.path == "/api/cancel":
                        status, response = routes.handle_cancel(session)
                    else:
                        status, response = (
                            HTTPStatus.NOT_FOUND,
                            {"ok": False, "message": "Not found."},
                        )
                except Exception:  # pragma: no cover - defensive server guard
                    LOGGER.exception(
                        "Unhandled exception while processing %s %s",
                        self.command,
                        parsed.path,
                    )
                    status, response = (
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {
                            "ok": False,
                            "message": "Internal server error.",
                        },
                    )
                self._write_json(status, response)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return

            def _serve_static(self, path: Path, content_type: str) -> None:
                if not path.is_file():
                    self._write_json(
                        HTTPStatus.NOT_FOUND, {"ok": False, "message": "Not found."}
                    )
                    return
                body = path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self._write_no_cache_headers()
                self.end_headers()
                self.wfile.write(body)

            def _serve_index(self, path: Path, asset_version: str) -> None:
                body_text = path.read_text(encoding="utf-8").replace(
                    "__ASSET_VERSION__", asset_version
                )
                body = body_text.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self._write_no_cache_headers()
                self.end_headers()
                self.wfile.write(body)

            def _write_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self._write_no_cache_headers()
                self.end_headers()
                self.wfile.write(body)

            def _write_no_cache_headers(self) -> None:
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")

        return RequestHandler

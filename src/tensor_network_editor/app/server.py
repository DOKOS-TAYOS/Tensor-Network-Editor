from __future__ import annotations

import json
import logging
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import routes
from ._protocol import internal_server_error_response, not_found_response
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
                self._write_json(*self._dispatch_get(parsed.path))

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                body = self._read_request_body()
                try:
                    payload = routes.read_json(body)
                except ValueError as exc:
                    LOGGER.warning(
                        "Rejected malformed JSON request for %s: %s",
                        parsed.path,
                        exc,
                    )
                    self._write_json(
                        HTTPStatus.BAD_REQUEST, {"ok": False, "message": str(exc)}
                    )
                    return
                try:
                    status, response = self._dispatch_post(parsed.path, payload)
                except Exception:  # pragma: no cover - defensive server guard
                    LOGGER.exception(
                        "Unhandled exception while processing %s %s",
                        self.command,
                        parsed.path,
                    )
                    status, response = internal_server_error_response()
                self._write_json(status, response)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return

            def _dispatch_get(self, path: str) -> tuple[int, dict[str, Any]]:
                if path == "/api/bootstrap":
                    return routes.handle_bootstrap(session)
                if path == "/":
                    return self._index_response(
                        static_dir / "index.html", asset_version
                    )
                return self._static_response(path)

            def _dispatch_post(
                self, path: str, payload: dict[str, Any]
            ) -> tuple[int, dict[str, Any]]:
                if path == "/api/validate":
                    return routes.handle_validate(session, payload)
                if path == "/api/template":
                    return routes.handle_template(session, payload)
                if path == "/api/generate":
                    return routes.handle_generate(session, payload)
                if path == "/api/analyze-contraction":
                    return routes.handle_analyze_contraction(session, payload)
                if path == "/api/complete":
                    return routes.handle_complete(session, payload)
                if path == "/api/cancel":
                    return routes.handle_cancel(session)
                return not_found_response()

            def _static_response(self, request_path: str) -> tuple[int, dict[str, Any]]:
                static_path = self._resolve_static_path(request_path)
                if static_path is None:
                    return not_found_response()
                body = static_path.read_bytes()
                return HTTPStatus.OK, {
                    "__binary_body__": body,
                    "__content_type__": self._content_type_for_path(static_path),
                }

            def _index_response(
                self, path: Path, asset_version: str
            ) -> tuple[int, dict[str, Any]]:
                body_text = path.read_text(encoding="utf-8").replace(
                    "__ASSET_VERSION__", asset_version
                )
                return HTTPStatus.OK, {
                    "__binary_body__": body_text.encode("utf-8"),
                    "__content_type__": "text/html; charset=utf-8",
                }

            def _resolve_static_path(self, request_path: str) -> Path | None:
                candidate = (static_dir / request_path.lstrip("/")).resolve()
                if not str(candidate).startswith(str(static_dir.resolve())):
                    return None
                if not candidate.is_file():
                    return None
                return candidate

            def _content_type_for_path(self, path: Path) -> str:
                guessed_type, _ = mimetypes.guess_type(path.name)
                if path.suffix == ".js":
                    return "application/javascript; charset=utf-8"
                if path.suffix == ".css":
                    return "text/css; charset=utf-8"
                if path.suffix == ".html":
                    return "text/html; charset=utf-8"
                if guessed_type is None:
                    return "application/octet-stream"
                if guessed_type.startswith("text/"):
                    return f"{guessed_type}; charset=utf-8"
                return guessed_type

            def _read_request_body(self) -> bytes:
                return self.rfile.read(int(self.headers.get("Content-Length", "0")))

            def _write_json(self, status: int, payload: dict[str, Any]) -> None:
                binary_body = payload.pop("__binary_body__", None)
                content_type = payload.pop("__content_type__", None)
                if isinstance(binary_body, bytes) and isinstance(content_type, str):
                    self._write_bytes(status, binary_body, content_type)
                    return
                body = json.dumps(payload).encode("utf-8")
                self._write_bytes(status, body, "application/json; charset=utf-8")

            def _write_bytes(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self._write_no_cache_headers()
                self.end_headers()
                self.wfile.write(body)

            def _write_no_cache_headers(self) -> None:
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")

        return RequestHandler

"""Threaded local HTTP server that hosts the browser editor."""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BufferedReader
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlparse

from . import routes
from ._protocol import (
    JsonDict,
    JsonResponse,
    bad_request_response,
    internal_server_error_response,
    not_found_response,
    read_json,
)
from .session import EditorSession

LOGGER = logging.getLogger(__name__)
_SERVE_FOREVER_POLL_INTERVAL_SECONDS: float = 0.05
_MAX_REQUEST_BODY_BYTES: int = 1_048_576
_STATIC_ASSET_CACHE_LOCK = threading.Lock()
_STATIC_ASSET_CACHE_BY_ROOT: dict[Path, _StaticAssetCache] = {}


class SupportsReadBytes(Protocol):
    """Protocol for byte streams that support sized reads."""

    def read(self, size: int | None = -1, /) -> bytes:
        """Read up to ``size`` bytes from the underlying stream."""
        ...


def _parse_content_length(content_length_text: str | None) -> int:
    """Return a validated request body length from a Content-Length header."""
    if content_length_text is None:
        return 0
    try:
        content_length = int(content_length_text)
    except ValueError as exc:
        raise ValueError("Invalid Content-Length header.") from exc
    if content_length < 0:
        raise ValueError("Invalid Content-Length header: must be >= 0.")
    if content_length > _MAX_REQUEST_BODY_BYTES:
        raise ValueError("Request body exceeds maximum allowed size.")
    return content_length


def _read_request_body_bytes(reader: SupportsReadBytes, content_length: int) -> bytes:
    """Read exactly ``content_length`` bytes or raise when the stream ends early."""
    if content_length == 0:
        return b""
    chunks: list[bytes] = []
    remaining = content_length
    while remaining > 0:
        chunk = reader.read(remaining)
        if not chunk:
            raise ValueError("Request body ended before all bytes arrived.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass(slots=True, frozen=True)
class _BinaryResponse:
    """Internal response container for pre-encoded bytes."""

    status: int
    body: bytes
    content_type: str


@dataclass(slots=True, frozen=True)
class _StaticAssetCache:
    """Process-wide cache of static editor assets for one static root."""

    asset_version: str
    body_by_relative_path: dict[str, bytes]
    content_type_by_relative_path: dict[str, str]
    index_body: bytes
    source_signature: tuple[tuple[str, int, int], ...]


def _content_type_for_path(path: Path) -> str:
    """Guess the HTTP content type for one static asset path."""
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


def _scan_static_asset_files(
    static_dir: Path,
) -> list[tuple[Path, str, int, int]]:
    """Return sorted static asset metadata for one static directory."""
    resolved_static_dir = static_dir.resolve()
    return [
        (
            path,
            path.relative_to(resolved_static_dir).as_posix(),
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for path in sorted(
            path for path in resolved_static_dir.rglob("*") if path.is_file()
        )
    ]


def _build_static_asset_cache(static_dir: Path) -> _StaticAssetCache:
    """Read and cache the static editor asset tree for one process."""
    resolved_static_dir = static_dir.resolve()
    scanned_files = _scan_static_asset_files(resolved_static_dir)
    asset_version = (
        str(max(mtime_ns for _, _, mtime_ns, _ in scanned_files))
        if scanned_files
        else "0"
    )
    body_by_relative_path: dict[str, bytes] = {}
    content_type_by_relative_path: dict[str, str] = {}
    source_signature = tuple(
        (relative_path, mtime_ns, size)
        for _, relative_path, mtime_ns, size in scanned_files
    )

    for file_path, relative_path, _, _ in scanned_files:
        if relative_path == "index.html":
            continue
        body_by_relative_path[relative_path] = file_path.read_bytes()
        content_type_by_relative_path[relative_path] = _content_type_for_path(file_path)

    index_body = (
        (resolved_static_dir / "index.html")
        .read_text(encoding="utf-8")
        .replace("__ASSET_VERSION__", asset_version)
        .encode("utf-8")
    )
    return _StaticAssetCache(
        asset_version=asset_version,
        body_by_relative_path=body_by_relative_path,
        content_type_by_relative_path=content_type_by_relative_path,
        index_body=index_body,
        source_signature=source_signature,
    )


def _get_static_asset_cache(static_dir: Path) -> _StaticAssetCache:
    """Return a shared static asset cache for one editor static directory."""
    resolved_static_dir = static_dir.resolve()
    current_signature = tuple(
        (relative_path, mtime_ns, size)
        for _, relative_path, mtime_ns, size in _scan_static_asset_files(
            resolved_static_dir
        )
    )
    with _STATIC_ASSET_CACHE_LOCK:
        cache = _STATIC_ASSET_CACHE_BY_ROOT.get(resolved_static_dir)
        if cache is None or cache.source_signature != current_signature:
            cache = _build_static_asset_cache(resolved_static_dir)
            _STATIC_ASSET_CACHE_BY_ROOT[resolved_static_dir] = cache
        return cache


class EditorServer:
    """Serve the browser app and JSON API for one editor session."""

    def __init__(
        self, session: EditorSession, host: str = "127.0.0.1", port: int = 0
    ) -> None:
        """Initialize the threaded local editor server.

        Args:
            session: Shared editor session state served by this HTTP server.
            host: Local host interface to bind.
            port: Local port to bind. Use ``0`` for an ephemeral port.
        """
        self.session = session
        self.session_id = session.session_id
        self.host = host
        self.port = port
        self._static_dir = Path(__file__).resolve().parent / "static"
        self._static_asset_cache = _get_static_asset_cache(self._static_dir)
        self._asset_version = self._static_asset_cache.asset_version
        self._server = ThreadingHTTPServer((host, port), self._build_handler())
        self._thread = threading.Thread(target=self._serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        """Return the local base URL assigned to the server."""
        server_address = self._server.server_address
        host = server_address[0]
        port = server_address[1]
        host_text = host.decode("utf-8") if isinstance(host, bytes) else str(host)
        return f"http://{host_text}:{port}"

    def start(self) -> None:
        """Start serving requests in a background thread."""
        self._thread.start()
        LOGGER.info(
            "[session=%s] Editor server started at %s",
            self.session_id,
            self.base_url,
        )

    def stop(self) -> None:
        """Stop the server and wait for the worker thread to exit."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        LOGGER.info("[session=%s] Editor server stopped", self.session_id)

    def _serve_forever(self) -> None:
        """Serve requests with a short shutdown polling interval."""
        self._server.serve_forever(poll_interval=_SERVE_FOREVER_POLL_INTERVAL_SECONDS)

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        """Build the request-handler class bound to this server instance."""
        session = self.session
        session_id = self.session_id
        static_dir = self._static_dir
        static_asset_cache = self._static_asset_cache

        class RequestHandler(BaseHTTPRequestHandler):
            """Serve static editor assets and JSON routes for one session."""

            def do_GET(self) -> None:
                """Handle one HTTP GET request for assets or bootstrap data."""
                parsed = urlparse(self.path)
                try:
                    response = self._dispatch_get(parsed.path)
                except Exception:  # pragma: no cover - defensive server guard
                    LOGGER.exception(
                        "[session=%s] Unhandled exception while processing %s %s",
                        session_id,
                        self.command,
                        parsed.path,
                    )
                    response = internal_server_error_response()
                self._write_response(response)

            def do_POST(self) -> None:
                """Handle one HTTP POST request for the editor JSON API."""
                parsed = urlparse(self.path)
                try:
                    body = self._read_request_body()
                except ValueError as exc:
                    LOGGER.warning(
                        "[session=%s] Rejected malformed request body for %s: %s",
                        session_id,
                        parsed.path,
                        exc,
                    )
                    self._drain_pending_request_body()
                    self.close_connection = True
                    self._write_response(bad_request_response(str(exc)))
                    return
                try:
                    payload = read_json(body)
                except ValueError as exc:
                    LOGGER.warning(
                        "[session=%s] Rejected malformed JSON request for %s: %s",
                        session_id,
                        parsed.path,
                        exc,
                    )
                    self.close_connection = True
                    self._write_response(bad_request_response(str(exc)))
                    return
                try:
                    response = self._dispatch_post(parsed.path, payload)
                except Exception:  # pragma: no cover - defensive server guard
                    LOGGER.exception(
                        "[session=%s] Unhandled exception while processing %s %s",
                        session_id,
                        self.command,
                        parsed.path,
                    )
                    response = internal_server_error_response()
                self._write_response(response)

            def log_message(self, format: str, *args: object) -> None:
                """Suppress default stderr logging for handled HTTP requests."""
                del format, args
                return

            def _dispatch_get(self, path: str) -> JsonResponse | _BinaryResponse:
                """Route one GET request to bootstrap, index, or static assets."""
                if path == "/api/bootstrap":
                    return routes.handle_bootstrap(session)
                if path == "/":
                    return self._index_response()
                return self._static_response(path)

            def _dispatch_post(self, path: str, payload: JsonDict) -> JsonResponse:
                """Route one POST request to the matching JSON API handler."""
                if path == "/api/validate":
                    return routes.handle_validate(session, payload)
                if path == "/api/template":
                    return routes.handle_template(session, payload)
                if path == "/api/template/promote":
                    return routes.handle_template_promote(session, payload)
                if path == "/api/template/rename":
                    return routes.handle_template_rename(session, payload)
                if path == "/api/template/delete":
                    return routes.handle_template_delete(session, payload)
                if path == "/api/subnetwork/extract":
                    return routes.handle_subnetwork_extract(session, payload)
                if path == "/api/subnetwork/prepare-insert":
                    return routes.handle_subnetwork_prepare_insert(session, payload)
                if path == "/api/generate":
                    return routes.handle_generate(session, payload)
                if path == "/api/analyze-contraction":
                    return routes.handle_analyze_contraction(session, payload)
                if path == "/api/complete":
                    return routes.handle_complete(session, payload)
                if path == "/api/cancel":
                    return routes.handle_cancel(session)
                LOGGER.debug("[session=%s] Unknown POST path: %s", session_id, path)
                return not_found_response()

            def _static_response(
                self, request_path: str
            ) -> JsonResponse | _BinaryResponse:
                """Return one static asset response when the path resolves safely."""
                relative_path = self._resolve_static_asset_relative_path(request_path)
                if relative_path is None:
                    LOGGER.debug(
                        "[session=%s] Static asset not found for path %s",
                        session_id,
                        request_path,
                    )
                    return not_found_response()
                return _BinaryResponse(
                    status=HTTPStatus.OK,
                    body=static_asset_cache.body_by_relative_path[relative_path],
                    content_type=static_asset_cache.content_type_by_relative_path[
                        relative_path
                    ],
                )

            def _index_response(self) -> _BinaryResponse:
                """Return the cached main HTML page for this editor session."""
                return _BinaryResponse(
                    status=HTTPStatus.OK,
                    body=static_asset_cache.index_body,
                    content_type="text/html; charset=utf-8",
                )

            def _resolve_static_asset_relative_path(
                self, request_path: str
            ) -> str | None:
                """Resolve one request path to a cached static asset key."""
                static_root = static_dir.resolve()
                candidate = (static_dir / request_path.lstrip("/")).resolve()
                try:
                    relative_path = candidate.relative_to(static_root)
                except ValueError:
                    return None
                relative_path_text = relative_path.as_posix()
                if relative_path_text not in static_asset_cache.body_by_relative_path:
                    return None
                return relative_path_text

            def _read_request_body(self) -> bytes:
                """Read one request body after validating the Content-Length header."""
                content_length = _parse_content_length(
                    self.headers.get("Content-Length")
                )
                return _read_request_body_bytes(self.rfile, content_length)

            def _drain_pending_request_body(self) -> None:
                """Best-effort drain of pending request bytes before closing."""
                previous_timeout = self.connection.gettimeout()
                buffered_reader = cast(BufferedReader, self.rfile)
                try:
                    self.connection.settimeout(0.01)
                    while True:
                        buffered = buffered_reader.peek(1)
                        if not buffered:
                            break
                        discarded = buffered_reader.read(len(buffered))
                        if not discarded:
                            break
                except OSError:
                    return
                finally:
                    self.connection.settimeout(previous_timeout)

            def _write_response(self, response: JsonResponse | _BinaryResponse) -> None:
                """Serialize and send either a JSON or pre-encoded binary response."""
                if isinstance(response, _BinaryResponse):
                    self._write_bytes(
                        response.status, response.body, response.content_type
                    )
                    return
                status, payload = response
                self._write_json(status, payload)

            def _write_json(self, status: int, payload: JsonDict) -> None:
                """Encode one JSON payload and send it as an HTTP response."""
                body = json.dumps(payload).encode("utf-8")
                self._write_bytes(status, body, "application/json; charset=utf-8")

            def _write_bytes(self, status: int, body: bytes, content_type: str) -> None:
                """Send one raw byte payload with the provided HTTP metadata."""
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                if self.close_connection:
                    self.send_header("Connection", "close")
                self._write_no_cache_headers()
                self.end_headers()
                self.wfile.write(body)

            def _write_no_cache_headers(self) -> None:
                """Emit headers that disable browser and intermediary caching."""
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")

        return RequestHandler

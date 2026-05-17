"""Threaded local HTTP server that hosts the browser editor."""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import mimetypes
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BufferedReader
from pathlib import Path
from typing import Protocol, TypeAlias, cast
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

from ..internal._logging import (
    bind_log_context,
    format_log_message,
    log_branch,
    log_operation,
)
from . import routes
from ._bootstrap_payloads import build_frontend_logging_payload
from ._protocol import (
    JsonDict,
    JsonResponse,
    bad_request_response,
    forbidden_response,
    internal_server_error_response,
    not_found_response,
    read_json,
    unsupported_media_type_response,
)
from .session import EditorSession

LOGGER = logging.getLogger(__name__)
_SERVE_FOREVER_POLL_INTERVAL_SECONDS: float = 0.05
_STARTUP_READY_TIMEOUT_SECONDS: float = 5.0
_STARTUP_READY_POLL_INTERVAL_SECONDS: float = 0.01
_STARTUP_READY_REQUEST_TIMEOUT_SECONDS: float = 0.2
_RESPONSE_WRITE_CHUNK_SIZE_BYTES: int = 64 * 1024
_STATIC_ASSET_CACHE_VALIDATION_INTERVAL_SECONDS: float = 0.5
_MAX_REQUEST_BODY_BYTES: int = 1_048_576
_STATIC_ASSET_CACHE_LOCK = threading.Lock()
_STATIC_ASSET_CACHE_BY_ROOT: dict[Path, _StaticAssetCache] = {}
_STATIC_ASSET_CACHE_LAST_VALIDATED_AT_BY_ROOT: dict[Path, float] = {}
_UNEXPECTED_INTERNAL_ERROR_MESSAGE = "Unexpected internal error."
_UNEXPECTED_INTERNAL_ERROR_GUIDANCE = (
    "Try again. If the problem continues, check the terminal output for this "
    "session or rerun with debug logging."
)
_QUIET_MISSING_STATIC_ASSET_PATHS: frozenset[str] = frozenset({"/favicon.ico"})
_ScannedStaticAssetFile: TypeAlias = tuple[Path, str, int, int]
_RUNTIME_CONFIG_PLACEHOLDER = "__TNE_RUNTIME_CONFIG__"
_CSP_NONCE_PLACEHOLDER = "__TNE_CSP_NONCE__"
_API_TOKEN_HEADER = "X-TNE-Session-Token"  # noqa: S105, RUF100 - header name.
_EXPECTED_JSON_CONTENT_TYPE = "application/json"
_PERMISSIONS_POLICY_HEADER = (
    "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
    "magnetometer=(), microphone=(), payment=(), usb=()"
)


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


def _is_loopback_host_name(host_name: str) -> bool:
    """Return whether a hostname literal is safe for local-only editor serving."""
    normalized_host = host_name.strip().strip("[]").rstrip(".").lower()
    if normalized_host in {"localhost"} or normalized_host.endswith(".localhost"):
        return True
    if "%" in normalized_host:
        normalized_host = normalized_host.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        return False
    return address.is_loopback


def _validate_bind_host(host: str, *, allow_remote: bool) -> None:
    """Reject non-loopback bind hosts unless remote serving is explicit."""
    if allow_remote or _is_loopback_host_name(host):
        return
    raise ValueError(
        "Refusing to bind the editor server to a non-loopback host. "
        "Use allow_remote=True only when you intentionally expose this local API."
    )


def _host_name_from_header(host_header: str | None) -> str | None:
    """Extract the hostname portion from one HTTP Host header."""
    if host_header is None:
        return None
    value = host_header.strip()
    if not value:
        return None
    if value.startswith("["):
        end_index = value.find("]")
        if end_index <= 1:
            return None
        return value[1:end_index]
    if value.count(":") == 1:
        host_name, port_text = value.rsplit(":", 1)
        if port_text.isdigit():
            return host_name
    return value


def _is_trusted_host_header(host_header: str | None, *, allow_remote: bool) -> bool:
    """Return whether one Host header is acceptable for this server."""
    if allow_remote:
        return bool(host_header and host_header.strip())
    host_name = _host_name_from_header(host_header)
    return host_name is not None and _is_loopback_host_name(host_name)


def _is_trusted_origin_header(origin_header: str | None, *, allow_remote: bool) -> bool:
    """Return whether one optional Origin header is acceptable for API writes."""
    if origin_header is None:
        return True
    parsed_origin = urlparse(origin_header)
    if parsed_origin.scheme not in {"http", "https"}:
        return False
    return _is_trusted_host_header(parsed_origin.netloc, allow_remote=allow_remote)


def _is_json_content_type(content_type: str | None) -> bool:
    """Return whether one Content-Type header identifies a JSON request body."""
    if content_type is None:
        return False
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type == _EXPECTED_JSON_CONTENT_TYPE


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
) -> list[_ScannedStaticAssetFile]:
    """Return sorted static asset metadata for one static directory."""
    resolved_static_dir = static_dir.resolve()
    scanned_files: list[_ScannedStaticAssetFile] = []
    for path in sorted(
        path for path in resolved_static_dir.rglob("*") if path.is_file()
    ):
        path_stat = path.stat()
        scanned_files.append(
            (
                path,
                path.relative_to(resolved_static_dir).as_posix(),
                path_stat.st_mtime_ns,
                path_stat.st_size,
            )
        )
    return scanned_files


def _build_static_asset_source_signature(
    scanned_files: list[_ScannedStaticAssetFile],
) -> tuple[tuple[str, int, int], ...]:
    """Return the stable change-detection signature for one asset scan."""
    return tuple(
        (relative_path, mtime_ns, size)
        for _, relative_path, mtime_ns, size in scanned_files
    )


def _build_static_asset_cache(
    static_dir: Path,
    *,
    scanned_files: list[_ScannedStaticAssetFile] | None = None,
) -> _StaticAssetCache:
    """Read and cache the static editor asset tree for one process."""
    resolved_static_dir = static_dir.resolve()
    resolved_scanned_files = (
        scanned_files
        if scanned_files is not None
        else _scan_static_asset_files(resolved_static_dir)
    )
    asset_version = (
        str(max(mtime_ns for _, _, mtime_ns, _ in resolved_scanned_files))
        if resolved_scanned_files
        else "0"
    )
    body_by_relative_path: dict[str, bytes] = {}
    content_type_by_relative_path: dict[str, str] = {}
    source_signature = _build_static_asset_source_signature(resolved_scanned_files)

    for file_path, relative_path, _, _ in resolved_scanned_files:
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
    with _STATIC_ASSET_CACHE_LOCK:
        validation_started_at = time.monotonic()
        cache = _STATIC_ASSET_CACHE_BY_ROOT.get(resolved_static_dir)
        last_validated_at = _STATIC_ASSET_CACHE_LAST_VALIDATED_AT_BY_ROOT.get(
            resolved_static_dir
        )
        if cache is None:
            scanned_files = _scan_static_asset_files(resolved_static_dir)
            with log_operation(
                LOGGER,
                "Static asset cache build",
                context={"path": resolved_static_dir},
            ) as success_context:
                cache = _build_static_asset_cache(
                    resolved_static_dir,
                    scanned_files=scanned_files,
                )
                _STATIC_ASSET_CACHE_BY_ROOT[resolved_static_dir] = cache
                _STATIC_ASSET_CACHE_LAST_VALIDATED_AT_BY_ROOT[resolved_static_dir] = (
                    validation_started_at
                )
                success_context["after"] = cache.asset_version
                success_context["asset_count"] = len(cache.body_by_relative_path)
                return cache
        if (
            last_validated_at is not None
            and validation_started_at - last_validated_at
            < _STATIC_ASSET_CACHE_VALIDATION_INTERVAL_SECONDS
        ):
            log_branch(
                LOGGER,
                "Static asset cache reused",
                context={
                    "path": resolved_static_dir,
                    "after": cache.asset_version,
                    "asset_count": len(cache.body_by_relative_path),
                },
            )
            return cache
        scanned_files = _scan_static_asset_files(resolved_static_dir)
        current_signature = _build_static_asset_source_signature(scanned_files)
        if cache.source_signature != current_signature:
            with log_operation(
                LOGGER,
                "Static asset cache refresh",
                context={
                    "path": resolved_static_dir,
                    "before": cache.asset_version,
                },
            ) as success_context:
                refreshed_cache = _build_static_asset_cache(
                    resolved_static_dir,
                    scanned_files=scanned_files,
                )
                _STATIC_ASSET_CACHE_BY_ROOT[resolved_static_dir] = refreshed_cache
                _STATIC_ASSET_CACHE_LAST_VALIDATED_AT_BY_ROOT[resolved_static_dir] = (
                    validation_started_at
                )
                success_context["after"] = refreshed_cache.asset_version
                success_context["asset_count"] = len(
                    refreshed_cache.body_by_relative_path
                )
                return refreshed_cache
        _STATIC_ASSET_CACHE_LAST_VALIDATED_AT_BY_ROOT[resolved_static_dir] = (
            validation_started_at
        )
        log_branch(
            LOGGER,
            "Static asset cache reused",
            context={
                "path": resolved_static_dir,
                "after": cache.asset_version,
                "asset_count": len(cache.body_by_relative_path),
            },
        )
        return cache


def _build_frontend_runtime_config_payload(
    session: EditorSession, *, api_token: str
) -> JsonDict:
    """Return the runtime configuration embedded into the editor HTML page."""
    return {
        "session_id": session.session_id,
        "api_token": api_token,
        "frontend_logging": build_frontend_logging_payload(session),
    }


def _serialize_frontend_runtime_config(
    session: EditorSession, *, api_token: str
) -> str:
    """Serialize one session runtime config safely for an inline JSON script."""
    return json.dumps(
        _build_frontend_runtime_config_payload(session, api_token=api_token)
    ).replace("</", "<\\/")


def _render_session_index_body(
    index_body: bytes,
    session: EditorSession,
    *,
    api_token: str,
    csp_nonce: str,
) -> bytes:
    """Return the per-session editor HTML body with embedded runtime config."""
    return index_body.replace(
        _RUNTIME_CONFIG_PLACEHOLDER.encode("utf-8"),
        _serialize_frontend_runtime_config(session, api_token=api_token).encode(
            "utf-8"
        ),
    ).replace(
        _CSP_NONCE_PLACEHOLDER.encode("utf-8"),
        csp_nonce.encode("utf-8"),
    )


def _build_content_security_policy(*, csp_nonce: str) -> str:
    """Return the editor CSP that permits only trusted local assets."""
    directives = [
        "default-src 'self'",
        "base-uri 'none'",
        "object-src 'none'",
        "frame-ancestors 'none'",
        "form-action 'none'",
        "connect-src 'self'",
        "img-src 'self' data: blob:",
        f"script-src 'self' 'nonce-{csp_nonce}'",
        "style-src 'self' 'unsafe-inline'",
        "font-src 'self' data:",
        "worker-src 'self' blob:",
    ]
    return "; ".join(directives)


def _unexpected_internal_error_response(session_id: str) -> JsonResponse:
    """Return an actionable but safe error payload for unexpected failures."""
    return internal_server_error_response(
        message=_UNEXPECTED_INTERNAL_ERROR_MESSAGE,
        guidance=_UNEXPECTED_INTERNAL_ERROR_GUIDANCE,
        reference=session_id,
    )


def _should_log_missing_static_asset(request_path: str) -> bool:
    """Return whether one missing static path should appear in debug logs."""
    return request_path not in _QUIET_MISSING_STATIC_ASSET_PATHS


def _normalize_static_asset_request_path(request_path: str) -> str | None:
    """Return a safe static asset cache key for one URL path."""
    decoded_path = unquote(request_path)
    if "\x00" in decoded_path or "\\" in decoded_path:
        return None
    relative_parts: list[str] = []
    for part in decoded_path.lstrip("/").split("/"):
        if not part or part in {".", ".."} or ":" in part:
            return None
        relative_parts.append(part)
    if not relative_parts:
        return None
    return "/".join(relative_parts)


class EditorServer:
    """Serve the browser app and JSON API for one editor session."""

    def __init__(
        self,
        session: EditorSession,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        allow_remote: bool = False,
        api_token: str | None = None,
    ) -> None:
        """Initialize the threaded local editor server.

        Args:
            session: Shared editor session state served by this HTTP server.
            host: Local host interface to bind.
            port: Local port to bind. Use ``0`` for an ephemeral port.
            allow_remote: Whether non-loopback bind hosts are allowed.
            api_token: Optional pre-generated API token for tests.
        """
        _validate_bind_host(host, allow_remote=allow_remote)
        self.session = session
        self.session_id = session.session_id
        self.host = host
        self.port = port
        self.allow_remote = allow_remote
        self.api_token = api_token or secrets.token_urlsafe(32)
        if not self.api_token.strip():
            raise ValueError("Editor API token cannot be empty.")
        self._csp_nonce = secrets.token_urlsafe(16)
        self._content_security_policy = _build_content_security_policy(
            csp_nonce=self._csp_nonce
        )
        self._static_dir = Path(__file__).resolve().parent / "static"
        self._static_asset_cache = _get_static_asset_cache(self._static_dir)
        self._index_body = _render_session_index_body(
            self._static_asset_cache.index_body,
            session,
            api_token=self.api_token,
            csp_nonce=self._csp_nonce,
        )
        self._server = ThreadingHTTPServer((host, port), self._build_handler())
        self._thread = threading.Thread(target=self._serve_forever, daemon=True)
        self._serve_forever_ready = threading.Event()

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
        try:
            self._wait_until_ready()
        except Exception:
            self._cleanup_failed_start()
            raise
        log_branch(
            LOGGER,
            f"Editor server started at {self.base_url}",
            level=logging.INFO,
            context={"session": self.session_id},
        )

    def stop(self) -> None:
        """Stop the server and wait for the worker thread to exit."""
        self._stop_server_worker()
        log_branch(
            LOGGER,
            "Editor server stopped",
            level=logging.INFO,
            context={"session": self.session_id},
        )

    def _serve_forever(self) -> None:
        """Serve requests with a short shutdown polling interval."""
        self._serve_forever_ready.set()
        self._server.serve_forever(poll_interval=_SERVE_FOREVER_POLL_INTERVAL_SECONDS)

    def _wait_until_ready(self) -> None:
        """Block until loopback requests can read one fully served asset."""
        deadline = time.monotonic() + _STARTUP_READY_TIMEOUT_SECONDS
        if not self._serve_forever_ready.wait(timeout=_STARTUP_READY_TIMEOUT_SECONDS):
            raise RuntimeError(
                "Editor server did not enter the serving loop before the startup timeout elapsed."
            )

        last_error: OSError | None = None
        while True:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                break
            request_timeout_seconds = min(
                _STARTUP_READY_REQUEST_TIMEOUT_SECONDS,
                remaining_seconds,
            )
            try:
                self._probe_loopback_readiness(request_timeout_seconds)
            except OSError as exc:
                last_error = exc
                time.sleep(min(_STARTUP_READY_POLL_INTERVAL_SECONDS, remaining_seconds))
                continue
            return

        if last_error is None:
            raise RuntimeError(
                "Editor server readiness probe timed out before any loopback request succeeded."
            )
        raise RuntimeError(
            "Editor server did not become ready to serve loopback requests before the startup timeout elapsed."
        ) from last_error

    def _probe_loopback_readiness(self, timeout_seconds: float) -> None:
        """Read one small static asset to verify the server serves full responses."""
        with urlopen(  # noqa: S310, RUF100 - probes this loopback server.
            f"{self.base_url}/favicon.ico", timeout=timeout_seconds
        ) as response:
            response.read()

    def _stop_server_worker(self) -> None:
        """Best-effort shutdown that is safe before the serve loop starts."""
        if self._thread.is_alive() and self._serve_forever_ready.is_set():
            self._server.shutdown()
        self._server.server_close()
        if self._thread.ident is not None:
            self._thread.join(timeout=5)

    def _cleanup_failed_start(self) -> None:
        """Best-effort cleanup when startup fails after allocating the server socket."""
        self._stop_server_worker()

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        """Build the request-handler class bound to this server instance."""
        session = self.session
        session_id = self.session_id
        static_dir = self._static_dir
        static_asset_cache = self._static_asset_cache
        index_body = self._index_body
        api_token = self.api_token
        allow_remote = self.allow_remote
        content_security_policy = self._content_security_policy

        def build_index_response() -> _BinaryResponse:
            """Return the cached main HTML page for this editor session."""
            return _BinaryResponse(
                status=HTTPStatus.OK,
                body=index_body,
                content_type="text/html; charset=utf-8",
            )

        def adapt_get_route(
            route_name: str,
        ) -> Callable[[], JsonResponse | _BinaryResponse]:
            """Adapt one session GET route to a zero-argument dispatch callback."""

            def handle_route() -> JsonResponse | _BinaryResponse:
                route_handler = cast(
                    Callable[[EditorSession], JsonResponse],
                    getattr(routes, route_name),
                )
                return route_handler(session)

            return handle_route

        def adapt_payload_route(route_name: str) -> Callable[[JsonDict], JsonResponse]:
            """Adapt one session POST route to a payload dispatch callback."""

            def handle_route(payload: JsonDict) -> JsonResponse:
                route_handler = cast(
                    Callable[[EditorSession, JsonDict], JsonResponse],
                    getattr(routes, route_name),
                )
                return route_handler(session, payload)

            return handle_route

        def adapt_session_only_route(
            route_name: str,
        ) -> Callable[[JsonDict], JsonResponse]:
            """Adapt one session-only POST route to the payload handler signature."""
            return lambda _payload: cast(
                Callable[[EditorSession], JsonResponse],
                getattr(routes, route_name),
            )(session)

        get_route_handlers: dict[str, Callable[[], JsonResponse | _BinaryResponse]] = {
            "/api/bootstrap": adapt_get_route("handle_bootstrap"),
            "/api/draft": adapt_get_route("handle_draft_load"),
            "/": build_index_response,
        }
        post_route_handlers: dict[str, Callable[[JsonDict], JsonResponse]] = {
            "/api/client-log": adapt_payload_route("handle_client_log"),
            "/api/validate": adapt_payload_route("handle_validate"),
            "/api/draft": adapt_payload_route("handle_draft_save"),
            "/api/draft/clear": adapt_session_only_route("handle_draft_clear"),
            "/api/template": adapt_payload_route("handle_template"),
            "/api/template/promote": adapt_payload_route("handle_template_promote"),
            "/api/template/rename": adapt_payload_route("handle_template_rename"),
            "/api/template/delete": adapt_payload_route("handle_template_delete"),
            "/api/subnetwork/extract": adapt_payload_route("handle_subnetwork_extract"),
            "/api/subnetwork/prepare-insert": adapt_payload_route(
                "handle_subnetwork_prepare_insert"
            ),
            "/api/subnetwork-library/save": adapt_payload_route(
                "handle_subnetwork_library_save"
            ),
            "/api/subnetwork-library/rename": adapt_payload_route(
                "handle_subnetwork_library_rename"
            ),
            "/api/subnetwork-library/delete": adapt_payload_route(
                "handle_subnetwork_library_delete"
            ),
            "/api/subnetwork-library/prepare-insert": adapt_payload_route(
                "handle_subnetwork_library_prepare_insert"
            ),
            "/api/generate": adapt_payload_route("handle_generate"),
            "/api/render": adapt_payload_route("handle_render"),
            "/api/analyze-contraction": adapt_payload_route(
                "handle_analyze_contraction"
            ),
            "/api/complete": adapt_payload_route("handle_complete"),
            "/api/cancel": adapt_session_only_route("handle_cancel"),
        }

        class RequestHandler(BaseHTTPRequestHandler):
            """Serve static editor assets and JSON routes for one session."""

            def do_GET(self) -> None:
                """Handle one HTTP GET request for assets or bootstrap data."""
                parsed = urlparse(self.path)
                with bind_log_context(session=session_id, route=parsed.path):
                    if self._reject_untrusted_host():
                        return
                    if self._reject_invalid_api_token(parsed.path):
                        return
                    try:
                        with log_operation(LOGGER, "Route request"):
                            response = self._dispatch_get(parsed.path)
                    except Exception:  # pragma: no cover - defensive server guard
                        LOGGER.exception(
                            format_log_message(
                                f"Unhandled exception while processing {self.command} {parsed.path}"
                            ),
                        )
                        response = _unexpected_internal_error_response(session_id)
                self._write_response(response)

            def do_POST(self) -> None:
                """Handle one HTTP POST request for the editor JSON API."""
                parsed = urlparse(self.path)
                with bind_log_context(session=session_id, route=parsed.path):
                    if self._reject_untrusted_host():
                        return
                    if self._reject_untrusted_origin():
                        return
                    if self._reject_invalid_api_token(parsed.path):
                        return
                    if self._reject_unsupported_content_type():
                        return
                    try:
                        with log_operation(LOGGER, "Route request"):
                            try:
                                body = self._read_request_body()
                            except ValueError as exc:
                                LOGGER.warning(
                                    format_log_message(
                                        f"Rejected malformed request body for {parsed.path}: {exc}"
                                    ),
                                )
                                self._drain_pending_request_body()
                                self.close_connection = True
                                self._write_response(bad_request_response(str(exc)))
                                return
                            try:
                                payload = read_json(body)
                            except ValueError as exc:
                                LOGGER.warning(
                                    format_log_message(
                                        f"Rejected malformed JSON request for {parsed.path}: {exc}"
                                    ),
                                )
                                self.close_connection = True
                                self._write_response(bad_request_response(str(exc)))
                                return
                            response = self._dispatch_post(parsed.path, payload)
                    except Exception:  # pragma: no cover - defensive server guard
                        LOGGER.exception(
                            format_log_message(
                                f"Unhandled exception while processing {self.command} {parsed.path}"
                            ),
                        )
                        response = _unexpected_internal_error_response(session_id)
                self._write_response(response)

            def log_message(self, format: str, *args: object) -> None:
                """Suppress default stderr logging for handled HTTP requests."""
                del format, args
                return

            def _dispatch_get(self, path: str) -> JsonResponse | _BinaryResponse:
                """Route one GET request to bootstrap, index, or static assets."""
                route_handler = get_route_handlers.get(path)
                if route_handler is not None:
                    return route_handler()
                return self._static_response(path)

            def _dispatch_post(self, path: str, payload: JsonDict) -> JsonResponse:
                """Route one POST request to the matching JSON API handler."""
                route_handler = post_route_handlers.get(path)
                if route_handler is not None:
                    return route_handler(payload)
                LOGGER.debug(format_log_message(f"Unknown POST path: {path}"))
                return not_found_response()

            def _reject_untrusted_host(self) -> bool:
                """Write a forbidden response when the Host header is not local."""
                if _is_trusted_host_header(
                    self.headers.get("Host"),
                    allow_remote=allow_remote,
                ):
                    return False
                LOGGER.warning(
                    format_log_message(
                        "Rejected request with untrusted Host header",
                        context={"host": self.headers.get("Host")},
                    ),
                )
                self._prepare_rejected_request_connection()
                self._write_response(forbidden_response("Untrusted Host header."))
                return True

            def _reject_untrusted_origin(self) -> bool:
                """Write a forbidden response when the Origin header is not local."""
                if _is_trusted_origin_header(
                    self.headers.get("Origin"),
                    allow_remote=allow_remote,
                ):
                    return False
                LOGGER.warning(
                    format_log_message(
                        "Rejected request with untrusted Origin header",
                        context={"origin": self.headers.get("Origin")},
                    ),
                )
                self._prepare_rejected_request_connection()
                self._write_response(forbidden_response("Untrusted Origin header."))
                return True

            def _reject_invalid_api_token(self, path: str) -> bool:
                """Write a forbidden response when an API request lacks the token."""
                if not path.startswith("/api/"):
                    return False
                header_value = self.headers.get(_API_TOKEN_HEADER)
                if header_value is not None and hmac.compare_digest(
                    header_value,
                    api_token,
                ):
                    return False
                LOGGER.warning(
                    format_log_message(
                        "Rejected API request with invalid session token"
                    ),
                )
                self._prepare_rejected_request_connection()
                self._write_response(
                    forbidden_response("Invalid editor session token.")
                )
                return True

            def _reject_unsupported_content_type(self) -> bool:
                """Write an unsupported-media response for non-JSON API writes."""
                if _is_json_content_type(self.headers.get("Content-Type")):
                    return False
                LOGGER.warning(
                    format_log_message(
                        "Rejected API request with unsupported Content-Type",
                        context={"content_type": self.headers.get("Content-Type")},
                    ),
                )
                self._prepare_rejected_request_connection()
                self._write_response(
                    unsupported_media_type_response(
                        "Expected Content-Type 'application/json'."
                    )
                )
                return True

            def _prepare_rejected_request_connection(self) -> None:
                """Drain rejected POST bodies before closing the connection."""
                if self.command == "POST":
                    self._drain_pending_request_body()
                self.close_connection = True

            def _static_response(
                self, request_path: str
            ) -> JsonResponse | _BinaryResponse:
                """Return one static asset response when the path resolves safely."""
                relative_path = self._resolve_static_asset_relative_path(request_path)
                if relative_path is None:
                    if _should_log_missing_static_asset(request_path):
                        LOGGER.debug(
                            format_log_message(
                                f"Static asset not found for path {request_path}"
                            ),
                        )
                    return not_found_response()
                return _BinaryResponse(
                    status=HTTPStatus.OK,
                    body=static_asset_cache.body_by_relative_path[relative_path],
                    content_type=static_asset_cache.content_type_by_relative_path[
                        relative_path
                    ],
                )

            def _resolve_static_asset_relative_path(
                self, request_path: str
            ) -> str | None:
                """Resolve one request path to a cached static asset key."""
                static_root = static_dir.resolve()
                normalized_request_path = _normalize_static_asset_request_path(
                    request_path
                )
                if normalized_request_path is None:
                    return None
                candidate = (static_root / normalized_request_path).resolve()
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
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Content-Security-Policy", content_security_policy)
                self.send_header("Permissions-Policy", _PERMISSIONS_POLICY_HEADER)
                self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                if self.close_connection:
                    self.send_header("Connection", "close")
                self._write_no_cache_headers()
                self.end_headers()
                body_view = memoryview(body)
                for offset in range(
                    0, len(body_view), _RESPONSE_WRITE_CHUNK_SIZE_BYTES
                ):
                    next_offset = offset + _RESPONSE_WRITE_CHUNK_SIZE_BYTES
                    self.wfile.write(body_view[offset:next_offset])
                self.wfile.flush()

            def _write_no_cache_headers(self) -> None:
                """Emit headers that disable browser and intermediary caching."""
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")

        return RequestHandler

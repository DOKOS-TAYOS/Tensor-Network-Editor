from __future__ import annotations

import logging
import os
from http import HTTPStatus
from pathlib import Path
from typing import Protocol, cast

import pytest

from tensor_network_editor.app import server as app_server
from tensor_network_editor.app._protocol import JsonResponse
from tensor_network_editor.app.server import EditorServer
from tensor_network_editor.app.session import EditorSession
from tests.factories import build_sample_spec


class _RecordingHandler:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bytes, str]] = []

    def _write_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.calls.append((status, body, content_type))


class _HandlerClass(Protocol):
    def _write_response(
        self,
        handler: _RecordingHandler,
        response: app_server._BinaryResponse,
    ) -> None: ...


class _StaticHandler(Protocol):
    def _static_response(
        self, request_path: str
    ) -> app_server.JsonResponse | app_server._BinaryResponse: ...


class _GetHandler(Protocol):
    def _dispatch_get(
        self, path: str
    ) -> app_server.JsonResponse | app_server._BinaryResponse: ...


class _ChunkedBodyReader:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    def read(self, size: int | None = -1, /) -> bytes:
        del size
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def test_binary_response_writer_uses_explicit_response_object(
    editor_server: EditorServer,
) -> None:
    handler_class = cast(_HandlerClass, editor_server._build_handler())
    recorder = _RecordingHandler()
    response = app_server._BinaryResponse(
        status=HTTPStatus.OK,
        body=b"asset-body",
        content_type="text/plain; charset=utf-8",
    )

    handler_class._write_response(recorder, response)

    assert recorder.calls == [
        (HTTPStatus.OK, b"asset-body", "text/plain; charset=utf-8")
    ]
    assert response.body == b"asset-body"


def test_parse_content_length_accepts_missing_and_positive_values() -> None:
    assert app_server._parse_content_length(None) == 0
    assert app_server._parse_content_length("0") == 0
    assert app_server._parse_content_length("42") == 42


def test_parse_content_length_rejects_malformed_values() -> None:
    invalid_values = {
        "abc": "Invalid Content-Length header.",
        "-1": "Invalid Content-Length header: must be >= 0.",
    }

    for raw_value, expected_message in invalid_values.items():
        try:
            app_server._parse_content_length(raw_value)
        except ValueError as exc:
            assert str(exc) == expected_message
        else:
            raise AssertionError(f"Content-Length {raw_value!r} was accepted.")


def test_read_request_body_bytes_reads_exact_length_from_chunked_stream() -> None:
    body = app_server._read_request_body_bytes(
        _ChunkedBodyReader([b"ab", b"cd", b"ef"]),
        6,
    )

    assert body == b"abcdef"


def test_read_request_body_bytes_rejects_incomplete_stream() -> None:
    with pytest.raises(
        ValueError, match="Request body ended before all bytes arrived."
    ):
        app_server._read_request_body_bytes(
            _ChunkedBodyReader([b"ab", b"c"]),
            4,
        )


def test_editor_servers_reuse_static_asset_cache_between_instances() -> None:
    first_server = EditorServer(EditorSession(initial_spec=build_sample_spec()))
    second_server = EditorServer(EditorSession(initial_spec=build_sample_spec()))

    try:
        assert first_server._static_asset_cache is second_server._static_asset_cache
        assert (
            first_server._static_asset_cache.asset_version
            == second_server._static_asset_cache.asset_version
        )
        assert (
            first_server._static_asset_cache.index_body
            == second_server._static_asset_cache.index_body
        )
    finally:
        first_server._server.server_close()
        second_server._server.server_close()


def test_editor_index_response_embeds_session_runtime_config() -> None:
    first_server = EditorServer(EditorSession(initial_spec=build_sample_spec()))
    second_server = EditorServer(EditorSession(initial_spec=build_sample_spec()))

    try:
        handler_class = cast(type[_GetHandler], first_server._build_handler())
        first_handler = cast(_GetHandler, handler_class.__new__(handler_class))
        second_handler_class = cast(type[_GetHandler], second_server._build_handler())
        second_handler = cast(
            _GetHandler, second_handler_class.__new__(second_handler_class)
        )

        first_response = cast(
            app_server._BinaryResponse, first_handler._dispatch_get("/")
        )
        second_response = cast(
            app_server._BinaryResponse, second_handler._dispatch_get("/")
        )
        first_body = first_response.body.decode("utf-8")
        second_body = second_response.body.decode("utf-8")

        assert 'id="tne-runtime-config"' in first_body
        assert first_server.session_id in first_body
        assert second_server.session_id in second_body
        assert first_server.session_id != second_server.session_id
        assert first_body != second_body
        assert '"frontend_logging"' in first_body
        assert '"enabled": false' in first_body
        assert '"persist": false' in first_body
        assert '"/api/client-log"' in first_body
        assert "__ASSET_VERSION__" not in first_body
    finally:
        first_server._server.server_close()
        second_server._server.server_close()


def test_static_asset_cache_refreshes_when_static_files_change(
    tmp_path: Path,
) -> None:
    static_dir = tmp_path / "static"
    asset_path = static_dir / "js" / "app.js"
    asset_path.parent.mkdir(parents=True)
    (static_dir / "index.html").write_text(
        "<script src='js/app.js?v=__ASSET_VERSION__'></script>",
        encoding="utf-8",
    )
    asset_path.write_text("console.log('first');", encoding="utf-8")

    first_cache = app_server._get_static_asset_cache(static_dir)

    asset_path.write_text("console.log('second');", encoding="utf-8")
    future_timestamp_ns = (
        max(path.stat().st_mtime_ns for path in static_dir.rglob("*") if path.is_file())
        + 1_000_000_000
    )
    os.utime(asset_path, ns=(future_timestamp_ns, future_timestamp_ns))

    refreshed_cache = app_server._get_static_asset_cache(static_dir)

    assert (
        refreshed_cache.body_by_relative_path["js/app.js"] == b"console.log('second');"
    )
    assert refreshed_cache.asset_version != first_cache.asset_version


def test_static_asset_cache_reuses_one_scan_per_build_or_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static_dir = tmp_path / "static"
    resolved_static_dir = static_dir.resolve()
    asset_path = static_dir / "js" / "app.js"
    asset_path.parent.mkdir(parents=True)
    (static_dir / "index.html").write_text(
        "<script src='js/app.js?v=__ASSET_VERSION__'></script>",
        encoding="utf-8",
    )
    asset_path.write_text("console.log('first');", encoding="utf-8")

    scan_calls: list[Path] = []
    original_scan = app_server._scan_static_asset_files

    def recording_scan(path: Path) -> list[tuple[Path, str, int, int]]:
        scan_calls.append(path.resolve())
        return original_scan(path)

    monkeypatch.setattr(app_server, "_scan_static_asset_files", recording_scan)
    app_server._STATIC_ASSET_CACHE_BY_ROOT.pop(resolved_static_dir, None)

    first_cache = app_server._get_static_asset_cache(static_dir)

    assert first_cache.body_by_relative_path["js/app.js"] == b"console.log('first');"
    assert scan_calls == [resolved_static_dir]

    asset_path.write_text("console.log('second');", encoding="utf-8")
    future_timestamp_ns = (
        max(path.stat().st_mtime_ns for path in static_dir.rglob("*") if path.is_file())
        + 1_000_000_000
    )
    os.utime(asset_path, ns=(future_timestamp_ns, future_timestamp_ns))

    refreshed_cache = app_server._get_static_asset_cache(static_dir)

    assert (
        refreshed_cache.body_by_relative_path["js/app.js"] == b"console.log('second');"
    )
    assert scan_calls == [resolved_static_dir, resolved_static_dir]


def test_static_asset_cache_logs_build_and_reuse(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    static_dir = tmp_path / "static"
    asset_path = static_dir / "js" / "app.js"
    resolved_static_dir = static_dir.resolve()
    asset_path.parent.mkdir(parents=True)
    (static_dir / "index.html").write_text(
        "<script src='js/app.js?v=__ASSET_VERSION__'></script>",
        encoding="utf-8",
    )
    asset_path.write_text("console.log('first');", encoding="utf-8")
    app_server._STATIC_ASSET_CACHE_BY_ROOT.pop(resolved_static_dir, None)

    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
        first_cache = app_server._get_static_asset_cache(static_dir)
        second_cache = app_server._get_static_asset_cache(static_dir)

    assert first_cache is second_cache
    assert "Static asset cache build started" in caplog.text
    assert "Static asset cache build finished" in caplog.text
    assert "Static asset cache reused" in caplog.text
    assert f"path={resolved_static_dir}" in caplog.text


def test_static_asset_cache_logs_refresh_with_version_context(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    static_dir = tmp_path / "static"
    asset_path = static_dir / "js" / "app.js"
    resolved_static_dir = static_dir.resolve()
    asset_path.parent.mkdir(parents=True)
    (static_dir / "index.html").write_text(
        "<script src='js/app.js?v=__ASSET_VERSION__'></script>",
        encoding="utf-8",
    )
    asset_path.write_text("console.log('first');", encoding="utf-8")
    app_server._STATIC_ASSET_CACHE_BY_ROOT.pop(resolved_static_dir, None)
    first_cache = app_server._get_static_asset_cache(static_dir)

    asset_path.write_text("console.log('second');", encoding="utf-8")
    future_timestamp_ns = (
        max(path.stat().st_mtime_ns for path in static_dir.rglob("*") if path.is_file())
        + 1_000_000_000
    )
    os.utime(asset_path, ns=(future_timestamp_ns, future_timestamp_ns))

    with caplog.at_level(logging.DEBUG, logger="tensor_network_editor"):
        refreshed_cache = app_server._get_static_asset_cache(static_dir)

    assert refreshed_cache.asset_version != first_cache.asset_version
    assert "Static asset cache refresh started" in caplog.text
    assert "Static asset cache refresh finished" in caplog.text
    assert f"before={first_cache.asset_version}" in caplog.text
    assert f"after={refreshed_cache.asset_version}" in caplog.text


def test_favicon_request_resolves_to_static_asset_without_debug_log(
    editor_server: EditorServer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler_class = cast(type[_StaticHandler], editor_server._build_handler())
    handler = cast(_StaticHandler, handler_class.__new__(handler_class))

    with caplog.at_level(logging.DEBUG, logger=app_server.LOGGER.name):
        response = handler._static_response("/favicon.ico")

    favicon_response = cast(app_server._BinaryResponse, response)
    assert favicon_response.status == HTTPStatus.OK
    assert favicon_response.body
    assert favicon_response.content_type.startswith("image/")
    assert "Static asset not found for path /favicon.ico" not in caplog.text


def test_missing_non_favicon_request_keeps_debug_log(
    editor_server: EditorServer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler_class = cast(type[_StaticHandler], editor_server._build_handler())
    handler = cast(_StaticHandler, handler_class.__new__(handler_class))

    with caplog.at_level(logging.DEBUG, logger=app_server.LOGGER.name):
        response = handler._static_response("/missing-asset.js")

    missing_response = cast(JsonResponse, response)
    assert missing_response[0] == HTTPStatus.NOT_FOUND
    assert "Static asset not found for path /missing-asset.js" in caplog.text

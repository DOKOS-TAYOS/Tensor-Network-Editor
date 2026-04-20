from __future__ import annotations

from http import HTTPStatus
from typing import Protocol, cast

from tensor_network_editor.app import server as app_server
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

from __future__ import annotations

from http import HTTPStatus
from typing import Protocol, cast

from tensor_network_editor.app import server as app_server
from tensor_network_editor.app.server import EditorServer


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

from __future__ import annotations

from unittest.mock import patch

from tests import app_support


class _FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")
        self.status = 200
        self.headers = {"Cache-Control": "no-store"}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        return None

    def read(self) -> bytes:
        return self._body


def test_request_text_uses_shared_asset_timeout() -> None:
    recorded_timeout: list[float] = []

    def fake_urlopen(url: str, timeout: float) -> _FakeResponse:
        recorded_timeout.append(timeout)
        assert url == "http://example.test/"
        return _FakeResponse("body")

    with patch("tests.app_support.urlopen", side_effect=fake_urlopen):
        body = app_support.request_text("http://example.test/")

    assert body == "body"
    assert recorded_timeout == [app_support._ASSET_REQUEST_TIMEOUT_SECONDS]


def test_request_with_headers_uses_shared_asset_timeout() -> None:
    recorded_timeout: list[float] = []

    def fake_urlopen(url: str, timeout: float) -> _FakeResponse:
        recorded_timeout.append(timeout)
        assert url == "http://example.test/app.css"
        return _FakeResponse("css")

    with patch("tests.app_support.urlopen", side_effect=fake_urlopen):
        body, headers = app_support.request_with_headers("http://example.test/app.css")

    assert body == "css"
    assert headers == {"Cache-Control": "no-store"}
    assert recorded_timeout == [app_support._ASSET_REQUEST_TIMEOUT_SECONDS]


def test_request_headers_uses_shared_asset_timeout_without_reading_body() -> None:
    recorded_timeout: list[float] = []
    response = _FakeResponse("body")

    def fake_urlopen(url: str, timeout: float) -> _FakeResponse:
        recorded_timeout.append(timeout)
        assert url == "http://example.test/vendor.js"
        return response

    with patch("tests.app_support.urlopen", side_effect=fake_urlopen):
        headers = app_support.request_headers("http://example.test/vendor.js")

    assert headers == {"Cache-Control": "no-store"}
    assert recorded_timeout == [app_support._ASSET_REQUEST_TIMEOUT_SECONDS]


def test_runtime_config_parser_matches_mixed_case_script_tags() -> None:
    html = (
        '<SCRIPT ID="tne-runtime-config" type="application/json">'
        '{"api_token": "token-demo"}'
        "</ScRiPt>"
    )

    config_text = app_support._extract_runtime_config_json(html)

    assert config_text == '{"api_token": "token-demo"}'


def test_runtime_config_parser_accepts_browser_tolerated_script_end_tags() -> None:
    html = (
        '<script id="tne-runtime-config" type="application/json">'
        '{"api_token": "token-demo"}'
        "</script >"
    )

    assert app_support._extract_runtime_config_json(html) == (
        '{"api_token": "token-demo"}'
    )


def test_read_asset_response_retries_transient_os_errors() -> None:
    attempts = 0

    def fake_urlopen(url: str, timeout: float) -> _FakeResponse:
        nonlocal attempts
        attempts += 1
        assert url == "http://example.test/retry.js"
        assert timeout == app_support._ASSET_REQUEST_TIMEOUT_SECONDS
        if attempts < 3:
            raise TimeoutError("temporary timeout")
        return _FakeResponse("ok")

    with patch("tests.app_support.urlopen", side_effect=fake_urlopen):
        body, headers = app_support._read_asset_response("http://example.test/retry.js")

    assert body == b"ok"
    assert headers == {"Cache-Control": "no-store"}
    assert attempts == 3


def test_request_bytes_uses_shared_asset_fetcher() -> None:
    with patch(
        "tests.app_support._read_asset_response",
        return_value=(b"icon", {"Content-Type": "image/x-icon"}),
    ) as read_asset_response_mock:
        body = app_support.request_bytes("http://example.test/favicon.ico")

    assert body == b"icon"
    read_asset_response_mock.assert_called_once_with("http://example.test/favicon.ico")

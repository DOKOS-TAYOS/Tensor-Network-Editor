from __future__ import annotations

import json
import re
import time
from typing import Any, cast
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

_ASSET_REQUEST_TIMEOUT_SECONDS = 15.0
_ASSET_REQUEST_RETRY_COUNT = 3
_ASSET_REQUEST_RETRY_DELAY_SECONDS = 0.1
_RUNTIME_CONFIG_RE = re.compile(
    r'<script\b(?=[^>]*\bid="tne-runtime-config")[^>]*>(.*?)</script>',
    re.DOTALL,
)
_SESSION_TOKEN_BY_ORIGIN: dict[str, str | None] = {}


def request_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    status, response = request_json_with_status(
        url,
        method=method,
        payload=payload,
        timeout=timeout,
    )
    if status >= 400:
        raise AssertionError(f"Expected success response for {url}, received {status}.")
    return response


def request_json_with_status(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    raw_body: bytes | None = None,
    session_token: str | None = None,
    include_session_token: bool = True,
    timeout: float = 5.0,
) -> tuple[int, dict[str, Any]]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None and raw_body is not None:
        raise ValueError("payload and raw_body cannot be combined.")
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif raw_body is not None:
        data = raw_body
        headers["Content-Type"] = "application/json"
    if include_session_token:
        resolved_session_token = (
            session_token if session_token is not None else _session_token_for_url(url)
        )
        if resolved_session_token:
            headers["X-TNE-Session-Token"] = resolved_session_token
    request = Request(url=url, method=method, data=data, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _session_token_for_url(url: str) -> str | None:
    """Read the embedded editor API token for a local test server URL."""
    origin = _origin_for_url(url)
    if origin is None:
        return None
    if origin in _SESSION_TOKEN_BY_ORIGIN:
        return _SESSION_TOKEN_BY_ORIGIN[origin]
    try:
        with urlopen(f"{origin}/", timeout=_ASSET_REQUEST_TIMEOUT_SECONDS) as response:
            html = response.read().decode("utf-8")
    except OSError:
        _SESSION_TOKEN_BY_ORIGIN[origin] = None
        return None
    match = _RUNTIME_CONFIG_RE.search(html)
    if match is None:
        _SESSION_TOKEN_BY_ORIGIN[origin] = None
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        _SESSION_TOKEN_BY_ORIGIN[origin] = None
        return None
    token = payload.get("api_token") if isinstance(payload, dict) else None
    _SESSION_TOKEN_BY_ORIGIN[origin] = token if isinstance(token, str) else None
    return _SESSION_TOKEN_BY_ORIGIN[origin]


def _origin_for_url(url: str) -> str | None:
    """Return the scheme/authority origin for an absolute URL."""
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _read_asset_response(url: str) -> tuple[bytes, dict[str, str]]:
    """Read one asset request with retries for transient local-server hiccups."""
    last_error: OSError | None = None
    for attempt_index in range(_ASSET_REQUEST_RETRY_COUNT):
        try:
            with urlopen(url, timeout=_ASSET_REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read()
                headers = {key: value for key, value in response.headers.items()}
                return body, headers
        except OSError as exc:
            last_error = exc
            if attempt_index + 1 >= _ASSET_REQUEST_RETRY_COUNT:
                raise
            time.sleep(_ASSET_REQUEST_RETRY_DELAY_SECONDS)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Asset request retry loop ended unexpectedly.")


def request_text(url: str) -> str:
    body, _headers = _read_asset_response(url)
    return cast(str, body.decode("utf-8"))


def request_with_headers(url: str) -> tuple[str, dict[str, str]]:
    body, headers = _read_asset_response(url)
    return body.decode("utf-8"), headers


def request_headers(url: str) -> dict[str, str]:
    last_error: OSError | None = None
    for attempt_index in range(_ASSET_REQUEST_RETRY_COUNT):
        try:
            with urlopen(url, timeout=_ASSET_REQUEST_TIMEOUT_SECONDS) as response:
                return {key: value for key, value in response.headers.items()}
        except OSError as exc:
            last_error = exc
            if attempt_index + 1 >= _ASSET_REQUEST_RETRY_COUNT:
                raise
            time.sleep(_ASSET_REQUEST_RETRY_DELAY_SECONDS)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Asset header request retry loop ended unexpectedly.")


def request_bytes(url: str) -> bytes:
    body, _headers = _read_asset_response(url)
    return body

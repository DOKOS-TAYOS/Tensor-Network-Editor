from __future__ import annotations

import json
import time
from typing import Any, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

_ASSET_REQUEST_TIMEOUT_SECONDS = 15.0
_ASSET_REQUEST_RETRY_COUNT = 3
_ASSET_REQUEST_RETRY_DELAY_SECONDS = 0.1


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
    request = Request(url=url, method=method, data=data, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


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

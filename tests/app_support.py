from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request_json(
    url: str, method: str = "GET", payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    status, response = request_json_with_status(url, method=method, payload=payload)
    if status >= 400:
        raise AssertionError(f"Expected success response for {url}, received {status}.")
    return response


def request_json_with_status(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    raw_body: bytes | None = None,
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
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def request_text(url: str) -> str:
    with urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def request_with_headers(url: str) -> tuple[str, dict[str, str]]:
    with urlopen(url, timeout=5) as response:
        body = response.read().decode("utf-8")
        headers = {key: value for key, value in response.headers.items()}
        return body, headers

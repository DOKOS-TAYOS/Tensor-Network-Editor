"""Subprocess entry point for live Python object imports."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import cast

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from tensor_network_editor.internal.io._python_live_import import (  # noqa: E402
    build_live_import_result_from_namespace,
)


def main() -> int:
    """Execute one live-import request and emit a JSON response."""
    try:
        request_payload = _read_request_payload()
        response_payload = _run_request(request_payload)
    except BaseException as exc:  # pragma: no cover - defensive subprocess guard
        _write_response(
            {
                "ok": False,
                "message": str(exc)
                or f"{type(exc).__name__} raised during live import.",
            }
        )
        return 0
    _write_response(response_payload)
    return 0


def _read_request_payload() -> dict[str, object]:
    """Read and validate the JSON request from standard input."""
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        raise ValueError("Live import subprocess received invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Live import subprocess expected a JSON object request.")
    return cast(dict[str, object], payload)


def _run_request(request_payload: dict[str, object]) -> dict[str, object]:
    """Execute one live-import request and serialize the resulting spec."""
    code = request_payload.get("code")
    filename = request_payload.get("filename")
    source_profile = request_payload.get("source_profile", "auto")
    python_object_name = request_payload.get("python_object_name")
    if not isinstance(code, str):
        raise ValueError("Live import subprocess requires a string 'code' field.")
    if not isinstance(filename, str):
        raise ValueError("Live import subprocess requires a string 'filename' field.")
    if not isinstance(source_profile, str):
        raise ValueError(
            "Live import subprocess requires a string 'source_profile' field."
        )
    if python_object_name is not None and not isinstance(python_object_name, str):
        raise ValueError(
            "Live import subprocess requires 'python_object_name' to be a string when provided."
        )

    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

    namespace: dict[str, object] = {
        "__builtins__": __builtins__,
        "__file__": filename,
        "__name__": "__main__",
        "__package__": None,
    }
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    compiled_code = compile(code, filename, "exec")
    with (
        contextlib.redirect_stdout(stdout_buffer),
        contextlib.redirect_stderr(stderr_buffer),
    ):
        exec(compiled_code, namespace, namespace)
    result = build_live_import_result_from_namespace(
        namespace,
        source_profile=source_profile,
        python_object_name=python_object_name,
    )
    return {
        "ok": True,
        "network": result.spec.to_dict(),
        "warnings": result.warnings,
    }


def _write_response(response_payload: dict[str, object]) -> None:
    """Write one JSON response to standard output."""
    sys.stdout.write(json.dumps(response_payload))
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())

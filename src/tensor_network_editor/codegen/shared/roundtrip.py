"""Helpers for embedding editor round-trip metadata in generated code."""

from __future__ import annotations

import base64
import json

from ...models import CodegenResult, NetworkSpec

_ROUNDTRIP_MARKER_PREFIX = "# TNE_SPEC_B64: "
_ROUNDTRIP_MARKER_WIDTH = 96
_ROUNDTRIP_SCHEMA_VERSION = 2


def render_roundtrip_spec_marker_lines(spec: NetworkSpec) -> list[str]:
    """Return comment lines carrying a serialized spec round-trip payload."""
    payload = {
        "schema_version": _ROUNDTRIP_SCHEMA_VERSION,
        "network": spec.to_dict(),
    }
    payload_text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    encoded_payload = base64.b64encode(payload_text.encode("utf-8")).decode("ascii")
    return [
        f"{_ROUNDTRIP_MARKER_PREFIX}{encoded_payload[offset : offset + _ROUNDTRIP_MARKER_WIDTH]}"
        for offset in range(0, len(encoded_payload), _ROUNDTRIP_MARKER_WIDTH)
    ]


def with_roundtrip_spec_marker(
    result: CodegenResult,
    *,
    spec: NetworkSpec,
) -> CodegenResult:
    """Return ``result`` with a leading serialized-spec comment marker."""
    marker = "\n".join(render_roundtrip_spec_marker_lines(spec))
    return CodegenResult(
        engine=result.engine,
        code=f"{marker}\n{result.code}",
        warnings=list(result.warnings),
        artifacts=dict(result.artifacts),
    )


__all__ = ["render_roundtrip_spec_marker_lines", "with_roundtrip_spec_marker"]

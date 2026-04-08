"""Serialize, deserialize, and persist tensor-network specifications."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ._io import read_utf8_text, write_utf8_text
from ._payloads import coerce_int
from ._python_roundtrip import parse_generated_python_network
from .errors import SerializationError
from .models import NetworkSpec
from .types import JSONValue, StrPath
from .validation import ensure_valid_spec

SCHEMA_VERSION = 4
LOGGER = logging.getLogger(__name__)


def serialize_spec(spec: NetworkSpec) -> dict[str, JSONValue]:
    """Return the schema-wrapped JSON payload for ``spec``."""
    ensure_valid_spec(spec)
    return {
        "schema_version": SCHEMA_VERSION,
        "network": spec.to_dict(),
    }


def deserialize_spec(
    payload: dict[str, object], *, validate: bool = True
) -> NetworkSpec:
    """Build a ``NetworkSpec`` from a schema-wrapped JSON payload."""
    if "schema_version" not in payload:
        raise SerializationError(
            "Serialized payload must contain a valid schema version."
        )
    schema_version_raw = payload.get("schema_version")
    if isinstance(schema_version_raw, bool) or not isinstance(
        schema_version_raw, (int, float, str)
    ):
        raise SerializationError(
            "Serialized payload must contain a valid schema version."
        )
    try:
        schema_version = coerce_int(
            schema_version_raw,
            field_name="schema_version",
        )
    except TypeError as exc:
        raise SerializationError(
            "Serialized payload must contain a valid schema version."
        ) from exc
    if schema_version != SCHEMA_VERSION:
        raise SerializationError(
            f"Unsupported schema version {schema_version}. Expected {SCHEMA_VERSION}."
        )

    network_payload = payload.get("network")
    if not isinstance(network_payload, dict):
        raise SerializationError("Serialized payload must contain a 'network' object.")

    try:
        spec = NetworkSpec.from_dict(network_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise SerializationError(
            "Serialized payload contains a malformed network object."
        ) from exc
    return ensure_valid_spec(spec) if validate else spec


def save_spec(spec: NetworkSpec, path: StrPath) -> None:
    """Write ``spec`` to ``path`` as formatted UTF-8 JSON."""
    payload = serialize_spec(spec)
    try:
        body = json.dumps(payload, indent=2)
    except TypeError as exc:
        raise SerializationError(
            "Could not serialize the network specification to JSON."
        ) from exc
    write_utf8_text(path, body, description="network specification JSON")


def load_spec(path: StrPath) -> NetworkSpec:
    """Load a saved JSON spec or a supported generated Python file from disk."""
    if Path(path).suffix.lower() == ".py":
        body = read_utf8_text(path, description="generated Python code")
        LOGGER.debug("Loaded generated Python code payload from %s", path)
        return load_spec_from_python_code(body)

    body = read_utf8_text(path, description="network specification JSON")
    LOGGER.debug("Loaded serialized network payload from %s", path)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SerializationError("Could not parse network specification JSON.") from exc
    if not isinstance(payload, dict):
        raise SerializationError("Serialized network must be a JSON object.")
    return deserialize_spec(payload)


def deserialize_spec_from_python_code(
    code: str, *, validate: bool = True
) -> NetworkSpec:
    """Parse supported generated Python source into a ``NetworkSpec``."""
    if "# Tensor Network Editor linear periodic mode" in code:
        raise SerializationError(
            "Loading generated Python from linear periodic mode is not supported."
        )
    spec = parse_generated_python_network(code)
    return ensure_valid_spec(spec) if validate else spec


def load_spec_from_python_code(code: str) -> NetworkSpec:
    """Parse and validate supported generated Python source."""
    return deserialize_spec_from_python_code(code, validate=True)

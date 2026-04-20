"""Small coercion helpers used while reading serialized payloads."""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import cast
from uuid import uuid4

from ...types import JSONValue, MetadataDict


def new_identifier(prefix: str) -> str:
    """Return a short random identifier with the given prefix."""
    return f"{prefix}_{uuid4().hex[:8]}"


def coerce_float(value: object, *, field_name: str) -> float:
    """Coerce ``value`` to ``float`` or raise a typed payload error."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{field_name} must be a number.")
    try:
        numeric_value = float(value)
    except (OverflowError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a number.") from exc
    if not math.isfinite(numeric_value):
        raise TypeError(f"{field_name} must be a number.")
    return numeric_value


def coerce_int(value: object, *, field_name: str) -> int:
    """Coerce ``value`` to ``int`` or raise a typed payload error."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{field_name} must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise TypeError(f"{field_name} must be an integer.")

    stripped_value = value.strip()
    if not stripped_value:
        raise TypeError(f"{field_name} must be an integer.")
    try:
        numeric_value = Decimal(stripped_value)
    except InvalidOperation as exc:
        raise TypeError(f"{field_name} must be an integer.") from exc
    if (
        not numeric_value.is_finite()
        or numeric_value != numeric_value.to_integral_value()
    ):
        raise TypeError(f"{field_name} must be an integer.")
    return int(numeric_value)


def coerce_string(value: object, *, field_name: str) -> str:
    """Require that ``value`` is a string."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    return value


def require_dict(value: object, *, field_name: str) -> dict[str, object]:
    """Require that ``value`` is a dictionary."""
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object.")
    return value


def require_list(value: object, *, field_name: str) -> list[object]:
    """Require that ``value`` is a list."""
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list.")
    return value


def coerce_metadata(value: object, *, field_name: str) -> MetadataDict:
    """Validate and cast metadata to the package metadata type."""
    payload = require_dict(value, field_name=field_name)
    normalized_payload: dict[str, JSONValue] = {}
    for metadata_key, metadata_value in payload.items():
        if not isinstance(metadata_key, str):
            raise TypeError(f"{field_name} must contain string keys.")
        normalized_payload[metadata_key] = _coerce_json_value(
            metadata_value,
            field_name=field_name,
        )
    return normalized_payload


def _coerce_json_value(value: object, *, field_name: str) -> JSONValue:
    """Coerce one metadata value to the declared JSON-compatible union."""
    if value is None or isinstance(value, (bool, int, str)):
        return cast(JSONValue, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{field_name} must contain JSON-compatible values.")
        return value
    if isinstance(value, list):
        return [_coerce_json_value(item, field_name=field_name) for item in value]
    if isinstance(value, dict):
        normalized_object: dict[str, JSONValue] = {}
        for nested_key, nested_value in value.items():
            if not isinstance(nested_key, str):
                raise TypeError(f"{field_name} must contain string keys.")
            normalized_object[nested_key] = _coerce_json_value(
                nested_value,
                field_name=field_name,
            )
        return normalized_object
    raise TypeError(f"{field_name} must contain JSON-compatible values.")

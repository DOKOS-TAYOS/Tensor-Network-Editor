"""Small coercion helpers used while reading serialized payloads."""

from __future__ import annotations

from typing import cast
from uuid import uuid4

from .types import MetadataDict


def new_identifier(prefix: str) -> str:
    """Return a short random identifier with the given prefix."""
    return f"{prefix}_{uuid4().hex[:8]}"


def coerce_float(value: object, *, field_name: str) -> float:
    """Coerce ``value`` to ``float`` or raise a typed payload error."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{field_name} must be a number.")
    return float(value)


def coerce_int(value: object, *, field_name: str) -> int:
    """Coerce ``value`` to ``int`` or raise a typed payload error."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{field_name} must be an integer.")
    return int(value)


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
    return cast(MetadataDict, dict(require_dict(value, field_name=field_name)))

"""Validate abstract tensor-network specifications."""

from __future__ import annotations

from .internal.validation._validation_spec import (
    ensure_valid_spec,
    validate_spec,
    validate_spec_with_analysis,
)

_validate_spec_with_analysis = validate_spec_with_analysis

__all__ = [
    "ensure_valid_spec",
    "validate_spec",
]

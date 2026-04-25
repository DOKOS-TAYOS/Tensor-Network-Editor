"""Validate abstract tensor-network specifications."""

from __future__ import annotations

from .internal.validation._validation_spec import (
    ensure_valid_spec,
    validate_spec,
)
from .internal.validation._validation_spec import (
    validate_spec_with_analysis as validate_spec_with_analysis,
)

__all__ = [
    "ensure_valid_spec",
    "validate_spec",
]

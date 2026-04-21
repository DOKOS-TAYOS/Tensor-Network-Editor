"""Analysis request helpers for editor routes."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ..errors import SpecValidationError
from ..internal.analysis._contraction_analysis import _analyze_validated_contraction
from ..internal.analysis._contraction_analysis_types import ContractionAnalysisResult
from ..models import NetworkSpec, ValidationIssue


def analyze_serialized_contraction(
    serialized_spec: Mapping[str, object],
    *,
    deserialize_spec_fn: Callable[[Mapping[str, object]], NetworkSpec],
    validate_spec_fn: Callable[[NetworkSpec], list[ValidationIssue]],
) -> ContractionAnalysisResult:
    """Deserialize, validate, and analyze contraction data for one payload."""
    spec = deserialize_spec_fn(serialized_spec)
    issues = validate_spec_fn(spec)
    if issues:
        raise SpecValidationError(issues)
    return _analyze_validated_contraction(spec)

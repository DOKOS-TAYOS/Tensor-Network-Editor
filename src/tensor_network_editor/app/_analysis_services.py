"""Analysis request helpers for editor routes."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from ..errors import SpecValidationError
from ..internal._logging import (
    log_branch,
    log_operation,
    summarize_contraction_analysis,
    summarize_spec_counts,
)
from ..internal.analysis._contraction_analysis import _analyze_validated_contraction
from ..internal.analysis._contraction_analysis_types import ContractionAnalysisResult
from ..models import NetworkSpec, ValidationIssue

LOGGER = logging.getLogger(__name__)


def analyze_serialized_contraction(
    serialized_spec: Mapping[str, object],
    *,
    deserialize_spec_fn: Callable[[Mapping[str, object]], NetworkSpec],
    validate_spec_fn: Callable[[NetworkSpec], list[ValidationIssue]],
) -> ContractionAnalysisResult:
    """Deserialize, validate, and analyze contraction data for one payload."""
    with log_operation(
        LOGGER,
        "Serialized contraction analysis",
        emit_start=False,
    ) as success_context:
        spec = deserialize_spec_fn(serialized_spec)
        issues = validate_spec_fn(spec)
        if issues:
            log_branch(
                LOGGER,
                "Contraction-analysis payload failed validation",
                level=logging.WARNING,
                context={
                    "analysis_status": "issues",
                    "issue_count": len(issues),
                },
            )
            raise SpecValidationError(issues)
        result = _analyze_validated_contraction(spec)
        success_context.update(summarize_spec_counts(spec))
        success_context.update(
            {
                "memory_dtype": result.memory_dtype,
                **summarize_contraction_analysis(result),
            }
        )
        return result

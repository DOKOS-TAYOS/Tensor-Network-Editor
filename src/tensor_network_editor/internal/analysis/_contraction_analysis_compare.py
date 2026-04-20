"""Comparison helpers for contraction analysis results."""

from __future__ import annotations

from ._contraction_analysis_types import (
    AutomaticContractionPlanAnalysis,
    ContractionComparison,
    ContractionStepAnalysis,
    ManualContractionPlanAnalysis,
)


def _build_contraction_comparisons(
    *,
    manual: ManualContractionPlanAnalysis,
    automatic_full: AutomaticContractionPlanAnalysis,
    automatic_past: AutomaticContractionPlanAnalysis,
    memory_dtype: str,
) -> dict[str, ContractionComparison]:
    """Build comparison payloads between manual and automatic analyses."""
    return {
        "manual_vs_automatic_full": _compare_plan_analyses(
            baseline_label="manual",
            baseline_analysis=manual,
            candidate_label="automatic_full",
            candidate_analysis=automatic_full,
            memory_dtype=memory_dtype,
        ),
        "manual_remaining_vs_automatic_future": ContractionComparison(
            status="unavailable",
            baseline_label="manual_remaining",
            candidate_label="automatic_future",
            memory_dtype=memory_dtype,
            message=(
                "The saved manual plan does not expose a separate remaining suffix to compare yet."
            ),
        ),
        "manual_subtrees_vs_automatic_past": _compare_plan_analyses(
            baseline_label="manual_subtrees",
            baseline_analysis=manual,
            candidate_label="automatic_past",
            candidate_analysis=automatic_past,
            memory_dtype=memory_dtype,
        ),
    }


def _compare_plan_analyses(
    *,
    baseline_label: str,
    baseline_analysis: ManualContractionPlanAnalysis | AutomaticContractionPlanAnalysis,
    candidate_label: str,
    candidate_analysis: ManualContractionPlanAnalysis
    | AutomaticContractionPlanAnalysis,
    memory_dtype: str,
) -> ContractionComparison:
    """Build deltas between two contraction analyses when both are available."""
    if baseline_analysis.status == "unavailable":
        return ContractionComparison(
            status="unavailable",
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            memory_dtype=memory_dtype,
            message=baseline_analysis.message,
        )
    if candidate_analysis.status == "unavailable":
        return ContractionComparison(
            status="unavailable",
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            memory_dtype=memory_dtype,
            message=candidate_analysis.message,
        )

    baseline_peak_size = baseline_analysis.summary.peak_intermediate_size
    candidate_peak_size = candidate_analysis.summary.peak_intermediate_size
    baseline_peak_step = _find_peak_step(baseline_analysis.steps)
    candidate_peak_step = _find_peak_step(candidate_analysis.steps)
    baseline_peak_bytes = baseline_analysis.summary.peak_intermediate_bytes
    candidate_peak_bytes = candidate_analysis.summary.peak_intermediate_bytes
    return ContractionComparison(
        status="complete",
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        memory_dtype=memory_dtype,
        baseline_peak_intermediate_bytes=baseline_peak_bytes,
        candidate_peak_intermediate_bytes=candidate_peak_bytes,
        delta_total_estimated_flops=(
            candidate_analysis.summary.total_estimated_flops
            - baseline_analysis.summary.total_estimated_flops
        ),
        delta_total_estimated_macs=(
            candidate_analysis.summary.total_estimated_macs
            - baseline_analysis.summary.total_estimated_macs
        ),
        delta_peak_intermediate_size=candidate_peak_size - baseline_peak_size,
        delta_peak_intermediate_bytes=candidate_peak_bytes - baseline_peak_bytes,
        baseline_peak_step_id=baseline_peak_step.step_id
        if baseline_peak_step
        else None,
        candidate_peak_step_id=candidate_peak_step.step_id
        if candidate_peak_step
        else None,
        baseline_bottleneck_labels=(
            _build_bottleneck_labels(baseline_peak_step) if baseline_peak_step else ()
        ),
        candidate_bottleneck_labels=(
            _build_bottleneck_labels(candidate_peak_step) if candidate_peak_step else ()
        ),
    )


def _find_peak_step(
    steps: list[ContractionStepAnalysis],
) -> ContractionStepAnalysis | None:
    """Return the step that creates the largest intermediate tensor."""
    if not steps:
        return None
    return max(steps, key=lambda step: step.intermediate_size)


def _build_bottleneck_labels(step: ContractionStepAnalysis) -> tuple[str, ...]:
    """Return the labels that participate in the peak intermediate step."""
    return tuple(dict.fromkeys(step.contracted_labels + step.surviving_labels))

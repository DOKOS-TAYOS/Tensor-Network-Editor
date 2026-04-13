"""Manual-plan helpers for contraction analysis."""

from __future__ import annotations

from dataclasses import dataclass

from ._contraction_analysis_types import (
    ContractionStepAnalysis,
    ManualContractionPlanAnalysis,
    ManualContractionSummary,
)
from ._contraction_plan import simulate_contraction_plan
from .models import ContractionPlanSpec, ContractionStepSpec, NetworkSpec


@dataclass(slots=True)
class ManualOperandState:
    """Manual-plan state carried into automatic preview calculations."""

    active_operand_ids: tuple[str, ...]
    remaining_operands: dict[str, tuple[str, ...]]
    source_tensor_ids_by_operand_id: dict[str, tuple[str, ...]]


def _build_manual_operand_state(
    *,
    spec: NetworkSpec,
    initial_operands: dict[str, tuple[str, ...]],
    initial_axis_names: dict[str, tuple[str, ...]],
    dimension_by_label: dict[str, int],
) -> ManualOperandState:
    """Simulate the saved manual plan and keep its remaining operands."""
    simulation = simulate_contraction_plan(
        initial_operand_ids=tuple(initial_operands),
        initial_operands=initial_operands,
        initial_axis_names=initial_axis_names,
        dimension_by_label=dimension_by_label,
        plan=spec.contraction_plan,
    )

    return ManualOperandState(
        active_operand_ids=simulation.remaining_operand_ids,
        remaining_operands=simulation.remaining_operands,
        source_tensor_ids_by_operand_id=simulation.source_tensor_ids_by_operand_id,
    )


def _analyze_manual_plan(
    *,
    spec: NetworkSpec,
    initial_operands: dict[str, tuple[str, ...]],
    dimension_by_label: dict[str, int],
    bytes_per_element: int,
) -> ManualContractionPlanAnalysis:
    """Analyze the saved manual plan, or derive a trivial summary when absent."""
    plan = spec.contraction_plan
    if plan is None or not plan.steps:
        summary = _build_manual_summary_from_operands(
            remaining_operands=initial_operands,
            status="complete" if len(initial_operands) <= 1 else "incomplete",
            total_estimated_flops=0,
            total_estimated_macs=0,
            peak_intermediate_size=0,
            dimension_by_label=dimension_by_label,
            bytes_per_element=bytes_per_element,
        )
        return ManualContractionPlanAnalysis(
            status=summary.completion_status,
            steps=[],
            summary=summary,
        )

    return _simulate_plan_steps(
        steps=plan.steps,
        initial_operands=initial_operands,
        dimension_by_label=dimension_by_label,
        bytes_per_element=bytes_per_element,
    )


def _simulate_plan_steps(
    *,
    steps: list[ContractionStepSpec],
    initial_operands: dict[str, tuple[str, ...]],
    dimension_by_label: dict[str, int],
    bytes_per_element: int,
) -> ManualContractionPlanAnalysis:
    """Simulate each saved step and accumulate manual-plan metrics."""
    simulation = simulate_contraction_plan(
        initial_operand_ids=tuple(initial_operands),
        initial_operands=initial_operands,
        initial_axis_names={
            operand_id: labels for operand_id, labels in initial_operands.items()
        },
        dimension_by_label=dimension_by_label,
        plan=ContractionPlanSpec(steps=steps),
    )
    step_results = [
        ContractionStepAnalysis(
            step_id=step.step_id,
            left_operand_id=step.left_operand_id,
            right_operand_id=step.right_operand_id,
            result_operand_id=step.step_id,
            contracted_labels=step.contracted_labels,
            surviving_labels=step.surviving_labels,
            result_shape=step.result_shape,
            result_rank=step.result_rank,
            estimated_flops=step.estimated_flops,
            estimated_macs=step.estimated_macs,
            intermediate_size=step.intermediate_size,
        )
        for step in simulation.steps
    ]
    total_estimated_flops = sum(step.estimated_flops for step in step_results)
    total_estimated_macs = sum(step.estimated_macs for step in step_results)
    peak_intermediate_size = max(
        (step.intermediate_size for step in step_results),
        default=0,
    )
    status = "complete" if len(simulation.remaining_operands) <= 1 else "incomplete"
    summary = _build_manual_summary_from_operands(
        remaining_operands=simulation.remaining_operands,
        status=status,
        total_estimated_flops=total_estimated_flops,
        total_estimated_macs=total_estimated_macs,
        peak_intermediate_size=peak_intermediate_size,
        dimension_by_label=dimension_by_label,
        bytes_per_element=bytes_per_element,
        last_result_shape=step_results[-1].result_shape if step_results else None,
    )
    return ManualContractionPlanAnalysis(
        status=status,
        steps=step_results,
        summary=summary,
    )


def _build_manual_summary_from_operands(
    *,
    remaining_operands: dict[str, tuple[str, ...]],
    status: str,
    total_estimated_flops: int,
    total_estimated_macs: int,
    peak_intermediate_size: int,
    dimension_by_label: dict[str, int],
    bytes_per_element: int,
    last_result_shape: tuple[int, ...] | None = None,
) -> ManualContractionSummary:
    """Build the summary payload for the current manual-plan state."""
    final_shape = last_result_shape
    if final_shape is None and len(remaining_operands) == 1:
        labels = next(iter(remaining_operands.values()))
        final_shape = tuple(dimension_by_label[label] for label in labels)
    elif final_shape is None and len(remaining_operands) == 0:
        final_shape = ()
    return ManualContractionSummary(
        total_estimated_flops=total_estimated_flops,
        total_estimated_macs=total_estimated_macs,
        peak_intermediate_size=peak_intermediate_size,
        final_shape=final_shape if status == "complete" or last_result_shape else None,
        completion_status=status,
        remaining_operand_ids=tuple(remaining_operands),
        peak_intermediate_bytes=peak_intermediate_size * bytes_per_element,
    )

"""Manual-plan helpers for contraction analysis."""

from __future__ import annotations

from dataclasses import dataclass

from ...models import NetworkSpec
from ._contraction_analysis_types import (
    ContractionStepAnalysis,
    ManualContractionPlanAnalysis,
    ManualContractionSummary,
)
from ._contraction_plan import (
    PreparedContractionInputs,
    SimulatedContractionPlan,
    simulate_contraction_plan,
)


@dataclass(slots=True)
class ManualOperandState:
    """Manual-plan state carried into automatic preview calculations."""

    active_operand_ids: tuple[str, ...]
    remaining_operands: dict[str, tuple[str, ...]]
    source_tensor_ids_by_operand_id: dict[str, tuple[str, ...]]


def _analyze_manual_plan_and_state(
    *,
    spec: NetworkSpec,
    contraction_inputs: PreparedContractionInputs,
    bytes_per_element: int,
) -> tuple[ManualContractionPlanAnalysis, ManualOperandState]:
    """Analyze the manual plan and derive its remaining operands from one simulation."""
    simulation = simulate_contraction_plan(
        initial_operand_ids=contraction_inputs.initial_operand_ids,
        initial_operands=contraction_inputs.initial_operands,
        initial_axis_names=contraction_inputs.initial_axis_names,
        dimension_by_label=contraction_inputs.dimension_by_label,
        plan=spec.contraction_plan,
    )
    return (
        _build_manual_analysis_from_simulation(
            simulation=simulation,
            dimension_by_label=contraction_inputs.dimension_by_label,
            bytes_per_element=bytes_per_element,
        ),
        _build_manual_operand_state_from_simulation(simulation),
    )


def _build_manual_analysis_from_simulation(
    *,
    simulation: SimulatedContractionPlan,
    dimension_by_label: dict[str, int],
    bytes_per_element: int,
) -> ManualContractionPlanAnalysis:
    """Build a manual analysis payload from one finished simulation."""
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


def _build_manual_operand_state_from_simulation(
    simulation: SimulatedContractionPlan,
) -> ManualOperandState:
    """Build the remaining-manual-operands state from one simulation."""
    return ManualOperandState(
        active_operand_ids=simulation.remaining_operand_ids,
        remaining_operands=simulation.remaining_operands,
        source_tensor_ids_by_operand_id=simulation.source_tensor_ids_by_operand_id,
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

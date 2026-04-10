"""Analyze manual and automatic contraction paths for a network spec."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from string import ascii_letters
from typing import Any, cast

from ._contraction_analysis_types import (
    AutomaticContractionPlanAnalysis,
    AutomaticContractionSummary,
    ContractionAnalysisResult,
    ContractionComparison,
    ContractionStepAnalysis,
    ManualContractionPlanAnalysis,
    ManualContractionSummary,
)
from ._contraction_plan import (
    build_dimension_by_label,
    build_initial_operand_axis_names,
    build_initial_operand_labels,
    simulate_contraction_plan,
    simulate_contraction_step,
)
from ._memory_dtypes import DEFAULT_MEMORY_DTYPE, dtype_size_in_bytes
from .codegen.common import prepare_network
from .models import ContractionPlanSpec, ContractionStepSpec, NetworkSpec


def analyze_contraction(
    spec: NetworkSpec,
    *,
    memory_dtype: str = DEFAULT_MEMORY_DTYPE,
) -> ContractionAnalysisResult:
    """Analyze the saved manual plan and available automatic greedy previews."""
    bytes_per_element = dtype_size_in_bytes(memory_dtype)
    prepared = prepare_network(spec)
    dimension_by_label = build_dimension_by_label(prepared)
    initial_operands = build_initial_operand_labels(prepared)
    initial_axis_names = build_initial_operand_axis_names(prepared)
    network_output_shape = tuple(
        index.spec.dimension for index in prepared.open_indices
    )
    manual = _analyze_manual_plan(
        spec=spec,
        initial_operands=initial_operands,
        dimension_by_label=dimension_by_label,
        bytes_per_element=bytes_per_element,
    )
    manual_operand_state = _build_manual_operand_state(
        spec=spec,
        initial_operands=initial_operands,
        initial_axis_names=initial_axis_names,
        dimension_by_label=dimension_by_label,
    )
    automatic_full = _analyze_automatic_operands(
        operand_order=list(initial_operands),
        operands=initial_operands,
        dimension_by_label=dimension_by_label,
        step_id_prefix="auto_full_step_",
        bytes_per_element=bytes_per_element,
    )
    automatic_future = _analyze_future_automatic_plan(
        initial_operands=initial_operands,
        manual_operand_state=manual_operand_state,
        dimension_by_label=dimension_by_label,
        bytes_per_element=bytes_per_element,
    )
    automatic_past = _analyze_past_automatic_plan(
        spec=spec,
        initial_operands=initial_operands,
        manual_operand_state=manual_operand_state,
        dimension_by_label=dimension_by_label,
        bytes_per_element=bytes_per_element,
    )
    message = (
        automatic_future.message
        if automatic_future.status == "unavailable"
        else automatic_past.message
        if automatic_past.status == "unavailable"
        else None
    )
    return ContractionAnalysisResult(
        network_output_shape=network_output_shape,
        manual=manual,
        automatic_full=automatic_full,
        automatic_future=automatic_future,
        automatic_past=automatic_past,
        memory_dtype=memory_dtype,
        comparisons=_build_contraction_comparisons(
            manual=manual,
            automatic_full=automatic_full,
            automatic_future=automatic_future,
            automatic_past=automatic_past,
            memory_dtype=memory_dtype,
        ),
        automatic_strategy="greedy",
        message=message,
    )


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


def _contract_operands(
    *,
    step_id: str,
    left_operand_id: str,
    right_operand_id: str,
    left_labels: tuple[str, ...],
    right_labels: tuple[str, ...],
    dimension_by_label: dict[str, int],
) -> ContractionStepAnalysis:
    """Estimate the metrics for one pairwise contraction."""
    simulated_step = simulate_contraction_step(
        step=ContractionStepSpec(
            id=step_id,
            left_operand_id=left_operand_id,
            right_operand_id=right_operand_id,
        ),
        left_labels=left_labels,
        right_labels=right_labels,
        left_axis_names=left_labels,
        right_axis_names=right_labels,
        dimension_by_label=dimension_by_label,
    )
    return ContractionStepAnalysis(
        step_id=simulated_step.step_id,
        left_operand_id=simulated_step.left_operand_id,
        right_operand_id=simulated_step.right_operand_id,
        result_operand_id=simulated_step.step_id,
        contracted_labels=simulated_step.contracted_labels,
        surviving_labels=simulated_step.surviving_labels,
        result_shape=simulated_step.result_shape,
        result_rank=simulated_step.result_rank,
        estimated_flops=simulated_step.estimated_flops,
        estimated_macs=simulated_step.estimated_macs,
        intermediate_size=simulated_step.intermediate_size,
    )


def _build_automatic_summary(
    *,
    total_estimated_flops: int,
    total_estimated_macs: int,
    peak_intermediate_size: int,
    bytes_per_element: int,
) -> AutomaticContractionSummary:
    """Build a summary payload for automatic path analysis."""
    return AutomaticContractionSummary(
        total_estimated_flops=total_estimated_flops,
        total_estimated_macs=total_estimated_macs,
        peak_intermediate_size=peak_intermediate_size,
        peak_intermediate_bytes=peak_intermediate_size * bytes_per_element,
    )


def _build_contraction_comparisons(
    *,
    manual: ManualContractionPlanAnalysis,
    automatic_full: AutomaticContractionPlanAnalysis,
    automatic_future: AutomaticContractionPlanAnalysis,
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


def _analyze_future_automatic_plan(
    *,
    initial_operands: dict[str, tuple[str, ...]],
    manual_operand_state: ManualOperandState,
    dimension_by_label: dict[str, int],
    bytes_per_element: int,
) -> AutomaticContractionPlanAnalysis:
    """Analyze the greedy path that continues from the current manual state."""
    del initial_operands
    return _analyze_automatic_operands(
        operand_order=list(manual_operand_state.active_operand_ids),
        operands=manual_operand_state.remaining_operands,
        dimension_by_label=dimension_by_label,
        step_id_prefix="auto_future_step_",
        bytes_per_element=bytes_per_element,
    )


def _analyze_past_automatic_plan(
    *,
    spec: NetworkSpec,
    initial_operands: dict[str, tuple[str, ...]],
    manual_operand_state: ManualOperandState,
    dimension_by_label: dict[str, int],
    bytes_per_element: int,
) -> AutomaticContractionPlanAnalysis:
    """Analyze greedy paths for already contracted manual subtrees."""
    del spec
    base_tensor_ids = set(initial_operands)
    contracted_root_operand_ids = [
        operand_id
        for operand_id in manual_operand_state.active_operand_ids
        if operand_id not in base_tensor_ids
        and len(
            manual_operand_state.source_tensor_ids_by_operand_id.get(operand_id, ())
        )
        > 1
    ]
    if not contracted_root_operand_ids:
        return _unavailable_automatic_analysis(
            "Contract at least one tensor pair to unlock the auto past preview.",
            bytes_per_element=bytes_per_element,
        )

    all_steps: list[ContractionStepAnalysis] = []
    total_estimated_flops = 0
    total_estimated_macs = 0
    peak_intermediate_size = 0

    for root_operand_id in contracted_root_operand_ids:
        root_tensor_ids = manual_operand_state.source_tensor_ids_by_operand_id.get(
            root_operand_id, ()
        )
        analysis = _analyze_automatic_operands(
            operand_order=list(root_tensor_ids),
            operands={
                tensor_id: initial_operands[tensor_id] for tensor_id in root_tensor_ids
            },
            dimension_by_label=dimension_by_label,
            step_id_prefix=f"{root_operand_id}__auto_past_",
            final_step_id=root_operand_id,
            bytes_per_element=bytes_per_element,
        )
        if analysis.status == "unavailable":
            return analysis
        all_steps.extend(analysis.steps)
        total_estimated_flops += analysis.summary.total_estimated_flops
        total_estimated_macs += analysis.summary.total_estimated_macs
        peak_intermediate_size = max(
            peak_intermediate_size, analysis.summary.peak_intermediate_size
        )

    return AutomaticContractionPlanAnalysis(
        status="complete",
        steps=all_steps,
        summary=_build_automatic_summary(
            total_estimated_flops=total_estimated_flops,
            total_estimated_macs=total_estimated_macs,
            peak_intermediate_size=peak_intermediate_size,
            bytes_per_element=bytes_per_element,
        ),
    )


def _analyze_automatic_operands(
    *,
    operand_order: list[str],
    operands: dict[str, tuple[str, ...]],
    dimension_by_label: dict[str, int],
    step_id_prefix: str,
    bytes_per_element: int,
    final_step_id: str | None = None,
) -> AutomaticContractionPlanAnalysis:
    """Run automatic greedy analysis for the provided operand set."""
    if len(operand_order) <= 1:
        return AutomaticContractionPlanAnalysis(
            status="complete",
            steps=[],
            summary=_build_automatic_summary(
                total_estimated_flops=0,
                total_estimated_macs=0,
                peak_intermediate_size=0,
                bytes_per_element=bytes_per_element,
            ),
        )

    try:
        contract_path = cast(
            Any,
            cast(Any, import_module("opt_einsum")).contract_path,
        )
    except ImportError:
        return _unavailable_automatic_analysis(
            "Install the planner extra to enable automatic greedy path suggestions.",
            bytes_per_element=bytes_per_element,
        )

    label_order: list[str] = []
    for operand_id in operand_order:
        for label in operands[operand_id]:
            if label not in label_order:
                label_order.append(label)
    if len(label_order) > len(ascii_letters):
        return _unavailable_automatic_analysis(
            "Automatic greedy path analysis currently supports up to 52 distinct labels.",
            bytes_per_element=bytes_per_element,
        )

    symbol_map = {
        label: ascii_letters[offset]
        for offset, label in enumerate(label_order[: len(ascii_letters)])
    }
    label_counts = {
        label: sum(operand_labels.count(label) for operand_labels in operands.values())
        for label in label_order
    }
    output_labels = [label for label in label_order if label_counts[label] == 1]
    equation = (
        ",".join(
            "".join(symbol_map[label] for label in operands[operand_id])
            for operand_id in operand_order
        )
        + "->"
        + "".join(symbol_map[label] for label in output_labels)
    )
    shapes = [
        tuple(dimension_by_label[label] for label in operands[operand_id])
        for operand_id in operand_order
    ]

    try:
        path, _ = contract_path(
            equation,
            *shapes,
            shapes=True,
            optimize="greedy",
        )
    except Exception as exc:  # pragma: no cover - optional dependency behavior
        return _unavailable_automatic_analysis(
            f"Automatic greedy path analysis failed: {exc}",
            bytes_per_element=bytes_per_element,
        )

    remaining_order = list(operand_order)
    remaining_operands = dict(operands)
    steps: list[ContractionStepAnalysis] = []
    total_estimated_flops = 0
    total_estimated_macs = 0
    peak_intermediate_size = 0

    for step_index, raw_indices in enumerate(path, start=1):
        indices = tuple(int(value) for value in raw_indices)
        if len(indices) != 2:
            return _unavailable_automatic_analysis(
                "Automatic greedy path produced a non-pairwise contraction step.",
                bytes_per_element=bytes_per_element,
            )
        left_operand_id = remaining_order[indices[0]]
        right_operand_id = remaining_order[indices[1]]
        step_id = (
            final_step_id
            if final_step_id is not None and step_index == len(path)
            else f"{step_id_prefix}{step_index}"
        )
        step_result = _contract_operands(
            step_id=step_id,
            left_operand_id=left_operand_id,
            right_operand_id=right_operand_id,
            left_labels=remaining_operands.pop(left_operand_id),
            right_labels=remaining_operands.pop(right_operand_id),
            dimension_by_label=dimension_by_label,
        )
        steps.append(step_result)
        remaining_operands[step_id] = step_result.surviving_labels
        total_estimated_flops += step_result.estimated_flops
        total_estimated_macs += step_result.estimated_macs
        peak_intermediate_size = max(
            peak_intermediate_size, step_result.intermediate_size
        )
        for operand_index in sorted(indices, reverse=True):
            remaining_order.pop(operand_index)
        remaining_order.append(step_id)

    status = "complete" if len(remaining_operands) <= 1 else "incomplete"
    return AutomaticContractionPlanAnalysis(
        status=status,
        steps=steps,
        summary=_build_automatic_summary(
            total_estimated_flops=total_estimated_flops,
            total_estimated_macs=total_estimated_macs,
            peak_intermediate_size=peak_intermediate_size,
            bytes_per_element=bytes_per_element,
        ),
    )


def _unavailable_automatic_analysis(
    message: str,
    *,
    bytes_per_element: int,
) -> AutomaticContractionPlanAnalysis:
    """Return a standardized unavailable-analysis payload."""
    return AutomaticContractionPlanAnalysis(
        status="unavailable",
        steps=[],
        summary=_build_automatic_summary(
            total_estimated_flops=0,
            total_estimated_macs=0,
            peak_intermediate_size=0,
            bytes_per_element=bytes_per_element,
        ),
        message=message,
    )

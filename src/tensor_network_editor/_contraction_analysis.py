from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import import_module
from string import ascii_letters
from typing import Any, cast

from ._contraction_plan import (
    build_dimension_by_label,
    build_initial_operand_axis_names,
    build_initial_operand_labels,
    simulate_contraction_plan,
    simulate_contraction_step,
)
from .codegen.common import prepare_network
from .models import ContractionPlanSpec, ContractionStepSpec, NetworkSpec
from .types import JSONValue


@dataclass(slots=True)
class ContractionStepAnalysis:
    step_id: str
    left_operand_id: str
    right_operand_id: str
    result_operand_id: str
    contracted_labels: tuple[str, ...]
    surviving_labels: tuple[str, ...]
    result_shape: tuple[int, ...]
    result_rank: int
    estimated_flops: int
    estimated_macs: int
    intermediate_size: int

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "step_id": self.step_id,
            "left_operand_id": self.left_operand_id,
            "right_operand_id": self.right_operand_id,
            "result_operand_id": self.result_operand_id,
            "contracted_labels": cast(JSONValue, list(self.contracted_labels)),
            "surviving_labels": cast(JSONValue, list(self.surviving_labels)),
            "result_shape": cast(JSONValue, list(self.result_shape)),
            "result_rank": self.result_rank,
            "estimated_flops": self.estimated_flops,
            "estimated_macs": self.estimated_macs,
            "intermediate_size": self.intermediate_size,
        }


@dataclass(slots=True)
class ManualContractionSummary:
    total_estimated_flops: int
    total_estimated_macs: int
    peak_intermediate_size: int
    final_shape: tuple[int, ...] | None
    completion_status: str
    remaining_operand_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "total_estimated_flops": self.total_estimated_flops,
            "total_estimated_macs": self.total_estimated_macs,
            "peak_intermediate_size": self.peak_intermediate_size,
            "final_shape": (
                cast(JSONValue, list(self.final_shape))
                if self.final_shape is not None
                else None
            ),
            "completion_status": self.completion_status,
            "remaining_operand_ids": cast(JSONValue, list(self.remaining_operand_ids)),
        }


@dataclass(slots=True)
class AutomaticContractionSummary:
    total_estimated_flops: int
    total_estimated_macs: int
    peak_intermediate_size: int

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "total_estimated_flops": self.total_estimated_flops,
            "total_estimated_macs": self.total_estimated_macs,
            "peak_intermediate_size": self.peak_intermediate_size,
        }


@dataclass(slots=True)
class ManualContractionPlanAnalysis:
    status: str
    steps: list[ContractionStepAnalysis]
    summary: ManualContractionSummary
    message: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "status": self.status,
            "steps": [step.to_dict() for step in self.steps],
            "summary": self.summary.to_dict(),
        }
        if self.message is not None:
            payload["message"] = self.message
        return payload


@dataclass(slots=True)
class AutomaticContractionPlanAnalysis:
    status: str
    steps: list[ContractionStepAnalysis]
    summary: AutomaticContractionSummary
    message: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "status": self.status,
            "steps": [step.to_dict() for step in self.steps],
            "summary": self.summary.to_dict(),
        }
        if self.message is not None:
            payload["message"] = self.message
        return payload


@dataclass(slots=True)
class ContractionAnalysisResult:
    network_output_shape: tuple[int, ...]
    manual: ManualContractionPlanAnalysis
    automatic_future: AutomaticContractionPlanAnalysis
    automatic_past: AutomaticContractionPlanAnalysis
    automatic_strategy: str = "greedy"
    message: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "network_output_shape": cast(JSONValue, list(self.network_output_shape)),
            "manual": self.manual.to_dict(),
            "automatic_future": self.automatic_future.to_dict(),
            "automatic_past": self.automatic_past.to_dict(),
            "automatic_strategy": self.automatic_strategy,
        }
        if self.message is not None:
            payload["message"] = self.message
        return payload


def analyze_contraction(spec: NetworkSpec) -> ContractionAnalysisResult:
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
    )
    manual_operand_state = _build_manual_operand_state(
        spec=spec,
        initial_operands=initial_operands,
        initial_axis_names=initial_axis_names,
        dimension_by_label=dimension_by_label,
    )
    automatic_future = _analyze_future_automatic_plan(
        initial_operands=initial_operands,
        manual_operand_state=manual_operand_state,
        dimension_by_label=dimension_by_label,
    )
    automatic_past = _analyze_past_automatic_plan(
        spec=spec,
        initial_operands=initial_operands,
        manual_operand_state=manual_operand_state,
        dimension_by_label=dimension_by_label,
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
        automatic_future=automatic_future,
        automatic_past=automatic_past,
        automatic_strategy="greedy",
        message=message,
    )


@dataclass(slots=True)
class ManualOperandState:
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
) -> ManualContractionPlanAnalysis:
    plan = spec.contraction_plan
    if plan is None or not plan.steps:
        summary = _build_manual_summary_from_operands(
            remaining_operands=initial_operands,
            status="complete" if len(initial_operands) <= 1 else "incomplete",
            total_estimated_flops=0,
            total_estimated_macs=0,
            peak_intermediate_size=0,
            dimension_by_label=dimension_by_label,
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
    )


def _simulate_plan_steps(
    *,
    steps: list[ContractionStepSpec],
    initial_operands: dict[str, tuple[str, ...]],
    dimension_by_label: dict[str, int],
) -> ManualContractionPlanAnalysis:
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
    last_result_shape: tuple[int, ...] | None = None,
) -> ManualContractionSummary:
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
) -> AutomaticContractionSummary:
    return AutomaticContractionSummary(
        total_estimated_flops=total_estimated_flops,
        total_estimated_macs=total_estimated_macs,
        peak_intermediate_size=peak_intermediate_size,
    )


def _analyze_future_automatic_plan(
    *,
    initial_operands: dict[str, tuple[str, ...]],
    manual_operand_state: ManualOperandState,
    dimension_by_label: dict[str, int],
) -> AutomaticContractionPlanAnalysis:
    del initial_operands
    return _analyze_automatic_operands(
        operand_order=list(manual_operand_state.active_operand_ids),
        operands=manual_operand_state.remaining_operands,
        dimension_by_label=dimension_by_label,
        step_id_prefix="auto_future_step_",
    )


def _analyze_past_automatic_plan(
    *,
    spec: NetworkSpec,
    initial_operands: dict[str, tuple[str, ...]],
    manual_operand_state: ManualOperandState,
    dimension_by_label: dict[str, int],
) -> AutomaticContractionPlanAnalysis:
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
            "Contract at least one tensor pair to unlock the auto past preview."
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
        ),
    )


def _analyze_automatic_operands(
    *,
    operand_order: list[str],
    operands: dict[str, tuple[str, ...]],
    dimension_by_label: dict[str, int],
    step_id_prefix: str,
    final_step_id: str | None = None,
) -> AutomaticContractionPlanAnalysis:
    if len(operand_order) <= 1:
        return AutomaticContractionPlanAnalysis(
            status="complete",
            steps=[],
            summary=_build_automatic_summary(
                total_estimated_flops=0,
                total_estimated_macs=0,
                peak_intermediate_size=0,
            ),
        )

    try:
        contract_path = cast(
            Any,
            cast(Any, import_module("opt_einsum")).contract_path,
        )
    except ImportError:
        return _unavailable_automatic_analysis(
            "Install the planner extra to enable automatic greedy path suggestions."
        )

    label_order: list[str] = []
    for operand_id in operand_order:
        for label in operands[operand_id]:
            if label not in label_order:
                label_order.append(label)
    if len(label_order) > len(ascii_letters):
        return _unavailable_automatic_analysis(
            "Automatic greedy path analysis currently supports up to 52 distinct labels."
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
            f"Automatic greedy path analysis failed: {exc}"
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
                "Automatic greedy path produced a non-pairwise contraction step."
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
        ),
    )


def _unavailable_automatic_analysis(
    message: str,
) -> AutomaticContractionPlanAnalysis:
    return AutomaticContractionPlanAnalysis(
        status="unavailable",
        steps=[],
        summary=_build_automatic_summary(
            total_estimated_flops=0,
            total_estimated_macs=0,
            peak_intermediate_size=0,
        ),
        message=message,
    )


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result

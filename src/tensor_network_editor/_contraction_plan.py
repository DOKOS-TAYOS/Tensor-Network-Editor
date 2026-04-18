"""Helpers for simulating saved manual contraction plans."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import ContractionPlanSpec, ContractionStepSpec

if TYPE_CHECKING:
    from .codegen.common import PreparedNetwork


@dataclass(slots=True, frozen=True)
class SimulatedContractionStep:
    """Derived information for one simulated contraction step."""

    step_id: str
    left_operand_id: str
    right_operand_id: str
    left_labels: tuple[str, ...]
    right_labels: tuple[str, ...]
    left_axis_names: tuple[str, ...]
    right_axis_names: tuple[str, ...]
    contracted_labels: tuple[str, ...]
    surviving_labels: tuple[str, ...]
    result_axis_names: tuple[str, ...]
    union_labels: tuple[str, ...]
    result_shape: tuple[int, ...]
    result_rank: int
    estimated_flops: int
    estimated_macs: int
    intermediate_size: int

    @property
    def is_outer_product(self) -> bool:
        """Return ``True`` when the step contracts no shared labels."""
        return not self.contracted_labels


@dataclass(slots=True)
class SimulatedContractionPlan:
    """Simulation output for a whole manual contraction plan."""

    steps: list[SimulatedContractionStep]
    remaining_operand_ids: tuple[str, ...]
    remaining_operands: dict[str, tuple[str, ...]]
    remaining_axis_names: dict[str, tuple[str, ...]]
    source_tensor_ids_by_operand_id: dict[str, tuple[str, ...]]


@dataclass(slots=True, frozen=True)
class PreparedContractionInputs:
    """Reusable contraction inputs derived from a prepared network."""

    initial_operand_ids: tuple[str, ...]
    initial_operands: dict[str, tuple[str, ...]]
    initial_axis_names: dict[str, tuple[str, ...]]
    dimension_by_label: dict[str, int]


def prepare_contraction_inputs(
    prepared: PreparedNetwork,
) -> PreparedContractionInputs:
    """Build reusable contraction inputs from ``prepared`` once."""
    initial_operand_ids: list[str] = []
    initial_operands: dict[str, tuple[str, ...]] = {}
    initial_axis_names: dict[str, tuple[str, ...]] = {}
    dimension_by_label: dict[str, int] = {}

    for tensor in prepared.tensors:
        operand_id = tensor.spec.id
        initial_operand_ids.append(operand_id)
        operand_labels: list[str] = []
        axis_names: list[str] = []
        for index in tensor.indices:
            operand_labels.append(index.label)
            axis_names.append(index.spec.name)
            dimension_by_label[index.label] = index.spec.dimension
        initial_operands[operand_id] = tuple(operand_labels)
        initial_axis_names[operand_id] = tuple(axis_names)

    return PreparedContractionInputs(
        initial_operand_ids=tuple(initial_operand_ids),
        initial_operands=initial_operands,
        initial_axis_names=initial_axis_names,
        dimension_by_label=dimension_by_label,
    )


def build_dimension_by_label(prepared: PreparedNetwork) -> dict[str, int]:
    """Build a mapping from prepared index labels to their dimensions."""
    return prepare_contraction_inputs(prepared).dimension_by_label


def simulate_contraction_plan(
    *,
    initial_operand_ids: tuple[str, ...],
    initial_operands: dict[str, tuple[str, ...]],
    initial_axis_names: dict[str, tuple[str, ...]],
    dimension_by_label: dict[str, int],
    plan: ContractionPlanSpec | None,
) -> SimulatedContractionPlan:
    """Simulate a full manual contraction plan from the initial operands."""
    remaining_operands: OrderedDict[str, tuple[str, ...]] = OrderedDict(
        (operand_id, initial_operands[operand_id]) for operand_id in initial_operand_ids
    )
    remaining_axis_names: OrderedDict[str, tuple[str, ...]] = OrderedDict(
        (operand_id, initial_axis_names[operand_id])
        for operand_id in initial_operand_ids
    )
    source_tensor_ids_by_operand_id: dict[str, tuple[str, ...]] = {
        operand_id: (operand_id,) for operand_id in initial_operand_ids
    }

    if plan is None or not plan.steps:
        return SimulatedContractionPlan(
            steps=[],
            remaining_operand_ids=tuple(remaining_operands),
            remaining_operands=remaining_operands,
            remaining_axis_names=remaining_axis_names,
            source_tensor_ids_by_operand_id=source_tensor_ids_by_operand_id,
        )

    step_results: list[SimulatedContractionStep] = []
    for step in plan.steps:
        left_labels = remaining_operands.pop(step.left_operand_id)
        right_labels = remaining_operands.pop(step.right_operand_id)
        left_axis_names = remaining_axis_names.pop(step.left_operand_id)
        right_axis_names = remaining_axis_names.pop(step.right_operand_id)
        left_source_tensor_ids = source_tensor_ids_by_operand_id.pop(
            step.left_operand_id
        )
        right_source_tensor_ids = source_tensor_ids_by_operand_id.pop(
            step.right_operand_id
        )

        step_result = simulate_contraction_step(
            step=step,
            left_labels=left_labels,
            right_labels=right_labels,
            left_axis_names=left_axis_names,
            right_axis_names=right_axis_names,
            dimension_by_label=dimension_by_label,
        )
        step_results.append(step_result)

        remaining_operands[step.id] = step_result.surviving_labels
        remaining_operands.move_to_end(step.id, last=False)
        remaining_axis_names[step.id] = step_result.result_axis_names
        remaining_axis_names.move_to_end(step.id, last=False)
        source_tensor_ids_by_operand_id[step.id] = _merge_source_tensor_ids(
            left_source_tensor_ids,
            right_source_tensor_ids,
        )

    return SimulatedContractionPlan(
        steps=step_results,
        remaining_operand_ids=tuple(remaining_operands),
        remaining_operands=dict(remaining_operands),
        remaining_axis_names=dict(remaining_axis_names),
        source_tensor_ids_by_operand_id=source_tensor_ids_by_operand_id,
    )


def simulate_contraction_step(
    *,
    step: ContractionStepSpec,
    left_labels: tuple[str, ...],
    right_labels: tuple[str, ...],
    left_axis_names: tuple[str, ...],
    right_axis_names: tuple[str, ...],
    dimension_by_label: dict[str, int],
) -> SimulatedContractionStep:
    """Simulate one pairwise contraction step using label metadata only."""
    right_label_set = set(right_labels)
    left_label_set = set(left_labels)
    contracted_labels_list: list[str] = []
    surviving_labels_list: list[str] = []
    union_labels_list: list[str] = []
    seen_union_labels: set[str] = set()

    for label in left_labels:
        if label in right_label_set:
            contracted_labels_list.append(label)
        else:
            surviving_labels_list.append(label)
        if label not in seen_union_labels:
            seen_union_labels.add(label)
            union_labels_list.append(label)

    for label in right_labels:
        if label not in left_label_set:
            surviving_labels_list.append(label)
        if label not in seen_union_labels:
            seen_union_labels.add(label)
            union_labels_list.append(label)

    contracted_labels = tuple(contracted_labels_list)
    surviving_labels = tuple(surviving_labels_list)
    union_labels = tuple(union_labels_list)
    result_shape = tuple(dimension_by_label[label] for label in surviving_labels)
    estimated_macs = _product(dimension_by_label[label] for label in union_labels)
    intermediate_size = _product(result_shape)
    estimated_flops = estimated_macs * 2
    return SimulatedContractionStep(
        step_id=step.id,
        left_operand_id=step.left_operand_id,
        right_operand_id=step.right_operand_id,
        left_labels=left_labels,
        right_labels=right_labels,
        left_axis_names=left_axis_names,
        right_axis_names=right_axis_names,
        contracted_labels=contracted_labels,
        surviving_labels=surviving_labels,
        result_axis_names=surviving_labels,
        union_labels=union_labels,
        result_shape=result_shape,
        result_rank=len(result_shape),
        estimated_flops=estimated_flops,
        estimated_macs=estimated_macs,
        intermediate_size=intermediate_size,
    )


def _product(values: Iterable[int]) -> int:
    """Return the multiplicative product of ``values``."""
    result = 1
    for value in values:
        result *= int(value)
    return result


def _merge_source_tensor_ids(
    left_source_tensor_ids: tuple[str, ...],
    right_source_tensor_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Merge source-tensor ids while preserving their first-seen order."""
    merged_source_tensor_ids = list(left_source_tensor_ids)
    seen_source_tensor_ids = set(left_source_tensor_ids)
    for tensor_id in right_source_tensor_ids:
        if tensor_id in seen_source_tensor_ids:
            continue
        seen_source_tensor_ids.add(tensor_id)
        merged_source_tensor_ids.append(tensor_id)
    return tuple(merged_source_tensor_ids)

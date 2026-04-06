from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import ContractionPlanSpec, ContractionStepSpec

if TYPE_CHECKING:
    from .codegen.common import PreparedNetwork

_NON_IDENTIFIER_PATTERN = re.compile(r"[^0-9a-zA-Z_]+")


@dataclass(slots=True, frozen=True)
class SimulatedContractionStep:
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
        return not self.contracted_labels


@dataclass(slots=True)
class SimulatedContractionPlan:
    steps: list[SimulatedContractionStep]
    remaining_operand_ids: tuple[str, ...]
    remaining_operands: dict[str, tuple[str, ...]]
    remaining_axis_names: dict[str, tuple[str, ...]]
    source_tensor_ids_by_operand_id: dict[str, tuple[str, ...]]


def sanitize_python_identifier(value: str, prefix: str) -> str:
    collapsed = _NON_IDENTIFIER_PATTERN.sub("_", value.strip()).strip("_").lower()
    if not collapsed:
        collapsed = prefix
    if collapsed[0].isdigit():
        collapsed = f"{prefix}_{collapsed}"
    return collapsed


def build_dimension_by_label(prepared: PreparedNetwork) -> dict[str, int]:
    dimension_by_label: dict[str, int] = {}
    for tensor in prepared.tensors:
        for index in tensor.indices:
            dimension_by_label[index.label] = index.spec.dimension
    return dimension_by_label


def build_initial_operand_labels(
    prepared: PreparedNetwork,
) -> dict[str, tuple[str, ...]]:
    return {
        tensor.spec.id: tuple(index.label for index in tensor.indices)
        for tensor in prepared.tensors
    }


def build_initial_operand_axis_names(
    prepared: PreparedNetwork,
) -> dict[str, tuple[str, ...]]:
    return {
        tensor.spec.id: tuple(index.spec.name for index in tensor.indices)
        for tensor in prepared.tensors
    }


def simulate_contraction_plan(
    *,
    initial_operand_ids: tuple[str, ...],
    initial_operands: dict[str, tuple[str, ...]],
    initial_axis_names: dict[str, tuple[str, ...]],
    dimension_by_label: dict[str, int],
    plan: ContractionPlanSpec | None,
) -> SimulatedContractionPlan:
    remaining_operands = dict(initial_operands)
    remaining_axis_names = dict(initial_axis_names)
    remaining_operand_ids = list(initial_operand_ids)
    source_tensor_ids_by_operand_id: dict[str, tuple[str, ...]] = {
        operand_id: (operand_id,) for operand_id in initial_operand_ids
    }

    if plan is None or not plan.steps:
        return SimulatedContractionPlan(
            steps=[],
            remaining_operand_ids=tuple(remaining_operand_ids),
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

        remaining_operands = {
            step.id: step_result.surviving_labels,
            **remaining_operands,
        }
        remaining_axis_names = {
            step.id: step_result.result_axis_names,
            **remaining_axis_names,
        }
        source_tensor_ids_by_operand_id = {
            step.id: tuple(
                dict.fromkeys(left_source_tensor_ids + right_source_tensor_ids)
            ),
            **source_tensor_ids_by_operand_id,
        }
        remaining_operand_ids = [
            step.id,
            *[
                operand_id
                for operand_id in remaining_operand_ids
                if operand_id not in {step.left_operand_id, step.right_operand_id}
            ],
        ]

    return SimulatedContractionPlan(
        steps=step_results,
        remaining_operand_ids=tuple(remaining_operand_ids),
        remaining_operands=remaining_operands,
        remaining_axis_names=remaining_axis_names,
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
    right_label_set = set(right_labels)
    contracted_labels = tuple(
        label for label in left_labels if label in right_label_set
    )
    surviving_labels = tuple(
        label for label in left_labels if label not in contracted_labels
    ) + tuple(label for label in right_labels if label not in contracted_labels)
    union_labels = tuple(dict.fromkeys(left_labels + right_labels))
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
    result = 1
    for value in values:
        result *= int(value)
    return result

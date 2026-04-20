"""Shared carry-mode helpers for linear periodic validation and codegen."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ...models import ContractionStepSpec
from ..analysis._contraction_plan import (
    SimulatedContractionStep,
    simulate_contraction_step,
)


@dataclass(slots=True, frozen=True)
class LinearPeriodicCarryOperandState:
    """Track the labels and axis names of one carry operand."""

    labels: tuple[str, ...]
    axis_names: tuple[str, ...]


def linear_periodic_step_uses_reserved_operand(
    step: ContractionStepSpec,
    operand_id: str,
) -> bool:
    """Return whether ``step`` references one reserved carry operand id."""
    return operand_id in {step.left_operand_id, step.right_operand_id}


def linear_periodic_carry_partner_operand_id(
    step: ContractionStepSpec,
    reserved_operand_id: str,
) -> str:
    """Return the non-reserved operand paired with one boundary operand."""
    if step.left_operand_id == reserved_operand_id:
        return step.right_operand_id
    return step.left_operand_id


def resolve_linear_periodic_carry_operand_order(
    *,
    step: ContractionStepSpec,
    reserved_operand_id: str,
    reserved_state: LinearPeriodicCarryOperandState,
    partner_state: LinearPeriodicCarryOperandState,
) -> tuple[LinearPeriodicCarryOperandState, LinearPeriodicCarryOperandState]:
    """Return operand states in the left/right order used by ``step``."""
    if step.left_operand_id == reserved_operand_id:
        return reserved_state, partner_state
    return partner_state, reserved_state


def simulate_linear_periodic_carry_step(
    *,
    step: ContractionStepSpec,
    left_state: LinearPeriodicCarryOperandState,
    right_state: LinearPeriodicCarryOperandState,
    dimension_by_label: dict[str, int],
) -> tuple[SimulatedContractionStep, tuple[str, ...]]:
    """Simulate one carry-mode contraction while preserving axis-name metadata."""
    simulation = simulate_contraction_step(
        step=step,
        left_labels=left_state.labels,
        right_labels=right_state.labels,
        left_axis_names=left_state.axis_names,
        right_axis_names=right_state.axis_names,
        dimension_by_label=dimension_by_label,
    )
    axis_name_by_label: dict[str, str] = {}
    for label, axis_name in zip(
        left_state.labels,
        left_state.axis_names,
        strict=True,
    ):
        axis_name_by_label[label] = axis_name
    for label, axis_name in zip(
        right_state.labels,
        right_state.axis_names,
        strict=True,
    ):
        axis_name_by_label.setdefault(label, axis_name)
    result_axis_names = tuple(
        axis_name_by_label[label] for label in simulation.surviving_labels
    )
    return replace(simulation, result_axis_names=result_axis_names), result_axis_names

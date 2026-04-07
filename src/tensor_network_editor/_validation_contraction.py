"""Validation helpers for manual contraction plans and saved snapshots."""

from __future__ import annotations

import math

from ._validation_common import (
    append_duplicate_id_issues,
    append_issue,
    is_valid_name,
    validate_metadata,
)
from .models import (
    ContractionOperandLayoutSpec,
    ContractionPlanSpec,
    ContractionStepSpec,
    ContractionViewSnapshotSpec,
    ValidationIssue,
)


def validate_contraction_plan(
    plan: ContractionPlanSpec,
    *,
    tensor_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    """Validate the manual contraction plan stored on a network."""
    if not is_valid_name(plan.name):
        append_issue(
            issues,
            code="invalid-name",
            message=f"Contraction plan '{plan.id}' has an empty name.",
            path="contraction_plan.name",
        )
    validate_metadata("contraction_plan.metadata", plan.metadata, issues)
    append_duplicate_id_issues(
        (step.id for step in plan.steps),
        code="duplicate-contraction-step-id",
        path="contraction_plan.steps",
        message_prefix="Contraction step id",
        issues=issues,
    )

    available_operand_ids = set(tensor_ids)
    consumed_operand_ids: set[str] = set()

    for snapshot in plan.view_snapshots:
        validate_contraction_view_snapshot(snapshot, issues=issues)

    for step in plan.steps:
        validate_contraction_step(
            step,
            available_operand_ids=available_operand_ids,
            consumed_operand_ids=consumed_operand_ids,
            issues=issues,
        )


def validate_contraction_view_snapshot(
    snapshot: ContractionViewSnapshotSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Validate one saved contraction-scene snapshot."""
    snapshot_path = f"contraction_plan.view_snapshots.{snapshot.applied_step_count}"
    if snapshot.applied_step_count < 0:
        append_issue(
            issues,
            code="invalid-contraction-view-snapshot",
            message="Contraction view snapshots must use a non-negative step count.",
            path=f"{snapshot_path}.applied_step_count",
        )

    seen_operand_ids: set[str] = set()
    for operand_layout in snapshot.operand_layouts:
        validate_contraction_operand_layout(
            operand_layout,
            snapshot_path=snapshot_path,
            seen_operand_ids=seen_operand_ids,
            issues=issues,
        )


def validate_contraction_operand_layout(
    operand_layout: ContractionOperandLayoutSpec,
    *,
    snapshot_path: str,
    seen_operand_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    """Validate one operand layout entry inside a snapshot."""
    operand_path = f"{snapshot_path}.operand_layouts.{operand_layout.operand_id or '_'}"
    if not is_valid_name(operand_layout.operand_id):
        append_issue(
            issues,
            code="invalid-contraction-view-snapshot",
            message="Contraction operand layouts must use a non-empty operand id.",
            path=f"{operand_path}.operand_id",
        )
    elif operand_layout.operand_id in seen_operand_ids:
        append_issue(
            issues,
            code="invalid-contraction-view-snapshot",
            message=(
                f"Contraction view snapshot duplicates operand id "
                f"'{operand_layout.operand_id}'."
            ),
            path=f"{operand_path}.operand_id",
        )
    else:
        seen_operand_ids.add(operand_layout.operand_id)

    if not math.isfinite(operand_layout.position.x) or not math.isfinite(
        operand_layout.position.y
    ):
        append_issue(
            issues,
            code="invalid-contraction-view-snapshot",
            message=(
                f"Contraction operand layout '{operand_layout.operand_id}' has a "
                "non-finite position."
            ),
            path=f"{operand_path}.position",
        )
    if (
        not math.isfinite(operand_layout.size.width)
        or not math.isfinite(operand_layout.size.height)
        or operand_layout.size.width <= 0
        or operand_layout.size.height <= 0
    ):
        append_issue(
            issues,
            code="invalid-size",
            message=(
                f"Contraction operand layout '{operand_layout.operand_id}' must "
                "have a positive finite size."
            ),
            path=f"{operand_path}.size",
        )


def validate_contraction_step(
    step: ContractionStepSpec,
    *,
    available_operand_ids: set[str],
    consumed_operand_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    """Validate a single manual contraction step against available operands."""
    step_path = f"contraction_plan.steps.{step.id}"
    validate_metadata(f"{step_path}.metadata", step.metadata, issues)

    if not is_valid_name(step.id):
        append_issue(
            issues,
            code="invalid-name",
            message="Contraction step id cannot be empty.",
            path=f"{step_path}.id",
        )
        return

    if step.id in available_operand_ids:
        append_issue(
            issues,
            code="duplicate-contraction-step-id",
            message=(
                f"Contraction step id '{step.id}' conflicts with an existing operand id."
            ),
            path=f"{step_path}.id",
        )
        return

    operand_ids = [step.left_operand_id, step.right_operand_id]
    if step.left_operand_id == step.right_operand_id:
        append_issue(
            issues,
            code="invalid-contraction-operand",
            message=(
                f"Contraction step '{step.id}' must use two distinct operand ids."
            ),
            path=f"{step_path}.left_operand_id",
        )
        return

    has_invalid_operand = False
    for attribute_name, operand_id in (
        ("left_operand_id", step.left_operand_id),
        ("right_operand_id", step.right_operand_id),
    ):
        if not is_valid_name(operand_id):
            append_issue(
                issues,
                code="invalid-contraction-operand",
                message=f"Contraction step '{step.id}' has an empty operand id.",
                path=f"{step_path}.{attribute_name}",
            )
            has_invalid_operand = True
            continue
        if operand_id in consumed_operand_ids:
            append_issue(
                issues,
                code="contraction-operand-reused",
                message=(
                    f"Operand '{operand_id}' in contraction step '{step.id}' was already consumed."
                ),
                path=f"{step_path}.{attribute_name}",
            )
            has_invalid_operand = True
            continue
        if operand_id not in available_operand_ids:
            append_issue(
                issues,
                code="invalid-contraction-operand",
                message=(
                    f"Operand '{operand_id}' in contraction step '{step.id}' is not available."
                ),
                path=f"{step_path}.{attribute_name}",
            )
            has_invalid_operand = True

    if has_invalid_operand:
        return

    for operand_id in operand_ids:
        available_operand_ids.remove(operand_id)
        consumed_operand_ids.add(operand_id)
    available_operand_ids.add(step.id)

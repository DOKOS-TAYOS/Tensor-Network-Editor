"""Validation helpers for the typed tree periodic mode."""

from __future__ import annotations

from collections import Counter

from ...models import (
    LinearPeriodicCellSpec,
    TreePeriodicCellName,
    TreePeriodicTensorRole,
    TreePeriodicTreeSpec,
    ValidationIssue,
)
from ..analysis._analysis import analyze_network
from ..modes._tree_periodic import (
    iter_tree_periodic_cells,
    tree_periodic_boundary_tensors,
    tree_periodic_cell_as_network,
    tree_periodic_reserved_operand_id_for_tensor,
)
from ._validation_common import append_issue, validate_metadata
from ._validation_contraction import validate_contraction_plan
from ._validation_edges import validate_edge
from ._validation_entities import (
    validate_group,
    validate_network,
    validate_note,
    validate_tensor,
)

_EXPECTED_PARENT_COUNT: dict[TreePeriodicCellName, int] = {
    TreePeriodicCellName.ROOT: 0,
    TreePeriodicCellName.BRANCH: 1,
    TreePeriodicCellName.LEAF: 1,
}

_EXPECTED_CHILD_COUNT_BY_CELL: dict[TreePeriodicCellName, bool] = {
    TreePeriodicCellName.ROOT: True,
    TreePeriodicCellName.BRANCH: True,
    TreePeriodicCellName.LEAF: False,
}


def validate_tree_periodic_tree(
    tree: TreePeriodicTreeSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Validate all cells and tree-specific boundary structure."""
    validate_metadata("tree_periodic_tree.metadata", tree.metadata, issues)
    if tree.branching_factor < 2:
        append_issue(
            issues,
            code="tree-periodic-branching-factor",
            message="Tree periodic mode requires a branching factor of at least 2.",
            path="tree_periodic_tree.branching_factor",
        )

    for cell_name, cell in iter_tree_periodic_cells(tree):
        _validate_tree_periodic_cell(
            cell_name,
            cell,
            branching_factor=tree.branching_factor,
            issues=issues,
        )


def _validate_tree_periodic_cell(
    cell_name: TreePeriodicCellName,
    cell: LinearPeriodicCellSpec,
    *,
    branching_factor: int,
    issues: list[ValidationIssue],
) -> None:
    """Validate one tree periodic cell."""
    prefix = _tree_periodic_cell_prefix(cell_name)
    cell_network = tree_periodic_cell_as_network(cell, cell_name=cell_name)

    local_issues: list[ValidationIssue] = []
    validate_network(cell_network, local_issues)
    analysis = analyze_network(cell_network)
    tensor_ids = set(analysis.tensor_map)

    for tensor in cell.tensors:
        validate_tensor(tensor, issues=local_issues)

    for group in cell.groups:
        validate_group(group, tensor_ids=tensor_ids, issues=local_issues)

    for note in cell.notes:
        validate_note(note, issues=local_issues)

    connected_indices: set[str] = set()
    for edge in cell.edges:
        validate_edge(
            edge,
            analysis_tensor_map=analysis.tensor_map,
            analysis_index_map=analysis.index_map,
            connected_indices=connected_indices,
            issues=local_issues,
        )

    issues.extend(_prefix_validation_issues(prefix, local_issues))
    _validate_tree_periodic_boundary_roles(
        cell_name,
        cell,
        branching_factor=branching_factor,
        issues=issues,
    )
    if cell.contraction_plan is not None:
        plan_issues: list[ValidationIssue] = []
        validate_contraction_plan(
            cell.contraction_plan,
            tensor_ids={
                tensor.id
                for tensor in cell.tensors
                if tensor.tree_periodic_role is None
            }
            | {
                reserved_operand_id
                for tensor in cell.tensors
                if (
                    reserved_operand_id := tree_periodic_reserved_operand_id_for_tensor(
                        tensor
                    )
                )
                is not None
            },
            issues=plan_issues,
        )
        issues.extend(_prefix_validation_issues(prefix, plan_issues))


def _validate_tree_periodic_boundary_roles(
    cell_name: TreePeriodicCellName,
    cell: LinearPeriodicCellSpec,
    *,
    branching_factor: int,
    issues: list[ValidationIssue],
) -> None:
    """Ensure each cell exposes the expected tree virtual boundaries."""
    cell_prefix = _tree_periodic_cell_prefix(cell_name)
    parent_tensors = tree_periodic_boundary_tensors(
        cell,
        role=TreePeriodicTensorRole.PARENT,
    )
    expected_parent_count = _EXPECTED_PARENT_COUNT[cell_name]
    if len(parent_tensors) != expected_parent_count:
        append_issue(
            issues,
            code="tree-periodic-boundary-role",
            message=(
                f"Cell '{cell_name.value}' must contain exactly "
                f"{expected_parent_count} parent boundary tensor(s)."
            ),
            path=f"{cell_prefix}.parent_boundary",
        )

    child_tensors = tree_periodic_boundary_tensors(
        cell,
        role=TreePeriodicTensorRole.CHILD,
    )
    expects_children = _EXPECTED_CHILD_COUNT_BY_CELL[cell_name]
    if not expects_children and child_tensors:
        append_issue(
            issues,
            code="tree-periodic-boundary-role",
            message=(
                f"Cell '{cell_name.value}' cannot contain child boundary tensors."
            ),
            path=f"{cell_prefix}.child_boundaries",
        )
        return

    if expects_children:
        child_counts = Counter(
            tensor.tree_periodic_child_index for tensor in child_tensors
        )
        missing_indices = [
            child_index
            for child_index in range(branching_factor)
            if child_counts.get(child_index, 0) == 0
        ]
        duplicate_indices = [
            child_index
            for child_index, count in child_counts.items()
            if child_index is not None and count > 1
        ]
        if len(child_tensors) != branching_factor:
            for child_index in missing_indices:
                append_issue(
                    issues,
                    code="tree-periodic-boundary-role",
                    message=(
                        f"Cell '{cell_name.value}' is missing child boundary "
                        f"slot {child_index}."
                    ),
                    path=f"{cell_prefix}.child_boundary_{child_index}",
                )
        if (
            any(tensor.tree_periodic_child_index is None for tensor in child_tensors)
            or duplicate_indices
        ):
            append_issue(
                issues,
                code="tree-periodic-child-index",
                message=(
                    f"Cell '{cell_name.value}' must expose exactly one child "
                    "boundary tensor for each child index."
                ),
                path=f"{cell_prefix}.child_boundaries",
            )


def _tree_periodic_cell_prefix(cell_name: TreePeriodicCellName) -> str:
    """Return the validation path prefix for one tree periodic cell."""
    return f"tree_periodic_tree.{cell_name.value}_cell"


def _prefix_validation_issues(
    prefix: str,
    issues: list[ValidationIssue],
) -> list[ValidationIssue]:
    """Return a copy of ``issues`` with every path nested below ``prefix``."""
    return [
        ValidationIssue(
            code=issue.code,
            message=issue.message,
            path=f"{prefix}.{issue.path}" if issue.path else prefix,
        )
        for issue in issues
    ]

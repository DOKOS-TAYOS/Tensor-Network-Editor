"""Validation helpers for the typed linear periodic-chain mode."""

from __future__ import annotations

from ._analysis import analyze_network
from ._linear_periodic import (
    build_linear_periodic_interface_ports,
    iter_linear_periodic_cells,
    linear_periodic_boundary_tensors,
    linear_periodic_cell_as_network,
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
from .models import (
    LinearPeriodicCellName,
    LinearPeriodicCellSpec,
    LinearPeriodicChainSpec,
    LinearPeriodicTensorRole,
    ValidationIssue,
)

_EXPECTED_BOUNDARY_ROLES: dict[
    LinearPeriodicCellName, tuple[LinearPeriodicTensorRole, ...]
] = {
    LinearPeriodicCellName.INITIAL: (LinearPeriodicTensorRole.NEXT,),
    LinearPeriodicCellName.PERIODIC: (
        LinearPeriodicTensorRole.PREVIOUS,
        LinearPeriodicTensorRole.NEXT,
    ),
    LinearPeriodicCellName.FINAL: (LinearPeriodicTensorRole.PREVIOUS,),
}


def validate_linear_periodic_chain(
    chain: LinearPeriodicChainSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Validate all cells and cross-cell interfaces in a periodic chain."""
    validate_metadata("linear_periodic_chain.metadata", chain.metadata, issues)

    for cell_name, cell in iter_linear_periodic_cells(chain):
        _validate_linear_periodic_cell(cell_name, cell, issues=issues)

    _validate_linear_periodic_interfaces(chain, issues=issues)


def _validate_linear_periodic_cell(
    cell_name: LinearPeriodicCellName,
    cell: LinearPeriodicCellSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Validate one cell using the existing plain-network validators."""
    prefix = f"linear_periodic_chain.{cell_name.value}_cell"
    cell_network = linear_periodic_cell_as_network(cell, cell_name=cell_name)

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

    if cell.contraction_plan is not None:
        validate_contraction_plan(
            cell.contraction_plan,
            tensor_ids={
                tensor.id
                for tensor in cell.tensors
                if tensor.linear_periodic_role is None
            },
            issues=local_issues,
        )

    issues.extend(_prefix_validation_issues(prefix, local_issues))
    _validate_linear_periodic_boundary_roles(cell_name, cell, issues=issues)


def _validate_linear_periodic_boundary_roles(
    cell_name: LinearPeriodicCellName,
    cell: LinearPeriodicCellSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Ensure each cell exposes the expected virtual boundary tensors."""
    allowed_roles = set(_EXPECTED_BOUNDARY_ROLES[cell_name])
    cell_prefix = f"linear_periodic_chain.{cell_name.value}_cell"

    for tensor in cell.tensors:
        role = tensor.linear_periodic_role
        if role is None:
            continue
        if role not in allowed_roles:
            append_issue(
                issues,
                code="linear-periodic-boundary-role",
                message=(
                    f"Cell '{cell_name.value}' does not allow a boundary tensor "
                    f"with role '{role.value}'."
                ),
                path=f"{cell_prefix}.tensors.{tensor.id}.linear_periodic_role",
            )

    for role in allowed_roles:
        matching_tensors = linear_periodic_boundary_tensors(cell, role=role)
        if len(matching_tensors) != 1:
            append_issue(
                issues,
                code="linear-periodic-boundary-role",
                message=(
                    f"Cell '{cell_name.value}' must contain exactly one "
                    f"'{role.value}' boundary tensor."
                ),
                path=f"{cell_prefix}.{role.value}_boundary",
            )


def _validate_linear_periodic_interfaces(
    chain: LinearPeriodicChainSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Validate the ordered boundary interfaces that connect adjacent cells."""
    interface_pairs = (
        (
            LinearPeriodicCellName.INITIAL,
            LinearPeriodicTensorRole.NEXT,
            LinearPeriodicCellName.PERIODIC,
            LinearPeriodicTensorRole.PREVIOUS,
            "linear_periodic_chain.initial_cell.next_interface",
            "initial.next -> periodic.previous",
        ),
        (
            LinearPeriodicCellName.PERIODIC,
            LinearPeriodicTensorRole.NEXT,
            LinearPeriodicCellName.PERIODIC,
            LinearPeriodicTensorRole.PREVIOUS,
            "linear_periodic_chain.periodic_cell.next_interface",
            "periodic.next -> periodic.previous",
        ),
        (
            LinearPeriodicCellName.PERIODIC,
            LinearPeriodicTensorRole.NEXT,
            LinearPeriodicCellName.FINAL,
            LinearPeriodicTensorRole.PREVIOUS,
            "linear_periodic_chain.periodic_cell.next_interface",
            "periodic.next -> final.previous",
        ),
    )
    cell_by_name = dict(iter_linear_periodic_cells(chain))

    for (
        source_cell_name,
        source_role,
        target_cell_name,
        target_role,
        path,
        pair_label,
    ) in interface_pairs:
        source_cell = cell_by_name[source_cell_name]
        target_cell = cell_by_name[target_cell_name]
        if len(linear_periodic_boundary_tensors(source_cell, role=source_role)) != 1:
            continue
        if len(linear_periodic_boundary_tensors(target_cell, role=target_role)) != 1:
            continue

        source_ports = build_linear_periodic_interface_ports(
            source_cell,
            cell_name=source_cell_name,
            role=source_role,
        )
        target_ports = build_linear_periodic_interface_ports(
            target_cell,
            cell_name=target_cell_name,
            role=target_role,
        )
        source_dimensions = tuple(port.dimension for port in source_ports)
        target_dimensions = tuple(port.dimension for port in target_ports)
        if source_dimensions != target_dimensions:
            append_issue(
                issues,
                code="linear-periodic-interface-mismatch",
                message=(
                    f"Linear periodic interface mismatch for {pair_label}: "
                    f"{source_dimensions} != {target_dimensions}."
                ),
                path=path,
            )


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

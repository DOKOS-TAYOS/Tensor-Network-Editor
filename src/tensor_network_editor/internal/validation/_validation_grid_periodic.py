"""Validation helpers for the typed bidimensional periodic-grid mode."""

from __future__ import annotations

from ...models import (
    GridPeriodicCellName,
    GridPeriodicGridSpec,
    GridPeriodicTensorRole,
    LinearPeriodicCellSpec,
    ValidationIssue,
)
from ..analysis._analysis import analyze_network
from ..modes._grid_periodic import (
    build_grid_periodic_interface_ports,
    grid_periodic_boundary_tensors,
    grid_periodic_cell_as_network,
    grid_periodic_reserved_operand_id_for_role,
    iter_grid_periodic_cells,
)
from ._validation_common import (
    append_issue,
    prefix_validation_issues,
    validate_metadata,
)
from ._validation_contraction import validate_contraction_plan
from ._validation_edges import validate_edge
from ._validation_entities import (
    validate_group,
    validate_network,
    validate_note,
    validate_tensor,
)

_EXPECTED_BOUNDARY_ROLES: dict[
    GridPeriodicCellName, tuple[GridPeriodicTensorRole, ...]
] = {
    GridPeriodicCellName.TOP_LEFT: (
        GridPeriodicTensorRole.RIGHT,
        GridPeriodicTensorRole.DOWN,
    ),
    GridPeriodicCellName.TOP: (
        GridPeriodicTensorRole.LEFT,
        GridPeriodicTensorRole.RIGHT,
        GridPeriodicTensorRole.DOWN,
    ),
    GridPeriodicCellName.TOP_RIGHT: (
        GridPeriodicTensorRole.LEFT,
        GridPeriodicTensorRole.DOWN,
    ),
    GridPeriodicCellName.LEFT: (
        GridPeriodicTensorRole.UP,
        GridPeriodicTensorRole.RIGHT,
        GridPeriodicTensorRole.DOWN,
    ),
    GridPeriodicCellName.CENTER: (
        GridPeriodicTensorRole.UP,
        GridPeriodicTensorRole.RIGHT,
        GridPeriodicTensorRole.DOWN,
        GridPeriodicTensorRole.LEFT,
    ),
    GridPeriodicCellName.RIGHT: (
        GridPeriodicTensorRole.UP,
        GridPeriodicTensorRole.DOWN,
        GridPeriodicTensorRole.LEFT,
    ),
    GridPeriodicCellName.BOTTOM_LEFT: (
        GridPeriodicTensorRole.UP,
        GridPeriodicTensorRole.RIGHT,
    ),
    GridPeriodicCellName.BOTTOM: (
        GridPeriodicTensorRole.UP,
        GridPeriodicTensorRole.LEFT,
        GridPeriodicTensorRole.RIGHT,
    ),
    GridPeriodicCellName.BOTTOM_RIGHT: (
        GridPeriodicTensorRole.UP,
        GridPeriodicTensorRole.LEFT,
    ),
}

_INTERFACE_FAMILIES: tuple[
    tuple[str, tuple[tuple[GridPeriodicCellName, GridPeriodicTensorRole], ...]], ...
] = (
    (
        "grid_periodic_grid.horizontal_interfaces.top_row",
        (
            (GridPeriodicCellName.TOP_LEFT, GridPeriodicTensorRole.RIGHT),
            (GridPeriodicCellName.TOP, GridPeriodicTensorRole.LEFT),
            (GridPeriodicCellName.TOP, GridPeriodicTensorRole.RIGHT),
            (GridPeriodicCellName.TOP_RIGHT, GridPeriodicTensorRole.LEFT),
        ),
    ),
    (
        "grid_periodic_grid.horizontal_interfaces.middle_row",
        (
            (GridPeriodicCellName.LEFT, GridPeriodicTensorRole.RIGHT),
            (GridPeriodicCellName.CENTER, GridPeriodicTensorRole.LEFT),
            (GridPeriodicCellName.CENTER, GridPeriodicTensorRole.RIGHT),
            (GridPeriodicCellName.RIGHT, GridPeriodicTensorRole.LEFT),
        ),
    ),
    (
        "grid_periodic_grid.horizontal_interfaces.bottom_row",
        (
            (GridPeriodicCellName.BOTTOM_LEFT, GridPeriodicTensorRole.RIGHT),
            (GridPeriodicCellName.BOTTOM, GridPeriodicTensorRole.LEFT),
            (GridPeriodicCellName.BOTTOM, GridPeriodicTensorRole.RIGHT),
            (GridPeriodicCellName.BOTTOM_RIGHT, GridPeriodicTensorRole.LEFT),
        ),
    ),
    (
        "grid_periodic_grid.vertical_interfaces.left_column",
        (
            (GridPeriodicCellName.TOP_LEFT, GridPeriodicTensorRole.DOWN),
            (GridPeriodicCellName.LEFT, GridPeriodicTensorRole.UP),
            (GridPeriodicCellName.LEFT, GridPeriodicTensorRole.DOWN),
            (GridPeriodicCellName.BOTTOM_LEFT, GridPeriodicTensorRole.UP),
        ),
    ),
    (
        "grid_periodic_grid.vertical_interfaces.center_column",
        (
            (GridPeriodicCellName.TOP, GridPeriodicTensorRole.DOWN),
            (GridPeriodicCellName.CENTER, GridPeriodicTensorRole.UP),
            (GridPeriodicCellName.CENTER, GridPeriodicTensorRole.DOWN),
            (GridPeriodicCellName.BOTTOM, GridPeriodicTensorRole.UP),
        ),
    ),
    (
        "grid_periodic_grid.vertical_interfaces.right_column",
        (
            (GridPeriodicCellName.TOP_RIGHT, GridPeriodicTensorRole.DOWN),
            (GridPeriodicCellName.RIGHT, GridPeriodicTensorRole.UP),
            (GridPeriodicCellName.RIGHT, GridPeriodicTensorRole.DOWN),
            (GridPeriodicCellName.BOTTOM_RIGHT, GridPeriodicTensorRole.UP),
        ),
    ),
)


def validate_grid_periodic_grid(
    grid: GridPeriodicGridSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Validate all cells and interface families in a periodic grid."""
    validate_metadata("grid_periodic_grid.metadata", grid.metadata, issues)

    for cell_name, cell in iter_grid_periodic_cells(grid):
        _validate_grid_periodic_cell(cell_name, cell, issues=issues)

    _validate_grid_periodic_interfaces(grid, issues=issues)


def _validate_grid_periodic_cell(
    cell_name: GridPeriodicCellName,
    cell: LinearPeriodicCellSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Validate one cell using the existing plain-network validators."""
    prefix = _grid_periodic_cell_prefix(cell_name)
    cell_network = grid_periodic_cell_as_network(cell, cell_name=cell_name)

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

    issues.extend(prefix_validation_issues(prefix, local_issues))
    _validate_grid_periodic_boundary_roles(cell_name, cell, issues=issues)
    if cell.contraction_plan is None:
        return
    plan_issues: list[ValidationIssue] = []
    validate_contraction_plan(
        cell.contraction_plan,
        tensor_ids={
            tensor.id for tensor in cell.tensors if tensor.grid_periodic_role is None
        }
        | {
            grid_periodic_reserved_operand_id_for_role(tensor.grid_periodic_role)
            for tensor in cell.tensors
            if tensor.grid_periodic_role is not None
        },
        issues=plan_issues,
    )
    issues.extend(prefix_validation_issues(prefix, plan_issues))


def _validate_grid_periodic_boundary_roles(
    cell_name: GridPeriodicCellName,
    cell: LinearPeriodicCellSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Ensure each cell exposes the expected 2D virtual boundary tensors."""
    allowed_roles = set(_EXPECTED_BOUNDARY_ROLES[cell_name])
    cell_prefix = _grid_periodic_cell_prefix(cell_name)

    for tensor in cell.tensors:
        role = tensor.grid_periodic_role
        if role is None:
            continue
        if role not in allowed_roles:
            append_issue(
                issues,
                code="grid-periodic-boundary-role",
                message=(
                    f"Cell '{cell_name.value}' does not allow a boundary tensor "
                    f"with role '{role.value}'."
                ),
                path=f"{cell_prefix}.tensors.{tensor.id}.grid_periodic_role",
            )

    for role in allowed_roles:
        matching_tensors = grid_periodic_boundary_tensors(cell, role=role)
        if len(matching_tensors) != 1:
            append_issue(
                issues,
                code="grid-periodic-boundary-role",
                message=(
                    f"Cell '{cell_name.value}' must contain exactly one "
                    f"'{role.value}' boundary tensor."
                ),
                path=f"{cell_prefix}.{role.value}_boundary",
            )


def _validate_grid_periodic_interfaces(
    grid: GridPeriodicGridSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Validate the shared interface families that connect the nine cells."""
    cell_by_name = dict(iter_grid_periodic_cells(grid))
    for path, members in _INTERFACE_FAMILIES:
        dimensions_by_member: list[tuple[str, tuple[int, ...]]] = []
        skip_family = False
        for cell_name, role in members:
            cell = cell_by_name[cell_name]
            if len(grid_periodic_boundary_tensors(cell, role=role)) != 1:
                skip_family = True
                break
            ports = build_grid_periodic_interface_ports(
                cell,
                cell_name=cell_name,
                role=role,
            )
            dimensions_by_member.append(
                (
                    f"{cell_name.value}.{role.value}",
                    tuple(port.dimension for port in ports),
                )
            )
        if skip_family or not dimensions_by_member:
            continue
        reference_dimensions = dimensions_by_member[0][1]
        if any(
            dimensions != reference_dimensions
            for _, dimensions in dimensions_by_member[1:]
        ):
            append_issue(
                issues,
                code="grid-periodic-interface-mismatch",
                message=(
                    "Grid periodic interface mismatch for "
                    + ", ".join(
                        f"{member_label}={dimensions}"
                        for member_label, dimensions in dimensions_by_member
                    )
                    + "."
                ),
                path=path,
            )


def _grid_periodic_cell_prefix(cell_name: GridPeriodicCellName) -> str:
    """Return the validation path prefix for one grid periodic cell."""
    return f"grid_periodic_grid.{cell_name.value}_cell"

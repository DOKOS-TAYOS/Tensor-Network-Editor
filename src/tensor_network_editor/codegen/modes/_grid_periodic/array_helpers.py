"""Shared array-rendering helpers for periodic-grid code generation."""

from __future__ import annotations

from ....internal.modes._grid_periodic import (
    GridPeriodicInterfacePort,
    build_grid_periodic_interface_ports,
)
from ....models import (
    GridPeriodicCellName,
    GridPeriodicTensorRole,
    LinearPeriodicCellSpec,
)
from ...shared.common import PreparedNetwork
from .shared import GRID_PERIODIC_CELL_ORDER

_GRID_CELL_KIND_OFFSET: dict[GridPeriodicCellName, int] = {
    cell_name: offset for offset, cell_name in enumerate(GRID_PERIODIC_CELL_ORDER)
}


def _build_ports_by_role(
    *,
    cell: LinearPeriodicCellSpec,
    cell_name: GridPeriodicCellName,
) -> dict[GridPeriodicTensorRole, tuple[GridPeriodicInterfacePort, ...]]:
    """Build every interface port family for one grid cell."""
    return {
        role: build_grid_periodic_interface_ports(
            cell,
            cell_name=cell_name,
            role=role,
        )
        for role in GridPeriodicTensorRole
    }


def _build_interface_slot_by_label(
    *,
    prepared: PreparedNetwork,
    ports_by_role: dict[GridPeriodicTensorRole, tuple[GridPeriodicInterfacePort, ...]],
) -> dict[str, tuple[GridPeriodicTensorRole, int]]:
    """Return the interface role/slot metadata for each prepared label."""
    prepared_label_by_index_id = {
        index.spec.id: index.label
        for tensor in prepared.tensors
        for index in tensor.indices
    }
    interface_slot_by_label: dict[str, tuple[GridPeriodicTensorRole, int]] = {}
    for role, ports in ports_by_role.items():
        for slot_index, port in enumerate(ports):
            internal_index_id = port.internal_index_id
            if internal_index_id not in prepared_label_by_index_id:
                continue
            interface_slot_by_label[prepared_label_by_index_id[internal_index_id]] = (
                role,
                slot_index,
            )
    return interface_slot_by_label


def _build_local_label_offsets(
    *,
    prepared: PreparedNetwork,
    interface_slot_by_label: dict[str, tuple[GridPeriodicTensorRole, int]],
) -> dict[str, int]:
    """Assign stable per-cell offsets to non-interface labels."""
    return {
        label: offset
        for offset, label in enumerate(
            dict.fromkeys(
                index.label
                for tensor in prepared.tensors
                for index in tensor.indices
                if index.label not in interface_slot_by_label
            )
        )
    }


def _quimb_interface_expression(
    *,
    role: GridPeriodicTensorRole,
    slot_index: int,
    column_expression: str,
    row_expression: str,
) -> str:
    """Render one runtime ``quimb`` label expression for an interface slot."""
    if role is GridPeriodicTensorRole.LEFT:
        return f"horizontal_label(({column_expression}) - 1, {row_expression}, {slot_index})"
    if role is GridPeriodicTensorRole.RIGHT:
        return f"horizontal_label({column_expression}, {row_expression}, {slot_index})"
    if role is GridPeriodicTensorRole.UP:
        return (
            f"vertical_label({column_expression}, ({row_expression}) - 1, {slot_index})"
        )
    return f"vertical_label({column_expression}, {row_expression}, {slot_index})"


def _einsum_interface_expression(
    *,
    role: GridPeriodicTensorRole,
    slot_index: int,
    column_expression: str,
    row_expression: str,
) -> str:
    """Render one runtime integer-label expression for an interface slot."""
    return _quimb_interface_expression(
        role=role,
        slot_index=slot_index,
        column_expression=column_expression,
        row_expression=row_expression,
    )


def _runtime_cell_coordinate_expressions(
    cell_name: GridPeriodicCellName,
) -> tuple[str, str]:
    """Return the runtime ``(column, row)`` expressions for one helper."""
    if cell_name is GridPeriodicCellName.TOP_LEFT:
        return "0", "0"
    if cell_name is GridPeriodicCellName.TOP:
        return "column_index", "0"
    if cell_name is GridPeriodicCellName.TOP_RIGHT:
        return "column_index", "0"
    if cell_name is GridPeriodicCellName.LEFT:
        return "0", "row_index"
    if cell_name is GridPeriodicCellName.CENTER:
        return "column_index", "row_index"
    if cell_name is GridPeriodicCellName.RIGHT:
        return "column_index", "row_index"
    if cell_name is GridPeriodicCellName.BOTTOM_LEFT:
        return "0", "row_index"
    if cell_name is GridPeriodicCellName.BOTTOM:
        return "column_index", "row_index"
    return "column_index", "row_index"

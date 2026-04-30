"""Shared cell-preparation helpers for grid-periodic array codegen."""

from __future__ import annotations

from dataclasses import dataclass

from ....internal.modes._grid_periodic import (
    GridPeriodicInterfacePort,
    build_internal_grid_periodic_cell_network,
)
from ....models import (
    GridPeriodicCellName,
    GridPeriodicGridSpec,
    GridPeriodicTensorRole,
    TensorCollectionFormat,
)
from ...shared.common import (
    PreparedNetwork,
    container_name_for_format,
    prepare_network,
    render_tensor_collection_assignment,
    render_tensor_collection_initialization,
)
from .array_helpers import _build_ports_by_role
from .common import _cell_from_grid


@dataclass(slots=True, frozen=True)
class GridArrayCellContext:
    """Prepared render context shared by grid-periodic array helper builders."""

    prepared: PreparedNetwork
    collection_format: TensorCollectionFormat
    collection_name: str
    ports_by_role: dict[GridPeriodicTensorRole, tuple[GridPeriodicInterfacePort, ...]]
    interface_index_ids: frozenset[str]


def build_grid_array_cell_context(
    *,
    grid: GridPeriodicGridSpec,
    cell_name: GridPeriodicCellName,
    collection_format: TensorCollectionFormat,
) -> GridArrayCellContext:
    """Build the shared prepared context for one array-backed grid cell helper."""
    cell = _cell_from_grid(grid, cell_name)
    prepared = prepare_network(
        build_internal_grid_periodic_cell_network(
            cell,
            cell_name=cell_name,
            include_contraction_plan=False,
        )
    )
    ports_by_role = _build_ports_by_role(cell=cell, cell_name=cell_name)
    return GridArrayCellContext(
        prepared=prepared,
        collection_format=collection_format,
        collection_name=container_name_for_format(collection_format),
        ports_by_role=ports_by_role,
        interface_index_ids=frozenset(
            port.internal_index_id for ports in ports_by_role.values() for port in ports
        ),
    )


def render_grid_array_tensor_sections(
    *,
    context: GridArrayCellContext,
    tensor_value_by_id: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Render the shared tensor collection sections for one grid cell helper."""
    tensor_collection_lines = render_tensor_collection_initialization(
        context.collection_name,
        context.collection_format,
    )
    tensor_construction_lines = render_tensor_collection_assignment(
        collection_name=context.collection_name,
        collection_format=context.collection_format,
        prepared=context.prepared,
        tensor_value_by_id=tensor_value_by_id,
        include_initialization=False,
    )
    return tensor_collection_lines, tensor_construction_lines

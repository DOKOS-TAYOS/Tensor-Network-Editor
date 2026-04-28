"""Shared helpers for the typed bidimensional periodic-grid editor mode."""

from __future__ import annotations

from dataclasses import dataclass

from ...models import (
    CanvasNoteSpec,
    EdgeSpec,
    GridPeriodicCellName,
    GridPeriodicGridSpec,
    GridPeriodicTensorRole,
    GroupSpec,
    IndexSpec,
    LinearPeriodicCellSpec,
    NetworkSpec,
    TensorSpec,
)
from ..analysis._analysis import analyze_network
from ._common import remap_analysis_edge_endpoint

GRID_PERIODIC_UP_OPERAND_ID = "__grid_up__"
GRID_PERIODIC_RIGHT_OPERAND_ID = "__grid_right__"
GRID_PERIODIC_DOWN_OPERAND_ID = "__grid_down__"
GRID_PERIODIC_LEFT_OPERAND_ID = "__grid_left__"
GRID_PERIODIC_RESERVED_OPERAND_ID_BY_ROLE: dict[GridPeriodicTensorRole, str] = {
    GridPeriodicTensorRole.UP: GRID_PERIODIC_UP_OPERAND_ID,
    GridPeriodicTensorRole.RIGHT: GRID_PERIODIC_RIGHT_OPERAND_ID,
    GridPeriodicTensorRole.DOWN: GRID_PERIODIC_DOWN_OPERAND_ID,
    GridPeriodicTensorRole.LEFT: GRID_PERIODIC_LEFT_OPERAND_ID,
}
GRID_PERIODIC_RESERVED_OPERAND_IDS = frozenset(
    GRID_PERIODIC_RESERVED_OPERAND_ID_BY_ROLE.values()
)


@dataclass(slots=True, frozen=True)
class GridPeriodicInterfacePort:
    """One connected slot on a 2D virtual boundary tensor."""

    boundary_tensor_id: str
    boundary_index_id: str
    boundary_index_name: str
    dimension: int
    internal_tensor_id: str
    internal_index_id: str
    internal_index_name: str


def iter_grid_periodic_cells(
    grid: GridPeriodicGridSpec,
) -> tuple[tuple[GridPeriodicCellName, LinearPeriodicCellSpec], ...]:
    """Return the nine grid cells in canonical reading order."""
    return (
        (GridPeriodicCellName.TOP_LEFT, grid.top_left_cell),
        (GridPeriodicCellName.TOP, grid.top_cell),
        (GridPeriodicCellName.TOP_RIGHT, grid.top_right_cell),
        (GridPeriodicCellName.LEFT, grid.left_cell),
        (GridPeriodicCellName.CENTER, grid.center_cell),
        (GridPeriodicCellName.RIGHT, grid.right_cell),
        (GridPeriodicCellName.BOTTOM_LEFT, grid.bottom_left_cell),
        (GridPeriodicCellName.BOTTOM, grid.bottom_cell),
        (GridPeriodicCellName.BOTTOM_RIGHT, grid.bottom_right_cell),
    )


def grid_periodic_cell_as_network(
    cell: LinearPeriodicCellSpec,
    *,
    cell_name: GridPeriodicCellName,
) -> NetworkSpec:
    """Wrap one cell as a regular ``NetworkSpec`` for shared analysis helpers."""
    return NetworkSpec(
        id=f"grid_periodic_{cell_name.value}",
        name=f"grid_periodic_{cell_name.value}",
        tensors=list(cell.tensors),
        groups=list(cell.groups),
        edges=list(cell.edges),
        notes=list(cell.notes),
        contraction_plan=cell.contraction_plan,
        metadata=dict(cell.metadata),
    )


def grid_periodic_active_cell_as_analysis_network(
    grid: GridPeriodicGridSpec,
) -> NetworkSpec:
    """Wrap the active grid cell as a regular network for path analysis."""
    cell_by_name = dict(iter_grid_periodic_cells(grid))
    cell_name = grid.active_cell
    return grid_periodic_cell_as_analysis_network(
        cell_by_name[cell_name],
        cell_name=cell_name,
    )


def grid_periodic_cell_as_analysis_network(
    cell: LinearPeriodicCellSpec,
    *,
    cell_name: GridPeriodicCellName,
) -> NetworkSpec:
    """Return a cell network where virtual boundaries are plain operands."""
    tensor_id_by_original_id = {
        tensor.id: _analysis_tensor_id(tensor) for tensor in cell.tensors
    }
    tensor_ids = set(tensor_id_by_original_id.values())
    groups = [
        GroupSpec(
            id=group.id,
            name=group.name,
            tensor_ids=[
                tensor_id
                for tensor_id in dict.fromkeys(
                    tensor_id_by_original_id[tensor_id]
                    for tensor_id in group.tensor_ids
                    if tensor_id in tensor_id_by_original_id
                )
                if tensor_id in tensor_ids
            ],
            metadata=dict(group.metadata),
        )
        for group in cell.groups
    ]

    return NetworkSpec(
        id=f"grid_periodic_analysis_{cell_name.value}",
        name=f"grid_periodic_analysis_{cell_name.value}",
        tensors=[
            TensorSpec(
                id=tensor_id_by_original_id[tensor.id],
                name=tensor.name,
                position=tensor.position,
                size=tensor.size,
                indices=[
                    IndexSpec(
                        id=index.id,
                        name=index.name,
                        dimension=index.dimension,
                        offset=index.offset,
                        metadata=dict(index.metadata),
                    )
                    for index in tensor.indices
                ],
                linear_periodic_role=None,
                grid_periodic_role=None,
                metadata=dict(tensor.metadata),
            )
            for tensor in cell.tensors
        ],
        groups=[group for group in groups if group.tensor_ids],
        edges=[
            EdgeSpec(
                id=edge.id,
                name=edge.name,
                left=remap_analysis_edge_endpoint(
                    edge.left,
                    tensor_id_by_original_id=tensor_id_by_original_id,
                ),
                right=remap_analysis_edge_endpoint(
                    edge.right,
                    tensor_id_by_original_id=tensor_id_by_original_id,
                ),
                metadata=dict(edge.metadata),
            )
            for edge in cell.edges
        ],
        notes=[
            CanvasNoteSpec(
                id=note.id,
                text=note.text,
                position=note.position,
                metadata=dict(note.metadata),
            )
            for note in cell.notes
        ],
        contraction_plan=cell.contraction_plan,
        metadata=dict(cell.metadata),
    )


def build_internal_grid_periodic_cell_network(
    cell: LinearPeriodicCellSpec,
    *,
    cell_name: GridPeriodicCellName,
    include_contraction_plan: bool = True,
) -> NetworkSpec:
    """Return the cell network without editor-only virtual boundary tensors."""
    real_tensors = [
        tensor for tensor in cell.tensors if tensor.grid_periodic_role is None
    ]
    real_tensor_ids = {tensor.id for tensor in real_tensors}
    real_edges = [
        edge
        for edge in cell.edges
        if edge.left.tensor_id in real_tensor_ids
        and edge.right.tensor_id in real_tensor_ids
    ]
    real_groups = [
        GroupSpec(
            id=group.id,
            name=group.name,
            tensor_ids=[
                tensor_id
                for tensor_id in group.tensor_ids
                if tensor_id in real_tensor_ids
            ],
            metadata=dict(group.metadata),
        )
        for group in cell.groups
    ]
    return NetworkSpec(
        id=f"grid_periodic_internal_{cell_name.value}",
        name=f"grid_periodic_internal_{cell_name.value}",
        tensors=[
            TensorSpec(
                id=tensor.id,
                name=tensor.name,
                position=tensor.position,
                size=tensor.size,
                indices=list(tensor.indices),
                linear_periodic_role=None,
                grid_periodic_role=None,
                metadata=dict(tensor.metadata),
            )
            for tensor in real_tensors
        ],
        groups=real_groups,
        edges=list(real_edges),
        notes=[
            CanvasNoteSpec(
                id=note.id,
                text=note.text,
                position=note.position,
                metadata=dict(note.metadata),
            )
            for note in cell.notes
        ],
        contraction_plan=cell.contraction_plan if include_contraction_plan else None,
        metadata=dict(cell.metadata),
    )


def grid_periodic_boundary_tensors(
    cell: LinearPeriodicCellSpec,
    *,
    role: GridPeriodicTensorRole,
) -> list[TensorSpec]:
    """Return all boundary tensors in ``cell`` with the requested role."""
    return [tensor for tensor in cell.tensors if tensor.grid_periodic_role is role]


def build_grid_periodic_interface_ports(
    cell: LinearPeriodicCellSpec,
    *,
    cell_name: GridPeriodicCellName,
    role: GridPeriodicTensorRole,
) -> tuple[GridPeriodicInterfacePort, ...]:
    """Return connected virtual-boundary slots in stable tensor-index order."""
    boundary_tensors = grid_periodic_boundary_tensors(cell, role=role)
    if len(boundary_tensors) != 1:
        return ()

    boundary_tensor = boundary_tensors[0]
    analysis = analyze_network(grid_periodic_cell_as_network(cell, cell_name=cell_name))
    internal_endpoint_by_boundary_index_id: dict[str, tuple[TensorSpec, IndexSpec]] = {}

    for edge in cell.edges:
        left_item = analysis.index_map.get(edge.left.index_id)
        right_item = analysis.index_map.get(edge.right.index_id)
        if left_item is None or right_item is None:
            continue

        left_tensor, left_index = left_item
        right_tensor, right_index = right_item

        if (
            left_tensor.id == boundary_tensor.id
            and right_tensor.grid_periodic_role is None
        ):
            internal_endpoint_by_boundary_index_id[left_index.id] = (
                right_tensor,
                right_index,
            )
        elif (
            right_tensor.id == boundary_tensor.id
            and left_tensor.grid_periodic_role is None
        ):
            internal_endpoint_by_boundary_index_id[right_index.id] = (
                left_tensor,
                left_index,
            )

    ports: list[GridPeriodicInterfacePort] = []
    for boundary_index in boundary_tensor.indices:
        internal_endpoint = internal_endpoint_by_boundary_index_id.get(
            boundary_index.id
        )
        if internal_endpoint is None:
            continue
        internal_tensor, internal_index = internal_endpoint
        ports.append(
            GridPeriodicInterfacePort(
                boundary_tensor_id=boundary_tensor.id,
                boundary_index_id=boundary_index.id,
                boundary_index_name=boundary_index.name,
                dimension=boundary_index.dimension,
                internal_tensor_id=internal_tensor.id,
                internal_index_id=internal_index.id,
                internal_index_name=internal_index.name,
            )
        )
    return tuple(ports)


def grid_periodic_reserved_operand_id_for_role(
    role: GridPeriodicTensorRole,
) -> str:
    """Return the planner operand id used for one 2D boundary role."""
    return GRID_PERIODIC_RESERVED_OPERAND_ID_BY_ROLE[role]


def is_grid_periodic_reserved_operand_id(operand_id: str) -> bool:
    """Return ``True`` when ``operand_id`` is a reserved 2D boundary operand."""
    return operand_id in GRID_PERIODIC_RESERVED_OPERAND_IDS


def _analysis_tensor_id(tensor: TensorSpec) -> str:
    """Return the analysis operand id for a grid-periodic cell tensor."""
    if tensor.grid_periodic_role is None:
        return tensor.id
    return grid_periodic_reserved_operand_id_for_role(tensor.grid_periodic_role)

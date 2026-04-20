"""Shared helpers for the typed tree periodic editor mode."""

from __future__ import annotations

from dataclasses import dataclass

from ...models import (
    CanvasNoteSpec,
    GroupSpec,
    IndexSpec,
    LinearPeriodicCellSpec,
    NetworkSpec,
    TensorSpec,
    TreePeriodicCellName,
    TreePeriodicTensorRole,
    TreePeriodicTreeSpec,
)
from ..analysis._analysis import analyze_network


@dataclass(slots=True, frozen=True)
class TreePeriodicInterfacePort:
    """One connected slot on a tree virtual-boundary tensor."""

    boundary_tensor_id: str
    boundary_index_id: str
    boundary_index_name: str
    dimension: int
    internal_tensor_id: str
    internal_index_id: str
    internal_index_name: str
    child_index: int | None


def iter_tree_periodic_cells(
    tree: TreePeriodicTreeSpec,
) -> tuple[tuple[TreePeriodicCellName, LinearPeriodicCellSpec], ...]:
    """Return the three tree cells in canonical top-down order."""
    return (
        (TreePeriodicCellName.ROOT, tree.root_cell),
        (TreePeriodicCellName.BRANCH, tree.branch_cell),
        (TreePeriodicCellName.LEAF, tree.leaf_cell),
    )


def tree_periodic_cell_as_network(
    cell: LinearPeriodicCellSpec,
    *,
    cell_name: TreePeriodicCellName,
) -> NetworkSpec:
    """Wrap one tree cell as a regular ``NetworkSpec``."""
    return NetworkSpec(
        id=f"tree_periodic_{cell_name.value}",
        name=f"tree_periodic_{cell_name.value}",
        tensors=list(cell.tensors),
        groups=list(cell.groups),
        edges=list(cell.edges),
        notes=list(cell.notes),
        contraction_plan=cell.contraction_plan,
        metadata=dict(cell.metadata),
    )


def tree_periodic_active_cell_as_analysis_network(
    tree: TreePeriodicTreeSpec,
) -> NetworkSpec:
    """Wrap the active tree cell as a regular network for shared analysis."""
    cell_by_name = dict(iter_tree_periodic_cells(tree))
    cell_name = tree.active_cell
    return tree_periodic_cell_as_network(cell_by_name[cell_name], cell_name=cell_name)


def build_internal_tree_periodic_cell_network(
    cell: LinearPeriodicCellSpec,
    *,
    cell_name: TreePeriodicCellName,
    include_contraction_plan: bool = True,
) -> NetworkSpec:
    """Return the cell network without editor-only virtual boundary tensors."""
    real_tensors = [
        tensor for tensor in cell.tensors if tensor.tree_periodic_role is None
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
        id=f"tree_periodic_internal_{cell_name.value}",
        name=f"tree_periodic_internal_{cell_name.value}",
        tensors=[
            TensorSpec(
                id=tensor.id,
                name=tensor.name,
                position=tensor.position,
                size=tensor.size,
                indices=list(tensor.indices),
                linear_periodic_role=None,
                grid_periodic_role=None,
                tree_periodic_role=None,
                tree_periodic_child_index=None,
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


def tree_periodic_boundary_tensors(
    cell: LinearPeriodicCellSpec,
    *,
    role: TreePeriodicTensorRole,
    child_index: int | None = None,
) -> list[TensorSpec]:
    """Return all tree boundary tensors in ``cell`` with the requested role."""
    tensors = [tensor for tensor in cell.tensors if tensor.tree_periodic_role is role]
    if role is TreePeriodicTensorRole.CHILD and child_index is not None:
        return [
            tensor
            for tensor in tensors
            if tensor.tree_periodic_child_index == child_index
        ]
    return tensors


def build_tree_periodic_interface_ports(
    cell: LinearPeriodicCellSpec,
    *,
    cell_name: TreePeriodicCellName,
    role: TreePeriodicTensorRole,
    child_index: int | None = None,
) -> tuple[TreePeriodicInterfacePort, ...]:
    """Return connected tree-boundary slots in stable tensor-index order."""
    boundary_tensors = tree_periodic_boundary_tensors(
        cell,
        role=role,
        child_index=child_index,
    )
    if len(boundary_tensors) != 1:
        return ()

    boundary_tensor = boundary_tensors[0]
    analysis = analyze_network(tree_periodic_cell_as_network(cell, cell_name=cell_name))
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
            and right_tensor.tree_periodic_role is None
        ):
            internal_endpoint_by_boundary_index_id[left_index.id] = (
                right_tensor,
                right_index,
            )
        elif (
            right_tensor.id == boundary_tensor.id
            and left_tensor.tree_periodic_role is None
        ):
            internal_endpoint_by_boundary_index_id[right_index.id] = (
                left_tensor,
                left_index,
            )

    ports: list[TreePeriodicInterfacePort] = []
    for boundary_index in boundary_tensor.indices:
        internal_endpoint = internal_endpoint_by_boundary_index_id.get(
            boundary_index.id
        )
        if internal_endpoint is None:
            continue
        internal_tensor, internal_index = internal_endpoint
        ports.append(
            TreePeriodicInterfacePort(
                boundary_tensor_id=boundary_tensor.id,
                boundary_index_id=boundary_index.id,
                boundary_index_name=boundary_index.name,
                dimension=boundary_index.dimension,
                internal_tensor_id=internal_tensor.id,
                internal_index_id=internal_index.id,
                internal_index_name=internal_index.name,
                child_index=boundary_tensor.tree_periodic_child_index,
            )
        )
    return tuple(ports)

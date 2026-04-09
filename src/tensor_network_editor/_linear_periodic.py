"""Shared helpers for the typed linear periodic-chain editor mode."""

from __future__ import annotations

from dataclasses import dataclass

from ._analysis import analyze_network
from .models import (
    CanvasNoteSpec,
    ContractionStepSpec,
    GroupSpec,
    IndexSpec,
    LinearPeriodicCellName,
    LinearPeriodicCellSpec,
    LinearPeriodicChainSpec,
    LinearPeriodicTensorRole,
    NetworkSpec,
    TensorSpec,
)

LINEAR_PERIODIC_PREVIOUS_OPERAND_ID = "__linear_previous__"
LINEAR_PERIODIC_NEXT_OPERAND_ID = "__linear_next__"
LINEAR_PERIODIC_RESERVED_OPERAND_IDS = frozenset(
    {
        LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
        LINEAR_PERIODIC_NEXT_OPERAND_ID,
    }
)


@dataclass(slots=True, frozen=True)
class LinearPeriodicInterfacePort:
    """One connected slot on a virtual boundary tensor."""

    boundary_tensor_id: str
    boundary_index_id: str
    boundary_index_name: str
    dimension: int
    internal_tensor_id: str
    internal_index_id: str
    internal_index_name: str


def iter_linear_periodic_cells(
    chain: LinearPeriodicChainSpec,
) -> tuple[tuple[LinearPeriodicCellName, LinearPeriodicCellSpec], ...]:
    """Return the three periodic-chain cells in their canonical order."""
    return (
        (LinearPeriodicCellName.INITIAL, chain.initial_cell),
        (LinearPeriodicCellName.PERIODIC, chain.periodic_cell),
        (LinearPeriodicCellName.FINAL, chain.final_cell),
    )


def linear_periodic_cell_as_network(
    cell: LinearPeriodicCellSpec,
    *,
    cell_name: LinearPeriodicCellName,
) -> NetworkSpec:
    """Wrap one cell as a regular ``NetworkSpec`` for shared analysis helpers."""
    return NetworkSpec(
        id=f"linear_periodic_{cell_name.value}",
        name=f"linear_periodic_{cell_name.value}",
        tensors=list(cell.tensors),
        groups=list(cell.groups),
        edges=list(cell.edges),
        notes=list(cell.notes),
        contraction_plan=cell.contraction_plan,
        metadata=dict(cell.metadata),
    )


def build_internal_linear_periodic_cell_network(
    cell: LinearPeriodicCellSpec,
    *,
    cell_name: LinearPeriodicCellName,
    include_contraction_plan: bool = True,
) -> NetworkSpec:
    """Return the cell network without editor-only virtual boundary tensors."""
    real_tensors = [
        tensor for tensor in cell.tensors if tensor.linear_periodic_role is None
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
        id=f"linear_periodic_internal_{cell_name.value}",
        name=f"linear_periodic_internal_{cell_name.value}",
        tensors=[
            TensorSpec(
                id=tensor.id,
                name=tensor.name,
                position=tensor.position,
                size=tensor.size,
                indices=list(tensor.indices),
                linear_periodic_role=None,
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


def linear_periodic_boundary_tensors(
    cell: LinearPeriodicCellSpec,
    *,
    role: LinearPeriodicTensorRole,
) -> list[TensorSpec]:
    """Return all boundary tensors in ``cell`` with the requested role."""
    return [tensor for tensor in cell.tensors if tensor.linear_periodic_role is role]


def build_linear_periodic_interface_ports(
    cell: LinearPeriodicCellSpec,
    *,
    cell_name: LinearPeriodicCellName,
    role: LinearPeriodicTensorRole,
) -> tuple[LinearPeriodicInterfacePort, ...]:
    """Return connected virtual-boundary slots in stable tensor-index order."""
    boundary_tensors = linear_periodic_boundary_tensors(cell, role=role)
    if len(boundary_tensors) != 1:
        return ()

    boundary_tensor = boundary_tensors[0]
    analysis = analyze_network(
        linear_periodic_cell_as_network(cell, cell_name=cell_name)
    )
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
            and right_tensor.linear_periodic_role is None
        ):
            internal_endpoint_by_boundary_index_id[left_index.id] = (
                right_tensor,
                right_index,
            )
        elif (
            right_tensor.id == boundary_tensor.id
            and left_tensor.linear_periodic_role is None
        ):
            internal_endpoint_by_boundary_index_id[right_index.id] = (
                left_tensor,
                left_index,
            )

    ports: list[LinearPeriodicInterfacePort] = []
    for boundary_index in boundary_tensor.indices:
        internal_endpoint = internal_endpoint_by_boundary_index_id.get(
            boundary_index.id
        )
        if internal_endpoint is None:
            continue
        internal_tensor, internal_index = internal_endpoint
        ports.append(
            LinearPeriodicInterfacePort(
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


def is_linear_periodic_reserved_operand_id(operand_id: str) -> bool:
    """Return ``True`` when ``operand_id`` is a reserved carry-mode operand."""
    return operand_id in LINEAR_PERIODIC_RESERVED_OPERAND_IDS


def linear_periodic_step_uses_reserved_operand(
    step: ContractionStepSpec,
) -> bool:
    """Return ``True`` when a step references ``previous`` or ``next``."""
    return is_linear_periodic_reserved_operand_id(
        step.left_operand_id
    ) or is_linear_periodic_reserved_operand_id(step.right_operand_id)


def linear_periodic_cell_uses_carry_mode(cell: LinearPeriodicCellSpec) -> bool:
    """Return ``True`` when a cell plan references carry-mode operands."""
    if cell.contraction_plan is None:
        return False
    return any(
        linear_periodic_step_uses_reserved_operand(step)
        for step in cell.contraction_plan.steps
    )


def linear_periodic_chain_uses_carry_mode(chain: LinearPeriodicChainSpec) -> bool:
    """Return ``True`` when any cell in the chain uses carry-mode operands."""
    return any(
        linear_periodic_cell_uses_carry_mode(cell)
        for _, cell in iter_linear_periodic_cells(chain)
    )

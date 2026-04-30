"""Shared cell-preparation helpers for tree-periodic array codegen."""

from __future__ import annotations

from dataclasses import dataclass

from ....internal.modes._tree_periodic import (
    TreePeriodicInterfacePort,
    build_internal_tree_periodic_cell_network,
    build_tree_periodic_interface_ports,
)
from ....models import (
    TensorCollectionFormat,
    TreePeriodicCellName,
    TreePeriodicTensorRole,
    TreePeriodicTreeSpec,
)
from ...shared.common import (
    PreparedNetwork,
    container_name_for_format,
    prepare_network,
    render_tensor_collection_assignment,
    render_tensor_collection_initialization,
)
from .common import _build_child_ports_by_index, _cell_from_tree


@dataclass(slots=True, frozen=True)
class TreeArrayCellContext:
    """Prepared render context shared by tree-periodic array helper builders."""

    prepared: PreparedNetwork
    collection_format: TensorCollectionFormat
    collection_name: str
    parent_ports: tuple[TreePeriodicInterfacePort, ...]
    child_ports_by_index: dict[int, tuple[TreePeriodicInterfacePort, ...]]
    interface_index_ids: frozenset[str]


def build_tree_array_cell_context(
    *,
    tree: TreePeriodicTreeSpec,
    cell_name: TreePeriodicCellName,
    collection_format: TensorCollectionFormat,
) -> TreeArrayCellContext:
    """Build the shared prepared context for one array-backed tree cell helper."""
    cell = _cell_from_tree(tree, cell_name)
    prepared = prepare_network(
        build_internal_tree_periodic_cell_network(
            cell,
            cell_name=cell_name,
            include_contraction_plan=False,
        )
    )
    parent_ports = build_tree_periodic_interface_ports(
        cell,
        cell_name=cell_name,
        role=TreePeriodicTensorRole.PARENT,
    )
    child_ports_by_index = _build_child_ports_by_index(
        tree=tree,
        cell=cell,
        cell_name=cell_name,
    )
    interface_index_ids = frozenset(
        {port.internal_index_id for port in parent_ports}
        | {
            port.internal_index_id
            for ports in child_ports_by_index.values()
            for port in ports
        }
    )
    return TreeArrayCellContext(
        prepared=prepared,
        collection_format=collection_format,
        collection_name=container_name_for_format(collection_format),
        parent_ports=parent_ports,
        child_ports_by_index=child_ports_by_index,
        interface_index_ids=interface_index_ids,
    )


def render_tree_array_tensor_sections(
    *,
    context: TreeArrayCellContext,
    tensor_value_by_id: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Render the shared tensor collection sections for one tree cell helper."""
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

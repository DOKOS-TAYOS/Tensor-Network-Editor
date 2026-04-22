"""Common tree-periodic helpers shared across backends."""

from __future__ import annotations

from ....internal.modes._tree_periodic import (
    TreePeriodicInterfacePort,
    build_tree_periodic_interface_ports,
)
from ....models import (
    LinearPeriodicCellSpec,
    TreePeriodicCellName,
    TreePeriodicTensorRole,
    TreePeriodicTreeSpec,
)

_TREE_CELL_KIND_OFFSET: dict[TreePeriodicCellName, int] = {
    TreePeriodicCellName.ROOT: 0,
    TreePeriodicCellName.BRANCH: 1,
    TreePeriodicCellName.LEAF: 2,
}


def _cell_from_tree(
    tree: TreePeriodicTreeSpec,
    cell_name: TreePeriodicCellName,
) -> LinearPeriodicCellSpec:
    """Return the matching cell from ``tree``."""
    if cell_name is TreePeriodicCellName.ROOT:
        return tree.root_cell
    if cell_name is TreePeriodicCellName.BRANCH:
        return tree.branch_cell
    return tree.leaf_cell


def _build_child_ports_by_index(
    *,
    tree: TreePeriodicTreeSpec,
    cell: LinearPeriodicCellSpec,
    cell_name: TreePeriodicCellName,
) -> dict[int, tuple[TreePeriodicInterfacePort, ...]]:
    """Build every child-interface family for one tree cell."""
    return {
        child_index: build_tree_periodic_interface_ports(
            cell,
            cell_name=cell_name,
            role=TreePeriodicTensorRole.CHILD,
            child_index=child_index,
        )
        for child_index in range(tree.branching_factor)
    }


def _render_parent_interface_validation(
    parent_ports: tuple[TreePeriodicInterfacePort, ...],
) -> list[str]:
    """Render the runtime validation for non-root parent interfaces."""
    if not parent_ports:
        return []
    return [
        f"if len(parent_interface) != {len(parent_ports)}:",
        "    raise ValueError('The provided parent interface does not match this tree cell.')",
    ]


def _runtime_coordinate_expressions(
    cell_name: TreePeriodicCellName,
) -> tuple[str, str]:
    """Return the runtime ``(level, node_index)`` expressions for one helper."""
    if cell_name is TreePeriodicCellName.ROOT:
        return "0", "0"
    return "level", "node_index"

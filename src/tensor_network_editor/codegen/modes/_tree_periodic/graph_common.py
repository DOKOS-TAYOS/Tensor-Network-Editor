"""Shared graph orchestration for tree-periodic code generation."""

from __future__ import annotations

from ....models import (
    CodegenResult,
    EngineName,
    TensorCollectionFormat,
    TreePeriodicCellName,
    TreePeriodicTreeSpec,
)
from .common import _manual_plan_step_ids_for_tree, _render_partial_network_output_lines
from .graph_cells import _render_tree_graph_cell_helper
from .shared import (
    render_tree_periodic_script,
    render_tree_periodic_shared_helpers,
    tree_periodic_helper_name,
    tree_periodic_helper_signature,
)


def generate_graph_tree_periodic_code(
    *,
    tree: TreePeriodicTreeSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate helper-based tree-periodic code for graph backends."""
    connect_call = "tn.connect" if engine is EngineName.TENSORNETWORK else "tk.connect"
    import_lines = (
        ["import numpy as np", "import tensornetwork as tn"]
        if engine is EngineName.TENSORNETWORK
        else ["import torch", "import tensorkrowch as tk"]
    )
    cell_lines_by_name = {
        cell_name: _render_tree_graph_cell_helper(
            tree=tree,
            cell_name=cell_name,
            helper_name=tree_periodic_helper_name(cell_name),
            helper_signature=tree_periodic_helper_signature(
                cell_name,
                interface_annotation="list[object]",
            ),
            engine=engine,
            collection_format=collection_format,
        ).lines
        for cell_name in (
            TreePeriodicCellName.ROOT,
            TreePeriodicCellName.BRANCH,
            TreePeriodicCellName.LEAF,
        )
    }
    manual_step_ids = _manual_plan_step_ids_for_tree(tree)
    main_loop_lines = [
        "validate_tree_depth(n)",
        "root_cell = build_root_cell()",
        "network_nodes = list(root_cell['nodes'])",
        "open_edges = list(root_cell['open_edges'])",
        "frontier = list(root_cell['child_interfaces'])",
        "",
        "for level in range(1, n - 1):",
        "    next_frontier: list[list[object]] = []",
        "    for node_index, parent_interface in enumerate(frontier):",
        "        branch_cell = build_branch_cell(level, node_index, parent_interface)",
        "        connect_tree_interfaces(parent_interface, branch_cell['parent_interface'])",
        "        network_nodes.extend(branch_cell['nodes'])",
        "        open_edges.extend(branch_cell['open_edges'])",
        "        next_frontier.extend(branch_cell['child_interfaces'])",
        "    frontier = next_frontier",
        "",
        "for node_index, parent_interface in enumerate(frontier):",
        "    leaf_cell = build_leaf_cell(n - 1, node_index, parent_interface)",
        "    connect_tree_interfaces(parent_interface, leaf_cell['parent_interface'])",
        "    network_nodes.extend(leaf_cell['nodes'])",
        "    open_edges.extend(leaf_cell['open_edges'])",
    ]
    if manual_step_ids:
        main_loop_lines.extend(_render_tree_bottom_up_marker_lines())
    output_lines = (
        _render_partial_network_output_lines(
            operand_expression="network_nodes",
            step_ids=manual_step_ids,
            key_prefix="tree_node",
            mode_message="Manual tree cell plans are assembled from leaves toward the root.",
        )
        if manual_step_ids
        else ["result = network_nodes[0] if len(network_nodes) == 1 else None"]
    )
    return CodegenResult(
        engine=engine,
        code=render_tree_periodic_script(
            import_lines=[
                *import_lines,
                "",
                "# Tensor Network Editor tree periodic mode",
            ],
            shared_helper_lines=render_tree_periodic_shared_helpers(
                extra_lines=[
                    f"branching_factor = {tree.branching_factor}",
                    "",
                    "def connect_tree_interfaces(parent_interface: list[object], child_interface: list[object]) -> list[object]:",
                    "    if len(parent_interface) != len(child_interface):",
                    "        raise ValueError('Tree parent and child interfaces must have the same size.')",
                    "    connections = []",
                    "    for parent_edge, child_edge in zip(parent_interface, child_interface, strict=True):",
                    f"        connections.append({connect_call}(parent_edge, child_edge))",
                    "    return connections",
                ]
            ),
            cell_lines_by_name=cell_lines_by_name,
            main_loop_lines=main_loop_lines,
            output_lines=output_lines,
        ),
    )


def _render_tree_bottom_up_marker_lines() -> list[str]:
    """Render the explicit bottom-up pass marker for manual tree plans."""
    return [
        "",
        "# Manual tree cell plans are assembled from leaves toward the root.",
        "for level in range(n - 1, 0, -1):",
        "    pass",
    ]

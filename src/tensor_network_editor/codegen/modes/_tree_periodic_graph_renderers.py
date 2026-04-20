"""Graph-backend renderers for tree periodic code generation."""

from __future__ import annotations

from ...internal.modes._linear_periodic import LinearPeriodicCellSpec
from ...internal.modes._tree_periodic import (
    TreePeriodicInterfacePort,
    build_internal_tree_periodic_cell_network,
    build_tree_periodic_interface_ports,
)
from ...models import (
    CodegenResult,
    EngineName,
    TensorCollectionFormat,
    TreePeriodicCellName,
    TreePeriodicTensorRole,
    TreePeriodicTreeSpec,
)
from ..shared._linear_periodic_expressions import _axis_names_for_engine
from ..shared.common import (
    CodeSection,
    PreparedNetwork,
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
    tensor_collection_reference,
)
from ._linear_periodic_graph_renderers import _render_cell_setup_sections
from ._tree_periodic_shared import (
    _RenderedTreeCellHelper,
    render_tree_periodic_helper,
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
    output_lines = ["result = network_nodes[0] if len(network_nodes) == 1 else None"]
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


def _render_tree_graph_cell_helper(
    *,
    tree: TreePeriodicTreeSpec,
    cell_name: TreePeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> _RenderedTreeCellHelper:
    cell = _cell_from_tree(tree, cell_name)
    prepared = prepare_network(
        build_internal_tree_periodic_cell_network(cell, cell_name=cell_name)
    )
    collection_name = container_name_for_format(collection_format)
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
    interface_index_ids = {port.internal_index_id for port in parent_ports} | {
        port.internal_index_id
        for ports in child_ports_by_index.values()
        for port in ports
    }
    (
        tensor_collection_lines,
        tensor_construction_lines,
        network_connection_lines,
    ) = _render_cell_setup_sections(
        prepared=prepared,
        engine=engine,
        collection_format=collection_format,
        collection_name=collection_name,
    )
    edge_expression_by_index_id = _build_edge_expression_by_index_id(
        prepared=prepared,
        engine=engine,
        collection_format=collection_format,
        collection_name=collection_name,
    )
    output_lines = _render_parent_interface_validation(parent_ports)
    output_lines.extend(
        [
            "network_nodes = "
            + flattened_tensor_collection_expression(
                collection_format, collection_name
            ),
            "parent_interface = "
            + _render_python_list_expression(
                [
                    edge_expression_by_index_id[port.internal_index_id]
                    for port in parent_ports
                    if port.internal_index_id in edge_expression_by_index_id
                ]
            ),
            *_render_child_interface_lines(
                child_ports_by_index=child_ports_by_index,
                edge_expression_by_index_id=edge_expression_by_index_id,
            ),
            "open_edges = "
            + _render_python_list_expression(
                [
                    edge_expression_by_index_id[index.spec.id]
                    for index in prepared.open_indices
                    if index.spec.id not in interface_index_ids
                    and index.spec.id in edge_expression_by_index_id
                ]
            ),
            "return {",
            "    'nodes': list(network_nodes),",
            "    'parent_interface': parent_interface,",
            "    'child_interfaces': child_interfaces,",
            "    'open_edges': open_edges,",
            "}",
        ]
    )
    return render_tree_periodic_helper(
        helper_name=helper_name,
        helper_signature=helper_signature,
        return_annotation="dict[str, object]",
        sections=[
            CodeSection(title="Tensor collection", lines=tensor_collection_lines),
            CodeSection(title="Tensor construction", lines=tensor_construction_lines),
            CodeSection(title="Network connections", lines=network_connection_lines),
            CodeSection(title="Outputs", lines=output_lines),
        ],
    )


def _cell_from_tree(
    tree: TreePeriodicTreeSpec,
    cell_name: TreePeriodicCellName,
) -> LinearPeriodicCellSpec:
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
    return {
        child_index: build_tree_periodic_interface_ports(
            cell,
            cell_name=cell_name,
            role=TreePeriodicTensorRole.CHILD,
            child_index=child_index,
        )
        for child_index in range(tree.branching_factor)
    }


def _build_edge_expression_by_index_id(
    *,
    prepared: PreparedNetwork,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> dict[str, str]:
    edge_expression_by_index_id: dict[str, str] = {}
    for tensor in prepared.tensors:
        tensor_reference = tensor_collection_reference(
            tensor,
            collection_format,
            collection_name,
        )
        runtime_axis_names = _axis_names_for_engine(
            engine,
            tuple(index.spec.name for index in tensor.indices),
        )
        for index, axis_name in zip(
            tensor.indices,
            runtime_axis_names,
            strict=True,
        ):
            edge_expression_by_index_id[index.spec.id] = (
                f"{tensor_reference}[{axis_name!r}]"
            )
    return edge_expression_by_index_id


def _render_parent_interface_validation(
    parent_ports: tuple[TreePeriodicInterfacePort, ...],
) -> list[str]:
    if not parent_ports:
        return []
    return [
        f"if len(parent_interface) != {len(parent_ports)}:",
        "    raise ValueError('The provided parent interface does not match this tree cell.')",
    ]


def _render_child_interface_lines(
    *,
    child_ports_by_index: dict[int, tuple[TreePeriodicInterfacePort, ...]],
    edge_expression_by_index_id: dict[str, str],
) -> list[str]:
    lines = ["child_interfaces = []"]
    for child_index in sorted(child_ports_by_index):
        child_interface_expression = _render_python_list_expression(
            [
                edge_expression_by_index_id[port.internal_index_id]
                for port in child_ports_by_index[child_index]
                if port.internal_index_id in edge_expression_by_index_id
            ]
        )
        lines.append(f"child_interfaces.append({child_interface_expression})")
    return lines


def _render_python_list_expression(values: list[str]) -> str:
    return "[" + ", ".join(values) + "]"

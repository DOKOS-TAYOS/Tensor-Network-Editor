"""Graph cell renderers for tree-periodic code generation."""

from __future__ import annotations

from ....internal.modes._tree_periodic import (
    TreePeriodicTensorRole,
    build_internal_tree_periodic_cell_network,
    build_tree_periodic_interface_ports,
)
from ....models import (
    EngineName,
    TensorCollectionFormat,
    TreePeriodicCellName,
    TreePeriodicTreeSpec,
)
from ...shared.common import (
    CodeSection,
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
)
from .._linear_periodic.graph_common import _render_cell_setup_sections
from .common import (
    _build_child_ports_by_index,
    _cell_from_tree,
    _render_parent_interface_validation,
)
from .graph_expressions import (
    _build_edge_expression_by_index_id,
    _render_child_interface_lines,
    _render_python_list_expression,
)
from .shared import _RenderedTreeCellHelper, render_tree_periodic_helper


def _render_tree_graph_cell_helper(
    *,
    tree: TreePeriodicTreeSpec,
    cell_name: TreePeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> _RenderedTreeCellHelper:
    """Render one graph-backend tree cell helper."""
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

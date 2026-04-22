"""Graph cell renderers for periodic-grid code generation."""

from __future__ import annotations

from ....internal.modes._grid_periodic import (
    build_grid_periodic_interface_ports,
    build_internal_grid_periodic_cell_network,
)
from ....models import (
    EngineName,
    GridPeriodicCellName,
    GridPeriodicGridSpec,
    GridPeriodicTensorRole,
    TensorCollectionFormat,
)
from ...shared.common import (
    CodeSection,
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
    render_tensor_collection_assignment,
    render_tensor_collection_initialization,
)
from .common import _cell_from_grid
from .graph_expressions import (
    _build_label_expression_map,
    _render_network_connection_lines,
    _tensor_value_expression,
)
from .shared import _RenderedCellHelper, render_grid_periodic_helper


def _render_cell_helper(
    *,
    grid: GridPeriodicGridSpec,
    cell_name: GridPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> _RenderedCellHelper:
    """Render one non-contracting grid cell helper."""
    cell = _cell_from_grid(grid, cell_name)
    internal_spec = build_internal_grid_periodic_cell_network(
        cell,
        cell_name=cell_name,
    )
    prepared = prepare_network(internal_spec)
    collection_name = container_name_for_format(collection_format)
    ports_by_role = {
        role: build_grid_periodic_interface_ports(
            cell,
            cell_name=cell_name,
            role=role,
        )
        for role in GridPeriodicTensorRole
    }
    prepared_label_by_index_id = {
        index.spec.id: index.label
        for tensor in prepared.tensors
        for index in tensor.indices
    }
    interface_index_ids = {
        port.internal_index_id for ports in ports_by_role.values() for port in ports
    }
    tensor_collection_lines = render_tensor_collection_initialization(
        collection_name,
        collection_format,
    )
    tensor_construction_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id={
            tensor.spec.id: _tensor_value_expression(
                prepared=prepared,
                tensor_id=tensor.spec.id,
                engine=engine,
            )
            for tensor in prepared.tensors
        },
        include_initialization=False,
    )
    network_connection_lines = _render_network_connection_lines(
        prepared=prepared,
        engine=engine,
        collection_format=collection_format,
        collection_name=collection_name,
    )
    label_expression_by_label = _build_label_expression_map(
        prepared=prepared,
        engine=engine,
        collection_format=collection_format,
        collection_name=collection_name,
    )
    output_lines = [
        "network_nodes = "
        + flattened_tensor_collection_expression(collection_format, collection_name)
    ]
    for role in (
        GridPeriodicTensorRole.UP,
        GridPeriodicTensorRole.RIGHT,
        GridPeriodicTensorRole.DOWN,
        GridPeriodicTensorRole.LEFT,
    ):
        interface_expressions = [
            label_expression_by_label[
                prepared_label_by_index_id[port.internal_index_id]
            ]
            for port in ports_by_role[role]
            if port.internal_index_id in prepared_label_by_index_id
        ]
        output_lines.append(
            f"{role.value}_interface = [" + ", ".join(interface_expressions) + "]"
        )
    open_edge_expressions = [
        label_expression_by_label[index.label]
        for index in prepared.open_indices
        if index.spec.id not in interface_index_ids
    ]
    output_lines.append("open_edges = [" + ", ".join(open_edge_expressions) + "]")
    output_lines.extend(
        [
            "return {",
            "    'nodes': network_nodes,",
            "    'up_interface': up_interface,",
            "    'right_interface': right_interface,",
            "    'down_interface': down_interface,",
            "    'left_interface': left_interface,",
            "    'open_edges': open_edges,",
            "}",
        ]
    )
    return render_grid_periodic_helper(
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

"""Standard graph-backend renderers for linear-periodic code generation."""

from __future__ import annotations

from ....internal.modes._linear_periodic import (
    LinearPeriodicTensorRole,
    build_internal_linear_periodic_cell_network,
    build_linear_periodic_interface_ports,
)
from ....models import (
    EngineName,
    LinearPeriodicCellName,
    LinearPeriodicChainSpec,
    TensorCollectionFormat,
)
from ...backends.tensorkrowch import TensorKrowchCodeGenerator
from ...backends.tensornetwork import TensorNetworkCodeGenerator
from ...shared.common import (
    CodeSection,
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
)
from .common import _cell_from_chain, _RenderedCellHelper, render_linear_periodic_helper
from .graph_common import (
    _build_label_expression_map,
    _render_cell_setup_sections,
)


def _render_cell_helper(
    *,
    chain: LinearPeriodicChainSpec,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> _RenderedCellHelper:
    """Render one non-carry cell helper and the expressions it returns."""
    cell = _cell_from_chain(chain, cell_name)
    internal_spec = build_internal_linear_periodic_cell_network(
        cell,
        cell_name=cell_name,
    )
    prepared = prepare_network(internal_spec)
    collection_name = container_name_for_format(collection_format)
    incoming_ports = build_linear_periodic_interface_ports(
        cell,
        cell_name=cell_name,
        role=LinearPeriodicTensorRole.PREVIOUS,
    )
    outgoing_ports = build_linear_periodic_interface_ports(
        cell,
        cell_name=cell_name,
        role=LinearPeriodicTensorRole.NEXT,
    )
    prepared_label_by_index_id = {
        index.spec.id: index.label
        for tensor in prepared.tensors
        for index in tensor.indices
    }
    interface_index_ids = {
        port.internal_index_id for port in (*incoming_ports, *outgoing_ports)
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
    label_expression_by_label = _build_label_expression_map(
        prepared=prepared,
        engine=engine,
        collection_format=collection_format,
        collection_name=collection_name,
    )
    incoming_expressions = [
        label_expression_by_label[prepared_label_by_index_id[port.internal_index_id]]
        for port in incoming_ports
        if port.internal_index_id in prepared_label_by_index_id
    ]
    outgoing_expressions = [
        label_expression_by_label[prepared_label_by_index_id[port.internal_index_id]]
        for port in outgoing_ports
        if port.internal_index_id in prepared_label_by_index_id
    ]
    open_edge_expressions = [
        label_expression_by_label[index.label]
        for index in prepared.open_indices
        if index.spec.id not in interface_index_ids
    ]
    output_lines: list[str]
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        generator = (
            TensorNetworkCodeGenerator()
            if engine is EngineName.TENSORNETWORK
            else TensorKrowchCodeGenerator()
        )
        contraction_lines, output_lines = generator._render_manual_plan(
            prepared=prepared,
            collection_format=collection_format,
            collection_name=collection_name,
        )
        if engine is EngineName.TENSORKROWCH:
            output_lines.append("network_nodes = list(remaining_operands.values())")
    else:
        contraction_lines = [
            "network_nodes = "
            + flattened_tensor_collection_expression(collection_format, collection_name)
        ]
        output_lines = []

    output_lines.append(
        "incoming_interface = [" + ", ".join(incoming_expressions) + "]"
    )
    output_lines.append(
        "outgoing_interface = [" + ", ".join(outgoing_expressions) + "]"
    )
    output_lines.append("open_edges = [" + ", ".join(open_edge_expressions) + "]")
    output_lines.extend(
        [
            "return {",
            "    'nodes': network_nodes,",
            "    'incoming_interface': incoming_interface,",
            "    'outgoing_interface': outgoing_interface,",
            "    'open_edges': open_edges,",
            "}",
        ]
    )
    sections = [
        CodeSection(title="Tensor collection", lines=tensor_collection_lines),
        CodeSection(title="Tensor construction", lines=tensor_construction_lines),
        CodeSection(title="Network connections", lines=network_connection_lines),
    ]
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        sections.append(
            CodeSection(title="Manual contraction", lines=contraction_lines)
        )
    else:
        sections.append(CodeSection(title="Cell assembly", lines=contraction_lines))
    sections.append(CodeSection(title="Outputs", lines=output_lines))
    return render_linear_periodic_helper(
        helper_name=helper_name,
        helper_signature=helper_signature,
        return_annotation="dict[str, object]",
        sections=sections,
    )

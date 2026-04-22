"""Einsum array helpers for tree-periodic code generation."""

from __future__ import annotations

from ....internal.modes._tree_periodic import (
    build_internal_tree_periodic_cell_network,
    build_tree_periodic_interface_ports,
)
from ....models import (
    EngineName,
    TensorCollectionFormat,
    TreePeriodicCellName,
    TreePeriodicTensorRole,
    TreePeriodicTreeSpec,
)
from ...shared._linear_periodic_expressions import _render_python_list_expression
from ...shared.common import (
    CodeSection,
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
    render_tensor_collection_assignment,
    render_tensor_collection_initialization,
)
from .array_helpers import (
    _build_einsum_label_expression_map,
    _render_child_interface_lines,
)
from .common import (
    _build_child_ports_by_index,
    _cell_from_tree,
    _render_parent_interface_validation,
)
from .shared import _RenderedTreeCellHelper, render_tree_periodic_helper


def _render_einsum_cell_helper(
    *,
    tree: TreePeriodicTreeSpec,
    cell_name: TreePeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> _RenderedTreeCellHelper:
    """Render one tree cell helper for an einsum backend."""
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
    label_expression_by_label = _build_einsum_label_expression_map(
        prepared=prepared,
        cell_name=cell_name,
        parent_ports=parent_ports,
        child_ports_by_index=child_ports_by_index,
    )
    interface_index_ids = {port.internal_index_id for port in parent_ports} | {
        port.internal_index_id
        for ports in child_ports_by_index.values()
        for port in ports
    }
    module_alias = "np" if engine is EngineName.EINSUM_NUMPY else "torch"
    zero_suffix = (
        ", dtype=torch.float32)"
        if engine is EngineName.EINSUM_TORCH
        else ", dtype=float)"
    )
    tensor_collection_lines = render_tensor_collection_initialization(
        collection_name,
        collection_format,
    )
    tensor_value_by_id = {
        tensor.spec.id: f"{module_alias}.zeros({tensor.spec.shape!r}{zero_suffix}"
        for tensor in prepared.tensors
    }
    tensor_construction_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id=tensor_value_by_id,
        include_initialization=False,
    )
    output_lines = _render_parent_interface_validation(parent_ports)
    output_lines.extend(
        [
            "cell_operands = "
            + flattened_tensor_collection_expression(
                collection_format, collection_name
            ),
            "cell_operand_labels = [",
            *[
                "    "
                + _render_python_list_expression(
                    [label_expression_by_label[index.label] for index in tensor.indices]
                )
                + ","
                for tensor in prepared.tensors
            ],
            "]",
            *_render_child_interface_lines(
                cell_name=cell_name,
                child_ports_by_index=child_ports_by_index,
            ),
            "open_labels = "
            + _render_python_list_expression(
                [
                    label_expression_by_label[index.label]
                    for index in prepared.open_indices
                    if index.spec.id not in interface_index_ids
                ]
            ),
            "return {",
            "    'operands': list(cell_operands),",
            "    'operand_labels': cell_operand_labels,",
            "    'open_labels': open_labels,",
            "    'child_interfaces': child_interfaces,",
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
            CodeSection(title="Outputs", lines=output_lines),
        ],
    )

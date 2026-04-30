"""Quimb array helpers for tree-periodic code generation."""

from __future__ import annotations

from ....models import (
    TensorCollectionFormat,
    TreePeriodicCellName,
    TreePeriodicTreeSpec,
)
from ...shared._linear_periodic_expressions import _render_python_tuple_expression
from ...shared.common import (
    CodeSection,
    flattened_tensor_collection_expression,
)
from .array_helpers import (
    _build_quimb_label_expression_map,
    _render_child_interface_lines,
)
from .array_shared import (
    build_tree_array_cell_context,
    render_tree_array_tensor_sections,
)
from .common import _render_parent_interface_validation
from .shared import _RenderedTreeCellHelper, render_tree_periodic_helper


def _render_quimb_cell_helper(
    *,
    tree: TreePeriodicTreeSpec,
    cell_name: TreePeriodicCellName,
    helper_name: str,
    helper_signature: str,
    collection_format: TensorCollectionFormat,
) -> _RenderedTreeCellHelper:
    """Render one tree cell helper for the ``quimb`` backend."""
    context = build_tree_array_cell_context(
        tree=tree,
        cell_name=cell_name,
        collection_format=collection_format,
    )
    label_expression_by_label = _build_quimb_label_expression_map(
        prepared=context.prepared,
        cell_name=cell_name,
        parent_ports=context.parent_ports,
        child_ports_by_index=context.child_ports_by_index,
    )
    tensor_collection_lines, tensor_construction_lines = (
        render_tree_array_tensor_sections(
            context=context,
            tensor_value_by_id={
                tensor.spec.id: (
                    f"qtn.Tensor(data=np.zeros({tensor.spec.shape!r}, dtype=float), "
                    f"inds={_render_python_tuple_expression([label_expression_by_label[index.label] for index in tensor.indices])}, "
                    f"tags={(tensor.spec.name, tensor.spec.id)!r})"
                )
                for tensor in context.prepared.tensors
            },
        )
    )
    output_lines = _render_parent_interface_validation(context.parent_ports)
    output_lines.extend(
        [
            "network_tensors = "
            + flattened_tensor_collection_expression(
                collection_format, context.collection_name
            ),
            *_render_child_interface_lines(
                cell_name=cell_name,
                child_ports_by_index=context.child_ports_by_index,
            ),
            "open_inds = "
            + _render_python_tuple_expression(
                [
                    label_expression_by_label[index.label]
                    for index in context.prepared.open_indices
                    if index.spec.id not in context.interface_index_ids
                ]
            ),
            "return {",
            "    'tensors': list(network_tensors),",
            "    'open_inds': open_inds,",
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

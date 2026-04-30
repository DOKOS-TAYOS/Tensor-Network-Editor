"""Einsum array helpers for periodic-grid code generation."""

from __future__ import annotations

from ....internal.modes._grid_periodic import GridPeriodicInterfacePort
from ....models import (
    EngineName,
    GridPeriodicCellName,
    GridPeriodicGridSpec,
    GridPeriodicTensorRole,
    TensorCollectionFormat,
)
from ...shared._linear_periodic_expressions import _render_python_list_expression
from ...shared.common import (
    CodeSection,
    PreparedNetwork,
    tensor_collection_reference_by_id,
)
from .array_helpers import (
    _GRID_CELL_KIND_OFFSET,
    _build_interface_slot_by_label,
    _build_local_label_offsets,
    _einsum_interface_expression,
    _runtime_cell_coordinate_expressions,
)
from .array_shared import (
    build_grid_array_cell_context,
    render_grid_array_tensor_sections,
)
from .shared import _RenderedCellHelper, render_grid_periodic_helper


def _render_einsum_cell_helper(
    *,
    grid: GridPeriodicGridSpec,
    cell_name: GridPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> _RenderedCellHelper:
    """Render one grid cell helper for an einsum backend."""
    context = build_grid_array_cell_context(
        grid=grid,
        cell_name=cell_name,
        collection_format=collection_format,
    )
    label_expression_by_label = _build_einsum_label_expression_map(
        prepared=context.prepared,
        cell_name=cell_name,
        ports_by_role=context.ports_by_role,
    )
    module_alias = "np" if engine is EngineName.EINSUM_NUMPY else "torch"
    zero_suffix = (
        ", dtype=torch.float32)"
        if engine is EngineName.EINSUM_TORCH
        else ", dtype=float)"
    )
    tensor_collection_lines, tensor_construction_lines = (
        render_grid_array_tensor_sections(
            context=context,
            tensor_value_by_id={
                tensor.spec.id: f"{module_alias}.zeros({tensor.spec.shape!r}{zero_suffix}"
                for tensor in context.prepared.tensors
            },
        )
    )
    output_lines = ["cell_operands = []", "cell_operand_labels = []"]
    for tensor in context.prepared.tensors:
        output_lines.append(
            "cell_operands.append("
            + tensor_collection_reference_by_id(
                context.prepared,
                tensor.spec.id,
                collection_format,
                context.collection_name,
            )
            + ")"
        )
        output_lines.append(
            "cell_operand_labels.append("
            + _render_python_list_expression(
                [label_expression_by_label[index.label] for index in tensor.indices]
            )
            + ")"
        )
    output_lines.extend(
        [
            "open_labels = "
            + _render_python_list_expression(
                [
                    label_expression_by_label[index.label]
                    for index in context.prepared.open_indices
                    if index.spec.id not in context.interface_index_ids
                ]
            ),
            "return {",
            "    'operands': cell_operands,",
            "    'operand_labels': cell_operand_labels,",
            "    'open_labels': open_labels,",
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
            CodeSection(title="Outputs", lines=output_lines),
        ],
    )


def _build_einsum_label_expression_map(
    *,
    prepared: PreparedNetwork,
    cell_name: GridPeriodicCellName,
    ports_by_role: dict[GridPeriodicTensorRole, tuple[GridPeriodicInterfacePort, ...]],
) -> dict[str, str]:
    """Map prepared labels to runtime integer-label expressions for einsum."""
    interface_slot_by_label = _build_interface_slot_by_label(
        prepared=prepared,
        ports_by_role=ports_by_role,
    )
    local_label_offsets = _build_local_label_offsets(
        prepared=prepared,
        interface_slot_by_label=interface_slot_by_label,
    )
    column_expression, row_expression = _runtime_cell_coordinate_expressions(cell_name)
    kind_offset = _GRID_CELL_KIND_OFFSET[cell_name]
    label_expression_by_label: dict[str, str] = {}
    for tensor in prepared.tensors:
        for index in tensor.indices:
            interface_item = interface_slot_by_label.get(index.label)
            if interface_item is not None:
                role, slot_index = interface_item
                label_expression_by_label[index.label] = _einsum_interface_expression(
                    role=role,
                    slot_index=slot_index,
                    column_expression=column_expression,
                    row_expression=row_expression,
                )
                continue
            label_expression_by_label[index.label] = (
                f"local_label({kind_offset}, {column_expression}, {row_expression}, "
                f"{local_label_offsets[index.label]})"
            )
    return label_expression_by_label

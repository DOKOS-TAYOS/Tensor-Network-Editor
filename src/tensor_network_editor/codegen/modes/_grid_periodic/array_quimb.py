"""Quimb array helpers for periodic-grid code generation."""

from __future__ import annotations

from ....internal.modes._grid_periodic import GridPeriodicInterfacePort
from ....models import (
    GridPeriodicCellName,
    GridPeriodicGridSpec,
    GridPeriodicTensorRole,
    TensorCollectionFormat,
)
from ...shared._linear_periodic_expressions import _render_python_tuple_expression
from ...shared.common import (
    CodeSection,
    PreparedNetwork,
    flattened_tensor_collection_expression,
)
from .array_helpers import (
    _build_interface_slot_by_label,
    _build_local_label_offsets,
    _quimb_interface_expression,
    _runtime_cell_coordinate_expressions,
)
from .array_shared import (
    build_grid_array_cell_context,
    render_grid_array_tensor_sections,
)
from .shared import _RenderedCellHelper, render_grid_periodic_helper


def _render_quimb_cell_helper(
    *,
    grid: GridPeriodicGridSpec,
    cell_name: GridPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    collection_format: TensorCollectionFormat,
) -> _RenderedCellHelper:
    """Render one grid cell helper for the ``quimb`` backend."""
    context = build_grid_array_cell_context(
        grid=grid,
        cell_name=cell_name,
        collection_format=collection_format,
    )
    label_expression_by_label = _build_quimb_label_expression_map(
        prepared=context.prepared,
        cell_name=cell_name,
        ports_by_role=context.ports_by_role,
    )
    tensor_collection_lines, tensor_construction_lines = (
        render_grid_array_tensor_sections(
            context=context,
            tensor_value_by_id={
                tensor.spec.id: (
                    f"qtn.Tensor(data=np.zeros({tensor.spec.shape!r}, dtype=float), "
                    f"inds={_render_python_tuple_expression([label_expression_by_label[index.label] for index in tensor.indices])}, "
                    f"tags={(tensor.spec.name,)!r})"
                )
                for tensor in context.prepared.tensors
            },
        )
    )
    output_lines = [
        "cell_tensors = "
        + flattened_tensor_collection_expression(
            collection_format,
            context.collection_name,
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
        "    'tensors': cell_tensors,",
        "    'open_inds': open_inds,",
        "}",
    ]
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


def _build_quimb_label_expression_map(
    *,
    prepared: PreparedNetwork,
    cell_name: GridPeriodicCellName,
    ports_by_role: dict[GridPeriodicTensorRole, tuple[GridPeriodicInterfacePort, ...]],
) -> dict[str, str]:
    """Map prepared labels to runtime ``quimb`` index-label expressions."""
    interface_slot_by_label = _build_interface_slot_by_label(
        prepared=prepared,
        ports_by_role=ports_by_role,
    )
    local_label_offsets = _build_local_label_offsets(
        prepared=prepared,
        interface_slot_by_label=interface_slot_by_label,
    )
    column_expression, row_expression = _runtime_cell_coordinate_expressions(cell_name)
    label_expression_by_label: dict[str, str] = {}
    for tensor in prepared.tensors:
        for index in tensor.indices:
            interface_item = interface_slot_by_label.get(index.label)
            if interface_item is not None:
                role, slot_index = interface_item
                label_expression_by_label[index.label] = _quimb_interface_expression(
                    role=role,
                    slot_index=slot_index,
                    column_expression=column_expression,
                    row_expression=row_expression,
                )
                continue
            label_expression_by_label[index.label] = (
                f"cell_label({cell_name.value!r}, {column_expression}, {row_expression}, "
                f"{local_label_offsets[index.label]})"
            )
    return label_expression_by_label

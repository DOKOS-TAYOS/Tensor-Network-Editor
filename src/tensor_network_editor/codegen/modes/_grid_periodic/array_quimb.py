"""Quimb array helpers for periodic-grid code generation."""

from __future__ import annotations

from ....internal.modes._grid_periodic import (
    GridPeriodicInterfacePort,
    build_internal_grid_periodic_cell_network,
)
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
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
    render_tensor_collection_assignment,
    render_tensor_collection_initialization,
)
from .array_helpers import (
    _build_interface_slot_by_label,
    _build_local_label_offsets,
    _build_ports_by_role,
    _quimb_interface_expression,
    _runtime_cell_coordinate_expressions,
)
from .common import _cell_from_grid
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
    cell = _cell_from_grid(grid, cell_name)
    internal_spec = build_internal_grid_periodic_cell_network(
        cell,
        cell_name=cell_name,
    )
    prepared = prepare_network(internal_spec)
    collection_name = container_name_for_format(collection_format)
    ports_by_role = _build_ports_by_role(cell=cell, cell_name=cell_name)
    interface_index_ids = {
        port.internal_index_id for ports in ports_by_role.values() for port in ports
    }
    label_expression_by_label = _build_quimb_label_expression_map(
        prepared=prepared,
        cell_name=cell_name,
        ports_by_role=ports_by_role,
    )
    tensor_collection_lines = render_tensor_collection_initialization(
        collection_name,
        collection_format,
    )
    tensor_construction_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id={
            tensor.spec.id: (
                f"qtn.Tensor(data=np.zeros({tensor.spec.shape!r}, dtype=float), "
                f"inds={_render_python_tuple_expression([label_expression_by_label[index.label] for index in tensor.indices])}, "
                f"tags={(tensor.spec.name,)!r})"
            )
            for tensor in prepared.tensors
        },
        include_initialization=False,
    )
    output_lines = [
        "cell_tensors = "
        + flattened_tensor_collection_expression(collection_format, collection_name),
        "open_inds = "
        + _render_python_tuple_expression(
            [
                label_expression_by_label[index.label]
                for index in prepared.open_indices
                if index.spec.id not in interface_index_ids
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

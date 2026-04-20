"""Array-backend renderers for bidimensional periodic-grid code generation."""

from __future__ import annotations

from ...internal.modes._grid_periodic import (
    GridPeriodicInterfacePort,
    build_grid_periodic_interface_ports,
    build_internal_grid_periodic_cell_network,
)
from ...models import (
    CodegenResult,
    EngineName,
    GridPeriodicCellName,
    GridPeriodicGridSpec,
    GridPeriodicTensorRole,
    LinearPeriodicCellSpec,
    TensorCollectionFormat,
)
from ..shared._linear_periodic_expressions import (
    _render_python_list_expression,
    _render_python_tuple_expression,
)
from ..shared.common import (
    CodeSection,
    PreparedNetwork,
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
    render_tensor_collection_assignment,
    render_tensor_collection_initialization,
    tensor_collection_reference_by_id,
)
from ._grid_periodic_shared import (
    GRID_PERIODIC_CELL_ORDER,
    _RenderedCellHelper,
    grid_periodic_helper_name,
    grid_periodic_helper_signature,
    render_grid_periodic_helper,
    render_grid_periodic_script,
    render_grid_periodic_shared_helpers,
)

_GRID_CELL_KIND_OFFSET: dict[GridPeriodicCellName, int] = {
    cell_name: offset for offset, cell_name in enumerate(GRID_PERIODIC_CELL_ORDER)
}


def generate_array_grid_periodic_code(
    *,
    grid: GridPeriodicGridSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate grid-periodic code for Quimb and einsum backends."""
    if engine is EngineName.QUIMB:
        return _generate_quimb_grid_periodic_code(
            grid=grid,
            collection_format=collection_format,
        )
    return _generate_einsum_grid_periodic_code(
        grid=grid,
        engine=engine,
        collection_format=collection_format,
    )


def _generate_quimb_grid_periodic_code(
    *,
    grid: GridPeriodicGridSpec,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate non-contracting grid-periodic code for the ``quimb`` backend."""
    cell_lines_by_name = {
        cell_name: _render_quimb_cell_helper(
            grid=grid,
            cell_name=cell_name,
            helper_name=grid_periodic_helper_name(cell_name),
            helper_signature=grid_periodic_helper_signature(cell_name),
            collection_format=collection_format,
        ).lines
        for cell_name in GRID_PERIODIC_CELL_ORDER
    }
    return CodegenResult(
        engine=EngineName.QUIMB,
        code=render_grid_periodic_script(
            import_lines=[
                "# Tensor Network Editor grid periodic mode",
                "import numpy as np",
                "import quimb.tensor as qtn",
            ],
            shared_helper_lines=_render_quimb_shared_helper_lines(),
            cell_lines_by_name=cell_lines_by_name,
            main_loop_lines=_render_quimb_main_loop_lines(),
            output_lines=[
                "network = qtn.TensorNetwork(network_tensors)",
                "result = network_tensors[0] if len(network_tensors) == 1 else None",
            ],
        ),
    )


def _generate_einsum_grid_periodic_code(
    *,
    grid: GridPeriodicGridSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate grid-periodic code for one einsum backend."""
    module_alias = "np" if engine is EngineName.EINSUM_NUMPY else "torch"
    import_line = (
        "import numpy as np" if engine is EngineName.EINSUM_NUMPY else "import torch"
    )
    cell_lines_by_name = {
        cell_name: _render_einsum_cell_helper(
            grid=grid,
            cell_name=cell_name,
            helper_name=grid_periodic_helper_name(cell_name),
            helper_signature=grid_periodic_helper_signature(cell_name),
            engine=engine,
            collection_format=collection_format,
        ).lines
        for cell_name in GRID_PERIODIC_CELL_ORDER
    }
    return CodegenResult(
        engine=engine,
        code=render_grid_periodic_script(
            import_lines=[
                "# Tensor Network Editor grid periodic mode",
                import_line,
            ],
            shared_helper_lines=_render_einsum_shared_helper_lines(),
            cell_lines_by_name=cell_lines_by_name,
            main_loop_lines=_render_einsum_main_loop_lines(),
            output_lines=[
                "dense_label_values: list[int] = []",
                "for operand_labels in einsum_operands[1::2]:",
                "    for label in operand_labels:",
                "        if label not in dense_label_values:",
                "            dense_label_values.append(label)",
                "for label in output_labels:",
                "    if label not in dense_label_values:",
                "        dense_label_values.append(label)",
                "dense_label_by_value = {",
                "    label: offset for offset, label in enumerate(dense_label_values)",
                "}",
                "dense_einsum_operands: list[object] = []",
                "for operand_index, operand in enumerate(einsum_operands):",
                "    if operand_index % 2 == 0:",
                "        dense_einsum_operands.append(operand)",
                "        continue",
                "    dense_einsum_operands.append(",
                "        [dense_label_by_value[label] for label in operand]",
                "    )",
                "dense_output_labels = [dense_label_by_value[label] for label in output_labels]",
                f"result = {module_alias}.einsum(*dense_einsum_operands, dense_output_labels)",
            ],
        ),
    )


def _render_quimb_shared_helper_lines() -> list[str]:
    """Render shared helper functions for the ``quimb`` script."""
    return render_grid_periodic_shared_helpers(
        extra_lines=[
            "def horizontal_label(column_index: int, row_index: int, slot_index: int) -> str:",
            "    return f'gp_h_{column_index}_{row_index}_{slot_index}'",
            "",
            "def vertical_label(column_index: int, row_index: int, slot_index: int) -> str:",
            "    return f'gp_v_{column_index}_{row_index}_{slot_index}'",
            "",
            "def cell_label(cell_kind: str, column_index: int, row_index: int, label_offset: int) -> str:",
            "    return f'gp_{cell_kind}_{column_index}_{row_index}_{label_offset}'",
        ]
    )


def _render_einsum_shared_helper_lines() -> list[str]:
    """Render shared helper functions for an einsum grid-periodic script."""
    return render_grid_periodic_shared_helpers(
        extra_lines=[
            "def horizontal_label(column_index: int, row_index: int, slot_index: int) -> int:",
            "    return 1_000_000_000 + column_index * 1_000_000 + row_index * 10_000 + slot_index",
            "",
            "def vertical_label(column_index: int, row_index: int, slot_index: int) -> int:",
            "    return 2_000_000_000 + column_index * 1_000_000 + row_index * 10_000 + slot_index",
            "",
            "def local_label(kind_offset: int, column_index: int, row_index: int, label_offset: int) -> int:",
            "    return 3_000_000_000 + kind_offset * 100_000_000 + column_index * 1_000_000 + row_index * 10_000 + label_offset",
        ]
    )


def _render_quimb_main_loop_lines() -> list[str]:
    """Render the outer flow for the ``quimb`` backend."""
    return [
        "validate_grid_shape(n, m)",
        "network_tensors: list[object] = []",
        "open_inds: list[str] = []",
        "",
        "top_left_cell = build_top_left_cell()",
        "network_tensors.extend(top_left_cell['tensors'])",
        "open_inds.extend(top_left_cell['open_inds'])",
        "",
        "for column_index in range(1, n - 1):",
        "    top_cell = build_top_cell(column_index)",
        "    network_tensors.extend(top_cell['tensors'])",
        "    open_inds.extend(top_cell['open_inds'])",
        "",
        "top_right_cell = build_top_right_cell(n - 1)",
        "network_tensors.extend(top_right_cell['tensors'])",
        "open_inds.extend(top_right_cell['open_inds'])",
        "",
        "for row_index in range(1, m - 1):",
        "    left_cell = build_left_cell(row_index)",
        "    network_tensors.extend(left_cell['tensors'])",
        "    open_inds.extend(left_cell['open_inds'])",
        "",
        "    for column_index in range(1, n - 1):",
        "        center_cell = build_center_cell(column_index, row_index)",
        "        network_tensors.extend(center_cell['tensors'])",
        "        open_inds.extend(center_cell['open_inds'])",
        "",
        "    right_cell = build_right_cell(n - 1, row_index)",
        "    network_tensors.extend(right_cell['tensors'])",
        "    open_inds.extend(right_cell['open_inds'])",
        "",
        "bottom_left_cell = build_bottom_left_cell(m - 1)",
        "network_tensors.extend(bottom_left_cell['tensors'])",
        "open_inds.extend(bottom_left_cell['open_inds'])",
        "",
        "for column_index in range(1, n - 1):",
        "    bottom_cell = build_bottom_cell(column_index, m - 1)",
        "    network_tensors.extend(bottom_cell['tensors'])",
        "    open_inds.extend(bottom_cell['open_inds'])",
        "",
        "bottom_right_cell = build_bottom_right_cell(n - 1, m - 1)",
        "network_tensors.extend(bottom_right_cell['tensors'])",
        "open_inds.extend(bottom_right_cell['open_inds'])",
        "open_inds = tuple(open_inds)",
    ]


def _render_einsum_main_loop_lines() -> list[str]:
    """Render the outer flow for einsum backends."""
    return [
        "validate_grid_shape(n, m)",
        "einsum_operands: list[object] = []",
        "output_labels: list[int] = []",
        "",
        "top_left_cell = build_top_left_cell()",
        "output_labels.extend(top_left_cell['open_labels'])",
        "for operand, operand_labels in zip(",
        "    top_left_cell['operands'],",
        "    top_left_cell['operand_labels'],",
        "    strict=True,",
        "):",
        "    einsum_operands.append(operand)",
        "    einsum_operands.append(operand_labels)",
        "",
        "for column_index in range(1, n - 1):",
        "    top_cell = build_top_cell(column_index)",
        "    output_labels.extend(top_cell['open_labels'])",
        "    for operand, operand_labels in zip(",
        "        top_cell['operands'],",
        "        top_cell['operand_labels'],",
        "        strict=True,",
        "    ):",
        "        einsum_operands.append(operand)",
        "        einsum_operands.append(operand_labels)",
        "",
        "top_right_cell = build_top_right_cell(n - 1)",
        "output_labels.extend(top_right_cell['open_labels'])",
        "for operand, operand_labels in zip(",
        "    top_right_cell['operands'],",
        "    top_right_cell['operand_labels'],",
        "    strict=True,",
        "):",
        "    einsum_operands.append(operand)",
        "    einsum_operands.append(operand_labels)",
        "",
        "for row_index in range(1, m - 1):",
        "    left_cell = build_left_cell(row_index)",
        "    output_labels.extend(left_cell['open_labels'])",
        "    for operand, operand_labels in zip(",
        "        left_cell['operands'],",
        "        left_cell['operand_labels'],",
        "        strict=True,",
        "    ):",
        "        einsum_operands.append(operand)",
        "        einsum_operands.append(operand_labels)",
        "",
        "    for column_index in range(1, n - 1):",
        "        center_cell = build_center_cell(column_index, row_index)",
        "        output_labels.extend(center_cell['open_labels'])",
        "        for operand, operand_labels in zip(",
        "            center_cell['operands'],",
        "            center_cell['operand_labels'],",
        "            strict=True,",
        "        ):",
        "            einsum_operands.append(operand)",
        "            einsum_operands.append(operand_labels)",
        "",
        "    right_cell = build_right_cell(n - 1, row_index)",
        "    output_labels.extend(right_cell['open_labels'])",
        "    for operand, operand_labels in zip(",
        "        right_cell['operands'],",
        "        right_cell['operand_labels'],",
        "        strict=True,",
        "    ):",
        "        einsum_operands.append(operand)",
        "        einsum_operands.append(operand_labels)",
        "",
        "bottom_left_cell = build_bottom_left_cell(m - 1)",
        "output_labels.extend(bottom_left_cell['open_labels'])",
        "for operand, operand_labels in zip(",
        "    bottom_left_cell['operands'],",
        "    bottom_left_cell['operand_labels'],",
        "    strict=True,",
        "):",
        "    einsum_operands.append(operand)",
        "    einsum_operands.append(operand_labels)",
        "",
        "for column_index in range(1, n - 1):",
        "    bottom_cell = build_bottom_cell(column_index, m - 1)",
        "    output_labels.extend(bottom_cell['open_labels'])",
        "    for operand, operand_labels in zip(",
        "        bottom_cell['operands'],",
        "        bottom_cell['operand_labels'],",
        "        strict=True,",
        "    ):",
        "        einsum_operands.append(operand)",
        "        einsum_operands.append(operand_labels)",
        "",
        "bottom_right_cell = build_bottom_right_cell(n - 1, m - 1)",
        "output_labels.extend(bottom_right_cell['open_labels'])",
        "for operand, operand_labels in zip(",
        "    bottom_right_cell['operands'],",
        "    bottom_right_cell['operand_labels'],",
        "    strict=True,",
        "):",
        "    einsum_operands.append(operand)",
        "    einsum_operands.append(operand_labels)",
    ]


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
    label_expression_by_label = _build_einsum_label_expression_map(
        prepared=prepared,
        cell_name=cell_name,
        ports_by_role=ports_by_role,
    )
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
    tensor_construction_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id={
            tensor.spec.id: f"{module_alias}.zeros({tensor.spec.shape!r}{zero_suffix}"
            for tensor in prepared.tensors
        },
        include_initialization=False,
    )
    output_lines = ["cell_operands = []", "cell_operand_labels = []"]
    for tensor in prepared.tensors:
        output_lines.append(
            "cell_operands.append("
            + tensor_collection_reference_by_id(
                prepared,
                tensor.spec.id,
                collection_format,
                collection_name,
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
                    for index in prepared.open_indices
                    if index.spec.id not in interface_index_ids
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


def _build_ports_by_role(
    *,
    cell: LinearPeriodicCellSpec,
    cell_name: GridPeriodicCellName,
) -> dict[GridPeriodicTensorRole, tuple[GridPeriodicInterfacePort, ...]]:
    """Build every interface port family for one grid cell."""
    return {
        role: build_grid_periodic_interface_ports(
            cell,
            cell_name=cell_name,
            role=role,
        )
        for role in GridPeriodicTensorRole
    }


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


def _build_interface_slot_by_label(
    *,
    prepared: PreparedNetwork,
    ports_by_role: dict[GridPeriodicTensorRole, tuple[GridPeriodicInterfacePort, ...]],
) -> dict[str, tuple[GridPeriodicTensorRole, int]]:
    """Return the interface role/slot metadata for each prepared label."""
    prepared_label_by_index_id = {
        index.spec.id: index.label
        for tensor in prepared.tensors
        for index in tensor.indices
    }
    interface_slot_by_label: dict[str, tuple[GridPeriodicTensorRole, int]] = {}
    for role, ports in ports_by_role.items():
        for slot_index, port in enumerate(ports):
            internal_index_id = port.internal_index_id
            if internal_index_id not in prepared_label_by_index_id:
                continue
            interface_slot_by_label[prepared_label_by_index_id[internal_index_id]] = (
                role,
                slot_index,
            )
    return interface_slot_by_label


def _build_local_label_offsets(
    *,
    prepared: PreparedNetwork,
    interface_slot_by_label: dict[str, tuple[GridPeriodicTensorRole, int]],
) -> dict[str, int]:
    """Assign stable per-cell offsets to non-interface labels."""
    return {
        label: offset
        for offset, label in enumerate(
            dict.fromkeys(
                index.label
                for tensor in prepared.tensors
                for index in tensor.indices
                if index.label not in interface_slot_by_label
            )
        )
    }


def _quimb_interface_expression(
    *,
    role: GridPeriodicTensorRole,
    slot_index: int,
    column_expression: str,
    row_expression: str,
) -> str:
    """Render one runtime ``quimb`` label expression for an interface slot."""
    if role is GridPeriodicTensorRole.LEFT:
        return f"horizontal_label(({column_expression}) - 1, {row_expression}, {slot_index})"
    if role is GridPeriodicTensorRole.RIGHT:
        return f"horizontal_label({column_expression}, {row_expression}, {slot_index})"
    if role is GridPeriodicTensorRole.UP:
        return (
            f"vertical_label({column_expression}, ({row_expression}) - 1, {slot_index})"
        )
    return f"vertical_label({column_expression}, {row_expression}, {slot_index})"


def _einsum_interface_expression(
    *,
    role: GridPeriodicTensorRole,
    slot_index: int,
    column_expression: str,
    row_expression: str,
) -> str:
    """Render one runtime integer-label expression for an interface slot."""
    return _quimb_interface_expression(
        role=role,
        slot_index=slot_index,
        column_expression=column_expression,
        row_expression=row_expression,
    )


def _runtime_cell_coordinate_expressions(
    cell_name: GridPeriodicCellName,
) -> tuple[str, str]:
    """Return the runtime ``(column, row)`` expressions for one helper."""
    if cell_name is GridPeriodicCellName.TOP_LEFT:
        return "0", "0"
    if cell_name is GridPeriodicCellName.TOP:
        return "column_index", "0"
    if cell_name is GridPeriodicCellName.TOP_RIGHT:
        return "column_index", "0"
    if cell_name is GridPeriodicCellName.LEFT:
        return "0", "row_index"
    if cell_name is GridPeriodicCellName.CENTER:
        return "column_index", "row_index"
    if cell_name is GridPeriodicCellName.RIGHT:
        return "column_index", "row_index"
    if cell_name is GridPeriodicCellName.BOTTOM_LEFT:
        return "0", "row_index"
    if cell_name is GridPeriodicCellName.BOTTOM:
        return "column_index", "row_index"
    return "column_index", "row_index"


def _cell_from_grid(
    grid: GridPeriodicGridSpec,
    cell_name: GridPeriodicCellName,
) -> LinearPeriodicCellSpec:
    """Return the matching cell from ``grid``."""
    if cell_name is GridPeriodicCellName.TOP_LEFT:
        return grid.top_left_cell
    if cell_name is GridPeriodicCellName.TOP:
        return grid.top_cell
    if cell_name is GridPeriodicCellName.TOP_RIGHT:
        return grid.top_right_cell
    if cell_name is GridPeriodicCellName.LEFT:
        return grid.left_cell
    if cell_name is GridPeriodicCellName.CENTER:
        return grid.center_cell
    if cell_name is GridPeriodicCellName.RIGHT:
        return grid.right_cell
    if cell_name is GridPeriodicCellName.BOTTOM_LEFT:
        return grid.bottom_left_cell
    if cell_name is GridPeriodicCellName.BOTTOM:
        return grid.bottom_cell
    return grid.bottom_right_cell

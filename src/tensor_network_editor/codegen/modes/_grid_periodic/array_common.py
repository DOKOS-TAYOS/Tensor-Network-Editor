"""Shared array orchestration for periodic-grid code generation."""

from __future__ import annotations

from ....models import (
    CodegenResult,
    EngineName,
    GridPeriodicGridSpec,
    TensorCollectionFormat,
)
from .array_einsum import _render_einsum_cell_helper
from .array_quimb import _render_quimb_cell_helper
from .shared import (
    GRID_PERIODIC_CELL_ORDER,
    grid_periodic_helper_name,
    grid_periodic_helper_signature,
    render_grid_periodic_script,
    render_grid_periodic_shared_helpers,
)


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

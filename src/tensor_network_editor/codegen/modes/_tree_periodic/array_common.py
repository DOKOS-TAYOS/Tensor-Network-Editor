"""Shared array orchestration for tree-periodic code generation."""

from __future__ import annotations

from ....models import (
    CodegenResult,
    EngineName,
    TensorCollectionFormat,
    TreePeriodicCellName,
    TreePeriodicTreeSpec,
)
from .array_einsum import _render_einsum_cell_helper
from .array_quimb import _render_quimb_cell_helper
from .shared import (
    render_tree_periodic_script,
    render_tree_periodic_shared_helpers,
    tree_periodic_helper_name,
    tree_periodic_helper_signature,
)


def generate_array_tree_periodic_code(
    *,
    tree: TreePeriodicTreeSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate helper-based tree-periodic code for array backends."""
    if engine is EngineName.QUIMB:
        return _generate_quimb_tree_periodic_code(
            tree=tree,
            collection_format=collection_format,
        )
    return _generate_einsum_tree_periodic_code(
        tree=tree,
        engine=engine,
        collection_format=collection_format,
    )


def _generate_quimb_tree_periodic_code(
    *,
    tree: TreePeriodicTreeSpec,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    cell_lines_by_name = {
        cell_name: _render_quimb_cell_helper(
            tree=tree,
            cell_name=cell_name,
            helper_name=tree_periodic_helper_name(cell_name),
            helper_signature=tree_periodic_helper_signature(
                cell_name,
                interface_annotation="list[str]",
            ),
            collection_format=collection_format,
        ).lines
        for cell_name in (
            TreePeriodicCellName.ROOT,
            TreePeriodicCellName.BRANCH,
            TreePeriodicCellName.LEAF,
        )
    }
    main_loop_lines = [
        "validate_tree_depth(n)",
        "root_cell = build_root_cell()",
        "network_tensors = list(root_cell['tensors'])",
        "open_inds = list(root_cell['open_inds'])",
        "frontier = list(root_cell['child_interfaces'])",
        "",
        "for level in range(1, n - 1):",
        "    next_frontier: list[list[str]] = []",
        "    for node_index, parent_interface in enumerate(frontier):",
        "        branch_cell = build_branch_cell(level, node_index, parent_interface)",
        "        network_tensors.extend(branch_cell['tensors'])",
        "        open_inds.extend(branch_cell['open_inds'])",
        "        next_frontier.extend(branch_cell['child_interfaces'])",
        "    frontier = next_frontier",
        "",
        "for node_index, parent_interface in enumerate(frontier):",
        "    leaf_cell = build_leaf_cell(n - 1, node_index, parent_interface)",
        "    network_tensors.extend(leaf_cell['tensors'])",
        "    open_inds.extend(leaf_cell['open_inds'])",
    ]
    output_lines = [
        "open_inds = tuple(open_inds)",
        "network = qtn.TensorNetwork(network_tensors) if network_tensors else None",
        "result = network_tensors[0] if len(network_tensors) == 1 else None",
    ]
    return CodegenResult(
        engine=EngineName.QUIMB,
        code=render_tree_periodic_script(
            import_lines=[
                "import numpy as np",
                "import quimb.tensor as qtn",
                "",
                "# Tensor Network Editor tree periodic mode",
            ],
            shared_helper_lines=render_tree_periodic_shared_helpers(
                extra_lines=[
                    f"branching_factor = {tree.branching_factor}",
                    "",
                    "def child_label(level: int, node_index: int, child_index: int, slot_index: int) -> str:",
                    "    return f'tp_child_{level}_{node_index}_{child_index}_{slot_index}'",
                    "",
                    "def local_label(cell_kind: str, level: int, node_index: int, label_offset: int) -> str:",
                    "    return f'tp_{cell_kind}_{level}_{node_index}_{label_offset}'",
                ]
            ),
            cell_lines_by_name=cell_lines_by_name,
            main_loop_lines=main_loop_lines,
            output_lines=output_lines,
        ),
    )


def _generate_einsum_tree_periodic_code(
    *,
    tree: TreePeriodicTreeSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    module_alias = "np" if engine is EngineName.EINSUM_NUMPY else "torch"
    scalar_expression = (
        "np.array(1.0)" if engine is EngineName.EINSUM_NUMPY else "torch.tensor(1.0)"
    )
    cell_lines_by_name = {
        cell_name: _render_einsum_cell_helper(
            tree=tree,
            cell_name=cell_name,
            helper_name=tree_periodic_helper_name(cell_name),
            helper_signature=tree_periodic_helper_signature(
                cell_name,
                interface_annotation="list[int]",
            ),
            engine=engine,
            collection_format=collection_format,
        ).lines
        for cell_name in (
            TreePeriodicCellName.ROOT,
            TreePeriodicCellName.BRANCH,
            TreePeriodicCellName.LEAF,
        )
    }
    main_loop_lines = [
        "validate_tree_depth(n)",
        "root_cell = build_root_cell()",
        "einsum_operands: list[object] = []",
        "output_labels = list(root_cell['open_labels'])",
        "for operand, operand_labels in zip(",
        "    root_cell['operands'],",
        "    root_cell['operand_labels'],",
        "    strict=True,",
        "):",
        "    einsum_operands.append(operand)",
        "    einsum_operands.append(operand_labels)",
        "frontier = list(root_cell['child_interfaces'])",
        "",
        "for level in range(1, n - 1):",
        "    next_frontier: list[list[int]] = []",
        "    for node_index, parent_interface in enumerate(frontier):",
        "        branch_cell = build_branch_cell(level, node_index, parent_interface)",
        "        output_labels.extend(branch_cell['open_labels'])",
        "        for operand, operand_labels in zip(",
        "            branch_cell['operands'],",
        "            branch_cell['operand_labels'],",
        "            strict=True,",
        "        ):",
        "            einsum_operands.append(operand)",
        "            einsum_operands.append(operand_labels)",
        "        next_frontier.extend(branch_cell['child_interfaces'])",
        "    frontier = next_frontier",
        "",
        "for node_index, parent_interface in enumerate(frontier):",
        "    leaf_cell = build_leaf_cell(n - 1, node_index, parent_interface)",
        "    output_labels.extend(leaf_cell['open_labels'])",
        "    for operand, operand_labels in zip(",
        "        leaf_cell['operands'],",
        "        leaf_cell['operand_labels'],",
        "        strict=True,",
        "    ):",
        "        einsum_operands.append(operand)",
        "        einsum_operands.append(operand_labels)",
    ]
    output_lines = [
        "if not einsum_operands:",
        f"    result = {scalar_expression}",
        "else:",
        "    dense_label_values: list[int] = []",
        "    for operand_labels in einsum_operands[1::2]:",
        "        for label in operand_labels:",
        "            if label not in dense_label_values:",
        "                dense_label_values.append(label)",
        "    for label in output_labels:",
        "        if label not in dense_label_values:",
        "            dense_label_values.append(label)",
        "    dense_label_by_value = {",
        "        label: offset for offset, label in enumerate(dense_label_values)",
        "    }",
        "    dense_einsum_operands: list[object] = []",
        "    for operand_index, operand in enumerate(einsum_operands):",
        "        if operand_index % 2 == 0:",
        "            dense_einsum_operands.append(operand)",
        "            continue",
        "        dense_einsum_operands.append(",
        "            [dense_label_by_value[label] for label in operand]",
        "        )",
        "    dense_output_labels = [dense_label_by_value[label] for label in output_labels]",
        f"    result = {module_alias}.einsum(*dense_einsum_operands, dense_output_labels)",
    ]
    return CodegenResult(
        engine=engine,
        code=render_tree_periodic_script(
            import_lines=[
                "import numpy as np"
                if engine is EngineName.EINSUM_NUMPY
                else "import torch",
                "",
                "# Tensor Network Editor tree periodic mode",
            ],
            shared_helper_lines=render_tree_periodic_shared_helpers(
                extra_lines=[
                    f"branching_factor = {tree.branching_factor}",
                    "",
                    "def child_label(level: int, node_index: int, child_index: int, slot_index: int) -> int:",
                    "    return 1_000_000_000_000 + level * 10_000_000_000 + node_index * 100_000 + child_index * 1_000 + slot_index",
                    "",
                    "def local_label(kind_offset: int, level: int, node_index: int, label_offset: int) -> int:",
                    "    return 9_000_000_000_000 + kind_offset * 1_000_000_000_000 + level * 10_000_000_000 + node_index * 100_000 + label_offset",
                ]
            ),
            cell_lines_by_name=cell_lines_by_name,
            main_loop_lines=main_loop_lines,
            output_lines=output_lines,
        ),
    )

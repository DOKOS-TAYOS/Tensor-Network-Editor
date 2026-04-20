"""Array-backend renderers for tree periodic code generation."""

from __future__ import annotations

from .._tree_periodic import (
    TreePeriodicInterfacePort,
    build_internal_tree_periodic_cell_network,
    build_tree_periodic_interface_ports,
)
from ..models import (
    CodegenResult,
    EngineName,
    TensorCollectionFormat,
    TreePeriodicCellName,
    TreePeriodicTensorRole,
    TreePeriodicTreeSpec,
)
from ._linear_periodic_expressions import (
    _render_python_list_expression,
    _render_python_tuple_expression,
)
from ._tree_periodic_shared import (
    render_tree_periodic_helper,
    render_tree_periodic_script,
    render_tree_periodic_shared_helpers,
    tree_periodic_helper_name,
    tree_periodic_helper_signature,
)
from .common import (
    CodeSection,
    PreparedNetwork,
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
    render_tensor_collection_assignment,
    render_tensor_collection_initialization,
)

_TREE_CELL_KIND_OFFSET: dict[TreePeriodicCellName, int] = {
    TreePeriodicCellName.ROOT: 0,
    TreePeriodicCellName.BRANCH: 1,
    TreePeriodicCellName.LEAF: 2,
}


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


def _render_quimb_cell_helper(
    *,
    tree: TreePeriodicTreeSpec,
    cell_name: TreePeriodicCellName,
    helper_name: str,
    helper_signature: str,
    collection_format: TensorCollectionFormat,
):
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
    label_expression_by_label = _build_quimb_label_expression_map(
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
    tensor_collection_lines = render_tensor_collection_initialization(
        collection_name,
        collection_format,
    )
    tensor_value_by_id = {
        tensor.spec.id: (
            f"qtn.Tensor(data=np.zeros({tensor.spec.shape!r}, dtype=float), "
            f"inds={_render_python_tuple_expression([label_expression_by_label[index.label] for index in tensor.indices])}, "
            f"tags={(tensor.spec.name, tensor.spec.id)!r})"
        )
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
            "network_tensors = "
            + flattened_tensor_collection_expression(
                collection_format, collection_name
            ),
            *_render_child_interface_lines(
                cell_name=cell_name,
                child_ports_by_index=child_ports_by_index,
            ),
            "open_inds = "
            + _render_python_tuple_expression(
                [
                    label_expression_by_label[index.label]
                    for index in prepared.open_indices
                    if index.spec.id not in interface_index_ids
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


def _render_einsum_cell_helper(
    *,
    tree: TreePeriodicTreeSpec,
    cell_name: TreePeriodicCellName,
    helper_name: str,
    helper_signature: str,
    collection_format: TensorCollectionFormat,
):
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
    tensor_collection_lines = render_tensor_collection_initialization(
        collection_name,
        collection_format,
    )
    tensor_value_by_id = {
        tensor.spec.id: f"np.zeros({tensor.spec.shape!r}, dtype=float)"
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


def _cell_from_tree(
    tree: TreePeriodicTreeSpec,
    cell_name: TreePeriodicCellName,
):
    if cell_name is TreePeriodicCellName.ROOT:
        return tree.root_cell
    if cell_name is TreePeriodicCellName.BRANCH:
        return tree.branch_cell
    return tree.leaf_cell


def _build_child_ports_by_index(
    *,
    tree: TreePeriodicTreeSpec,
    cell,
    cell_name: TreePeriodicCellName,
) -> dict[int, tuple[TreePeriodicInterfacePort, ...]]:
    return {
        child_index: build_tree_periodic_interface_ports(
            cell,
            cell_name=cell_name,
            role=TreePeriodicTensorRole.CHILD,
            child_index=child_index,
        )
        for child_index in range(tree.branching_factor)
    }


def _build_interface_slot_by_label(
    *,
    prepared: PreparedNetwork,
    parent_ports: tuple[TreePeriodicInterfacePort, ...],
    child_ports_by_index: dict[int, tuple[TreePeriodicInterfacePort, ...]],
) -> dict[str, tuple[str, int, int | None]]:
    prepared_label_by_index_id = {
        index.spec.id: index.label
        for tensor in prepared.tensors
        for index in tensor.indices
    }
    interface_slot_by_label: dict[str, tuple[str, int, int | None]] = {}
    for slot_index, port in enumerate(parent_ports):
        internal_index_id = port.internal_index_id
        if internal_index_id in prepared_label_by_index_id:
            interface_slot_by_label[prepared_label_by_index_id[internal_index_id]] = (
                "parent",
                slot_index,
                None,
            )
    for child_index, ports in child_ports_by_index.items():
        for slot_index, port in enumerate(ports):
            internal_index_id = port.internal_index_id
            if internal_index_id in prepared_label_by_index_id:
                interface_slot_by_label[
                    prepared_label_by_index_id[internal_index_id]
                ] = (
                    "child",
                    slot_index,
                    child_index,
                )
    return interface_slot_by_label


def _build_local_label_offsets(
    *,
    prepared: PreparedNetwork,
    interface_slot_by_label: dict[str, tuple[str, int, int | None]],
) -> dict[str, int]:
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


def _build_quimb_label_expression_map(
    *,
    prepared: PreparedNetwork,
    cell_name: TreePeriodicCellName,
    parent_ports: tuple[TreePeriodicInterfacePort, ...],
    child_ports_by_index: dict[int, tuple[TreePeriodicInterfacePort, ...]],
) -> dict[str, str]:
    interface_slot_by_label = _build_interface_slot_by_label(
        prepared=prepared,
        parent_ports=parent_ports,
        child_ports_by_index=child_ports_by_index,
    )
    local_label_offsets = _build_local_label_offsets(
        prepared=prepared,
        interface_slot_by_label=interface_slot_by_label,
    )
    level_expression, node_expression = _runtime_coordinate_expressions(cell_name)
    label_expression_by_label: dict[str, str] = {}
    for tensor in prepared.tensors:
        for index in tensor.indices:
            interface_item = interface_slot_by_label.get(index.label)
            if interface_item is not None:
                family, slot_index, child_index = interface_item
                if family == "parent":
                    label_expression_by_label[index.label] = (
                        f"parent_interface[{slot_index}]"
                    )
                else:
                    label_expression_by_label[index.label] = (
                        f"child_label({level_expression}, {node_expression}, {child_index}, {slot_index})"
                    )
                continue
            label_expression_by_label[index.label] = (
                f"local_label({cell_name.value!r}, {level_expression}, {node_expression}, "
                f"{local_label_offsets[index.label]})"
            )
    return label_expression_by_label


def _build_einsum_label_expression_map(
    *,
    prepared: PreparedNetwork,
    cell_name: TreePeriodicCellName,
    parent_ports: tuple[TreePeriodicInterfacePort, ...],
    child_ports_by_index: dict[int, tuple[TreePeriodicInterfacePort, ...]],
) -> dict[str, str]:
    interface_slot_by_label = _build_interface_slot_by_label(
        prepared=prepared,
        parent_ports=parent_ports,
        child_ports_by_index=child_ports_by_index,
    )
    local_label_offsets = _build_local_label_offsets(
        prepared=prepared,
        interface_slot_by_label=interface_slot_by_label,
    )
    level_expression, node_expression = _runtime_coordinate_expressions(cell_name)
    label_expression_by_label: dict[str, str] = {}
    kind_offset = _TREE_CELL_KIND_OFFSET[cell_name]
    for tensor in prepared.tensors:
        for index in tensor.indices:
            interface_item = interface_slot_by_label.get(index.label)
            if interface_item is not None:
                family, slot_index, child_index = interface_item
                if family == "parent":
                    label_expression_by_label[index.label] = (
                        f"parent_interface[{slot_index}]"
                    )
                else:
                    label_expression_by_label[index.label] = (
                        f"child_label({level_expression}, {node_expression}, {child_index}, {slot_index})"
                    )
                continue
            label_expression_by_label[index.label] = (
                f"local_label({kind_offset}, {level_expression}, {node_expression}, "
                f"{local_label_offsets[index.label]})"
            )
    return label_expression_by_label


def _render_parent_interface_validation(
    parent_ports: tuple[TreePeriodicInterfacePort, ...],
) -> list[str]:
    if not parent_ports:
        return []
    return [
        f"if len(parent_interface) != {len(parent_ports)}:",
        "    raise ValueError('The provided parent interface does not match this tree cell.')",
    ]


def _render_child_interface_lines(
    *,
    cell_name: TreePeriodicCellName,
    child_ports_by_index: dict[int, tuple[TreePeriodicInterfacePort, ...]],
) -> list[str]:
    if cell_name is TreePeriodicCellName.LEAF:
        return ["child_interfaces = []"]
    level_expression, node_expression = _runtime_coordinate_expressions(cell_name)
    lines = ["child_interfaces = []"]
    for child_index in sorted(child_ports_by_index):
        ports = child_ports_by_index[child_index]
        lines.append(
            "child_interfaces.append("
            + _render_python_list_expression(
                [
                    f"child_label({level_expression}, {node_expression}, {child_index}, {slot_index})"
                    for slot_index, _port in enumerate(ports)
                ]
            )
            + ")"
        )
    return lines


def _runtime_coordinate_expressions(
    cell_name: TreePeriodicCellName,
) -> tuple[str, str]:
    if cell_name is TreePeriodicCellName.ROOT:
        return "0", "0"
    return "level", "node_index"

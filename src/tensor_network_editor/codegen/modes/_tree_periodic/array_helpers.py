"""Shared array helpers for tree-periodic code generation."""

from __future__ import annotations

from ....internal.modes._tree_periodic import TreePeriodicInterfacePort
from ....models import TreePeriodicCellName
from ...shared._linear_periodic_expressions import _render_python_list_expression
from ...shared.common import PreparedNetwork
from .common import _TREE_CELL_KIND_OFFSET, _runtime_coordinate_expressions


def _build_interface_slot_by_label(
    *,
    prepared: PreparedNetwork,
    parent_ports: tuple[TreePeriodicInterfacePort, ...],
    child_ports_by_index: dict[int, tuple[TreePeriodicInterfacePort, ...]],
) -> dict[str, tuple[str, int, int | None]]:
    """Map prepared labels onto parent/child interface slots."""
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
    """Assign stable offsets to non-interface labels inside one cell."""
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
    """Map prepared labels to runtime ``quimb`` expressions."""
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
    """Map prepared labels to runtime integer expressions for einsum."""
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
    kind_offset = _TREE_CELL_KIND_OFFSET[cell_name]
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
                f"local_label({kind_offset}, {level_expression}, {node_expression}, "
                f"{local_label_offsets[index.label]})"
            )
    return label_expression_by_label


def _render_child_interface_lines(
    *,
    cell_name: TreePeriodicCellName,
    child_ports_by_index: dict[int, tuple[TreePeriodicInterfacePort, ...]],
) -> list[str]:
    """Render the runtime child-interface payloads for array backends."""
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

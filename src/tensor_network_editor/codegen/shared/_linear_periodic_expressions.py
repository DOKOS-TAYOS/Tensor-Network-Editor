"""Internal expression helpers for linear periodic code generation."""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from ...errors import CodeGenerationError
from ...internal.modes._linear_periodic import LinearPeriodicInterfacePort
from ...models import EngineName, LinearPeriodicCellName
from .common import PreparedNetwork, render_results_list_reference

if TYPE_CHECKING:
    from ..modes._linear_periodic.carry import _CarryOperandState


_TENSORKROWCH_AXIS_INDEX_SUFFIX_RE = re.compile(r"_\d+$")


def _deduplicate_tensorkrowch_axis_names(
    axis_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Mirror TensorKrowch's suffixing for duplicate surviving axis names."""
    axis_names = tuple(
        _TENSORKROWCH_AXIS_INDEX_SUFFIX_RE.sub("", axis_name)
        for axis_name in axis_names
    )
    return _deduplicate_axis_names(axis_names)


def _deduplicate_axis_names(
    axis_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Make duplicate axis names unique with stable numeric suffixes."""
    counts = Counter(axis_names)
    seen: dict[str, int] = {}
    resolved_axis_names: list[str] = []
    for axis_name in axis_names:
        if counts[axis_name] == 1:
            resolved_axis_names.append(axis_name)
            continue
        suffix = seen.get(axis_name, 0)
        seen[axis_name] = suffix + 1
        resolved_axis_names.append(f"{axis_name}_{suffix}")
    return tuple(resolved_axis_names)


def _axis_names_for_engine(
    engine: EngineName,
    axis_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the runtime axis names produced by the requested backend."""
    if engine is EngineName.TENSORKROWCH:
        return _deduplicate_tensorkrowch_axis_names(axis_names)
    return _deduplicate_axis_names(axis_names)


def _axis_name_for_engine(engine: EngineName, axis_name: str) -> str:
    """Return one runtime axis name for the requested backend."""
    return _axis_names_for_engine(engine, (axis_name,))[0]


def _build_remaining_label_expression_map(
    *,
    remaining_operand_ids: tuple[str, ...],
    remaining_operand_states: dict[str, _CarryOperandState],
    base_operand_expressions: dict[str, str],
    step_result_indexes: dict[str, int],
    latest_result_index: int | None,
) -> dict[str, str]:
    """Resolve surviving labels from the current operand state mapping."""
    label_expression_by_label: dict[str, str] = {}
    for operand_id in remaining_operand_ids:
        operand_state = remaining_operand_states.get(operand_id)
        if operand_state is None:
            continue
        operand_expression = _operand_expression(
            operand_id=operand_id,
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=step_result_indexes,
            latest_result_index=latest_result_index,
        )
        for label, axis_name in zip(
            operand_state.labels,
            operand_state.axis_names,
            strict=True,
        ):
            label_expression_by_label[label] = f"{operand_expression}[{axis_name!r}]"
    return label_expression_by_label


def _operand_expression(
    *,
    operand_id: str,
    base_operand_expressions: dict[str, str],
    step_result_indexes: dict[str, int],
    latest_result_index: int | None,
) -> str:
    """Resolve one simulated operand id to the generated Python expression."""
    if operand_id in base_operand_expressions:
        return base_operand_expressions[operand_id]
    if operand_id not in step_result_indexes:
        raise CodeGenerationError(
            f"Operand '{operand_id}' is not available while rendering linear periodic code."
        )
    return render_results_list_reference(
        step_result_indexes[operand_id],
        latest_result_index=latest_result_index,
    )


def _build_quimb_label_expression_map(
    *,
    prepared: PreparedNetwork,
    cell_name: LinearPeriodicCellName,
    incoming_ports: tuple[LinearPeriodicInterfacePort, ...],
    outgoing_ports: tuple[LinearPeriodicInterfacePort, ...],
) -> dict[str, str]:
    """Map prepared labels to runtime ``quimb`` index-label expressions."""
    prepared_label_by_index_id = {
        index.spec.id: index.label
        for tensor in prepared.tensors
        for index in tensor.indices
    }
    incoming_slot_by_label = {
        prepared_label_by_index_id[port.internal_index_id]: slot_index
        for slot_index, port in enumerate(incoming_ports)
        if port.internal_index_id in prepared_label_by_index_id
    }
    outgoing_slot_by_label = {
        prepared_label_by_index_id[port.internal_index_id]: slot_index
        for slot_index, port in enumerate(outgoing_ports)
        if port.internal_index_id in prepared_label_by_index_id
    }
    label_expression_by_label: dict[str, str] = {}
    for tensor in prepared.tensors:
        for index in tensor.indices:
            label_expression_by_label[index.label] = _quimb_label_expression(
                cell_name=cell_name,
                label=index.label,
                incoming_slot_by_label=incoming_slot_by_label,
                outgoing_slot_by_label=outgoing_slot_by_label,
            )
    return label_expression_by_label


def _build_einsum_label_expression_map(
    *,
    prepared: PreparedNetwork,
    cell_name: LinearPeriodicCellName,
    incoming_ports: tuple[LinearPeriodicInterfacePort, ...],
    outgoing_ports: tuple[LinearPeriodicInterfacePort, ...],
) -> dict[str, str]:
    """Map prepared labels to runtime integer-label expressions for einsum."""
    prepared_label_by_index_id = {
        index.spec.id: index.label
        for tensor in prepared.tensors
        for index in tensor.indices
    }
    incoming_slot_by_label = {
        prepared_label_by_index_id[port.internal_index_id]: slot_index
        for slot_index, port in enumerate(incoming_ports)
        if port.internal_index_id in prepared_label_by_index_id
    }
    outgoing_slot_by_label = {
        prepared_label_by_index_id[port.internal_index_id]: slot_index
        for slot_index, port in enumerate(outgoing_ports)
        if port.internal_index_id in prepared_label_by_index_id
    }
    local_label_offsets = {
        label: offset
        for offset, label in enumerate(
            dict.fromkeys(
                index.label
                for tensor in prepared.tensors
                for index in tensor.indices
                if index.label not in incoming_slot_by_label
                and index.label not in outgoing_slot_by_label
            )
        )
    }
    label_expression_by_label: dict[str, str] = {}
    for tensor in prepared.tensors:
        for index in tensor.indices:
            label_expression_by_label[index.label] = _einsum_label_expression(
                cell_name=cell_name,
                label=index.label,
                incoming_slot_by_label=incoming_slot_by_label,
                outgoing_slot_by_label=outgoing_slot_by_label,
                local_label_offsets=local_label_offsets,
            )
    return label_expression_by_label


def _quimb_label_expression(
    *,
    cell_name: LinearPeriodicCellName,
    label: str,
    incoming_slot_by_label: dict[str, int],
    outgoing_slot_by_label: dict[str, int],
) -> str:
    """Render one runtime ``quimb`` label expression."""
    cell_index_expression = _runtime_cell_index_expression(cell_name)
    if label in incoming_slot_by_label:
        return (
            f"interface_label({cell_index_expression} - 1, "
            f"{cell_index_expression}, {incoming_slot_by_label[label]})"
        )
    if label in outgoing_slot_by_label:
        if cell_name is LinearPeriodicCellName.INITIAL:
            return f"interface_label(0, 1, {outgoing_slot_by_label[label]})"
        return (
            f"interface_label({cell_index_expression}, "
            f"{cell_index_expression} + 1, {outgoing_slot_by_label[label]})"
        )
    return f"cell_label({cell_name.value!r}, {cell_index_expression}, {label!r})"


def _einsum_label_expression(
    *,
    cell_name: LinearPeriodicCellName,
    label: str,
    incoming_slot_by_label: dict[str, int],
    outgoing_slot_by_label: dict[str, int],
    local_label_offsets: dict[str, int],
) -> str:
    """Render one runtime integer-label expression for einsum."""
    cell_index_expression = _runtime_cell_index_expression(cell_name)
    if label in incoming_slot_by_label:
        return f"interface_label({cell_index_expression} - 1, {incoming_slot_by_label[label]})"
    if label in outgoing_slot_by_label:
        if cell_name is LinearPeriodicCellName.INITIAL:
            return f"interface_label(0, {outgoing_slot_by_label[label]})"
        return (
            f"interface_label({cell_index_expression}, {outgoing_slot_by_label[label]})"
        )
    label_offset = local_label_offsets[label]
    if cell_name is LinearPeriodicCellName.INITIAL:
        return f"initial_label({label_offset})"
    if cell_name is LinearPeriodicCellName.PERIODIC:
        return f"periodic_label({cell_index_expression}, {label_offset})"
    return f"final_label({cell_index_expression}, {label_offset})"


def _runtime_cell_index_expression(cell_name: LinearPeriodicCellName) -> str:
    """Return the runtime cell-index expression for the given helper."""
    if cell_name is LinearPeriodicCellName.INITIAL:
        return "0"
    return "cell_index"


def _render_python_tuple_expression(values: list[str]) -> str:
    """Render a Python tuple literal from pre-rendered item expressions."""
    if not values:
        return "()"
    if len(values) == 1:
        return f"({values[0]},)"
    return "(" + ", ".join(values) + ")"


def _render_python_list_expression(values: list[str]) -> str:
    """Render a Python list literal from pre-rendered item expressions."""
    return "[" + ", ".join(values) + "]"


def _carry_cell_key_prefix_expression(cell_name: LinearPeriodicCellName) -> str:
    """Return the runtime Python expression used to namespace remaining operands."""
    if cell_name is LinearPeriodicCellName.INITIAL:
        return "'initial'"
    if cell_name is LinearPeriodicCellName.PERIODIC:
        return "f'periodic_{cell_index}'"
    return "'final'"

"""Carry-mode graph-backend renderers for linear-periodic code generation."""

from __future__ import annotations

from ....errors import CodeGenerationError
from ....internal.modes._linear_periodic import LINEAR_PERIODIC_PREVIOUS_OPERAND_ID
from ....models import EngineName, LinearPeriodicCellName, TensorCollectionFormat
from ...backends.tensornetwork import TensorNetworkCodeGenerator
from ...shared._linear_periodic_expressions import (
    _axis_name_for_engine,
    _axis_names_for_engine,
    _build_remaining_label_expression_map,
    _carry_cell_key_prefix_expression,
    _operand_expression,
)
from ...shared.common import (
    CodeSection,
    container_name_for_format,
    tensor_collection_reference_by_id,
)
from .carry import _CarryPlanSimulation
from .common import _RenderedCellHelper, render_linear_periodic_helper
from .graph_common import _render_cell_setup_sections


def _render_carry_cell_helper(
    *,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    simulation: _CarryPlanSimulation,
) -> _RenderedCellHelper:
    """Render one carry-mode helper that threads ``previous_operand``."""
    collection_name = container_name_for_format(collection_format)
    (
        tensor_collection_lines,
        tensor_construction_lines,
        network_connection_lines,
    ) = _render_cell_setup_sections(
        prepared=simulation.prepared,
        engine=engine,
        collection_format=collection_format,
        collection_name=collection_name,
    )
    tracked_export_lines, tracked_export_expressions = (
        _render_tensorkrowch_export_tracking_lines(
            simulation=simulation,
            collection_format=collection_format,
            collection_name=collection_name,
        )
        if engine is EngineName.TENSORKROWCH
        else ([], {})
    )
    if tracked_export_lines:
        if network_connection_lines:
            network_connection_lines.append("")
        network_connection_lines.extend(tracked_export_lines)
    previous_interface_lines = _render_carry_boundary_setup(
        simulation=simulation,
        engine=engine,
        collection_format=collection_format,
        collection_name=collection_name,
    )
    contraction_lines, output_lines = _render_carry_plan_lines(
        simulation=simulation,
        cell_name=cell_name,
        engine=engine,
        collection_format=collection_format,
        collection_name=collection_name,
        tracked_export_expressions=tracked_export_expressions,
    )
    return render_linear_periodic_helper(
        helper_name=helper_name,
        helper_signature=helper_signature,
        return_annotation="dict[str, object] | object | None",
        sections=[
            CodeSection(title="Tensor collection", lines=tensor_collection_lines),
            CodeSection(title="Tensor construction", lines=tensor_construction_lines),
            CodeSection(title="Network connections", lines=network_connection_lines),
            CodeSection(title="Previous interface", lines=previous_interface_lines),
            CodeSection(title="Manual contraction", lines=contraction_lines),
            CodeSection(title="Outputs", lines=output_lines),
        ],
    )


def _render_carry_boundary_setup(
    *,
    simulation: _CarryPlanSimulation,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> list[str]:
    """Render boundary-slot wiring for ``previous`` and ``next`` carry operands."""
    lines: list[str] = []
    if simulation.incoming_ports:
        previous_operand_index = simulation.previous_operand_interface_index or 0
        lines.append(
            "previous_interface = list(previous_payload['outgoing_interface'])"
        )
        lines.append("previous_operands = list(previous_payload['outgoing_operands'])")
        lines.append(
            f"if len(previous_interface) != {len(simulation.incoming_ports)} or "
            f"len(previous_operands) != {len(simulation.incoming_ports)}:"
        )
        lines.append(
            "    raise ValueError('Previous payload interface does not match this cell.')"
        )
        lines.append(f"previous_operand = previous_operands[{previous_operand_index}]")
        lines.append("incoming_edges = []")
        for port_index, port in enumerate(simulation.incoming_ports):
            internal_axis_name = _axis_name_for_engine(
                engine,
                port.internal_index_name,
            )
            local_tensor = tensor_collection_reference_by_id(
                simulation.prepared,
                port.internal_tensor_id,
                collection_format,
                collection_name,
            )
            if engine is EngineName.TENSORNETWORK:
                lines.append(
                    "incoming_edges.append(tn.connect("
                    f"previous_interface[{port_index}], "
                    f"{local_tensor}[{internal_axis_name!r}], "
                    f"name={port.boundary_index_name!r}))"
                )
            else:
                lines.append(
                    "incoming_edges.append(("
                    f"{port.boundary_index_name!r}, "
                    f"tk.connect(previous_interface[{port_index}], "
                    f"{local_tensor}[{internal_axis_name!r}])"
                    "))"
                )
        lines.append("")
    return lines


def _render_tensorkrowch_export_tracking_lines(
    *,
    simulation: _CarryPlanSimulation,
    collection_format: TensorCollectionFormat,
    collection_name: str,
) -> tuple[list[str], dict[str, str]]:
    """Track only current-cell local-open edges before later contractions rename them."""
    if not simulation.local_open_labels:
        return [], {}

    label_expression_by_label: dict[str, str] = {}
    for tensor in simulation.prepared.tensors:
        tensor_expression = tensor_collection_reference_by_id(
            simulation.prepared,
            tensor.spec.id,
            collection_format,
            collection_name,
        )
        runtime_axis_names = _axis_names_for_engine(
            EngineName.TENSORKROWCH,
            tuple(index.spec.name for index in tensor.indices),
        )
        for index, runtime_axis_name in zip(
            tensor.indices,
            runtime_axis_names,
            strict=True,
        ):
            label_expression_by_label[index.label] = (
                f"{tensor_expression}[{runtime_axis_name!r}]"
            )

    lines = ["# Tracked current-cell edges"]
    tracked_expressions: dict[str, str] = {}
    for tracked_index, label in enumerate(simulation.local_open_labels):
        label_expression = label_expression_by_label.get(label)
        if label_expression is None:
            continue
        variable_name = f"tracked_edge_{tracked_index}"
        lines.append(f"{variable_name} = {label_expression}")
        tracked_expressions[label] = variable_name
    return lines, tracked_expressions


def _render_carry_plan_lines(
    *,
    simulation: _CarryPlanSimulation,
    cell_name: LinearPeriodicCellName,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
    tracked_export_expressions: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Render all carry-mode contractions and helper epilogue lines."""
    if engine is EngineName.TENSORKROWCH and any(
        step.is_outer_product for step in simulation.real_steps
    ):
        raise CodeGenerationError(
            "TensorKrowch cannot emit manual outer product steps with contract_between."
        )

    base_operand_expressions = {
        tensor.spec.id: tensor_collection_reference_by_id(
            simulation.prepared,
            tensor.spec.id,
            collection_format,
            collection_name,
        )
        for tensor in simulation.prepared.tensors
    }
    if cell_name is not LinearPeriodicCellName.INITIAL:
        base_operand_expressions[LINEAR_PERIODIC_PREVIOUS_OPERAND_ID] = (
            "previous_operand"
        )

    contraction_lines: list[str] = []
    for step_index, step in enumerate(simulation.real_steps):
        latest_result_index = step_index - 1 if step_index > 0 else None
        left_expression = _operand_expression(
            operand_id=step.left_operand_id,
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=simulation.result_index_by_step_id,
            latest_result_index=latest_result_index,
        )
        right_expression = _operand_expression(
            operand_id=step.right_operand_id,
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=simulation.result_index_by_step_id,
            latest_result_index=latest_result_index,
        )
        if not contraction_lines:
            contraction_lines.extend(["results_list = []", ""])
        contraction_lines.append(f"# Manual step {step.step_id}")
        if engine is EngineName.TENSORNETWORK:
            output_edge_order = TensorNetworkCodeGenerator._build_output_edge_order(
                left_expression=left_expression,
                right_expression=right_expression,
                left_labels=step.left_labels,
                right_labels=step.right_labels,
                left_axis_names=step.left_axis_names,
                right_axis_names=step.right_axis_names,
                contracted_labels=step.contracted_labels,
            )
            contraction_lines.append(
                "results_list.append(tn.contract_between("
                f"{left_expression}, "
                f"{right_expression}, "
                f"name={step.step_id!r}, "
                "allow_outer_product=True, "
                f"output_edge_order={output_edge_order}, "
                f"axis_names={list(step.result_axis_names)!r}))"
            )
        else:
            contraction_lines.append(
                "results_list.append(tk.contract_between("
                f"{left_expression}, {right_expression}))"
            )
            if step_index < len(simulation.real_steps) - 1:
                contraction_lines.append(
                    "results_list[-1].reattach_edges(override=True)"
                )
        contraction_lines.append("")

    final_result_index = (
        len(simulation.real_steps) - 1 if simulation.real_steps else None
    )
    output_lines: list[str] = []
    if engine is EngineName.TENSORKROWCH:
        for operand_id in dict.fromkeys(simulation.remaining_operand_ids):
            if operand_id not in simulation.result_index_by_step_id:
                continue
            operand_expression = _operand_expression(
                operand_id=operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=final_result_index,
            )
            output_lines.append(f"{operand_expression}.reattach_edges(override=True)")

    label_expression_by_label = _build_remaining_label_expression_map(
        remaining_operand_ids=simulation.remaining_operand_ids,
        remaining_operand_states=simulation.remaining_operand_states,
        base_operand_expressions=base_operand_expressions,
        step_result_indexes=simulation.result_index_by_step_id,
        latest_result_index=final_result_index,
    )
    if tracked_export_expressions:
        label_expression_by_label.update(tracked_export_expressions)
    local_open_expressions = [
        label_expression_by_label[label]
        for label in simulation.local_open_labels
        if label in label_expression_by_label
    ]
    if local_open_expressions:
        output_lines.append(
            "open_edges.extend([" + ", ".join(local_open_expressions) + "])"
        )

    local_remaining_operand_ids = [
        operand_id
        for operand_id in simulation.remaining_operand_ids
        if operand_id != simulation.carry_operand_id
    ]
    if local_remaining_operand_ids:
        output_lines.append(
            "cell_key_prefix = " + _carry_cell_key_prefix_expression(cell_name)
        )
        for operand_id in local_remaining_operand_ids:
            operand_expression = _operand_expression(
                operand_id=operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=final_result_index,
            )
            output_lines.append(
                f'remaining_operands[f"{{cell_key_prefix}}:{operand_id}"] = {operand_expression}'
            )

    if simulation.carry_operand_id is not None:
        carry_expression = _operand_expression(
            operand_id=simulation.carry_operand_id,
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=simulation.result_index_by_step_id,
            latest_result_index=final_result_index,
        )
        outgoing_interface_expressions = [
            label_expression_by_label[label] for label in simulation.outgoing_labels
        ]
        outgoing_operand_expressions = [
            _operand_expression(
                operand_id=operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=final_result_index,
            )
            for operand_id in simulation.outgoing_interface_operand_ids
        ]
        output_lines.append(
            "outgoing_interface = [" + ", ".join(outgoing_interface_expressions) + "]"
        )
        output_lines.append(
            "outgoing_operands = [" + ", ".join(outgoing_operand_expressions) + "]"
        )
        output_lines.extend(
            [
                "return {",
                f"    'operand': {carry_expression},",
                "    'outgoing_interface': outgoing_interface,",
                "    'outgoing_operands': outgoing_operands,",
                "}",
            ]
        )
        return contraction_lines, output_lines

    if local_remaining_operand_ids:
        final_expression = _operand_expression(
            operand_id=local_remaining_operand_ids[0],
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=simulation.result_index_by_step_id,
            latest_result_index=final_result_index,
        )
        output_lines.append(f"return {final_expression}")
    else:
        output_lines.append("return None")
    return contraction_lines, output_lines

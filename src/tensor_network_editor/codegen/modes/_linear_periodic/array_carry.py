"""Carry-mode array-backend renderers for linear-periodic code generation."""

from __future__ import annotations

from ....internal.modes._linear_periodic import LINEAR_PERIODIC_PREVIOUS_OPERAND_ID
from ....models import (
    CodegenResult,
    EngineName,
    LinearPeriodicCellName,
    LinearPeriodicChainSpec,
    TensorCollectionFormat,
)
from ...backends.quimb import QuimbCodeGenerator
from ...shared._linear_periodic_expressions import (
    _build_einsum_label_expression_map,
    _build_quimb_label_expression_map,
    _carry_cell_key_prefix_expression,
    _operand_expression,
    _render_python_list_expression,
    _render_python_tuple_expression,
)
from ...shared.common import (
    CodeSection,
    container_name_for_format,
    flattened_tensor_collection_expression,
    render_operand_expression,
    render_tensor_collection_assignment,
    render_tensor_collection_initialization,
    tensor_collection_reference_by_id,
)
from .array_common import (
    _einsum_generator_for_engine,
    _render_einsum_carry_main_flow_lines,
    _render_einsum_shared_helper_lines,
    _render_quimb_carry_main_flow_lines,
    _render_quimb_shared_helper_lines,
)
from .carry import _build_carry_simulation_map, _CarryPlanSimulation
from .common import (
    _RenderedCellHelper,
    render_linear_periodic_helper,
    render_linear_periodic_script,
)


def _generate_quimb_linear_periodic_carry_code(
    *,
    chain: LinearPeriodicChainSpec,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate carry-mode linear-periodic code for the ``quimb`` backend."""
    import_lines = [
        "# Tensor Network Editor linear periodic mode",
        "import numpy as np",
        "import quimb.tensor as qtn",
    ]
    shared_helper_lines = _render_quimb_shared_helper_lines()
    carry_simulation_by_cell_name = _build_carry_simulation_map(
        chain=chain,
        engine=EngineName.QUIMB,
    )
    helper_signature_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "",
        LinearPeriodicCellName.PERIODIC: "cell_index: int, previous_payload: dict[str, object]",
        LinearPeriodicCellName.FINAL: "cell_index: int, previous_payload: dict[str, object]",
    }
    helper_name_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "build_initial_cell",
        LinearPeriodicCellName.PERIODIC: "build_periodic_cell",
        LinearPeriodicCellName.FINAL: "build_final_cell",
    }
    main_loop_lines, output_lines = _render_quimb_carry_main_flow_lines()
    return CodegenResult(
        engine=EngineName.QUIMB,
        code=render_linear_periodic_script(
            import_lines=import_lines,
            shared_helper_lines=shared_helper_lines,
            initial_cell_lines=_render_quimb_carry_cell_helper(
                cell_name=LinearPeriodicCellName.INITIAL,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.INITIAL],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.INITIAL
                ],
                collection_format=collection_format,
                simulation=carry_simulation_by_cell_name[
                    LinearPeriodicCellName.INITIAL
                ],
            ).lines,
            periodic_cell_lines=_render_quimb_carry_cell_helper(
                cell_name=LinearPeriodicCellName.PERIODIC,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.PERIODIC],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.PERIODIC
                ],
                collection_format=collection_format,
                simulation=carry_simulation_by_cell_name[
                    LinearPeriodicCellName.PERIODIC
                ],
            ).lines,
            final_cell_lines=_render_quimb_carry_cell_helper(
                cell_name=LinearPeriodicCellName.FINAL,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.FINAL],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.FINAL
                ],
                collection_format=collection_format,
                simulation=carry_simulation_by_cell_name[LinearPeriodicCellName.FINAL],
            ).lines,
            main_loop_lines=main_loop_lines,
            output_lines=output_lines,
        ),
    )


def _generate_einsum_linear_periodic_carry_code(
    *,
    chain: LinearPeriodicChainSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate carry-mode linear-periodic code for one einsum backend."""
    generator = _einsum_generator_for_engine(engine)
    import_lines = [
        "# Tensor Network Editor linear periodic mode",
        generator.import_line,
    ]
    shared_helper_lines = _render_einsum_shared_helper_lines(engine)
    carry_simulation_by_cell_name = _build_carry_simulation_map(
        chain=chain,
        engine=engine,
    )
    helper_signature_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "",
        LinearPeriodicCellName.PERIODIC: "cell_index: int, previous_payload: dict[str, object]",
        LinearPeriodicCellName.FINAL: "cell_index: int, previous_payload: dict[str, object]",
    }
    helper_name_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "build_initial_cell",
        LinearPeriodicCellName.PERIODIC: "build_periodic_cell",
        LinearPeriodicCellName.FINAL: "build_final_cell",
    }
    main_loop_lines, output_lines = _render_einsum_carry_main_flow_lines()
    return CodegenResult(
        engine=engine,
        code=render_linear_periodic_script(
            import_lines=import_lines,
            shared_helper_lines=shared_helper_lines,
            initial_cell_lines=_render_einsum_carry_cell_helper(
                cell_name=LinearPeriodicCellName.INITIAL,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.INITIAL],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.INITIAL
                ],
                engine=engine,
                collection_format=collection_format,
                simulation=carry_simulation_by_cell_name[
                    LinearPeriodicCellName.INITIAL
                ],
            ).lines,
            periodic_cell_lines=_render_einsum_carry_cell_helper(
                cell_name=LinearPeriodicCellName.PERIODIC,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.PERIODIC],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.PERIODIC
                ],
                engine=engine,
                collection_format=collection_format,
                simulation=carry_simulation_by_cell_name[
                    LinearPeriodicCellName.PERIODIC
                ],
            ).lines,
            final_cell_lines=_render_einsum_carry_cell_helper(
                cell_name=LinearPeriodicCellName.FINAL,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.FINAL],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.FINAL
                ],
                engine=engine,
                collection_format=collection_format,
                simulation=carry_simulation_by_cell_name[LinearPeriodicCellName.FINAL],
            ).lines,
            main_loop_lines=main_loop_lines,
            output_lines=output_lines,
        ),
    )


def _render_quimb_carry_cell_helper(
    *,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    collection_format: TensorCollectionFormat,
    simulation: _CarryPlanSimulation,
) -> _RenderedCellHelper:
    """Render one carry-mode cell helper for ``quimb``."""
    prepared = simulation.prepared
    collection_name = container_name_for_format(collection_format)
    label_expression_by_label = _build_quimb_label_expression_map(
        prepared=prepared,
        cell_name=cell_name,
        incoming_ports=simulation.incoming_ports,
        outgoing_ports=simulation.outgoing_ports,
    )
    generator = QuimbCodeGenerator()
    tensor_value_by_id = {
        tensor.spec.id: (
            f"qtn.Tensor(data=np.zeros({tensor.spec.shape!r}, dtype=float), "
            f"inds={_render_python_tuple_expression([label_expression_by_label[index.label] for index in tensor.indices])}, "
            f"tags={(tensor.spec.name, generator._operand_tag(tensor.spec.id))!r})"
        )
        for tensor in prepared.tensors
    }
    tensor_collection_lines = render_tensor_collection_initialization(
        collection_name,
        collection_format,
    )
    tensor_construction_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id=tensor_value_by_id,
        include_initialization=False,
    )
    network_setup_lines = [
        "network_tensors = "
        + flattened_tensor_collection_expression(collection_format, collection_name)
    ]
    base_operand_expressions = {
        tensor.spec.id: tensor_collection_reference_by_id(
            prepared,
            tensor.spec.id,
            collection_format,
            collection_name,
        )
        for tensor in prepared.tensors
    }
    previous_interface_lines: list[str] = []
    if simulation.incoming_ports:
        previous_operand_index = simulation.previous_operand_interface_index or 0
        expected_interface = [
            label_expression_by_label[label] for label in simulation.incoming_labels
        ]
        previous_interface_lines = [
            "previous_interface = list(previous_payload['outgoing_interface'])",
            "previous_operands = list(previous_payload['outgoing_operands'])",
            "expected_interface = "
            + _render_python_list_expression(expected_interface),
            f"if previous_interface != expected_interface or len(previous_operands) != {len(simulation.incoming_ports)}:",
            "    raise ValueError('Previous payload interface does not match this cell.')",
            f"previous_operand = previous_operands[{previous_operand_index}]",
            f"previous_operand.add_tag({generator._operand_tag(LINEAR_PERIODIC_PREVIOUS_OPERAND_ID)!r})",
            "network_tensors = [previous_operand, *network_tensors]",
        ]
        base_operand_expressions[LINEAR_PERIODIC_PREVIOUS_OPERAND_ID] = (
            "previous_operand"
        )
    contraction_lines: list[str] = []
    for step in simulation.real_steps:
        left_tag = generator._operand_tag(step.left_operand_id)
        right_tag = generator._operand_tag(step.right_operand_id)
        step_tag = generator._operand_tag(step.step_id)
        if not contraction_lines:
            contraction_lines.extend(
                [
                    "network = qtn.TensorNetwork(network_tensors)",
                    "",
                    "results_list = []",
                    "",
                ]
            )
        contraction_lines.append(f"# Manual step {step.step_id}")
        contraction_lines.append(
            f"network.contract_between({left_tag!r}, {right_tag!r})"
        )
        contraction_lines.append(f"network[{left_tag!r}].add_tag({step_tag!r})")
        contraction_lines.append(f"results_list.append(network[{step_tag!r}])")
        contraction_lines.append("")
    latest_result_index = (
        len(simulation.real_steps) - 1 if simulation.real_steps else None
    )
    output_lines: list[str] = []
    local_open_expressions = [
        label_expression_by_label[label] for label in simulation.local_open_labels
    ]
    if local_open_expressions:
        output_lines.append(
            "open_inds.extend("
            + _render_python_list_expression(local_open_expressions)
            + ")"
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
            output_lines.append(
                f'remaining_operands[f"{{cell_key_prefix}}:{operand_id}"] = '
                + _operand_expression(
                    operand_id=operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=simulation.result_index_by_step_id,
                    latest_result_index=latest_result_index,
                )
            )
    if simulation.carry_operand_id is not None:
        outgoing_interface_expressions = [
            label_expression_by_label[label] for label in simulation.outgoing_labels
        ]
        outgoing_operand_expressions = [
            _operand_expression(
                operand_id=operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=latest_result_index,
            )
            for operand_id in simulation.outgoing_interface_operand_ids
        ]
        output_lines.extend(
            [
                "outgoing_interface = "
                + _render_python_list_expression(outgoing_interface_expressions),
                "outgoing_operands = "
                + _render_python_list_expression(outgoing_operand_expressions),
                "return {",
                "    'operand': "
                + _operand_expression(
                    operand_id=simulation.carry_operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=simulation.result_index_by_step_id,
                    latest_result_index=latest_result_index,
                )
                + ",",
                "    'outgoing_interface': outgoing_interface,",
                "    'outgoing_operands': outgoing_operands,",
                "}",
            ]
        )
    elif local_remaining_operand_ids:
        output_lines.append(
            "return "
            + _operand_expression(
                operand_id=local_remaining_operand_ids[0],
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=latest_result_index,
            )
        )
    else:
        output_lines.append("return None")
    sections = [
        CodeSection(title="Tensor collection", lines=tensor_collection_lines),
        CodeSection(title="Tensor construction", lines=tensor_construction_lines),
        CodeSection(title="Network setup", lines=network_setup_lines),
        CodeSection(title="Previous interface", lines=previous_interface_lines),
        CodeSection(title="Manual contraction", lines=contraction_lines),
        CodeSection(title="Outputs", lines=output_lines),
    ]
    return render_linear_periodic_helper(
        helper_name=helper_name,
        helper_signature=helper_signature,
        return_annotation="dict[str, object] | object | None",
        sections=sections,
    )


def _render_einsum_carry_cell_helper(
    *,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    simulation: _CarryPlanSimulation,
) -> _RenderedCellHelper:
    """Render one carry-mode cell helper for an einsum backend."""
    prepared = simulation.prepared
    collection_name = container_name_for_format(collection_format)
    label_expression_by_label = _build_einsum_label_expression_map(
        prepared=prepared,
        cell_name=cell_name,
        incoming_ports=simulation.incoming_ports,
        outgoing_ports=simulation.outgoing_ports,
    )
    generator = _einsum_generator_for_engine(engine)
    tensor_value_by_id = {
        tensor.spec.id: (
            f"{generator.module_alias}.zeros({tensor.spec.shape!r}"
            f"{generator.zero_initializer_suffix})"
        )
        for tensor in prepared.tensors
    }
    tensor_collection_lines = render_tensor_collection_initialization(
        collection_name,
        collection_format,
    )
    tensor_construction_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id=tensor_value_by_id,
        include_initialization=False,
    )
    base_operand_expressions = {
        tensor.spec.id: tensor_collection_reference_by_id(
            prepared,
            tensor.spec.id,
            collection_format,
            collection_name,
        )
        for tensor in prepared.tensors
    }
    previous_interface_lines: list[str] = []
    if simulation.incoming_ports:
        previous_operand_index = simulation.previous_operand_interface_index or 0
        expected_interface = [
            label_expression_by_label[label] for label in simulation.incoming_labels
        ]
        previous_interface_lines = [
            "previous_interface = list(previous_payload['outgoing_interface'])",
            "previous_operands = list(previous_payload['outgoing_operands'])",
            "expected_interface = "
            + _render_python_list_expression(expected_interface),
            "if previous_interface != expected_interface "
            + f"or len(previous_operands) != {len(simulation.incoming_ports)}:",
            "    raise ValueError('Previous payload interface does not match this cell.')",
            f"previous_operand = previous_operands[{previous_operand_index}]",
        ]
        base_operand_expressions[LINEAR_PERIODIC_PREVIOUS_OPERAND_ID] = (
            "previous_operand"
        )
    label_order = list(
        dict.fromkeys(
            [
                *(
                    index.label
                    for tensor in prepared.tensors
                    for index in tensor.indices
                ),
                *(
                    label
                    for step in simulation.real_steps
                    for label in (
                        *step.left_labels,
                        *step.right_labels,
                        *step.surviving_labels,
                    )
                ),
            ]
        )
    )
    label_to_int = {label: offset for offset, label in enumerate(label_order)}
    contraction_lines: list[str] = []
    for step_index, step in enumerate(simulation.real_steps):
        latest_result_index = step_index - 1 if step_index > 0 else None
        if not contraction_lines:
            contraction_lines.extend(["results_list = []", ""])
        contraction_lines.append(f"# Manual step {step.step_id}")
        contraction_lines.append(
            "results_list.append("
            + generator._render_manual_step_call(
                left_expression=render_operand_expression(
                    step.left_operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=simulation.result_index_by_step_id,
                    latest_result_index=latest_result_index,
                ),
                right_expression=render_operand_expression(
                    step.right_operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=simulation.result_index_by_step_id,
                    latest_result_index=latest_result_index,
                ),
                left_labels=step.left_labels,
                right_labels=step.right_labels,
                output_labels=step.surviving_labels,
                use_string_labels=False,
                symbol_map={},
                label_to_int=label_to_int,
            )
            + ")"
        )
        contraction_lines.append("")
    latest_result_index = (
        len(simulation.real_steps) - 1 if simulation.real_steps else None
    )
    output_lines: list[str] = []
    local_open_expressions = [
        label_expression_by_label[label] for label in simulation.local_open_labels
    ]
    if local_open_expressions:
        output_lines.append(
            "open_labels.extend("
            + _render_python_list_expression(local_open_expressions)
            + ")"
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
            output_lines.append(
                f'remaining_operands[f"{{cell_key_prefix}}:{operand_id}"] = '
                + _operand_expression(
                    operand_id=operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=simulation.result_index_by_step_id,
                    latest_result_index=latest_result_index,
                )
            )
    if simulation.carry_operand_id is not None:
        outgoing_interface_expressions = [
            label_expression_by_label[label] for label in simulation.outgoing_labels
        ]
        outgoing_operand_expressions = [
            _operand_expression(
                operand_id=operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=latest_result_index,
            )
            for operand_id in simulation.outgoing_interface_operand_ids
        ]
        output_lines.extend(
            [
                "outgoing_interface = "
                + _render_python_list_expression(outgoing_interface_expressions),
                "outgoing_operands = "
                + _render_python_list_expression(outgoing_operand_expressions),
                "return {",
                "    'operand': "
                + _operand_expression(
                    operand_id=simulation.carry_operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=simulation.result_index_by_step_id,
                    latest_result_index=latest_result_index,
                )
                + ",",
                "    'outgoing_interface': outgoing_interface,",
                "    'outgoing_operands': outgoing_operands,",
                "}",
            ]
        )
    elif local_remaining_operand_ids:
        output_lines.append(
            "return "
            + _operand_expression(
                operand_id=local_remaining_operand_ids[0],
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=latest_result_index,
            )
        )
    else:
        output_lines.append("return None")
    return render_linear_periodic_helper(
        helper_name=helper_name,
        helper_signature=helper_signature,
        return_annotation="dict[str, object] | object | None",
        sections=[
            CodeSection(title="Tensor collection", lines=tensor_collection_lines),
            CodeSection(title="Tensor construction", lines=tensor_construction_lines),
            CodeSection(title="Previous interface", lines=previous_interface_lines),
            CodeSection(title="Manual contraction", lines=contraction_lines),
            CodeSection(title="Outputs", lines=output_lines),
        ],
    )

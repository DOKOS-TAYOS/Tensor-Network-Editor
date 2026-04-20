"""Array-backend renderers for linear periodic code generation."""

from __future__ import annotations

from ...errors import CodeGenerationError
from ...internal.analysis._contraction_plan import (
    prepare_contraction_inputs,
    simulate_contraction_plan,
)
from ...internal.modes._linear_periodic import (
    LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
    LinearPeriodicTensorRole,
    build_internal_linear_periodic_cell_network,
    build_linear_periodic_interface_ports,
)
from ...models import (
    CodegenResult,
    EngineName,
    LinearPeriodicCellName,
    LinearPeriodicChainSpec,
    TensorCollectionFormat,
)
from ..backends.einsum_numpy import EinsumNumpyCodeGenerator
from ..backends.einsum_torch import EinsumTorchCodeGenerator
from ..backends.quimb import QuimbCodeGenerator
from ..shared._linear_periodic_expressions import (
    _build_einsum_label_expression_map,
    _build_quimb_label_expression_map,
    _carry_cell_key_prefix_expression,
    _operand_expression,
    _render_python_list_expression,
    _render_python_tuple_expression,
)
from ..shared.common import (
    CodeSection,
    PreparedNetwork,
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
    render_operand_expression,
    render_tensor_collection_assignment,
    render_tensor_collection_initialization,
    tensor_collection_reference_by_id,
)
from ._linear_periodic_shared import (
    _build_carry_simulation_map,
    _CarryPlanSimulation,
    _cell_from_chain,
    _RenderedCellHelper,
    render_linear_periodic_helper,
    render_linear_periodic_script,
    render_linear_periodic_shared_helpers,
)


def generate_array_linear_periodic_code(
    *,
    chain: LinearPeriodicChainSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    uses_carry_mode: bool,
) -> CodegenResult:
    """Generate linear-periodic code for Quimb and einsum backends."""
    if engine is EngineName.QUIMB:
        if uses_carry_mode:
            return _generate_quimb_linear_periodic_carry_code(
                chain=chain,
                collection_format=collection_format,
            )
        return _generate_quimb_linear_periodic_code(
            chain=chain,
            collection_format=collection_format,
        )
    if engine in {EngineName.EINSUM_NUMPY, EngineName.EINSUM_TORCH}:
        if uses_carry_mode:
            return _generate_einsum_linear_periodic_carry_code(
                chain=chain,
                engine=engine,
                collection_format=collection_format,
            )
        return _generate_einsum_linear_periodic_code(
            chain=chain,
            engine=engine,
            collection_format=collection_format,
        )
    raise CodeGenerationError(
        f"The {engine.value} backend does not support linear periodic code generation."
    )


def _render_quimb_shared_helper_lines() -> list[str]:
    """Render shared helper functions for the ``quimb`` script."""
    return render_linear_periodic_shared_helpers(
        extra_lines=[
            "def interface_label(left_cell: int, right_cell: int, slot_index: int) -> str:",
            "    return f'lp_link_{left_cell}_{right_cell}_{slot_index}'",
            "",
            "def cell_label(cell_kind: str, cell_index: int, label_name: str) -> str:",
            "    return f'lp_{cell_kind}_{cell_index}_{label_name}'",
        ]
    )


def _render_einsum_shared_helper_lines(engine: EngineName) -> list[str]:
    """Render shared helper functions for one einsum linear-periodic script."""
    del engine
    return render_linear_periodic_shared_helpers(
        extra_lines=[
            "def interface_label(left_cell: int, slot_index: int) -> int:",
            "    return 1_000_000_000 + left_cell * 10_000 + slot_index",
            "",
            "def initial_label(label_offset: int) -> int:",
            "    return 2_000_000_000 + label_offset",
            "",
            "def periodic_label(cell_index: int, label_offset: int) -> int:",
            "    return 3_000_000_000 + cell_index * 10_000 + label_offset",
            "",
            "def final_label(cell_index: int, label_offset: int) -> int:",
            "    return 4_000_000_000 + cell_index * 10_000 + label_offset",
        ]
    )


def _generate_quimb_linear_periodic_code(
    *,
    chain: LinearPeriodicChainSpec,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate non-carry linear-periodic code for the ``quimb`` backend."""
    import_lines = [
        "# Tensor Network Editor linear periodic mode",
        "import numpy as np",
        "import quimb.tensor as qtn",
    ]
    shared_helper_lines = _render_quimb_shared_helper_lines()
    helper_signature_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "",
        LinearPeriodicCellName.PERIODIC: "cell_index: int",
        LinearPeriodicCellName.FINAL: "cell_index: int",
    }
    helper_name_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "build_initial_cell",
        LinearPeriodicCellName.PERIODIC: "build_periodic_cell",
        LinearPeriodicCellName.FINAL: "build_final_cell",
    }
    main_loop_lines, output_lines = _render_quimb_main_flow_lines()
    return CodegenResult(
        engine=EngineName.QUIMB,
        code=render_linear_periodic_script(
            import_lines=import_lines,
            shared_helper_lines=shared_helper_lines,
            initial_cell_lines=_render_quimb_cell_helper(
                chain=chain,
                cell_name=LinearPeriodicCellName.INITIAL,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.INITIAL],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.INITIAL
                ],
                collection_format=collection_format,
            ).lines,
            periodic_cell_lines=_render_quimb_cell_helper(
                chain=chain,
                cell_name=LinearPeriodicCellName.PERIODIC,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.PERIODIC],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.PERIODIC
                ],
                collection_format=collection_format,
            ).lines,
            final_cell_lines=_render_quimb_cell_helper(
                chain=chain,
                cell_name=LinearPeriodicCellName.FINAL,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.FINAL],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.FINAL
                ],
                collection_format=collection_format,
            ).lines,
            main_loop_lines=main_loop_lines,
            output_lines=output_lines,
        ),
    )


def _generate_einsum_linear_periodic_code(
    *,
    chain: LinearPeriodicChainSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate non-carry linear-periodic code for one einsum backend."""
    generator = _einsum_generator_for_engine(engine)
    import_lines = [
        "# Tensor Network Editor linear periodic mode",
        generator.import_line,
    ]
    shared_helper_lines = _render_einsum_shared_helper_lines(engine)
    helper_signature_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "",
        LinearPeriodicCellName.PERIODIC: "cell_index: int",
        LinearPeriodicCellName.FINAL: "cell_index: int",
    }
    helper_name_by_cell_name: dict[LinearPeriodicCellName, str] = {
        LinearPeriodicCellName.INITIAL: "build_initial_cell",
        LinearPeriodicCellName.PERIODIC: "build_periodic_cell",
        LinearPeriodicCellName.FINAL: "build_final_cell",
    }
    main_loop_lines, output_lines = _render_einsum_main_flow_lines(engine)
    return CodegenResult(
        engine=engine,
        code=render_linear_periodic_script(
            import_lines=import_lines,
            shared_helper_lines=shared_helper_lines,
            initial_cell_lines=_render_einsum_cell_helper(
                chain=chain,
                cell_name=LinearPeriodicCellName.INITIAL,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.INITIAL],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.INITIAL
                ],
                engine=engine,
                collection_format=collection_format,
            ).lines,
            periodic_cell_lines=_render_einsum_cell_helper(
                chain=chain,
                cell_name=LinearPeriodicCellName.PERIODIC,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.PERIODIC],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.PERIODIC
                ],
                engine=engine,
                collection_format=collection_format,
            ).lines,
            final_cell_lines=_render_einsum_cell_helper(
                chain=chain,
                cell_name=LinearPeriodicCellName.FINAL,
                helper_name=helper_name_by_cell_name[LinearPeriodicCellName.FINAL],
                helper_signature=helper_signature_by_cell_name[
                    LinearPeriodicCellName.FINAL
                ],
                engine=engine,
                collection_format=collection_format,
            ).lines,
            main_loop_lines=main_loop_lines,
            output_lines=output_lines,
        ),
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
                chain=chain,
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
                chain=chain,
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
                chain=chain,
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
                chain=chain,
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
                chain=chain,
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
                chain=chain,
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


def _render_quimb_cell_helper(
    *,
    chain: LinearPeriodicChainSpec,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    collection_format: TensorCollectionFormat,
) -> _RenderedCellHelper:
    """Render one non-carry cell helper for ``quimb``."""
    cell = _cell_from_chain(chain, cell_name)
    internal_spec = build_internal_linear_periodic_cell_network(
        cell,
        cell_name=cell_name,
    )
    prepared = prepare_network(internal_spec)
    collection_name = container_name_for_format(collection_format)
    incoming_ports = build_linear_periodic_interface_ports(
        cell,
        cell_name=cell_name,
        role=LinearPeriodicTensorRole.PREVIOUS,
    )
    outgoing_ports = build_linear_periodic_interface_ports(
        cell,
        cell_name=cell_name,
        role=LinearPeriodicTensorRole.NEXT,
    )
    label_expression_by_label = _build_quimb_label_expression_map(
        prepared=prepared,
        cell_name=cell_name,
        incoming_ports=incoming_ports,
        outgoing_ports=outgoing_ports,
    )
    interface_index_ids = {
        port.internal_index_id for port in (*incoming_ports, *outgoing_ports)
    }
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
        + flattened_tensor_collection_expression(collection_format, collection_name),
    ]
    assembly_lines: list[str]
    output_lines: list[str]
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        assembly_lines, output_lines = generator._render_manual_plan(
            prepared=prepared,
            collection_format=collection_format,
            collection_name=collection_name,
        )
        assembly_lines = [
            "network = qtn.TensorNetwork(network_tensors)",
            "",
            *assembly_lines,
        ]
        output_lines = [
            line
            for line in output_lines
            if not line.startswith("result = ") and not line.startswith("open_inds = ")
        ]
        output_lines.append("cell_tensors = list(remaining_operands.values())")
    else:
        assembly_lines = ["cell_tensors = list(network_tensors)"]
        output_lines = []
    local_open_expressions = [
        label_expression_by_label[index.label]
        for index in prepared.open_indices
        if index.spec.id not in interface_index_ids
    ]
    output_lines.append(
        "open_inds = " + _render_python_tuple_expression(local_open_expressions)
    )
    output_lines.append("result = cell_tensors[0] if len(cell_tensors) == 1 else None")
    output_lines.extend(
        [
            "return {",
            "    'tensors': cell_tensors,",
            "    'open_inds': open_inds,",
            "    'result': result,",
            "}",
        ]
    )
    sections = [
        CodeSection(title="Tensor collection", lines=tensor_collection_lines),
        CodeSection(title="Tensor construction", lines=tensor_construction_lines),
        CodeSection(title="Network setup", lines=network_setup_lines),
    ]
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        sections.append(CodeSection(title="Manual contraction", lines=assembly_lines))
    else:
        sections.append(CodeSection(title="Cell assembly", lines=assembly_lines))
    sections.append(CodeSection(title="Outputs", lines=output_lines))
    return render_linear_periodic_helper(
        helper_name=helper_name,
        helper_signature=helper_signature,
        return_annotation="dict[str, object]",
        sections=sections,
    )


def _render_quimb_carry_cell_helper(
    *,
    chain: LinearPeriodicChainSpec,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    collection_format: TensorCollectionFormat,
    simulation: _CarryPlanSimulation,
) -> _RenderedCellHelper:
    """Render one carry-mode cell helper for ``quimb``."""
    del chain
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


def _render_einsum_cell_helper(
    *,
    chain: LinearPeriodicChainSpec,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> _RenderedCellHelper:
    """Render one non-carry cell helper for an einsum backend."""
    cell = _cell_from_chain(chain, cell_name)
    internal_spec = build_internal_linear_periodic_cell_network(
        cell,
        cell_name=cell_name,
    )
    prepared = prepare_network(internal_spec)
    collection_name = container_name_for_format(collection_format)
    incoming_ports = build_linear_periodic_interface_ports(
        cell,
        cell_name=cell_name,
        role=LinearPeriodicTensorRole.PREVIOUS,
    )
    outgoing_ports = build_linear_periodic_interface_ports(
        cell,
        cell_name=cell_name,
        role=LinearPeriodicTensorRole.NEXT,
    )
    label_expression_by_label = _build_einsum_label_expression_map(
        prepared=prepared,
        cell_name=cell_name,
        incoming_ports=incoming_ports,
        outgoing_ports=outgoing_ports,
    )
    interface_index_ids = {
        port.internal_index_id for port in (*incoming_ports, *outgoing_ports)
    }
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
    assembly_lines: list[str]
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        assembly_lines = _render_einsum_manual_plan_lines(
            prepared=prepared,
            engine=engine,
            collection_format=collection_format,
            collection_name=collection_name,
            label_expression_by_label=label_expression_by_label,
        )
    else:
        assembly_lines = [
            "cell_operands = []",
            "cell_operand_labels = []",
        ]
        for tensor in prepared.tensors:
            assembly_lines.append(
                "cell_operands.append("
                + tensor_collection_reference_by_id(
                    prepared,
                    tensor.spec.id,
                    collection_format,
                    collection_name,
                )
                + ")"
            )
            assembly_lines.append(
                "cell_operand_labels.append("
                + _render_python_list_expression(
                    [label_expression_by_label[index.label] for index in tensor.indices]
                )
                + ")"
            )
    output_lines = [
        "open_labels = "
        + _render_python_list_expression(
            [
                label_expression_by_label[index.label]
                for index in prepared.open_indices
                if index.spec.id not in interface_index_ids
            ]
        ),
    ]
    output_lines.extend(
        [
            "return {",
            "    'operands': cell_operands,",
            "    'operand_labels': cell_operand_labels,",
            "    'open_labels': open_labels,",
            "}",
        ]
    )
    sections = [
        CodeSection(title="Tensor collection", lines=tensor_collection_lines),
        CodeSection(title="Tensor construction", lines=tensor_construction_lines),
    ]
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        sections.append(CodeSection(title="Manual contraction", lines=assembly_lines))
    else:
        sections.append(CodeSection(title="Cell assembly", lines=assembly_lines))
    sections.append(CodeSection(title="Outputs", lines=output_lines))
    return render_linear_periodic_helper(
        helper_name=helper_name,
        helper_signature=helper_signature,
        return_annotation="dict[str, object]",
        sections=sections,
    )


def _render_einsum_carry_cell_helper(
    *,
    chain: LinearPeriodicChainSpec,
    cell_name: LinearPeriodicCellName,
    helper_name: str,
    helper_signature: str,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    simulation: _CarryPlanSimulation,
) -> _RenderedCellHelper:
    """Render one carry-mode cell helper for an einsum backend."""
    del chain
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


def _render_quimb_main_flow_lines() -> tuple[list[str], list[str]]:
    """Render the non-carry outer flow for the ``quimb`` backend."""
    return (
        [
            "validate_chain_length(n)",
            "initial_cell = build_initial_cell()",
            "network_tensors = list(initial_cell['tensors'])",
            "open_inds = list(initial_cell['open_inds'])",
            "",
            "for cell_index in range(1, n - 1):",
            "    periodic_cell = build_periodic_cell(cell_index)",
            "    network_tensors.extend(periodic_cell['tensors'])",
            "    open_inds.extend(periodic_cell['open_inds'])",
        ],
        [
            "final_cell = build_final_cell(n - 1)",
            "network_tensors.extend(final_cell['tensors'])",
            "open_inds.extend(final_cell['open_inds'])",
            "open_inds = tuple(open_inds)",
            "network = qtn.TensorNetwork(network_tensors)",
            "result = network_tensors[0] if len(network_tensors) == 1 else None",
        ],
    )


def _render_quimb_carry_main_flow_lines() -> tuple[list[str], list[str]]:
    """Render the carry-mode outer flow for the ``quimb`` backend."""
    return (
        [
            "validate_chain_length(n)",
            "remaining_operands = {}",
            "open_inds = []",
            "",
            "previous_payload = build_initial_cell()",
            "",
            "for cell_index in range(1, n - 1):",
            "    previous_payload = build_periodic_cell(cell_index, previous_payload)",
        ],
        [
            "result = build_final_cell(n - 1, previous_payload)",
            "network_tensors = list(remaining_operands.values())",
            "if result is not None:",
            "    network_tensors.append(result)",
            "open_inds = tuple(open_inds)",
            "network = qtn.TensorNetwork(network_tensors) if network_tensors else None",
        ],
    )


def _render_einsum_main_flow_lines(engine: EngineName) -> tuple[list[str], list[str]]:
    """Render the non-carry outer flow for one einsum backend."""
    module_alias = _einsum_generator_for_engine(engine).module_alias
    return (
        [
            "validate_chain_length(n)",
            "initial_cell = build_initial_cell()",
            "einsum_operands: list[object] = []",
            "output_labels = list(initial_cell['open_labels'])",
            "for operand, operand_labels in zip(",
            "    initial_cell['operands'],",
            "    initial_cell['operand_labels'],",
            "    strict=True,",
            "):",
            "    einsum_operands.append(operand)",
            "    einsum_operands.append(operand_labels)",
            "",
            "for cell_index in range(1, n - 1):",
            "    periodic_cell = build_periodic_cell(cell_index)",
            "    output_labels.extend(periodic_cell['open_labels'])",
            "    for operand, operand_labels in zip(",
            "        periodic_cell['operands'],",
            "        periodic_cell['operand_labels'],",
            "        strict=True,",
            "    ):",
            "        einsum_operands.append(operand)",
            "        einsum_operands.append(operand_labels)",
        ],
        [
            "final_cell = build_final_cell(n - 1)",
            "output_labels.extend(final_cell['open_labels'])",
            "for operand, operand_labels in zip(",
            "    final_cell['operands'],",
            "    final_cell['operand_labels'],",
            "    strict=True,",
            "):",
            "    einsum_operands.append(operand)",
            "    einsum_operands.append(operand_labels)",
            "",
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
    )


def _render_einsum_carry_main_flow_lines() -> tuple[list[str], list[str]]:
    """Render the carry-mode outer flow for einsum backends."""
    return (
        [
            "validate_chain_length(n)",
            "remaining_operands = {}",
            "open_labels = []",
            "",
            "previous_payload = build_initial_cell()",
            "",
            "for cell_index in range(1, n - 1):",
            "    previous_payload = build_periodic_cell(cell_index, previous_payload)",
        ],
        ["result = build_final_cell(n - 1, previous_payload)"],
    )


def _render_einsum_manual_plan_lines(
    *,
    prepared: PreparedNetwork,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
    collection_name: str,
    label_expression_by_label: dict[str, str],
) -> list[str]:
    """Render one cell-local manual plan for an einsum backend."""
    generator = _einsum_generator_for_engine(engine)
    label_order = list(
        dict.fromkeys(
            index.label for tensor in prepared.tensors for index in tensor.indices
        )
    )
    label_to_int = {label: offset for offset, label in enumerate(label_order)}
    contraction_inputs = prepare_contraction_inputs(prepared)
    simulation = simulate_contraction_plan(
        initial_operand_ids=contraction_inputs.initial_operand_ids,
        initial_operands=contraction_inputs.initial_operands,
        initial_axis_names=contraction_inputs.initial_axis_names,
        dimension_by_label=contraction_inputs.dimension_by_label,
        plan=prepared.spec.contraction_plan,
    )
    step_result_indexes = {
        step.step_id: result_index for result_index, step in enumerate(simulation.steps)
    }
    base_operand_expressions = {
        tensor.spec.id: tensor_collection_reference_by_id(
            prepared,
            tensor.spec.id,
            collection_format,
            collection_name,
        )
        for tensor in prepared.tensors
    }
    lines = ["results_list = []", "", "cell_operands = []", "cell_operand_labels = []"]
    for step_index, step in enumerate(simulation.steps):
        latest_result_index = step_index - 1 if step_index > 0 else None
        lines.append(f"# Manual step {step.step_id}")
        lines.append(
            "results_list.append("
            + generator._render_manual_step_call(
                left_expression=render_operand_expression(
                    step.left_operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=step_result_indexes,
                    latest_result_index=latest_result_index,
                ),
                right_expression=render_operand_expression(
                    step.right_operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=step_result_indexes,
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
        lines.append("")
    latest_result_index = len(simulation.steps) - 1 if simulation.steps else None
    for operand_id in simulation.remaining_operand_ids:
        lines.append(
            "cell_operands.append("
            + render_operand_expression(
                operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=step_result_indexes,
                latest_result_index=latest_result_index,
            )
            + ")"
        )
        lines.append(
            "cell_operand_labels.append("
            + _render_python_list_expression(
                [
                    label_expression_by_label[label]
                    for label in simulation.remaining_operands[operand_id]
                ]
            )
            + ")"
        )
    return lines


def _einsum_generator_for_engine(
    engine: EngineName,
) -> EinsumNumpyCodeGenerator | EinsumTorchCodeGenerator:
    """Return the concrete einsum generator instance for ``engine``."""
    if engine is EngineName.EINSUM_NUMPY:
        return EinsumNumpyCodeGenerator()
    if engine is EngineName.EINSUM_TORCH:
        return EinsumTorchCodeGenerator()
    raise CodeGenerationError(f"The {engine.value} backend is not an einsum engine.")

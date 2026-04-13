"""Array-backend renderers for linear periodic code generation."""

from __future__ import annotations

from .._contraction_plan import (
    build_dimension_by_label,
    build_initial_operand_axis_names,
    build_initial_operand_labels,
    simulate_contraction_plan,
)
from .._linear_periodic import (
    LINEAR_PERIODIC_PREVIOUS_OPERAND_ID,
    LinearPeriodicTensorRole,
    build_internal_linear_periodic_cell_network,
    build_linear_periodic_interface_ports,
)
from ..errors import CodeGenerationError
from ..models import (
    CodegenResult,
    EngineName,
    LinearPeriodicCellName,
    LinearPeriodicChainSpec,
    TensorCollectionFormat,
)
from ._linear_periodic_expressions import (
    _build_einsum_label_expression_map,
    _build_quimb_label_expression_map,
    _carry_cell_key_prefix_expression,
    _operand_expression,
    _render_python_list_expression,
    _render_python_tuple_expression,
)
from ._linear_periodic_shared import (
    _build_carry_simulation_map,
    _CarryPlanSimulation,
    _cell_from_chain,
    _RenderedCellHelper,
)
from .common import (
    PreparedNetwork,
    container_name_for_format,
    flattened_tensor_collection_expression,
    prepare_network,
    render_operand_expression,
    render_tensor_collection_assignment,
    tensor_collection_reference_by_id,
)
from .einsum_numpy import EinsumNumpyCodeGenerator
from .einsum_torch import EinsumTorchCodeGenerator
from .quimb import QuimbCodeGenerator


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


def _generate_quimb_linear_periodic_code(
    *,
    chain: LinearPeriodicChainSpec,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate non-carry linear-periodic code for the ``quimb`` backend."""
    lines = [
        "# Tensor Network Editor linear periodic mode",
        "import numpy as np",
        "import quimb.tensor as qtn",
        "",
        "def interface_label(left_cell: int, right_cell: int, slot_index: int) -> str:",
        "    return f'lp_link_{left_cell}_{right_cell}_{slot_index}'",
        "",
        "def cell_label(cell_kind: str, cell_index: int, label_name: str) -> str:",
        "    return f'lp_{cell_kind}_{cell_index}_{label_name}'",
        "",
    ]
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
    for cell_name in (
        LinearPeriodicCellName.INITIAL,
        LinearPeriodicCellName.PERIODIC,
        LinearPeriodicCellName.FINAL,
    ):
        lines.extend(
            _render_quimb_cell_helper(
                chain=chain,
                cell_name=cell_name,
                helper_name=helper_name_by_cell_name[cell_name],
                helper_signature=helper_signature_by_cell_name[cell_name],
                collection_format=collection_format,
            ).lines
        )
        if cell_name is not LinearPeriodicCellName.FINAL:
            lines.append("")
    lines.extend(_render_quimb_main_flow_lines())
    return CodegenResult(engine=EngineName.QUIMB, code="\n".join(lines).strip() + "\n")


def _generate_einsum_linear_periodic_code(
    *,
    chain: LinearPeriodicChainSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate non-carry linear-periodic code for one einsum backend."""
    generator = _einsum_generator_for_engine(engine)
    lines = [
        "# Tensor Network Editor linear periodic mode",
        generator.import_line,
        "",
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
        "",
    ]
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
    for cell_name in (
        LinearPeriodicCellName.INITIAL,
        LinearPeriodicCellName.PERIODIC,
        LinearPeriodicCellName.FINAL,
    ):
        lines.extend(
            _render_einsum_cell_helper(
                chain=chain,
                cell_name=cell_name,
                helper_name=helper_name_by_cell_name[cell_name],
                helper_signature=helper_signature_by_cell_name[cell_name],
                engine=engine,
                collection_format=collection_format,
            ).lines
        )
        if cell_name is not LinearPeriodicCellName.FINAL:
            lines.append("")
    lines.extend(_render_einsum_main_flow_lines(engine))
    return CodegenResult(engine=engine, code="\n".join(lines).strip() + "\n")


def _generate_quimb_linear_periodic_carry_code(
    *,
    chain: LinearPeriodicChainSpec,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate carry-mode linear-periodic code for the ``quimb`` backend."""
    lines = [
        "# Tensor Network Editor linear periodic mode",
        "import numpy as np",
        "import quimb.tensor as qtn",
        "",
        "def interface_label(left_cell: int, right_cell: int, slot_index: int) -> str:",
        "    return f'lp_link_{left_cell}_{right_cell}_{slot_index}'",
        "",
        "def cell_label(cell_kind: str, cell_index: int, label_name: str) -> str:",
        "    return f'lp_{cell_kind}_{cell_index}_{label_name}'",
        "",
    ]
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
    for cell_name in (
        LinearPeriodicCellName.INITIAL,
        LinearPeriodicCellName.PERIODIC,
        LinearPeriodicCellName.FINAL,
    ):
        lines.extend(
            _render_quimb_carry_cell_helper(
                chain=chain,
                cell_name=cell_name,
                helper_name=helper_name_by_cell_name[cell_name],
                helper_signature=helper_signature_by_cell_name[cell_name],
                collection_format=collection_format,
                simulation=carry_simulation_by_cell_name[cell_name],
            ).lines
        )
        if cell_name is not LinearPeriodicCellName.FINAL:
            lines.append("")
    lines.extend(_render_quimb_carry_main_flow_lines())
    return CodegenResult(engine=EngineName.QUIMB, code="\n".join(lines).strip() + "\n")


def _generate_einsum_linear_periodic_carry_code(
    *,
    chain: LinearPeriodicChainSpec,
    engine: EngineName,
    collection_format: TensorCollectionFormat,
) -> CodegenResult:
    """Generate carry-mode linear-periodic code for one einsum backend."""
    generator = _einsum_generator_for_engine(engine)
    lines = [
        "# Tensor Network Editor linear periodic mode",
        generator.import_line,
        "",
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
        "",
    ]
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
    for cell_name in (
        LinearPeriodicCellName.INITIAL,
        LinearPeriodicCellName.PERIODIC,
        LinearPeriodicCellName.FINAL,
    ):
        lines.extend(
            _render_einsum_carry_cell_helper(
                chain=chain,
                cell_name=cell_name,
                helper_name=helper_name_by_cell_name[cell_name],
                helper_signature=helper_signature_by_cell_name[cell_name],
                engine=engine,
                collection_format=collection_format,
                simulation=carry_simulation_by_cell_name[cell_name],
            ).lines
        )
        if cell_name is not LinearPeriodicCellName.FINAL:
            lines.append("")
    lines.extend(_render_einsum_carry_main_flow_lines())
    return CodegenResult(engine=engine, code="\n".join(lines).strip() + "\n")


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
    body_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id=tensor_value_by_id,
    )
    body_lines.append("")
    body_lines.append(
        "network_tensors = "
        + flattened_tensor_collection_expression(collection_format, collection_name)
    )
    body_lines.append("network = qtn.TensorNetwork(network_tensors)")
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        body_lines.extend(
            generator._render_manual_plan(
                prepared=prepared,
                collection_format=collection_format,
                collection_name=collection_name,
            )
        )
        body_lines.append("cell_tensors = list(remaining_operands.values())")
    else:
        body_lines.append("cell_tensors = list(network_tensors)")
    local_open_expressions = [
        label_expression_by_label[index.label]
        for index in prepared.open_indices
        if index.spec.id not in interface_index_ids
    ]
    body_lines.append(
        "open_inds = " + _render_python_tuple_expression(local_open_expressions)
    )
    body_lines.append("result = cell_tensors[0] if len(cell_tensors) == 1 else None")
    body_lines.extend(
        [
            "return {",
            "    'tensors': cell_tensors,",
            "    'open_inds': open_inds,",
            "    'result': result,",
            "}",
        ]
    )
    helper_lines = [f"def {helper_name}({helper_signature}) -> dict[str, object]:"]
    helper_lines.extend([f"    {line}" if line else "" for line in body_lines])
    return _RenderedCellHelper(lines=helper_lines)


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
    body_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id=tensor_value_by_id,
    )
    body_lines.append("")
    body_lines.append(
        "network_tensors = "
        + flattened_tensor_collection_expression(collection_format, collection_name)
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
    if simulation.incoming_ports:
        previous_operand_index = simulation.previous_operand_interface_index or 0
        expected_interface = [
            label_expression_by_label[label] for label in simulation.incoming_labels
        ]
        body_lines.extend(
            [
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
        )
        base_operand_expressions[LINEAR_PERIODIC_PREVIOUS_OPERAND_ID] = (
            "previous_operand"
        )
    body_lines.append("network = qtn.TensorNetwork(network_tensors)")
    body_lines.append("results_list = []")
    body_lines.append("")
    for step in simulation.real_steps:
        left_tag = generator._operand_tag(step.left_operand_id)
        right_tag = generator._operand_tag(step.right_operand_id)
        step_tag = generator._operand_tag(step.step_id)
        body_lines.append(f"# Manual step {step.step_id}")
        body_lines.append(f"network.contract_between({left_tag!r}, {right_tag!r})")
        body_lines.append(f"network[{left_tag!r}].add_tag({step_tag!r})")
        body_lines.append(f"results_list.append(network[{step_tag!r}])")
        body_lines.append("")
    latest_result_index = (
        len(simulation.real_steps) - 1 if simulation.real_steps else None
    )
    local_open_expressions = [
        label_expression_by_label[label] for label in simulation.local_open_labels
    ]
    if local_open_expressions:
        body_lines.append(
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
        body_lines.append(
            "cell_key_prefix = " + _carry_cell_key_prefix_expression(cell_name)
        )
        for operand_id in local_remaining_operand_ids:
            body_lines.append(
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
        body_lines.extend(
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
        body_lines.append(
            "return "
            + _operand_expression(
                operand_id=local_remaining_operand_ids[0],
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=latest_result_index,
            )
        )
    else:
        body_lines.append("return None")
    helper_lines = [
        f"def {helper_name}({helper_signature}) -> dict[str, object] | object | None:"
    ]
    helper_lines.extend([f"    {line}" if line else "" for line in body_lines])
    return _RenderedCellHelper(lines=helper_lines)


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
    body_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id=tensor_value_by_id,
    )
    body_lines.append("")
    if (
        prepared.spec.contraction_plan is not None
        and prepared.spec.contraction_plan.steps
    ):
        body_lines.extend(
            _render_einsum_manual_plan_lines(
                prepared=prepared,
                engine=engine,
                collection_format=collection_format,
                collection_name=collection_name,
                label_expression_by_label=label_expression_by_label,
            )
        )
    else:
        body_lines.extend(
            [
                "cell_operands = []",
                "cell_operand_labels = []",
            ]
        )
        for tensor in prepared.tensors:
            body_lines.append(
                "cell_operands.append("
                + tensor_collection_reference_by_id(
                    prepared,
                    tensor.spec.id,
                    collection_format,
                    collection_name,
                )
                + ")"
            )
            body_lines.append(
                "cell_operand_labels.append("
                + _render_python_list_expression(
                    [label_expression_by_label[index.label] for index in tensor.indices]
                )
                + ")"
            )
    local_open_expressions = [
        label_expression_by_label[index.label]
        for index in prepared.open_indices
        if index.spec.id not in interface_index_ids
    ]
    body_lines.append(
        "open_labels = " + _render_python_list_expression(local_open_expressions)
    )
    body_lines.extend(
        [
            "return {",
            "    'operands': cell_operands,",
            "    'operand_labels': cell_operand_labels,",
            "    'open_labels': open_labels,",
            "}",
        ]
    )
    helper_lines = [f"def {helper_name}({helper_signature}) -> dict[str, object]:"]
    helper_lines.extend([f"    {line}" if line else "" for line in body_lines])
    return _RenderedCellHelper(lines=helper_lines)


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
    body_lines = render_tensor_collection_assignment(
        collection_name=collection_name,
        collection_format=collection_format,
        prepared=prepared,
        tensor_value_by_id=tensor_value_by_id,
    )
    body_lines.append("")
    base_operand_expressions = {
        tensor.spec.id: tensor_collection_reference_by_id(
            prepared,
            tensor.spec.id,
            collection_format,
            collection_name,
        )
        for tensor in prepared.tensors
    }
    if simulation.incoming_ports:
        previous_operand_index = simulation.previous_operand_interface_index or 0
        expected_interface = [
            label_expression_by_label[label] for label in simulation.incoming_labels
        ]
        body_lines.extend(
            [
                "previous_interface = list(previous_payload['outgoing_interface'])",
                "previous_operands = list(previous_payload['outgoing_operands'])",
                "expected_interface = "
                + _render_python_list_expression(expected_interface),
                "if previous_interface != expected_interface "
                + f"or len(previous_operands) != {len(simulation.incoming_ports)}:",
                "    raise ValueError('Previous payload interface does not match this cell.')",
                f"previous_operand = previous_operands[{previous_operand_index}]",
            ]
        )
        base_operand_expressions[LINEAR_PERIODIC_PREVIOUS_OPERAND_ID] = (
            "previous_operand"
        )
    label_order = list(
        dict.fromkeys(
            index.label for tensor in prepared.tensors for index in tensor.indices
        )
    )
    for step in simulation.real_steps:
        for label in (*step.left_labels, *step.right_labels, *step.surviving_labels):
            if label not in label_order:
                label_order.append(label)
    label_to_int = {label: offset for offset, label in enumerate(label_order)}
    lines_for_plan = ["results_list = []", ""]
    for step_index, step in enumerate(simulation.real_steps):
        latest_result_index = step_index - 1 if step_index > 0 else None
        lines_for_plan.append(f"# Manual step {step.step_id}")
        lines_for_plan.append(
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
        lines_for_plan.append("")
    body_lines.extend(lines_for_plan)
    latest_result_index = (
        len(simulation.real_steps) - 1 if simulation.real_steps else None
    )
    local_open_expressions = [
        label_expression_by_label[label] for label in simulation.local_open_labels
    ]
    if local_open_expressions:
        body_lines.append(
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
        body_lines.append(
            "cell_key_prefix = " + _carry_cell_key_prefix_expression(cell_name)
        )
        for operand_id in local_remaining_operand_ids:
            body_lines.append(
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
        body_lines.extend(
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
        body_lines.append(
            "return "
            + _operand_expression(
                operand_id=local_remaining_operand_ids[0],
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=simulation.result_index_by_step_id,
                latest_result_index=latest_result_index,
            )
        )
    else:
        body_lines.append("return None")
    helper_lines = [
        f"def {helper_name}({helper_signature}) -> dict[str, object] | object | None:"
    ]
    helper_lines.extend([f"    {line}" if line else "" for line in body_lines])
    return _RenderedCellHelper(lines=helper_lines)


def _render_quimb_main_flow_lines() -> list[str]:
    """Render the non-carry outer flow for the ``quimb`` backend."""
    return [
        "",
        "if n < 2:",
        "    raise ValueError('n must be at least 2 for a linear periodic chain.')",
        "",
        "initial_cell = build_initial_cell()",
        "network_tensors = list(initial_cell['tensors'])",
        "open_inds = list(initial_cell['open_inds'])",
        "",
        "for cell_index in range(1, n - 1):",
        "    periodic_cell = build_periodic_cell(cell_index)",
        "    network_tensors.extend(periodic_cell['tensors'])",
        "    open_inds.extend(periodic_cell['open_inds'])",
        "",
        "final_cell = build_final_cell(n - 1)",
        "network_tensors.extend(final_cell['tensors'])",
        "open_inds.extend(final_cell['open_inds'])",
        "open_inds = tuple(open_inds)",
        "network = qtn.TensorNetwork(network_tensors)",
        "result = network_tensors[0] if len(network_tensors) == 1 else None",
    ]


def _render_quimb_carry_main_flow_lines() -> list[str]:
    """Render the carry-mode outer flow for the ``quimb`` backend."""
    return [
        "",
        "if n < 2:",
        "    raise ValueError('n must be at least 2 for a linear periodic chain.')",
        "",
        "remaining_operands = {}",
        "open_inds = []",
        "",
        "previous_payload = build_initial_cell()",
        "",
        "for cell_index in range(1, n - 1):",
        "    previous_payload = build_periodic_cell(cell_index, previous_payload)",
        "",
        "result = build_final_cell(n - 1, previous_payload)",
        "network_tensors = list(remaining_operands.values())",
        "if result is not None:",
        "    network_tensors.append(result)",
        "open_inds = tuple(open_inds)",
        "network = qtn.TensorNetwork(network_tensors) if network_tensors else None",
    ]


def _render_einsum_main_flow_lines(engine: EngineName) -> list[str]:
    """Render the non-carry outer flow for one einsum backend."""
    module_alias = _einsum_generator_for_engine(engine).module_alias
    return [
        "",
        "if n < 2:",
        "    raise ValueError('n must be at least 2 for a linear periodic chain.')",
        "",
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
        "",
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
    ]


def _render_einsum_carry_main_flow_lines() -> list[str]:
    """Render the carry-mode outer flow for einsum backends."""
    return [
        "",
        "if n < 2:",
        "    raise ValueError('n must be at least 2 for a linear periodic chain.')",
        "",
        "remaining_operands = {}",
        "open_labels = []",
        "",
        "previous_payload = build_initial_cell()",
        "",
        "for cell_index in range(1, n - 1):",
        "    previous_payload = build_periodic_cell(cell_index, previous_payload)",
        "",
        "result = build_final_cell(n - 1, previous_payload)",
    ]


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
    simulation = simulate_contraction_plan(
        initial_operand_ids=tuple(tensor.spec.id for tensor in prepared.tensors),
        initial_operands=build_initial_operand_labels(prepared),
        initial_axis_names=build_initial_operand_axis_names(prepared),
        dimension_by_label=build_dimension_by_label(prepared),
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

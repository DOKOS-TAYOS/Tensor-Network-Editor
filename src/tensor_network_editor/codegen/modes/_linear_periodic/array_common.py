"""Shared array-backend helpers for linear-periodic code generation."""

from __future__ import annotations

from ....errors import CodeGenerationError
from ....internal.analysis._contraction_plan import (
    prepare_contraction_inputs,
    simulate_contraction_plan,
)
from ....models import (
    CodegenResult,
    EngineName,
    LinearPeriodicChainSpec,
    TensorCollectionFormat,
)
from ...backends.einsum_numpy import EinsumNumpyCodeGenerator
from ...backends.einsum_torch import EinsumTorchCodeGenerator
from ...shared._linear_periodic_expressions import _render_python_list_expression
from ...shared.common import (
    PreparedNetwork,
    render_operand_expression,
    tensor_collection_reference_by_id,
)
from .common import render_linear_periodic_shared_helpers


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
            from .array_carry import _generate_quimb_linear_periodic_carry_code

            return _generate_quimb_linear_periodic_carry_code(
                chain=chain,
                collection_format=collection_format,
            )
        from .array_standard import _generate_quimb_linear_periodic_code

        return _generate_quimb_linear_periodic_code(
            chain=chain,
            collection_format=collection_format,
        )
    if engine in {EngineName.EINSUM_NUMPY, EngineName.EINSUM_TORCH}:
        if uses_carry_mode:
            from .array_carry import _generate_einsum_linear_periodic_carry_code

            return _generate_einsum_linear_periodic_carry_code(
                chain=chain,
                engine=engine,
                collection_format=collection_format,
            )
        from .array_standard import _generate_einsum_linear_periodic_code

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

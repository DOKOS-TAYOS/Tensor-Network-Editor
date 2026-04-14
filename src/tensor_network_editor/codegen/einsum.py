"""Shared einsum-based code generation helpers."""

from __future__ import annotations

from abc import ABC
from string import ascii_letters

from .._contraction_plan import (
    prepare_contraction_inputs,
    simulate_contraction_plan,
)
from ..models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat
from .base import CodeGenerator
from .common import (
    CodeSection,
    PreparedNetwork,
    PreparedTensor,
    container_name_for_format,
    prepare_network,
    render_code_sections,
    render_manual_step_comment,
    render_operand_expression,
    render_remaining_operands_mapping,
    render_tensor_collection_assignment,
    render_tensor_collection_initialization,
    tensor_collection_reference,
    tensor_display_name_by_id,
)


class BaseEinsumCodeGenerator(CodeGenerator, ABC):
    """Base generator for NumPy and PyTorch einsum backends."""

    engine: EngineName
    import_line: str
    module_alias: str
    zero_initializer_suffix: str = ""
    empty_network_expression: str

    def generate(
        self,
        spec: NetworkSpec,
        collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    ) -> CodegenResult:
        """Generate einsum-based Python code for ``spec``."""
        prepared = prepare_network(spec)
        collection_name = container_name_for_format(collection_format)
        tensor_collection_lines = render_tensor_collection_initialization(
            collection_name,
            collection_format,
        )
        tensor_construction_lines = render_tensor_collection_assignment(
            collection_name=collection_name,
            collection_format=collection_format,
            prepared=prepared,
            tensor_value_by_id={
                tensor.spec.id: (
                    f"{self.module_alias}.zeros({tensor.spec.shape!r}"
                    f"{self.zero_initializer_suffix})"
                )
                for tensor in prepared.tensors
            },
            include_initialization=False,
        )

        if spec.contraction_plan is not None and spec.contraction_plan.steps:
            contraction_lines, output_lines = self._render_manual_plan(
                prepared=prepared,
                collection_format=collection_format,
                collection_name=collection_name,
            )
            contraction_title = "Manual contraction"
        else:
            contraction_lines, output_lines = self._render_full_network_einsum(
                prepared=prepared,
                collection_format=collection_format,
                collection_name=collection_name,
            )
            contraction_title = "Contraction"

        return CodegenResult(
            engine=self.engine,
            code=render_code_sections(
                CodeSection(title=None, lines=[self.import_line]),
                CodeSection(title="Tensor collection", lines=tensor_collection_lines),
                CodeSection(
                    title="Tensor construction",
                    lines=tensor_construction_lines,
                ),
                CodeSection(title=contraction_title, lines=contraction_lines),
                CodeSection(title="Outputs", lines=output_lines),
            ),
        )

    def _render_full_network_einsum(
        self,
        *,
        prepared: PreparedNetwork,
        collection_format: TensorCollectionFormat,
        collection_name: str,
    ) -> tuple[list[str], list[str]]:
        """Render a single einsum call for the full network."""
        if not prepared.tensors:
            return (
                ["# Empty network contracts to the scalar identity."],
                [f"result = {self.empty_network_expression}"],
            )

        label_order: list[str] = []
        for tensor in prepared.tensors:
            for index in tensor.indices:
                if index.label not in label_order:
                    label_order.append(index.label)

        label_to_int = {label: offset for offset, label in enumerate(label_order)}
        output_labels = [index.label for index in prepared.open_indices]

        use_string_equation = len(label_order) <= len(ascii_letters)
        symbol_map = {
            label: ascii_letters[offset]
            for offset, label in enumerate(label_order[: len(ascii_letters)])
        }
        if use_string_equation:
            equation = self._build_equation(
                tensors=prepared.tensors,
                output_labels=output_labels,
                symbol_map=symbol_map,
            )
            operand_names = ", ".join(
                tensor_collection_reference(tensor, collection_format, collection_name)
                for tensor in prepared.tensors
            )
            return (
                [f"# Einsum equation: {equation}"],
                [f"result = {self.module_alias}.einsum({equation!r}, {operand_names})"],
            )

        sublist_args: list[str] = []
        for tensor in prepared.tensors:
            sublist_args.append(
                tensor_collection_reference(tensor, collection_format, collection_name)
            )
            sublist_args.append(
                str([label_to_int[index.label] for index in tensor.indices])
            )
        sublist_args.append(str([label_to_int[label] for label in output_labels]))
        return (
            [
                "# Einsum uses the integer-sublist form because the network uses many labels.",
            ],
            [f"result = {self.module_alias}.einsum(" + ", ".join(sublist_args) + ")"],
        )

    def _render_manual_plan(
        self,
        *,
        prepared: PreparedNetwork,
        collection_format: TensorCollectionFormat,
        collection_name: str,
    ) -> tuple[list[str], list[str]]:
        """Render step-by-step einsum calls for a saved manual plan."""
        label_order: list[str] = []
        for tensor in prepared.tensors:
            for index in tensor.indices:
                if index.label not in label_order:
                    label_order.append(index.label)
        use_string_labels = len(label_order) <= len(ascii_letters)
        symbol_map = {
            label: ascii_letters[offset]
            for offset, label in enumerate(label_order[: len(ascii_letters)])
        }
        contraction_inputs = prepare_contraction_inputs(prepared)
        label_to_int = {label: offset for offset, label in enumerate(label_order)}
        simulation = simulate_contraction_plan(
            initial_operand_ids=contraction_inputs.initial_operand_ids,
            initial_operands=contraction_inputs.initial_operands,
            initial_axis_names=contraction_inputs.initial_axis_names,
            dimension_by_label=contraction_inputs.dimension_by_label,
            plan=prepared.spec.contraction_plan,
        )
        step_result_indexes = {
            step.step_id: result_index
            for result_index, step in enumerate(simulation.steps)
        }
        base_operand_expressions = {
            tensor.spec.id: tensor_collection_reference(
                tensor,
                collection_format,
                collection_name,
            )
            for tensor in prepared.tensors
        }
        tensor_names_by_id = tensor_display_name_by_id(prepared)

        contraction_lines = ["results_list = []", ""]
        for step_index, step in enumerate(simulation.steps):
            latest_result_index = step_index - 1 if step_index > 0 else None
            contraction_lines.append(
                render_manual_step_comment(
                    step.step_id,
                    step.left_operand_id,
                    step.right_operand_id,
                )
            )
            contraction_lines.append(
                "results_list.append("
                + self._render_manual_step_call(
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
                    use_string_labels=use_string_labels,
                    symbol_map=symbol_map,
                    label_to_int=label_to_int,
                )
                + ")"
            )
            contraction_lines.append("")

        final_result_index = len(simulation.steps) - 1 if simulation.steps else None
        output_lines: list[str] = []
        if len(simulation.remaining_operand_ids) > 1:
            output_lines.append("remaining_operand_labels = {")
            for operand_id in simulation.remaining_operand_ids:
                operand_expression = render_operand_expression(
                    operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=step_result_indexes,
                    latest_result_index=final_result_index,
                )
                output_lines.append(
                    f"    {operand_expression!r}: "
                    f"{self._render_remaining_label_sequence(simulation.remaining_operands[operand_id], use_string_labels=use_string_labels, symbol_map=symbol_map, label_to_int=label_to_int)!r},"
                )
            output_lines.append("}")
            output_lines.append("")

        output_lines.extend(
            render_remaining_operands_mapping(
                operand_ids=simulation.remaining_operand_ids,
                source_tensor_ids_by_operand_id=simulation.source_tensor_ids_by_operand_id,
                tensor_names_by_id=tensor_names_by_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=step_result_indexes,
                latest_result_index=final_result_index,
            )
        )
        if len(simulation.remaining_operand_ids) == 1:
            output_lines.append(
                "result = "
                + render_operand_expression(
                    simulation.remaining_operand_ids[0],
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=step_result_indexes,
                    latest_result_index=final_result_index,
                )
            )
        return contraction_lines, output_lines

    def _render_manual_step_call(
        self,
        *,
        left_expression: str,
        right_expression: str,
        left_labels: tuple[str, ...],
        right_labels: tuple[str, ...],
        output_labels: tuple[str, ...],
        use_string_labels: bool,
        symbol_map: dict[str, str],
        label_to_int: dict[str, int],
    ) -> str:
        """Render the einsum call for one manual contraction step."""
        if use_string_labels:
            equation = (
                "".join(symbol_map[label] for label in left_labels)
                + ","
                + "".join(symbol_map[label] for label in right_labels)
                + "->"
                + "".join(symbol_map[label] for label in output_labels)
            )
            return (
                f"{self.module_alias}.einsum("
                f"{equation!r}, {left_expression}, {right_expression})"
            )

        return (
            f"{self.module_alias}.einsum("
            f"{left_expression}, "
            f"{[label_to_int[label] for label in left_labels]!r}, "
            f"{right_expression}, "
            f"{[label_to_int[label] for label in right_labels]!r}, "
            f"{[label_to_int[label] for label in output_labels]!r})"
        )

    @staticmethod
    def _render_remaining_label_sequence(
        labels: tuple[str, ...],
        *,
        use_string_labels: bool,
        symbol_map: dict[str, str],
        label_to_int: dict[str, int],
    ) -> list[str]:
        """Render surviving labels for partial manual-plan exports."""
        if use_string_labels:
            return [symbol_map[label] for label in labels]
        return [f"label_{label_to_int[label]}" for label in labels]

    @staticmethod
    def _build_equation(
        tensors: list[PreparedTensor],
        output_labels: list[str],
        symbol_map: dict[str, str],
    ) -> str:
        """Build a standard einsum equation string for the prepared tensors."""
        input_terms = [
            "".join(symbol_map[index.label] for index in tensor.indices)
            for tensor in tensors
        ]
        output_term = "".join(symbol_map[label] for label in output_labels)
        return ",".join(input_terms) + "->" + output_term

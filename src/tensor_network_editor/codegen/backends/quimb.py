"""Code generation for the ``quimb`` backend."""

from __future__ import annotations

from ...internal.analysis._contraction_plan import (
    prepare_contraction_inputs,
    simulate_contraction_plan,
)
from ...models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat
from ..shared.base import CodeGenerator
from ..shared.common import (
    CodeSection,
    PreparedNetwork,
    container_name_for_format,
    flattened_tensor_collection_expression,
    joined_tensor_display_name,
    prepare_network,
    render_code_sections,
    render_manual_step_comment,
    render_results_list_reference,
    render_tensor_collection_assignment,
    render_tensor_collection_initialization,
    render_tensor_data_assignments,
    tensor_collection_reference,
    tensor_display_name_by_id,
    uses_external_pt_tensor_data,
)


class QuimbCodeGenerator(CodeGenerator):
    """Generate ``quimb`` tensor-network code."""

    engine = EngineName.QUIMB

    def generate(
        self,
        spec: NetworkSpec,
        collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
        *,
        validate: bool = True,
    ) -> CodegenResult:
        """Generate ``quimb`` code for ``spec``."""
        prepared = prepare_network(spec, validate=validate)
        collection_name = container_name_for_format(collection_format)
        tensor_collection_lines = render_tensor_collection_initialization(
            collection_name,
            collection_format,
        )
        tensor_data_lines = render_tensor_data_assignments(
            prepared,
            module_alias="np",
            zeros_initializer_suffix=", dtype=float",
            literal_constructor_name="array",
        )
        tensor_construction_lines = render_tensor_collection_assignment(
            collection_name=collection_name,
            collection_format=collection_format,
            prepared=prepared,
            tensor_value_by_id={
                tensor.spec.id: (
                    f"qtn.Tensor(data={tensor.data_variable_name}, "
                    f"inds={tuple(index.label for index in tensor.indices)!r}, "
                    f"tags={(tensor.spec.name, self._operand_tag(tensor.spec.id))!r})"
                )
                for tensor in prepared.tensors
            },
            include_initialization=False,
        )
        network_setup_lines = [
            "network_tensors = "
            + flattened_tensor_collection_expression(
                collection_format, collection_name
            ),
            "network = qtn.TensorNetwork(network_tensors)",
        ]

        if spec.contraction_plan is not None and spec.contraction_plan.steps:
            contraction_lines, output_lines = self._render_manual_plan(
                prepared=prepared,
                collection_format=collection_format,
                collection_name=collection_name,
            )
        elif prepared.open_indices:
            contraction_lines = []
            output_lines = [
                "open_inds = ("
                + ", ".join(repr(index.label) for index in prepared.open_indices)
                + ",)"
            ]
        else:
            contraction_lines = []
            output_lines = ["open_inds = ()"]

        return CodegenResult(
            engine=self.engine,
            code=render_code_sections(
                CodeSection(
                    title=None,
                    lines=self._render_import_lines(prepared),
                ),
                CodeSection(title="Tensor collection", lines=tensor_collection_lines),
                CodeSection(title="Tensor data", lines=tensor_data_lines),
                CodeSection(
                    title="Tensor construction",
                    lines=tensor_construction_lines,
                ),
                CodeSection(title="Network setup", lines=network_setup_lines),
                CodeSection(title="Manual contraction", lines=contraction_lines),
                CodeSection(title="Outputs", lines=output_lines),
            ),
        )

    @staticmethod
    def _render_import_lines(prepared: PreparedNetwork) -> list[str]:
        """Render imports needed by the Quimb backend."""
        lines = ["import numpy as np"]
        if uses_external_pt_tensor_data(prepared):
            lines.append("import torch")
        lines.append("import quimb.tensor as qtn")
        return lines

    def _render_manual_plan(
        self,
        *,
        prepared: PreparedNetwork,
        collection_format: TensorCollectionFormat,
        collection_name: str,
    ) -> tuple[list[str], list[str]]:
        """Render a saved manual contraction plan against one ``TensorNetwork``."""
        contraction_inputs = prepare_contraction_inputs(prepared)
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
        for step in simulation.steps:
            left_tag = self._operand_tag(step.left_operand_id)
            right_tag = self._operand_tag(step.right_operand_id)
            step_tag = self._operand_tag(step.step_id)
            contraction_lines.append(
                render_manual_step_comment(
                    step.step_id,
                    step.left_operand_id,
                    step.right_operand_id,
                )
            )
            contraction_lines.append(
                f"network.contract_between({left_tag!r}, {right_tag!r})"
            )
            contraction_lines.append(f"network[{left_tag!r}].add_tag({step_tag!r})")
            contraction_lines.append(f"results_list.append(network[{step_tag!r}])")
            contraction_lines.append("")

        final_result_index = len(simulation.steps) - 1 if simulation.steps else None
        output_lines = self._render_remaining_operands(
            operand_ids=simulation.remaining_operand_ids,
            source_tensor_ids_by_operand_id=simulation.source_tensor_ids_by_operand_id,
            tensor_names_by_id=tensor_names_by_id,
            base_operand_expressions=base_operand_expressions,
            step_result_indexes=step_result_indexes,
            latest_result_index=final_result_index,
        )
        if len(simulation.remaining_operand_ids) == 1:
            output_lines.append(
                "result = "
                + self._operand_expression(
                    simulation.remaining_operand_ids[0],
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=step_result_indexes,
                    latest_result_index=final_result_index,
                )
            )
            output_lines.append("open_inds = tuple(result.inds)")
        return contraction_lines, output_lines

    @staticmethod
    def _operand_tag(operand_id: str) -> str:
        """Return the internal tag used to track an operand inside the network."""
        return f"__tne_operand_{operand_id}"

    @classmethod
    def _operand_expression(
        cls,
        operand_id: str,
        *,
        base_operand_expressions: dict[str, str],
        step_result_indexes: dict[str, int],
        latest_result_index: int | None,
    ) -> str:
        """Resolve an operand id to its generated Python expression."""
        if operand_id in step_result_indexes:
            return render_results_list_reference(
                step_result_indexes[operand_id],
                latest_result_index=latest_result_index,
            )
        return base_operand_expressions[operand_id]

    @classmethod
    def _render_remaining_operands(
        cls,
        *,
        operand_ids: tuple[str, ...],
        source_tensor_ids_by_operand_id: dict[str, tuple[str, ...]],
        tensor_names_by_id: dict[str, str],
        base_operand_expressions: dict[str, str],
        step_result_indexes: dict[str, int],
        latest_result_index: int | None,
    ) -> list[str]:
        """Render the ``remaining_operands`` mapping for partial plans."""
        lines = ["remaining_operands = {"]
        for operand_id in operand_ids:
            operand_name = joined_tensor_display_name(
                source_tensor_ids_by_operand_id[operand_id],
                tensor_names_by_id,
            )
            lines.append(
                f"    {operand_name!r}: "
                + cls._operand_expression(
                    operand_id,
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=step_result_indexes,
                    latest_result_index=latest_result_index,
                )
                + ","
            )
        lines.append("}")
        return lines

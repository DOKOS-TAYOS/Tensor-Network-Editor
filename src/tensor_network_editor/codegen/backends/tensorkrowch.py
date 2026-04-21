"""Code generation for the ``tensorkrowch`` backend."""

from __future__ import annotations

from ...errors import CodeGenerationError
from ...internal.analysis._contraction_plan import (
    prepare_contraction_inputs,
    simulate_contraction_plan,
)
from ...models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat
from ..shared.base import CodeGenerator
from ..shared.common import (
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
    render_tensor_data_assignments,
    tensor_collection_reference_by_id,
    tensor_display_name_by_id,
)


class TensorKrowchCodeGenerator(CodeGenerator):
    """Generate ``tensorkrowch`` code for a network specification."""

    engine = EngineName.TENSORKROWCH

    def generate(
        self,
        spec: NetworkSpec,
        collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
        *,
        validate: bool = True,
    ) -> CodegenResult:
        """Generate ``tensorkrowch`` code for ``spec``."""
        prepared = prepare_network(spec, validate=validate)
        collection_name = container_name_for_format(collection_format)
        tensor_collection_lines = render_tensor_collection_initialization(
            collection_name,
            collection_format,
        )
        network_setup_lines = ["network = tk.TensorNetwork()"]
        tensor_data_lines = render_tensor_data_assignments(
            prepared,
            module_alias="torch",
            zeros_initializer_suffix=", dtype=torch.float32",
            literal_constructor_name="tensor",
        )
        tensor_construction_lines = render_tensor_collection_assignment(
            collection_name=collection_name,
            collection_format=collection_format,
            prepared=prepared,
            tensor_value_by_id={
                tensor.spec.id: (
                    f"tk.Node(tensor={tensor.data_variable_name}, "
                    f"axes_names={tuple(index.spec.name for index in tensor.indices)!r}, "
                    f"name={self.node_name(tensor)!r}, network=network)"
                )
                for tensor in prepared.tensors
            },
            include_initialization=False,
        )

        connection_lines: list[str] = []
        if prepared.edges:
            connection_lines.append("edges_list = []")
            for edge in prepared.edges:
                left_tensor = tensor_collection_reference_by_id(
                    prepared,
                    edge.spec.left.tensor_id,
                    collection_format,
                    collection_name,
                )
                right_tensor = tensor_collection_reference_by_id(
                    prepared,
                    edge.spec.right.tensor_id,
                    collection_format,
                    collection_name,
                )
                connection_lines.append(f"# Edge {edge.spec.name}")
                connection_lines.append(
                    "edges_list.append(("
                    f"{edge.spec.name!r}, "
                    f"tk.connect({left_tensor}[{edge.left.spec.name!r}], {right_tensor}[{edge.right.spec.name!r}])"
                    "))"
                )

        if spec.contraction_plan is not None and spec.contraction_plan.steps:
            contraction_lines, output_lines = self._render_manual_plan(
                prepared=prepared,
                collection_format=collection_format,
                collection_name=collection_name,
            )
        else:
            contraction_lines = []
            output_lines = [
                "open_edges = ("
                + ", ".join(
                    f"{tensor_collection_reference_by_id(prepared, index.tensor.id, collection_format, collection_name)}[{index.spec.name!r}]"
                    for index in prepared.open_indices
                )
                + ("," if prepared.open_indices else "")
                + ")"
            ]

        return CodegenResult(
            engine=self.engine,
            code=render_code_sections(
                CodeSection(
                    title=None,
                    lines=["import torch", "import tensorkrowch as tk"],
                ),
                CodeSection(title="Tensor collection", lines=tensor_collection_lines),
                CodeSection(title="Network setup", lines=network_setup_lines),
                CodeSection(title="Tensor data", lines=tensor_data_lines),
                CodeSection(
                    title="Tensor construction",
                    lines=tensor_construction_lines,
                ),
                CodeSection(title="Network connections", lines=connection_lines),
                CodeSection(title="Manual contraction", lines=contraction_lines),
                CodeSection(title="Outputs", lines=output_lines),
            ),
        )

    @staticmethod
    def node_name(tensor: PreparedTensor) -> str:
        """Return a TensorKrowch-safe node name while preserving valid names."""
        if tensor.spec.name and not any(
            character.isspace() for character in tensor.spec.name
        ):
            return tensor.spec.name
        return tensor.variable_name

    def _render_manual_plan(
        self,
        *,
        prepared: PreparedNetwork,
        collection_format: TensorCollectionFormat,
        collection_name: str,
    ) -> tuple[list[str], list[str]]:
        """Render a saved manual plan, rejecting unsupported outer products."""
        contraction_inputs = prepare_contraction_inputs(prepared)
        simulation = simulate_contraction_plan(
            initial_operand_ids=contraction_inputs.initial_operand_ids,
            initial_operands=contraction_inputs.initial_operands,
            initial_axis_names=contraction_inputs.initial_axis_names,
            dimension_by_label=contraction_inputs.dimension_by_label,
            plan=prepared.spec.contraction_plan,
        )
        if any(step.is_outer_product for step in simulation.steps):
            raise CodeGenerationError(
                "TensorKrowch manual plans cannot include outer product steps; each manual step needs at least one shared index between its operands."
            )

        step_result_indexes = {
            step.step_id: result_index
            for result_index, step in enumerate(simulation.steps)
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
        tensor_names_by_id = tensor_display_name_by_id(prepared)
        contraction_lines = ["results_list = []", ""]
        for step_index, step in enumerate(simulation.steps):
            latest_result_index = step_index - 1 if step_index > 0 else None
            left_expression = render_operand_expression(
                step.left_operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=step_result_indexes,
                latest_result_index=latest_result_index,
            )
            right_expression = render_operand_expression(
                step.right_operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=step_result_indexes,
                latest_result_index=latest_result_index,
            )
            contraction_lines.append(
                render_manual_step_comment(
                    step.step_id,
                    step.left_operand_id,
                    step.right_operand_id,
                )
            )
            contraction_lines.append(
                "results_list.append(tk.contract_between("
                f"{left_expression}, {right_expression}))"
            )
            contraction_lines.append("")

        final_result_index = len(simulation.steps) - 1 if simulation.steps else None
        output_lines = render_remaining_operands_mapping(
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
                + render_operand_expression(
                    simulation.remaining_operand_ids[0],
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=step_result_indexes,
                    latest_result_index=final_result_index,
                )
            )
        return contraction_lines, output_lines

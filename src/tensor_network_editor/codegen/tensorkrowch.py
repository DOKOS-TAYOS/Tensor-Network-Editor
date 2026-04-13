"""Code generation for the ``tensorkrowch`` backend."""

from __future__ import annotations

from .._contraction_plan import (
    prepare_contraction_inputs,
    simulate_contraction_plan,
)
from ..errors import CodeGenerationError
from ..models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat
from .base import CodeGenerator
from .common import (
    PreparedNetwork,
    PreparedTensor,
    container_name_for_format,
    prepare_network,
    render_operand_expression,
    render_remaining_operands_mapping,
    render_tensor_collection_assignment,
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
    ) -> CodegenResult:
        """Generate ``tensorkrowch`` code for ``spec``."""
        prepared = prepare_network(spec)
        collection_name = container_name_for_format(collection_format)
        lines = [
            "import torch",
            "import tensorkrowch as tk",
            "",
            "network = tk.TensorNetwork()",
            "",
        ]

        lines.extend(
            render_tensor_collection_assignment(
                collection_name=collection_name,
                collection_format=collection_format,
                prepared=prepared,
                tensor_value_by_id={
                    tensor.spec.id: (
                        f"tk.Node(tensor=torch.zeros({tensor.spec.shape!r}, dtype=torch.float32), "
                        f"axes_names={tuple(index.spec.name for index in tensor.indices)!r}, "
                        f"name={self.node_name(tensor)!r}, network=network)"
                    )
                    for tensor in prepared.tensors
                },
            )
        )
        lines.append("")

        if prepared.edges:
            lines.append("edges_list = []")
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
                lines.append(f"# {edge.spec.name}")
                lines.append(
                    "edges_list.append(("
                    f"{edge.spec.name!r}, "
                    f"tk.connect({left_tensor}[{edge.left.spec.name!r}], {right_tensor}[{edge.right.spec.name!r}])"
                    "))"
                )
            lines.append("")
        if spec.contraction_plan is not None and spec.contraction_plan.steps:
            lines.extend(
                self._render_manual_plan(
                    prepared=prepared,
                    collection_format=collection_format,
                    collection_name=collection_name,
                )
            )
        else:
            lines.append(
                "open_edges = ("
                + ", ".join(
                    f"{tensor_collection_reference_by_id(prepared, index.tensor.id, collection_format, collection_name)}[{index.spec.name!r}]"
                    for index in prepared.open_indices
                )
                + ("," if prepared.open_indices else "")
                + ")"
            )

        return CodegenResult(engine=self.engine, code="\n".join(lines).strip() + "\n")

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
    ) -> list[str]:
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
        lines = ["results_list = []", ""]
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
            lines.append(f"# Manual step {step.step_id}")
            lines.append(
                "results_list.append(tk.contract_between("
                f"{left_expression}, {right_expression}))"
            )
            lines.append("")

        final_result_index = len(simulation.steps) - 1 if simulation.steps else None
        lines.extend(
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
            lines.append(
                "result = "
                + render_operand_expression(
                    simulation.remaining_operand_ids[0],
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=step_result_indexes,
                    latest_result_index=final_result_index,
                )
            )
        return lines

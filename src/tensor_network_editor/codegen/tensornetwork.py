from __future__ import annotations

from .._contraction_plan import (
    build_dimension_by_label,
    build_initial_operand_axis_names,
    build_initial_operand_labels,
    simulate_contraction_plan,
)
from ..models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat
from .base import CodeGenerator
from .common import (
    container_name_for_format,
    flattened_tensor_collection_expression,
    joined_tensor_display_name,
    prepare_network,
    render_results_list_reference,
    render_tensor_collection_assignment,
    tensor_collection_reference_by_id,
    tensor_display_name_by_id,
)


class TensorNetworkCodeGenerator(CodeGenerator):
    engine = EngineName.TENSORNETWORK

    def generate(
        self,
        spec: NetworkSpec,
        collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    ) -> CodegenResult:
        prepared = prepare_network(spec)
        collection_name = container_name_for_format(collection_format)
        lines = [
            "import numpy as np",
            "import tensornetwork as tn",
            "",
        ]

        lines.extend(
            render_tensor_collection_assignment(
                collection_name=collection_name,
                collection_format=collection_format,
                prepared=prepared,
                tensor_value_by_id={
                    tensor.spec.id: (
                        f"tn.Node(np.zeros({tensor.spec.shape!r}, dtype=float), "
                        f"name={tensor.spec.name!r}, "
                        f"axis_names={[index.spec.name for index in tensor.indices]!r})"
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
                lines.append(
                    "edges_list.append(tn.connect("
                    f"{left_tensor}[{edge.left.spec.name!r}], "
                    f"{right_tensor}[{edge.right.spec.name!r}], "
                    f"name={edge.spec.name!r}))"
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
                "network_nodes = "
                + flattened_tensor_collection_expression(
                    collection_format, collection_name
                )
            )
            lines.append(
                "open_edges = ["
                + ", ".join(
                    f"{tensor_collection_reference_by_id(prepared, index.tensor.id, collection_format, collection_name)}[{index.spec.name!r}]"
                    for index in prepared.open_indices
                )
                + "]"
            )

        return CodegenResult(engine=self.engine, code="\n".join(lines).strip() + "\n")

    def _render_manual_plan(
        self,
        *,
        prepared,
        collection_format: TensorCollectionFormat,
        collection_name: str,
    ) -> list[str]:
        simulation = simulate_contraction_plan(
            initial_operand_ids=tuple(tensor.spec.id for tensor in prepared.tensors),
            initial_operands=build_initial_operand_labels(prepared),
            initial_axis_names=build_initial_operand_axis_names(prepared),
            dimension_by_label=build_dimension_by_label(prepared),
            plan=prepared.spec.contraction_plan,
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
            left_expression = self._operand_expression(
                step.left_operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=step_result_indexes,
                latest_result_index=latest_result_index,
            )
            right_expression = self._operand_expression(
                step.right_operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=step_result_indexes,
                latest_result_index=latest_result_index,
            )
            output_edge_order = self._build_output_edge_order(
                left_expression=left_expression,
                right_expression=right_expression,
                left_labels=step.left_labels,
                right_labels=step.right_labels,
                left_axis_names=step.left_axis_names,
                right_axis_names=step.right_axis_names,
                contracted_labels=step.contracted_labels,
            )
            lines.append(f"# Manual step {step.step_id}")
            lines.append(
                "results_list.append(tn.contract_between("
                f"{left_expression}, "
                f"{right_expression}, "
                f"name={step.step_id!r}, "
                f"allow_outer_product=True, "
                f"output_edge_order={output_edge_order}, "
                f"axis_names={list(step.result_axis_names)!r}))"
            )
            lines.append("")

        final_result_index = len(simulation.steps) - 1 if simulation.steps else None
        lines.extend(
            self._render_remaining_operands(
                operand_ids=simulation.remaining_operand_ids,
                source_tensor_ids_by_operand_id=simulation.source_tensor_ids_by_operand_id,
                tensor_names_by_id=tensor_names_by_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=step_result_indexes,
                latest_result_index=final_result_index,
            )
        )
        lines.append("network_nodes = list(remaining_operands.values())")
        if len(simulation.remaining_operand_ids) == 1:
            lines.append(
                "result = "
                + self._operand_expression(
                    simulation.remaining_operand_ids[0],
                    base_operand_expressions=base_operand_expressions,
                    step_result_indexes=step_result_indexes,
                    latest_result_index=final_result_index,
                )
            )
        return lines

    @staticmethod
    def _build_output_edge_order(
        *,
        left_expression: str,
        right_expression: str,
        left_labels: tuple[str, ...],
        right_labels: tuple[str, ...],
        left_axis_names: tuple[str, ...],
        right_axis_names: tuple[str, ...],
        contracted_labels: tuple[str, ...],
    ) -> str:
        output_edges: list[str] = []
        for label, axis_name in zip(left_labels, left_axis_names, strict=True):
            if label not in contracted_labels:
                output_edges.append(f"{left_expression}[{axis_name!r}]")
        for label, axis_name in zip(right_labels, right_axis_names, strict=True):
            if label not in contracted_labels:
                output_edges.append(f"{right_expression}[{axis_name!r}]")
        return "[" + ", ".join(output_edges) + "]"

    @staticmethod
    def _operand_expression(
        operand_id: str,
        *,
        base_operand_expressions: dict[str, str],
        step_result_indexes: dict[str, int],
        latest_result_index: int | None,
    ) -> str:
        if operand_id in base_operand_expressions:
            return base_operand_expressions[operand_id]
        return render_results_list_reference(
            step_result_indexes[operand_id],
            latest_result_index=latest_result_index,
        )

    @staticmethod
    def _render_remaining_operands(
        *,
        operand_ids: tuple[str, ...],
        source_tensor_ids_by_operand_id: dict[str, tuple[str, ...]],
        tensor_names_by_id: dict[str, str],
        base_operand_expressions: dict[str, str],
        step_result_indexes: dict[str, int],
        latest_result_index: int | None,
    ) -> list[str]:
        lines = ["remaining_operands = {"]
        for operand_id in operand_ids:
            operand_expression = TensorNetworkCodeGenerator._operand_expression(
                operand_id,
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=step_result_indexes,
                latest_result_index=latest_result_index,
            )
            operand_name = joined_tensor_display_name(
                source_tensor_ids_by_operand_id[operand_id],
                tensor_names_by_id,
            )
            lines.append(f"    {operand_name!r}: {operand_expression},")
        lines.append("}")
        return lines

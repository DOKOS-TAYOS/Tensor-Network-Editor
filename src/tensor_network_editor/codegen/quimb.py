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
    PreparedNetwork,
    container_name_for_format,
    flattened_tensor_collection_expression,
    joined_tensor_display_name,
    prepare_network,
    render_results_list_reference,
    render_tensor_collection_assignment,
    tensor_collection_reference,
    tensor_display_name_by_id,
)


class QuimbCodeGenerator(CodeGenerator):
    engine = EngineName.QUIMB

    def generate(
        self,
        spec: NetworkSpec,
        collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    ) -> CodegenResult:
        prepared = prepare_network(spec)
        collection_name = container_name_for_format(collection_format)
        lines = [
            "import numpy as np",
            "import quimb.tensor as qtn",
            "",
        ]

        lines.extend(
            render_tensor_collection_assignment(
                collection_name=collection_name,
                collection_format=collection_format,
                prepared=prepared,
                tensor_value_by_id={
                    tensor.spec.id: (
                        f"qtn.Tensor(data=np.zeros({tensor.spec.shape!r}, dtype=float), "
                        f"inds={tuple(index.label for index in tensor.indices)!r}, "
                        f"tags={(tensor.spec.name, self._operand_tag(tensor.spec.id))!r})"
                    )
                    for tensor in prepared.tensors
                },
            )
        )
        lines.append("")
        lines.append(
            "network_tensors = "
            + flattened_tensor_collection_expression(collection_format, collection_name)
        )
        lines.append("network = qtn.TensorNetwork(network_tensors)")
        if spec.contraction_plan is not None and spec.contraction_plan.steps:
            lines.extend(
                self._render_manual_plan(
                    prepared=prepared,
                    collection_format=collection_format,
                    collection_name=collection_name,
                )
            )
        elif prepared.open_indices:
            lines.append(
                "open_inds = ("
                + ", ".join(repr(index.label) for index in prepared.open_indices)
                + ",)"
            )
        else:
            lines.append("open_inds = ()")

        return CodegenResult(engine=self.engine, code="\n".join(lines).strip() + "\n")

    def _render_manual_plan(
        self,
        *,
        prepared: PreparedNetwork,
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
            tensor.spec.id: tensor_collection_reference(
                tensor,
                collection_format,
                collection_name,
            )
            for tensor in prepared.tensors
        }
        tensor_names_by_id = tensor_display_name_by_id(prepared)

        lines = ["results_list = []", ""]
        for step in simulation.steps:
            left_tag = self._operand_tag(step.left_operand_id)
            right_tag = self._operand_tag(step.right_operand_id)
            step_tag = self._operand_tag(step.step_id)
            lines.append(f"# Manual step {step.step_id}")
            lines.append(f"network.contract_between({left_tag!r}, {right_tag!r})")
            lines.append(f"network[{left_tag!r}].add_tag({step_tag!r})")
            lines.append(f"results_list.append(network[{step_tag!r}])")
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
            lines.append("open_inds = tuple(result.inds)")
        return lines

    @staticmethod
    def _operand_tag(operand_id: str) -> str:
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

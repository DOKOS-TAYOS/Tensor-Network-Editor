"""Shared einsum-based code generation helpers."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from string import ascii_letters
from typing import Any, TypeAlias, cast

from ...internal.analysis._contraction_plan import (
    PreparedContractionInputs,
    prepare_contraction_inputs,
    simulate_contraction_plan,
    simulate_contraction_step,
)
from ...models import (
    CodegenResult,
    ContractionPlanSpec,
    ContractionStepSpec,
    EngineIdentifier,
    NetworkSpec,
    TensorCollectionFormat,
)
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
    tensor_collection_reference,
    tensor_display_name_by_id,
    uses_external_numpy_tensor_data,
    uses_external_pt_tensor_data,
)

_RandomPairwisePath: TypeAlias = tuple[tuple[int, ...], ...]
_RandomPairwiseSignature: TypeAlias = tuple[
    tuple[str, tuple[str, ...], tuple[int, ...]],
    ...,
]
_RandomPairwisePlanCacheKey: TypeAlias = tuple[
    int,
    int,
    _RandomPairwiseSignature,
]
_RandomPairwisePlanCache: TypeAlias = dict[
    _RandomPairwisePlanCacheKey,
    _RandomPairwisePath | None,
]


@dataclass(slots=True, frozen=True)
class PairwiseContractionCandidate:
    """One automatic pairwise route plus its simulated cost summary."""

    plan: ContractionPlanSpec
    simulation: object
    total_estimated_flops: int
    peak_intermediate_size: int


def _build_pairwise_candidate(
    contraction_inputs: PreparedContractionInputs,
    plan: ContractionPlanSpec,
) -> PairwiseContractionCandidate:
    """Simulate one pairwise plan and capture its headline metrics."""
    simulation = simulate_contraction_plan(
        initial_operand_ids=contraction_inputs.initial_operand_ids,
        initial_operands=contraction_inputs.initial_operands,
        initial_axis_names=contraction_inputs.initial_axis_names,
        dimension_by_label=contraction_inputs.dimension_by_label,
        plan=plan,
    )
    total_estimated_flops = sum(step.estimated_flops for step in simulation.steps)
    peak_intermediate_size = max(
        (step.intermediate_size for step in simulation.steps),
        default=0,
    )
    return PairwiseContractionCandidate(
        plan=plan,
        simulation=simulation,
        total_estimated_flops=total_estimated_flops,
        peak_intermediate_size=peak_intermediate_size,
    )


def _build_heuristic_pairwise_plan(
    contraction_inputs: PreparedContractionInputs,
) -> ContractionPlanSpec:
    """Build a deterministic connected-first pairwise contraction plan."""
    remaining_order = list(contraction_inputs.initial_operand_ids)
    remaining_operands = dict(contraction_inputs.initial_operands)
    steps: list[ContractionStepSpec] = []

    while len(remaining_order) > 1:
        left_operand_id, right_operand_id = _select_connected_pair(
            remaining_order=remaining_order,
            remaining_operands=remaining_operands,
        )
        step = ContractionStepSpec(
            id=f"auto_step_{len(steps) + 1}",
            left_operand_id=left_operand_id,
            right_operand_id=right_operand_id,
        )
        left_labels = remaining_operands.pop(left_operand_id)
        right_labels = remaining_operands.pop(right_operand_id)
        simulated_step = simulate_contraction_step(
            step=step,
            left_labels=left_labels,
            right_labels=right_labels,
            left_axis_names=left_labels,
            right_axis_names=right_labels,
            dimension_by_label=contraction_inputs.dimension_by_label,
        )
        remaining_order.remove(left_operand_id)
        remaining_order.remove(right_operand_id)
        remaining_order.append(step.id)
        remaining_operands[step.id] = simulated_step.surviving_labels
        steps.append(step)

    return ContractionPlanSpec(
        id="auto_pairwise_plan",
        name="Automatic pairwise path",
        steps=steps,
    )


def _select_connected_pair(
    *,
    remaining_order: list[str],
    remaining_operands: dict[str, tuple[str, ...]],
) -> tuple[str, str]:
    """Return the next operand pair, preferring shared-label contractions."""
    for left_index, left_operand_id in enumerate(remaining_order[:-1]):
        left_labels = set(remaining_operands[left_operand_id])
        for right_operand_id in remaining_order[left_index + 1 :]:
            if left_labels.intersection(remaining_operands[right_operand_id]):
                return left_operand_id, right_operand_id
    return remaining_order[0], remaining_order[1]


def _load_random_optimizer_tools(
    importer: Callable[[str], Any],
) -> tuple[Any, Any] | None:
    """Resolve the optional opt_einsum helpers used for route comparison."""
    try:
        opt_einsum_module = cast(Any, importer("opt_einsum"))
        path_random_module = cast(Any, importer("opt_einsum.path_random"))
    except ImportError:
        return None
    return opt_einsum_module.contract_path, path_random_module.RandomGreedy


def _build_random_pairwise_plan(
    contraction_inputs: PreparedContractionInputs,
    *,
    path_cache: _RandomPairwisePlanCache | None = None,
) -> ContractionPlanSpec | None:
    """Build an opt_einsum-assisted random pairwise plan when available."""
    if len(contraction_inputs.initial_operand_ids) <= 1:
        return None

    label_order = list(
        dict.fromkeys(
            label
            for operand_id in contraction_inputs.initial_operand_ids
            for label in contraction_inputs.initial_operands[operand_id]
        )
    )
    if len(label_order) > len(ascii_letters):
        return None

    random_tools = _load_random_optimizer_tools(import_module)
    if random_tools is None:
        return None
    contract_path, random_optimizer_type = random_tools
    signature = _build_random_pairwise_signature(contraction_inputs)
    cache_key = (id(contract_path), id(random_optimizer_type), signature)
    if path_cache is not None and cache_key in path_cache:
        cached_path = path_cache[cache_key]
        if cached_path is None:
            return None
        return _build_pairwise_plan_from_path(
            contraction_inputs=contraction_inputs,
            path=cached_path,
        )

    symbol_map = {
        label: ascii_letters[offset]
        for offset, label in enumerate(label_order[: len(ascii_letters)])
    }
    label_counts = {label: 0 for label in label_order}
    for operand_id in contraction_inputs.initial_operand_ids:
        for label in contraction_inputs.initial_operands[operand_id]:
            label_counts[label] += 1
    output_labels = [label for label in label_order if label_counts[label] == 1]
    equation = (
        ",".join(
            "".join(
                symbol_map[label]
                for label in contraction_inputs.initial_operands[operand_id]
            )
            for operand_id in contraction_inputs.initial_operand_ids
        )
        + "->"
        + "".join(symbol_map[label] for label in output_labels)
    )
    shapes = [
        tuple(
            contraction_inputs.dimension_by_label[label]
            for label in contraction_inputs.initial_operands[operand_id]
        )
        for operand_id in contraction_inputs.initial_operand_ids
    ]

    try:
        path, _ = contract_path(
            equation,
            *shapes,
            shapes=True,
            optimize=random_optimizer_type(max_time=0.05, minimize="flops"),
        )
    except (NotImplementedError, TypeError, ValueError):
        if path_cache is not None:
            path_cache[cache_key] = None
        return None

    normalized_path = tuple(
        tuple(int(value) for value in raw_indices) for raw_indices in path
    )
    if path_cache is not None:
        path_cache[cache_key] = normalized_path

    return _build_pairwise_plan_from_path(
        contraction_inputs=contraction_inputs,
        path=normalized_path,
    )


def _build_random_pairwise_signature(
    contraction_inputs: PreparedContractionInputs,
) -> _RandomPairwiseSignature:
    """Build a stable cache signature for random pairwise path generation."""
    return tuple(
        (
            operand_id,
            contraction_inputs.initial_operands[operand_id],
            tuple(
                contraction_inputs.dimension_by_label[label]
                for label in contraction_inputs.initial_operands[operand_id]
            ),
        )
        for operand_id in contraction_inputs.initial_operand_ids
    )


def _build_pairwise_plan_from_path(
    *,
    contraction_inputs: PreparedContractionInputs,
    path: _RandomPairwisePath,
) -> ContractionPlanSpec | None:
    """Translate an opt_einsum path into a stored pairwise plan."""
    remaining_order = list(contraction_inputs.initial_operand_ids)
    remaining_operands = dict(contraction_inputs.initial_operands)
    steps: list[ContractionStepSpec] = []

    for step_index, raw_indices in enumerate(path, start=1):
        indices = tuple(int(value) for value in raw_indices)
        if len(indices) != 2:
            return None
        try:
            left_operand_id = remaining_order[indices[0]]
            right_operand_id = remaining_order[indices[1]]
        except IndexError:
            return None
        step = ContractionStepSpec(
            id=f"auto_step_{step_index}",
            left_operand_id=left_operand_id,
            right_operand_id=right_operand_id,
        )
        left_labels = remaining_operands.pop(left_operand_id, None)
        right_labels = remaining_operands.pop(right_operand_id, None)
        if left_labels is None or right_labels is None:
            return None
        simulated_step = simulate_contraction_step(
            step=step,
            left_labels=left_labels,
            right_labels=right_labels,
            left_axis_names=left_labels,
            right_axis_names=right_labels,
            dimension_by_label=contraction_inputs.dimension_by_label,
        )
        higher_index, lower_index = sorted(indices, reverse=True)
        remaining_order.pop(higher_index)
        remaining_order.pop(lower_index)
        remaining_order.append(step.id)
        remaining_operands[step.id] = simulated_step.surviving_labels
        steps.append(step)

    return ContractionPlanSpec(
        id="auto_random_plan",
        name="Automatic random path",
        steps=steps,
    )


def _select_pairwise_candidate(
    contraction_inputs: PreparedContractionInputs,
    *,
    path_cache: _RandomPairwisePlanCache | None = None,
) -> PairwiseContractionCandidate:
    """Choose the best available automatic pairwise route."""
    heuristic_candidate = _build_pairwise_candidate(
        contraction_inputs,
        _build_heuristic_pairwise_plan(contraction_inputs),
    )
    random_plan = _build_random_pairwise_plan(
        contraction_inputs,
        path_cache=path_cache,
    )
    if random_plan is None:
        return heuristic_candidate
    random_candidate = _build_pairwise_candidate(contraction_inputs, random_plan)
    if _is_better_pairwise_candidate(random_candidate, heuristic_candidate):
        return random_candidate
    return heuristic_candidate


def _is_better_pairwise_candidate(
    candidate: PairwiseContractionCandidate,
    baseline: PairwiseContractionCandidate,
) -> bool:
    """Return whether ``candidate`` beats ``baseline`` for export quality."""
    if candidate.total_estimated_flops != baseline.total_estimated_flops:
        return candidate.total_estimated_flops < baseline.total_estimated_flops
    if candidate.peak_intermediate_size != baseline.peak_intermediate_size:
        return candidate.peak_intermediate_size < baseline.peak_intermediate_size
    return False


class BaseEinsumCodeGenerator(CodeGenerator, ABC):
    """Base generator for NumPy and PyTorch einsum backends."""

    engine: EngineIdentifier
    import_line: str
    module_alias: str
    zero_initializer_suffix: str = ""
    empty_network_expression: str

    def __init__(self) -> None:
        """Initialize per-generator caches used by repeated code generation."""
        self._random_pairwise_plan_cache: _RandomPairwisePlanCache = {}

    def generate(
        self,
        spec: NetworkSpec,
        collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
        *,
        validate: bool = True,
    ) -> CodegenResult:
        """Generate einsum-based Python code for ``spec``."""
        prepared = prepare_network(spec, validate=validate)
        collection_name = container_name_for_format(collection_format)
        tensor_collection_lines = render_tensor_collection_initialization(
            collection_name,
            collection_format,
        )
        tensor_data_lines = render_tensor_data_assignments(
            prepared,
            module_alias=self.module_alias,
            zeros_initializer_suffix=self.zero_initializer_suffix,
            literal_constructor_name=(
                "array" if self.module_alias == "np" else "tensor"
            ),
        )
        tensor_construction_lines = render_tensor_collection_assignment(
            collection_name=collection_name,
            collection_format=collection_format,
            prepared=prepared,
            tensor_value_by_id={
                tensor.spec.id: tensor.data_variable_name for tensor in prepared.tensors
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
                CodeSection(title=None, lines=self._render_import_lines(prepared)),
                CodeSection(title="Tensor collection", lines=tensor_collection_lines),
                CodeSection(title="Tensor data", lines=tensor_data_lines),
                CodeSection(
                    title="Tensor construction",
                    lines=tensor_construction_lines,
                ),
                CodeSection(title=contraction_title, lines=contraction_lines),
                CodeSection(title="Outputs", lines=output_lines),
            ),
        )

    def _render_import_lines(self, prepared: PreparedNetwork) -> list[str]:
        """Render imports needed by this einsum backend."""
        lines: list[str] = []
        if self.module_alias == "torch":
            if uses_external_numpy_tensor_data(prepared):
                lines.append("import numpy as np")
            lines.append(self.import_line)
            return lines
        lines.append(self.import_line)
        if uses_external_pt_tensor_data(prepared):
            lines.append("import torch")
        return lines

    def _render_full_network_einsum(
        self,
        *,
        prepared: PreparedNetwork,
        collection_format: TensorCollectionFormat,
        collection_name: str,
    ) -> tuple[list[str], list[str]]:
        """Render an automatic pairwise einsum contraction for the full network."""
        if not prepared.tensors:
            return (
                ["# Empty network contracts to the scalar identity."],
                [f"result = {self.empty_network_expression}"],
            )
        if len(prepared.tensors) == 1:
            only_tensor = prepared.tensors[0]
            return (
                ["# Single tensor already represents the result."],
                [
                    "result = "
                    + tensor_collection_reference(
                        only_tensor,
                        collection_format,
                        collection_name,
                    )
                ],
            )

        label_order = list(
            dict.fromkeys(
                index.label for tensor in prepared.tensors for index in tensor.indices
            )
        )
        use_string_equation = len(label_order) <= len(ascii_letters)
        symbol_map = {
            label: ascii_letters[offset]
            for offset, label in enumerate(label_order[: len(ascii_letters)])
        }
        label_to_int = {label: offset for offset, label in enumerate(label_order)}
        contraction_inputs = prepare_contraction_inputs(prepared)
        candidate = _select_pairwise_candidate(
            contraction_inputs,
            path_cache=self._random_pairwise_plan_cache,
        )
        simulation = cast(Any, candidate.simulation)
        base_operand_expressions = {
            tensor.spec.id: tensor_collection_reference(
                tensor,
                collection_format,
                collection_name,
            )
            for tensor in prepared.tensors
        }
        step_result_indexes = {
            step.step_id: result_index
            for result_index, step in enumerate(simulation.steps)
        }
        contraction_lines: list[str] = ["results_list = []", ""]
        if not use_string_equation:
            contraction_lines.insert(
                0,
                "# Pairwise einsum uses the integer-sublist form because the network uses many labels.",
            )
            contraction_lines.insert(1, "")

        for step_index, step in enumerate(simulation.steps):
            latest_result_index = step_index - 1 if step_index > 0 else None
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
                    use_string_labels=use_string_equation,
                    symbol_map=symbol_map,
                    label_to_int=label_to_int,
                )
                + ")"
            )
            contraction_lines.append("")

        final_result_index = len(simulation.steps) - 1 if simulation.steps else None
        output_lines = [
            "result = "
            + render_operand_expression(
                simulation.remaining_operand_ids[0],
                base_operand_expressions=base_operand_expressions,
                step_result_indexes=step_result_indexes,
                latest_result_index=final_result_index,
            )
        ]
        return contraction_lines, output_lines

    def _render_manual_plan(
        self,
        *,
        prepared: PreparedNetwork,
        collection_format: TensorCollectionFormat,
        collection_name: str,
    ) -> tuple[list[str], list[str]]:
        """Render step-by-step einsum calls for a saved manual plan."""
        label_order = list(
            dict.fromkeys(
                index.label for tensor in prepared.tensors for index in tensor.indices
            )
        )
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

"""Analyze manual and automatic contraction paths for a network spec."""

from __future__ import annotations

from ...models import NetworkSpec
from ...validation import ensure_valid_spec
from ..modes._grid_periodic import grid_periodic_active_cell_as_analysis_network
from ..modes._linear_periodic import linear_periodic_active_cell_as_analysis_network
from ..modes._tree_periodic import tree_periodic_active_cell_as_analysis_network
from ._analysis import analyze_network
from ._contraction_analysis_automatic import (
    _analyze_automatic_operands,
    _analyze_future_automatic_plan,
    _analyze_past_automatic_plan,
    _AutomaticPathCache,
)
from ._contraction_analysis_compare import _build_contraction_comparisons
from ._contraction_analysis_manual import (
    _analyze_manual_plan_and_state,
)
from ._contraction_analysis_types import (
    ContractionAnalysisResult,
    SyntheticContractionOperand,
)
from ._contraction_plan import (
    prepare_contraction_inputs,
)
from ._hyperedge_lowering import lower_hyperedges_to_pairwise_spec
from ._memory_dtypes import DEFAULT_MEMORY_DTYPE, dtype_size_in_bytes
from ._prepared_network import PreparedNetwork, prepare_analyzed_network

_HYPEREDGE_ANALYSIS_WARNING = (
    "Hyperedges are analyzed as generated copy tensors; the visual model is unchanged."
)


def analyze_contraction(
    spec: NetworkSpec,
    *,
    memory_dtype: str = DEFAULT_MEMORY_DTYPE,
) -> ContractionAnalysisResult:
    """Analyze the saved manual plan and available automatic greedy previews."""
    validated_spec = ensure_valid_spec(spec)
    return _analyze_validated_contraction(
        validated_spec,
        memory_dtype=memory_dtype,
    )


def _analyze_validated_contraction(
    spec: NetworkSpec,
    *,
    memory_dtype: str = DEFAULT_MEMORY_DTYPE,
) -> ContractionAnalysisResult:
    """Analyze contraction data for a spec that was already validated."""
    normalized_spec = _normalize_spec_for_contraction_analysis(spec, validate=False)
    prepared = prepare_analyzed_network(analyze_network(normalized_spec))
    warnings, synthetic_operands = _build_hyperedge_analysis_metadata(
        original_spec=spec,
        normalized_spec=normalized_spec,
    )
    return _analyze_prepared_contraction(
        prepared,
        memory_dtype=memory_dtype,
        warnings=warnings,
        synthetic_operands=synthetic_operands,
    )


def _normalize_spec_for_contraction_analysis(
    spec: NetworkSpec,
    *,
    validate: bool = True,
) -> NetworkSpec:
    """Return the validated spec variant consumed by contraction analysis."""
    resolved_spec = ensure_valid_spec(spec) if validate else spec
    if resolved_spec.linear_periodic_chain is not None:
        return linear_periodic_active_cell_as_analysis_network(
            resolved_spec.linear_periodic_chain
        )
    if resolved_spec.grid_periodic_grid is not None:
        return grid_periodic_active_cell_as_analysis_network(
            resolved_spec.grid_periodic_grid
        )
    if resolved_spec.tree_periodic_tree is not None:
        return tree_periodic_active_cell_as_analysis_network(
            resolved_spec.tree_periodic_tree
        )
    if resolved_spec.hyperedges:
        return lower_hyperedges_to_pairwise_spec(
            resolved_spec,
            preserve_contraction_plan=True,
        )
    return resolved_spec


def _analyze_prepared_contraction(
    prepared: PreparedNetwork,
    *,
    memory_dtype: str = DEFAULT_MEMORY_DTYPE,
    warnings: list[str] | None = None,
    synthetic_operands: tuple[SyntheticContractionOperand, ...] = (),
) -> ContractionAnalysisResult:
    """Analyze contraction paths using an already prepared network."""
    spec = prepared.spec
    bytes_per_element = dtype_size_in_bytes(memory_dtype)
    contraction_inputs = prepare_contraction_inputs(prepared)
    automatic_path_cache: _AutomaticPathCache = {}
    network_output_shape = tuple(
        index.spec.dimension for index in prepared.open_indices
    )
    manual, manual_operand_state = _analyze_manual_plan_and_state(
        spec=spec,
        contraction_inputs=contraction_inputs,
        bytes_per_element=bytes_per_element,
    )
    automatic_full = _analyze_automatic_operands(
        operand_order=contraction_inputs.initial_operand_ids,
        operands=contraction_inputs.initial_operands,
        dimension_by_label=contraction_inputs.dimension_by_label,
        step_id_prefix="auto_full_step_",
        bytes_per_element=bytes_per_element,
        path_cache=automatic_path_cache,
    )
    automatic_future = _analyze_future_automatic_plan(
        initial_operands=contraction_inputs.initial_operands,
        manual_operand_state=manual_operand_state,
        dimension_by_label=contraction_inputs.dimension_by_label,
        bytes_per_element=bytes_per_element,
        path_cache=automatic_path_cache,
    )
    automatic_past = _analyze_past_automatic_plan(
        spec=spec,
        initial_operands=contraction_inputs.initial_operands,
        manual_operand_state=manual_operand_state,
        dimension_by_label=contraction_inputs.dimension_by_label,
        bytes_per_element=bytes_per_element,
        path_cache=automatic_path_cache,
    )
    message = (
        automatic_future.message
        if automatic_future.status == "unavailable"
        else automatic_past.message
        if automatic_past.status == "unavailable"
        else None
    )
    return ContractionAnalysisResult(
        network_output_shape=network_output_shape,
        manual=manual,
        automatic_full=automatic_full,
        automatic_future=automatic_future,
        automatic_past=automatic_past,
        memory_dtype=memory_dtype,
        comparisons=_build_contraction_comparisons(
            manual=manual,
            automatic_full=automatic_full,
            automatic_past=automatic_past,
            memory_dtype=memory_dtype,
        ),
        automatic_strategy="greedy",
        message=message,
        warnings=list(warnings or []),
        synthetic_operands=synthetic_operands,
    )


def _build_hyperedge_analysis_metadata(
    *,
    original_spec: NetworkSpec,
    normalized_spec: NetworkSpec,
) -> tuple[list[str], tuple[SyntheticContractionOperand, ...]]:
    """Return warnings and synthetic operand metadata for lowered hyperedges."""
    if not original_spec.hyperedges:
        return [], ()
    hyperedge_by_id = {
        hyperedge.id: hyperedge for hyperedge in original_spec.hyperedges
    }
    operands: list[SyntheticContractionOperand] = []
    for tensor in normalized_spec.tensors:
        source_hyperedge_id = tensor.metadata.get("generated_for_hyperedge")
        if not isinstance(source_hyperedge_id, str):
            continue
        hyperedge = hyperedge_by_id.get(source_hyperedge_id)
        if hyperedge is None:
            continue
        operands.append(
            SyntheticContractionOperand(
                operand_id=tensor.id,
                name=tensor.name,
                kind="hyperedge_copy_tensor",
                source_hyperedge_id=hyperedge.id,
                source_tensor_ids=tuple(
                    endpoint.tensor_id for endpoint in hyperedge.endpoints
                ),
            )
        )
    return [_HYPEREDGE_ANALYSIS_WARNING], tuple(operands)

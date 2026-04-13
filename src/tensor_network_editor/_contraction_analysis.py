"""Analyze manual and automatic contraction paths for a network spec."""

from __future__ import annotations

from ._contraction_analysis_automatic import (
    _analyze_automatic_operands,
    _analyze_future_automatic_plan,
    _analyze_past_automatic_plan,
)
from ._contraction_analysis_compare import _build_contraction_comparisons
from ._contraction_analysis_manual import (
    _analyze_manual_plan_and_state,
)
from ._contraction_analysis_types import ContractionAnalysisResult
from ._contraction_plan import (
    prepare_contraction_inputs,
)
from ._linear_periodic import linear_periodic_active_cell_as_analysis_network
from ._memory_dtypes import DEFAULT_MEMORY_DTYPE, dtype_size_in_bytes
from .codegen.common import prepare_network
from .models import NetworkSpec


def analyze_contraction(
    spec: NetworkSpec,
    *,
    memory_dtype: str = DEFAULT_MEMORY_DTYPE,
) -> ContractionAnalysisResult:
    """Analyze the saved manual plan and available automatic greedy previews."""
    if spec.linear_periodic_chain is not None:
        from .validation import ensure_valid_spec

        validated_spec = ensure_valid_spec(spec)
        if validated_spec.linear_periodic_chain is not None:
            spec = linear_periodic_active_cell_as_analysis_network(
                validated_spec.linear_periodic_chain
            )

    bytes_per_element = dtype_size_in_bytes(memory_dtype)
    prepared = prepare_network(spec)
    contraction_inputs = prepare_contraction_inputs(prepared)
    network_output_shape = tuple(
        index.spec.dimension for index in prepared.open_indices
    )
    manual, manual_operand_state = _analyze_manual_plan_and_state(
        spec=spec,
        contraction_inputs=contraction_inputs,
        bytes_per_element=bytes_per_element,
    )
    automatic_full = _analyze_automatic_operands(
        operand_order=list(contraction_inputs.initial_operands),
        operands=contraction_inputs.initial_operands,
        dimension_by_label=contraction_inputs.dimension_by_label,
        step_id_prefix="auto_full_step_",
        bytes_per_element=bytes_per_element,
    )
    automatic_future = _analyze_future_automatic_plan(
        initial_operands=contraction_inputs.initial_operands,
        manual_operand_state=manual_operand_state,
        dimension_by_label=contraction_inputs.dimension_by_label,
        bytes_per_element=bytes_per_element,
    )
    automatic_past = _analyze_past_automatic_plan(
        spec=spec,
        initial_operands=contraction_inputs.initial_operands,
        manual_operand_state=manual_operand_state,
        dimension_by_label=contraction_inputs.dimension_by_label,
        bytes_per_element=bytes_per_element,
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
    )

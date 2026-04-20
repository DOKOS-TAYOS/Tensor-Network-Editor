"""Public headless analysis helpers for tensor-network specifications."""

from __future__ import annotations

from .codegen.common import prepare_analyzed_network
from .internal.analysis._analysis import analyze_network
from .internal.analysis._contraction_analysis import (
    _analyze_prepared_contraction,
    _normalize_spec_for_contraction_analysis,
    analyze_contraction,
)
from .internal.analysis._memory_dtypes import DEFAULT_MEMORY_DTYPE
from .internal.models._headless_models import NetworkSummary, SpecAnalysisReport
from .models import NetworkSpec
from .validation import ensure_valid_spec


def analyze_spec(
    spec: NetworkSpec,
    *,
    memory_dtype: str = DEFAULT_MEMORY_DTYPE,
) -> SpecAnalysisReport:
    """Return a structured summary for ``spec`` and its contraction metadata."""
    validated_spec = ensure_valid_spec(spec)
    if (
        validated_spec.linear_periodic_chain is None
        and validated_spec.grid_periodic_grid is None
        and validated_spec.tree_periodic_tree is None
    ):
        network = analyze_network(validated_spec)
        contraction_prepared = prepare_analyzed_network(network)
    else:
        contraction_spec = _normalize_spec_for_contraction_analysis(
            validated_spec,
            validate=False,
        )
        network = analyze_network(contraction_spec)
        contraction_prepared = prepare_analyzed_network(network)
    return SpecAnalysisReport(
        network=NetworkSummary(
            tensor_count=len(network.spec.tensors),
            edge_count=len(network.spec.edges),
            group_count=len(network.spec.groups),
            note_count=len(network.spec.notes),
            open_index_count=len(network.open_indices),
        ),
        contraction=_analyze_prepared_contraction(
            contraction_prepared,
            memory_dtype=memory_dtype,
        ),
    )


__all__ = [
    "NetworkSummary",
    "SpecAnalysisReport",
    "analyze_contraction",
    "analyze_spec",
]

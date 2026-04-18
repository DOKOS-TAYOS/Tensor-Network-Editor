"""Public headless analysis helpers for tensor-network specifications."""

from __future__ import annotations

from ._analysis import analyze_network
from ._contraction_analysis import (
    _analyze_prepared_contraction,
    _normalize_spec_for_contraction_analysis,
    analyze_contraction,
)
from ._headless_models import NetworkSummary, SpecAnalysisReport
from ._memory_dtypes import DEFAULT_MEMORY_DTYPE
from .codegen.common import prepare_analyzed_network
from .models import NetworkSpec
from .validation import ensure_valid_analysis


def analyze_spec(
    spec: NetworkSpec,
    *,
    memory_dtype: str = DEFAULT_MEMORY_DTYPE,
) -> SpecAnalysisReport:
    """Return a structured summary for ``spec`` and its contraction metadata."""
    network = ensure_valid_analysis(spec)
    if network.spec.linear_periodic_chain is None:
        contraction_prepared = prepare_analyzed_network(network)
    else:
        contraction_spec = _normalize_spec_for_contraction_analysis(
            network.spec,
            validate=False,
        )
        contraction_prepared = prepare_analyzed_network(
            analyze_network(contraction_spec)
        )
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

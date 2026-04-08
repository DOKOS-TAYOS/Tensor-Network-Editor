"""Public headless analysis helpers for tensor-network specifications."""

from __future__ import annotations

from ._analysis import analyze_network
from ._contraction_analysis import analyze_contraction
from ._headless_models import NetworkSummary, SpecAnalysisReport
from .models import NetworkSpec


def analyze_spec(spec: NetworkSpec) -> SpecAnalysisReport:
    """Return a structured summary for ``spec`` and its contraction metadata."""
    network = analyze_network(spec, validate=True)
    return SpecAnalysisReport(
        network=NetworkSummary(
            tensor_count=len(network.spec.tensors),
            edge_count=len(network.spec.edges),
            group_count=len(network.spec.groups),
            note_count=len(network.spec.notes),
            open_index_count=len(network.open_indices),
        ),
        contraction=analyze_contraction(network.spec),
    )


__all__ = [
    "NetworkSummary",
    "SpecAnalysisReport",
    "analyze_contraction",
    "analyze_spec",
]

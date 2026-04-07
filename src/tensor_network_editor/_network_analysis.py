"""Convenience wrappers around ``analyze_network`` for common lookups."""

from __future__ import annotations

from ._analysis import analyze_network
from ._model_graph import IndexSpec, NetworkSpec, TensorSpec


def tensor_map(spec: NetworkSpec) -> dict[str, TensorSpec]:
    """Return a mapping from tensor ids to tensors for ``spec``."""
    return analyze_network(spec).tensor_map


def index_map(spec: NetworkSpec) -> dict[str, tuple[TensorSpec, IndexSpec]]:
    """Return a mapping from index ids to owning tensor/index pairs."""
    return analyze_network(spec).index_map


def connected_index_ids(spec: NetworkSpec) -> set[str]:
    """Return the set of index ids that participate in edges."""
    return analyze_network(spec).connected_index_ids


def open_indices(spec: NetworkSpec) -> list[tuple[TensorSpec, IndexSpec]]:
    """Return tensor/index pairs that are not connected by any edge."""
    return analyze_network(spec).open_indices

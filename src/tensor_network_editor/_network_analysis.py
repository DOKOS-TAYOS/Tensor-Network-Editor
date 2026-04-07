from __future__ import annotations

from ._analysis import analyze_network
from ._model_graph import IndexSpec, NetworkSpec, TensorSpec


def tensor_map(spec: NetworkSpec) -> dict[str, TensorSpec]:
    return analyze_network(spec).tensor_map


def index_map(spec: NetworkSpec) -> dict[str, tuple[TensorSpec, IndexSpec]]:
    return analyze_network(spec).index_map


def connected_index_ids(spec: NetworkSpec) -> set[str]:
    return analyze_network(spec).connected_index_ids


def open_indices(spec: NetworkSpec) -> list[tuple[TensorSpec, IndexSpec]]:
    return analyze_network(spec).open_indices

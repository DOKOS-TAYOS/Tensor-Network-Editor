"""Derived lookup structures for validated or in-progress network specs."""

from __future__ import annotations

from dataclasses import dataclass

from ...models import IndexSpec, NetworkSpec, TensorSpec


@dataclass(slots=True)
class NetworkAnalysis:
    """Cached lookup data derived from a ``NetworkSpec``."""

    spec: NetworkSpec
    tensor_map: dict[str, TensorSpec]
    index_map: dict[str, tuple[TensorSpec, IndexSpec]]
    connected_index_ids: set[str]
    open_indices: list[tuple[TensorSpec, IndexSpec]]
    left_tensor_by_edge_id: dict[str, TensorSpec | None]
    left_index_by_edge_id: dict[str, IndexSpec | None]
    right_tensor_by_edge_id: dict[str, TensorSpec | None]
    right_index_by_edge_id: dict[str, IndexSpec | None]


def analyze_network(spec: NetworkSpec, *, validate: bool = False) -> NetworkAnalysis:
    """Build convenient lookup maps and open-index lists for ``spec``."""
    if validate:
        from ...validation import ensure_valid_spec

        spec = ensure_valid_spec(spec)

    tensor_map, index_map, all_indices = _build_tensor_and_index_maps(spec)
    (
        connected_index_ids,
        left_tensor_by_edge_id,
        left_index_by_edge_id,
        right_tensor_by_edge_id,
        right_index_by_edge_id,
    ) = _build_edge_analysis_maps(spec, index_map)
    open_indices = _build_open_indices(all_indices, connected_index_ids)

    return NetworkAnalysis(
        spec=spec,
        tensor_map=tensor_map,
        index_map=index_map,
        connected_index_ids=connected_index_ids,
        open_indices=open_indices,
        left_tensor_by_edge_id=left_tensor_by_edge_id,
        left_index_by_edge_id=left_index_by_edge_id,
        right_tensor_by_edge_id=right_tensor_by_edge_id,
        right_index_by_edge_id=right_index_by_edge_id,
    )


def _build_tensor_and_index_maps(
    spec: NetworkSpec,
) -> tuple[
    dict[str, TensorSpec],
    dict[str, tuple[TensorSpec, IndexSpec]],
    list[tuple[TensorSpec, IndexSpec]],
]:
    """Build tensor/index maps and preserve index ordering in one tensor pass."""
    tensor_map: dict[str, TensorSpec] = {}
    index_map: dict[str, tuple[TensorSpec, IndexSpec]] = {}
    all_indices: list[tuple[TensorSpec, IndexSpec]] = []
    for tensor in spec.tensors:
        tensor_map[tensor.id] = tensor
        for index in tensor.indices:
            pair = (tensor, index)
            index_map[index.id] = pair
            all_indices.append(pair)
    return tensor_map, index_map, all_indices


def _build_edge_analysis_maps(
    spec: NetworkSpec,
    index_map: dict[str, tuple[TensorSpec, IndexSpec]],
) -> tuple[
    set[str],
    dict[str, TensorSpec | None],
    dict[str, IndexSpec | None],
    dict[str, TensorSpec | None],
    dict[str, IndexSpec | None],
]:
    """Build edge-derived connected-index and endpoint lookup maps in one pass."""
    connected_index_ids: set[str] = set()
    left_tensor_by_edge_id: dict[str, TensorSpec | None] = {}
    left_index_by_edge_id: dict[str, IndexSpec | None] = {}
    right_tensor_by_edge_id: dict[str, TensorSpec | None] = {}
    right_index_by_edge_id: dict[str, IndexSpec | None] = {}

    for edge in spec.edges:
        connected_index_ids.add(edge.left.index_id)
        connected_index_ids.add(edge.right.index_id)
        left_item = index_map.get(edge.left.index_id)
        right_item = index_map.get(edge.right.index_id)

        left_tensor_by_edge_id[edge.id] = (
            left_item[0] if left_item is not None else None
        )
        left_index_by_edge_id[edge.id] = left_item[1] if left_item is not None else None
        right_tensor_by_edge_id[edge.id] = (
            right_item[0] if right_item is not None else None
        )
        right_index_by_edge_id[edge.id] = (
            right_item[1] if right_item is not None else None
        )
    return (
        connected_index_ids,
        left_tensor_by_edge_id,
        left_index_by_edge_id,
        right_tensor_by_edge_id,
        right_index_by_edge_id,
    )


def _build_open_indices(
    all_indices: list[tuple[TensorSpec, IndexSpec]],
    connected_index_ids: set[str],
) -> list[tuple[TensorSpec, IndexSpec]]:
    """Return tensor/index pairs that are still open in the graph."""
    return [
        (tensor, index)
        for tensor, index in all_indices
        if index.id not in connected_index_ids
    ]

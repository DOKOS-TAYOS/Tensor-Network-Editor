from __future__ import annotations

from dataclasses import dataclass

from .models import IndexSpec, NetworkSpec, TensorSpec


@dataclass(slots=True)
class NetworkAnalysis:
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
    if validate:
        from .validation import ensure_valid_spec

        spec = ensure_valid_spec(spec)

    tensor_map = _build_tensor_map(spec)
    index_map = _build_index_map(spec)
    connected_index_ids = _build_connected_index_ids(spec)
    (
        left_tensor_by_edge_id,
        left_index_by_edge_id,
        right_tensor_by_edge_id,
        right_index_by_edge_id,
    ) = _build_edge_endpoint_maps(spec, index_map)
    open_indices = _build_open_indices(spec, connected_index_ids)

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


def _build_tensor_map(spec: NetworkSpec) -> dict[str, TensorSpec]:
    return {tensor.id: tensor for tensor in spec.tensors}


def _build_index_map(spec: NetworkSpec) -> dict[str, tuple[TensorSpec, IndexSpec]]:
    index_map: dict[str, tuple[TensorSpec, IndexSpec]] = {}
    for tensor in spec.tensors:
        for index in tensor.indices:
            index_map[index.id] = (tensor, index)
    return index_map


def _build_connected_index_ids(spec: NetworkSpec) -> set[str]:
    connected_index_ids: set[str] = set()
    for edge in spec.edges:
        connected_index_ids.add(edge.left.index_id)
        connected_index_ids.add(edge.right.index_id)
    return connected_index_ids


def _build_edge_endpoint_maps(
    spec: NetworkSpec,
    index_map: dict[str, tuple[TensorSpec, IndexSpec]],
) -> tuple[
    dict[str, TensorSpec | None],
    dict[str, IndexSpec | None],
    dict[str, TensorSpec | None],
    dict[str, IndexSpec | None],
]:
    left_tensor_by_edge_id: dict[str, TensorSpec | None] = {}
    left_index_by_edge_id: dict[str, IndexSpec | None] = {}
    right_tensor_by_edge_id: dict[str, TensorSpec | None] = {}
    right_index_by_edge_id: dict[str, IndexSpec | None] = {}

    for edge in spec.edges:
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
        left_tensor_by_edge_id,
        left_index_by_edge_id,
        right_tensor_by_edge_id,
        right_index_by_edge_id,
    )


def _build_open_indices(
    spec: NetworkSpec, connected_index_ids: set[str]
) -> list[tuple[TensorSpec, IndexSpec]]:
    return [
        (tensor, index)
        for tensor in spec.tensors
        for index in tensor.indices
        if index.id not in connected_index_ids
    ]

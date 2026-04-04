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


def analyze_network(spec: NetworkSpec) -> NetworkAnalysis:
    tensor_map = {tensor.id: tensor for tensor in spec.tensors}

    index_map: dict[str, tuple[TensorSpec, IndexSpec]] = {}
    for tensor in spec.tensors:
        for index in tensor.indices:
            index_map[index.id] = (tensor, index)

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

        left_tensor_by_edge_id[edge.id] = left_item[0] if left_item is not None else None
        left_index_by_edge_id[edge.id] = left_item[1] if left_item is not None else None
        right_tensor_by_edge_id[edge.id] = (
            right_item[0] if right_item is not None else None
        )
        right_index_by_edge_id[edge.id] = (
            right_item[1] if right_item is not None else None
        )

    open_indices = [
        (tensor, index)
        for tensor in spec.tensors
        for index in tensor.indices
        if index.id not in connected_index_ids
    ]

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

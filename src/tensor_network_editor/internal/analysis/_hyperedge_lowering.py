"""Helpers for lowering first-class hyperedges into pairwise copy tensors."""

from __future__ import annotations

from collections.abc import Iterable

from ...models import (
    CanvasPosition,
    EdgeEndpointRef,
    EdgeSpec,
    HyperedgeSpec,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
)
from ..models._model_tensor_data import TensorNumericLiteral


def lower_hyperedges_to_pairwise_spec(spec: NetworkSpec) -> NetworkSpec:
    """Return a normal-mode spec where hyperedges become copy tensors plus edges."""
    if not spec.hyperedges:
        return spec

    lowered = NetworkSpec.from_dict(spec.to_dict())
    lowered.hyperedges = []
    lowered.contraction_plan = None
    existing_tensor_ids = {tensor.id for tensor in lowered.tensors}
    existing_index_ids = {
        index.id for tensor in lowered.tensors for index in tensor.indices
    }
    existing_edge_ids = {edge.id for edge in lowered.edges}

    index_lookup = lowered.index_map()
    for hyperedge in spec.hyperedges:
        lowered_copy_tensor = _build_hyperedge_copy_tensor(
            hyperedge=hyperedge,
            index_lookup=index_lookup,
            existing_tensor_ids=existing_tensor_ids,
            existing_index_ids=existing_index_ids,
        )
        lowered.tensors.append(lowered_copy_tensor)
        lowered.edges.extend(
            _build_hyperedge_edges(
                hyperedge=hyperedge,
                copy_tensor=lowered_copy_tensor,
                existing_edge_ids=existing_edge_ids,
            )
        )
    return lowered


def _build_hyperedge_copy_tensor(
    *,
    hyperedge: HyperedgeSpec,
    index_lookup: dict[str, tuple[TensorSpec, IndexSpec]],
    existing_tensor_ids: set[str],
    existing_index_ids: set[str],
) -> TensorSpec:
    """Build one synthetic copy tensor for a validated hyperedge."""
    resolved_indices = [
        index_lookup[endpoint.index_id][1] for endpoint in hyperedge.endpoints
    ]
    copy_tensor_id = _reserve_unique_id(
        f"hyperedge_copy_{hyperedge.id}",
        existing_tensor_ids,
    )
    copy_index_specs = [
        IndexSpec(
            id=_reserve_unique_id(
                f"{copy_tensor_id}_index_{index_position + 1}",
                existing_index_ids,
            ),
            name=f"slot_{index_position + 1}",
            dimension=index.dimension,
        )
        for index_position, index in enumerate(resolved_indices)
    ]
    return TensorSpec(
        id=copy_tensor_id,
        name=f"Copy {hyperedge.name}",
        position=_average_tensor_position(
            index_lookup[endpoint.index_id][0] for endpoint in hyperedge.endpoints
        ),
        indices=copy_index_specs,
        metadata={
            "generated_for_hyperedge": hyperedge.id,
            "generated_by": "hyperedge_lowering",
        },
    )


def _build_hyperedge_edges(
    *,
    hyperedge: HyperedgeSpec,
    copy_tensor: TensorSpec,
    existing_edge_ids: set[str],
) -> list[EdgeSpec]:
    """Build one binary edge per hyperedge endpoint."""
    return [
        EdgeSpec(
            id=_reserve_unique_id(
                f"{hyperedge.id}_edge_{endpoint_position + 1}",
                existing_edge_ids,
            ),
            name=f"{hyperedge.name}_{endpoint_position + 1}",
            left=EdgeEndpointRef(
                tensor_id=endpoint.tensor_id,
                index_id=endpoint.index_id,
            ),
            right=EdgeEndpointRef(
                tensor_id=copy_tensor.id,
                index_id=copy_tensor.indices[endpoint_position].id,
            ),
            metadata={
                "generated_for_hyperedge": hyperedge.id,
                "generated_by": "hyperedge_lowering",
            },
        )
        for endpoint_position, endpoint in enumerate(hyperedge.endpoints)
    ]


def _build_copy_tensor_values(
    dimension: int,
    rank: int,
) -> TensorNumericLiteral:
    """Build the explicit Kronecker-delta literal for one copy tensor."""
    return _build_copy_tensor_values_at_prefix(
        dimension=dimension,
        rank=rank,
        prefix=(),
    )


def _build_copy_tensor_values_at_prefix(
    *,
    dimension: int,
    rank: int,
    prefix: tuple[int, ...],
) -> TensorNumericLiteral:
    """Build one subtree of the copy tensor literal recursively."""
    if len(prefix) == rank:
        return 1.0 if len(set(prefix)) == 1 else 0.0
    return [
        _build_copy_tensor_values_at_prefix(
            dimension=dimension,
            rank=rank,
            prefix=(*prefix, value),
        )
        for value in range(dimension)
    ]


def _average_tensor_position(tensors: Iterable[TensorSpec]) -> CanvasPosition:
    """Return the geometric center of the provided tensor positions."""
    tensor_list = list(tensors)
    if not tensor_list:
        return CanvasPosition()
    return CanvasPosition(
        x=sum(tensor.position.x for tensor in tensor_list) / len(tensor_list),
        y=sum(tensor.position.y for tensor in tensor_list) / len(tensor_list),
    )


def _reserve_unique_id(candidate: str, existing_ids: set[str]) -> str:
    """Return a unique identifier derived from ``candidate``."""
    if candidate not in existing_ids:
        existing_ids.add(candidate)
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in existing_ids:
        suffix += 1
    resolved_candidate = f"{candidate}_{suffix}"
    existing_ids.add(resolved_candidate)
    return resolved_candidate

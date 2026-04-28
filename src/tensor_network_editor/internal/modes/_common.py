"""Shared helpers for typed periodic editor modes."""

from __future__ import annotations

from ...models import EdgeEndpointRef


def remap_analysis_edge_endpoint(
    endpoint: EdgeEndpointRef,
    *,
    tensor_id_by_original_id: dict[str, str],
) -> EdgeEndpointRef:
    """Return an edge endpoint with any boundary tensor id remapped."""
    return EdgeEndpointRef(
        tensor_id=tensor_id_by_original_id.get(endpoint.tensor_id, endpoint.tensor_id),
        index_id=endpoint.index_id,
    )

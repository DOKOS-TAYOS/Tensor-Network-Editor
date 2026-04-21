"""Headless helpers for extracting and reusing tensor-network fragments."""

from __future__ import annotations

from copy import deepcopy

from ...models import CanvasPosition, EdgeEndpointRef, NetworkSpec
from ..io._payloads import new_identifier


def extract_subnetwork_spec(
    spec: NetworkSpec,
    *,
    tensor_ids: list[str],
    name: str | None = None,
    spec_id: str | None = None,
) -> NetworkSpec:
    """Extract a tensor-only fragment as a standalone ``NetworkSpec``."""
    _require_normal_graph_mode(spec)
    selected_tensor_ids = _normalize_tensor_ids(spec, tensor_ids)
    selected_tensor_id_set = set(selected_tensor_ids)

    return NetworkSpec(
        id=spec_id or new_identifier("network"),
        name=name or spec.name,
        tensors=[
            deepcopy(tensor)
            for tensor in spec.tensors
            if tensor.id in selected_tensor_id_set
        ],
        groups=[
            deepcopy(group)
            for group in spec.groups
            if group.tensor_ids
            and all(
                tensor_id in selected_tensor_id_set for tensor_id in group.tensor_ids
            )
        ],
        edges=[
            deepcopy(edge)
            for edge in spec.edges
            if edge.left.tensor_id in selected_tensor_id_set
            and edge.right.tensor_id in selected_tensor_id_set
        ],
        hyperedges=[
            deepcopy(hyperedge)
            for hyperedge in spec.hyperedges
            if hyperedge.endpoints
            and all(
                endpoint.tensor_id in selected_tensor_id_set
                for endpoint in hyperedge.endpoints
            )
        ],
        notes=[],
        contraction_plan=None,
        metadata=deepcopy(spec.metadata),
    )


def prepare_subnetwork_for_insertion(
    spec: NetworkSpec,
    *,
    target_center: CanvasPosition,
) -> NetworkSpec:
    """Regenerate ids for a fragment and translate it to ``target_center``."""
    _require_normal_graph_mode(spec)
    prepared = NetworkSpec(
        id=new_identifier("network"),
        name=spec.name,
        tensors=deepcopy(spec.tensors),
        groups=deepcopy(spec.groups),
        edges=deepcopy(spec.edges),
        hyperedges=deepcopy(spec.hyperedges),
        notes=[],
        contraction_plan=None,
        metadata=deepcopy(spec.metadata),
    )
    tensor_id_map: dict[str, str] = {}
    index_id_map: dict[str, str] = {}

    for tensor in prepared.tensors:
        previous_tensor_id = tensor.id
        tensor.id = new_identifier("tensor")
        tensor_id_map[previous_tensor_id] = tensor.id
        for index in tensor.indices:
            previous_index_id = index.id
            index.id = new_identifier("index")
            index_id_map[previous_index_id] = index.id

    for edge in prepared.edges:
        if (
            edge.left.tensor_id not in tensor_id_map
            or edge.right.tensor_id not in tensor_id_map
        ):
            raise ValueError(
                f"Edge '{edge.id}' references tensors outside the fragment."
            )
        if (
            edge.left.index_id not in index_id_map
            or edge.right.index_id not in index_id_map
        ):
            raise ValueError(
                f"Edge '{edge.id}' references indices outside the fragment."
            )
        edge.id = new_identifier("edge")
        edge.left = EdgeEndpointRef(
            tensor_id=tensor_id_map[edge.left.tensor_id],
            index_id=index_id_map[edge.left.index_id],
        )
        edge.right = EdgeEndpointRef(
            tensor_id=tensor_id_map[edge.right.tensor_id],
            index_id=index_id_map[edge.right.index_id],
        )

    for hyperedge in prepared.hyperedges:
        hyperedge.id = new_identifier("hyperedge")
        hyperedge.endpoints = [
            EdgeEndpointRef(
                tensor_id=tensor_id_map[endpoint.tensor_id],
                index_id=index_id_map[endpoint.index_id],
            )
            for endpoint in hyperedge.endpoints
        ]

    next_groups = []
    for group in prepared.groups:
        if not group.tensor_ids:
            continue
        if any(tensor_id not in tensor_id_map for tensor_id in group.tensor_ids):
            raise ValueError(
                f"Group '{group.id}' references tensors outside the fragment."
            )
        group.id = new_identifier("group")
        group.tensor_ids = [tensor_id_map[tensor_id] for tensor_id in group.tensor_ids]
        next_groups.append(group)
    prepared.groups = next_groups

    _recenter_tensors(prepared, target_center)
    return prepared


def _normalize_tensor_ids(spec: NetworkSpec, tensor_ids: list[str]) -> list[str]:
    """Validate ``tensor_ids`` and return them in source-spec tensor order."""
    if not tensor_ids:
        raise ValueError("'tensor_ids' must be a non-empty list of tensor ids.")
    requested_tensor_ids = [str(tensor_id).strip() for tensor_id in tensor_ids]
    if any(not tensor_id for tensor_id in requested_tensor_ids):
        raise ValueError("'tensor_ids' must be a non-empty list of tensor ids.")
    tensor_id_set = set(requested_tensor_ids)
    known_tensor_ids = {tensor.id for tensor in spec.tensors}
    missing_tensor_ids = [
        tensor_id
        for tensor_id in requested_tensor_ids
        if tensor_id not in known_tensor_ids
    ]
    if missing_tensor_ids:
        raise ValueError(
            f"Unknown tensor ids: {', '.join(sorted(set(missing_tensor_ids)))}."
        )
    ordered_tensor_ids = [
        tensor.id for tensor in spec.tensors if tensor.id in tensor_id_set
    ]
    if not ordered_tensor_ids:
        raise ValueError("'tensor_ids' must select at least one tensor.")
    return ordered_tensor_ids


def _recenter_tensors(spec: NetworkSpec, target_center: CanvasPosition) -> None:
    """Translate all tensors so their bounding-box center matches ``target_center``."""
    if not spec.tensors:
        return
    left = min(tensor.position.x - tensor.size.width / 2 for tensor in spec.tensors)
    right = max(tensor.position.x + tensor.size.width / 2 for tensor in spec.tensors)
    top = min(tensor.position.y - tensor.size.height / 2 for tensor in spec.tensors)
    bottom = max(tensor.position.y + tensor.size.height / 2 for tensor in spec.tensors)
    delta_x = target_center.x - ((left + right) / 2)
    delta_y = target_center.y - ((top + bottom) / 2)
    for tensor in spec.tensors:
        tensor.position.x += delta_x
        tensor.position.y += delta_y


def _require_normal_graph_mode(spec: NetworkSpec) -> None:
    """Reject operations that are not yet available in For modes."""
    if (
        spec.linear_periodic_chain is not None
        or spec.grid_periodic_grid is not None
        or spec.tree_periodic_tree is not None
    ):
        raise ValueError(
            "Subnetwork extraction and insertion are only available in normal graph mode."
        )

"""Validation helpers for hyperedge specifications."""

from __future__ import annotations

from collections import Counter

from ...models import HyperedgeSpec, IndexSpec, TensorSpec, ValidationIssue
from ._validation_common import append_issue, is_valid_name, validate_metadata


def validate_hyperedge(
    hyperedge: HyperedgeSpec,
    *,
    analysis_tensor_map: dict[str, TensorSpec],
    analysis_index_map: dict[str, tuple[TensorSpec, IndexSpec]],
    connected_indices: set[str],
    issues: list[ValidationIssue],
) -> None:
    """Validate one hyperedge against analyzed tensor and index lookups."""
    if not is_valid_name(hyperedge.name):
        append_issue(
            issues,
            code="invalid-name",
            message=f"Hyperedge '{hyperedge.id}' has an empty name.",
            path=f"hyperedges.{hyperedge.id}.name",
        )
    validate_metadata(
        f"hyperedges.{hyperedge.id}.metadata",
        hyperedge.metadata,
        issues,
    )

    if len(hyperedge.endpoints) < 3:
        append_issue(
            issues,
            code="invalid-hyperedge",
            message=f"Hyperedge '{hyperedge.id}' must connect at least 3 endpoints.",
            path=f"hyperedges.{hyperedge.id}.endpoints",
        )

    endpoint_counts = Counter(
        (endpoint.tensor_id, endpoint.index_id) for endpoint in hyperedge.endpoints
    )
    duplicate_endpoints = [
        endpoint for endpoint, count in endpoint_counts.items() if count > 1
    ]
    if duplicate_endpoints:
        append_issue(
            issues,
            code="duplicate-hyperedge-endpoint",
            message=(
                f"Hyperedge '{hyperedge.id}' contains duplicate endpoints: "
                f"{duplicate_endpoints!r}."
            ),
            path=f"hyperedges.{hyperedge.id}.endpoints",
        )

    endpoint_dimensions: list[int] = []
    for endpoint in hyperedge.endpoints:
        endpoint_path = f"hyperedges.{hyperedge.id}.endpoints"
        endpoint_tensor = analysis_tensor_map.get(endpoint.tensor_id)
        endpoint_item = analysis_index_map.get(endpoint.index_id)

        if endpoint_tensor is None or endpoint_item is None:
            append_issue(
                issues,
                code="missing-endpoint",
                message=f"Hyperedge '{hyperedge.id}' refers to a missing endpoint.",
                path=endpoint_path,
            )
            continue

        endpoint_owner, endpoint_index = endpoint_item
        if endpoint_owner.id != endpoint.tensor_id:
            append_issue(
                issues,
                code="endpoint-tensor-mismatch",
                message=(
                    f"Hyperedge '{hyperedge.id}' endpoint does not belong to tensor "
                    f"'{endpoint.tensor_id}'."
                ),
                path=endpoint_path,
            )
            continue

        if endpoint.index_id in connected_indices:
            append_issue(
                issues,
                code="index-already-connected",
                message=f"Index '{endpoint.index_id}' is connected more than once.",
                path=endpoint_path,
            )
        connected_indices.add(endpoint.index_id)
        endpoint_dimensions.append(endpoint_index.dimension)

    if endpoint_dimensions and any(
        dimension != endpoint_dimensions[0] for dimension in endpoint_dimensions[1:]
    ):
        append_issue(
            issues,
            code="dimension-mismatch",
            message=(
                f"Hyperedge '{hyperedge.id}' connects endpoints with mismatched "
                "dimensions."
            ),
            path=f"hyperedges.{hyperedge.id}",
        )

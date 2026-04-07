"""Validation helpers for edge specifications."""

from __future__ import annotations

from ._validation_common import append_issue, is_valid_name, validate_metadata
from .models import EdgeSpec, IndexSpec, TensorSpec, ValidationIssue


def validate_edge(
    edge: EdgeSpec,
    *,
    analysis_tensor_map: dict[str, TensorSpec],
    analysis_index_map: dict[str, tuple[TensorSpec, IndexSpec]],
    connected_indices: set[str],
    issues: list[ValidationIssue],
) -> None:
    """Validate one edge against the analyzed tensor and index lookups."""
    if not is_valid_name(edge.name):
        append_issue(
            issues,
            code="invalid-name",
            message=f"Edge '{edge.id}' has an empty name.",
            path=f"edges.{edge.id}.name",
        )
    validate_metadata(f"edges.{edge.id}.metadata", edge.metadata, issues)

    left_tensor = analysis_tensor_map.get(edge.left.tensor_id)
    right_tensor = analysis_tensor_map.get(edge.right.tensor_id)
    left_item = analysis_index_map.get(edge.left.index_id)
    right_item = analysis_index_map.get(edge.right.index_id)

    if left_tensor is None or left_item is None:
        append_issue(
            issues,
            code="missing-endpoint",
            message=f"Edge '{edge.id}' refers to a missing left endpoint.",
            path=f"edges.{edge.id}.left",
        )
        return
    if right_tensor is None or right_item is None:
        append_issue(
            issues,
            code="missing-endpoint",
            message=f"Edge '{edge.id}' refers to a missing right endpoint.",
            path=f"edges.{edge.id}.right",
        )
        return

    left_owner, left_index = left_item
    right_owner, right_index = right_item
    if left_owner.id != edge.left.tensor_id:
        append_issue(
            issues,
            code="endpoint-tensor-mismatch",
            message=(
                f"Edge '{edge.id}' left endpoint does not belong to tensor "
                f"'{edge.left.tensor_id}'."
            ),
            path=f"edges.{edge.id}.left",
        )
    if right_owner.id != edge.right.tensor_id:
        append_issue(
            issues,
            code="endpoint-tensor-mismatch",
            message=(
                f"Edge '{edge.id}' right endpoint does not belong to tensor "
                f"'{edge.right.tensor_id}'."
            ),
            path=f"edges.{edge.id}.right",
        )

    for endpoint_path, index_id in (
        (f"edges.{edge.id}.left", edge.left.index_id),
        (f"edges.{edge.id}.right", edge.right.index_id),
    ):
        if index_id in connected_indices:
            append_issue(
                issues,
                code="index-already-connected",
                message=f"Index '{index_id}' is connected more than once.",
                path=endpoint_path,
            )
        connected_indices.add(index_id)

    if left_index.dimension != right_index.dimension:
        append_issue(
            issues,
            code="dimension-mismatch",
            message=(
                f"Edge '{edge.id}' connects dimensions {left_index.dimension} and "
                f"{right_index.dimension}, which must match."
            ),
            path=f"edges.{edge.id}",
        )

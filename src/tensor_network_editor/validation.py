from __future__ import annotations

import json
import math
from collections import Counter

from ._analysis import analyze_network
from .errors import SpecValidationError
from .models import NetworkSpec, ValidationIssue


def _is_valid_name(value: str) -> bool:
    return bool(value.strip())


def _validate_metadata(
    path: str, metadata: object, issues: list[ValidationIssue]
) -> None:
    try:
        json.dumps(metadata)
    except TypeError as exc:
        issues.append(
            ValidationIssue(
                code="metadata-not-serializable",
                message=f"Metadata at {path} is not JSON serializable: {exc}",
                path=path,
            )
        )


def validate_spec(spec: NetworkSpec) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if not _is_valid_name(spec.name):
        issues.append(
            ValidationIssue(
                code="invalid-name",
                message="Network name cannot be empty.",
                path="name",
            )
        )

    _validate_metadata("metadata", spec.metadata, issues)

    tensor_counts = Counter(tensor.id for tensor in spec.tensors)
    for tensor_id, count in tensor_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code="duplicate-tensor-id",
                    message=f"Tensor id '{tensor_id}' is duplicated.",
                    path="tensors",
                )
            )

    edge_counts = Counter(edge.id for edge in spec.edges)
    for edge_id, count in edge_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code="duplicate-edge-id",
                    message=f"Edge id '{edge_id}' is duplicated.",
                    path="edges",
                )
            )

    index_counts = Counter(
        index.id for tensor in spec.tensors for index in tensor.indices
    )
    for index_id, count in index_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code="duplicate-index-id",
                    message=f"Index id '{index_id}' is duplicated.",
                    path="tensors.indices",
                )
            )

    analysis = analyze_network(spec)

    for tensor in spec.tensors:
        if not _is_valid_name(tensor.name):
            issues.append(
                ValidationIssue(
                    code="invalid-name",
                    message=f"Tensor '{tensor.id}' has an empty name.",
                    path=f"tensors.{tensor.id}.name",
                )
            )
        _validate_metadata(f"tensors.{tensor.id}.metadata", tensor.metadata, issues)

        if not math.isfinite(tensor.position.x) or not math.isfinite(tensor.position.y):
            issues.append(
                ValidationIssue(
                    code="invalid-position",
                    message=f"Tensor '{tensor.id}' has a non-finite position.",
                    path=f"tensors.{tensor.id}.position",
                )
            )
        if (
            not math.isfinite(tensor.size.width)
            or not math.isfinite(tensor.size.height)
            or tensor.size.width <= 0
            or tensor.size.height <= 0
        ):
            issues.append(
                ValidationIssue(
                    code="invalid-size",
                    message=f"Tensor '{tensor.id}' must have a positive finite size.",
                    path=f"tensors.{tensor.id}.size",
                )
            )

        seen_tensor_index_ids = Counter(index.id for index in tensor.indices)
        for index_id, count in seen_tensor_index_ids.items():
            if count > 1:
                issues.append(
                    ValidationIssue(
                        code="duplicate-index-id",
                        message=f"Tensor '{tensor.id}' contains duplicate index id '{index_id}'.",
                        path=f"tensors.{tensor.id}.indices",
                    )
                )

        seen_tensor_index_names = Counter(
            index.name.strip() for index in tensor.indices if _is_valid_name(index.name)
        )
        for index_name, count in seen_tensor_index_names.items():
            if count > 1:
                issues.append(
                    ValidationIssue(
                        code="duplicate-index-name",
                        message=(
                            f"Tensor '{tensor.id}' contains duplicate index name "
                            f"'{index_name}'."
                        ),
                        path=f"tensors.{tensor.id}.indices",
                    )
                )

        for index in tensor.indices:
            if not _is_valid_name(index.name):
                issues.append(
                    ValidationIssue(
                        code="invalid-name",
                        message=f"Index '{index.id}' has an empty name.",
                        path=f"tensors.{tensor.id}.indices.{index.id}.name",
                    )
                )
            if index.dimension <= 0:
                issues.append(
                    ValidationIssue(
                        code="invalid-dimension",
                        message=f"Index '{index.id}' must have a positive dimension.",
                        path=f"tensors.{tensor.id}.indices.{index.id}.dimension",
                    )
                )
            if not math.isfinite(index.offset.x) or not math.isfinite(index.offset.y):
                issues.append(
                    ValidationIssue(
                        code="invalid-offset",
                        message=f"Index '{index.id}' has a non-finite offset.",
                        path=f"tensors.{tensor.id}.indices.{index.id}.offset",
                    )
                )
            _validate_metadata(
                f"tensors.{tensor.id}.indices.{index.id}.metadata",
                index.metadata,
                issues,
            )

    group_counts = Counter(group.id for group in spec.groups)
    for group_id, count in group_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code="duplicate-group-id",
                    message=f"Group id '{group_id}' is duplicated.",
                    path="groups",
                )
            )

    for group in spec.groups:
        if not _is_valid_name(group.name):
            issues.append(
                ValidationIssue(
                    code="invalid-name",
                    message=f"Group '{group.id}' has an empty name.",
                    path=f"groups.{group.id}.name",
                )
            )
        _validate_metadata(f"groups.{group.id}.metadata", group.metadata, issues)
        for tensor_id in group.tensor_ids:
            if tensor_id not in analysis.tensor_map:
                issues.append(
                    ValidationIssue(
                        code="missing-group-tensor",
                        message=(
                            f"Group '{group.id}' refers to missing tensor '{tensor_id}'."
                        ),
                        path=f"groups.{group.id}.tensor_ids",
                    )
                )

    connected_indices: set[str] = set()
    for edge in spec.edges:
        if not _is_valid_name(edge.name):
            issues.append(
                ValidationIssue(
                    code="invalid-name",
                    message=f"Edge '{edge.id}' has an empty name.",
                    path=f"edges.{edge.id}.name",
                )
            )
        _validate_metadata(f"edges.{edge.id}.metadata", edge.metadata, issues)

        left_tensor = analysis.tensor_map.get(edge.left.tensor_id)
        right_tensor = analysis.tensor_map.get(edge.right.tensor_id)
        left_item = analysis.index_map.get(edge.left.index_id)
        right_item = analysis.index_map.get(edge.right.index_id)

        if left_tensor is None or left_item is None:
            issues.append(
                ValidationIssue(
                    code="missing-endpoint",
                    message=f"Edge '{edge.id}' refers to a missing left endpoint.",
                    path=f"edges.{edge.id}.left",
                )
            )
            continue
        if right_tensor is None or right_item is None:
            issues.append(
                ValidationIssue(
                    code="missing-endpoint",
                    message=f"Edge '{edge.id}' refers to a missing right endpoint.",
                    path=f"edges.{edge.id}.right",
                )
            )
            continue

        left_owner, left_index = left_item
        right_owner, right_index = right_item
        if left_owner.id != edge.left.tensor_id:
            issues.append(
                ValidationIssue(
                    code="endpoint-tensor-mismatch",
                    message=f"Edge '{edge.id}' left endpoint does not belong to tensor '{edge.left.tensor_id}'.",
                    path=f"edges.{edge.id}.left",
                )
            )
        if right_owner.id != edge.right.tensor_id:
            issues.append(
                ValidationIssue(
                    code="endpoint-tensor-mismatch",
                    message=f"Edge '{edge.id}' right endpoint does not belong to tensor '{edge.right.tensor_id}'.",
                    path=f"edges.{edge.id}.right",
                )
            )

        for endpoint_path, index_id in (
            (f"edges.{edge.id}.left", edge.left.index_id),
            (f"edges.{edge.id}.right", edge.right.index_id),
        ):
            if index_id in connected_indices:
                issues.append(
                    ValidationIssue(
                        code="index-already-connected",
                        message=f"Index '{index_id}' is connected more than once.",
                        path=endpoint_path,
                    )
                )
            connected_indices.add(index_id)

        if left_index.dimension != right_index.dimension:
            issues.append(
                ValidationIssue(
                    code="dimension-mismatch",
                    message=(
                        f"Edge '{edge.id}' connects dimensions {left_index.dimension} and "
                        f"{right_index.dimension}, which must match."
                    ),
                    path=f"edges.{edge.id}",
                )
            )

    return issues


def ensure_valid_spec(spec: NetworkSpec) -> NetworkSpec:
    issues = validate_spec(spec)
    if issues:
        raise SpecValidationError(issues)
    return spec

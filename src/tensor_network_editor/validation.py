from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable

from ._analysis import analyze_network
from .errors import SpecValidationError
from .models import (
    EdgeSpec,
    GroupSpec,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
    ValidationIssue,
)


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


def _append_issue(
    issues: list[ValidationIssue], *, code: str, message: str, path: str
) -> None:
    issues.append(ValidationIssue(code=code, message=message, path=path))


def _append_duplicate_id_issues(
    values: Iterable[str],
    *,
    code: str,
    path: str,
    message_prefix: str,
    issues: list[ValidationIssue],
) -> None:
    counts = Counter(values)
    for value, count in counts.items():
        if count > 1:
            _append_issue(
                issues,
                code=code,
                message=f"{message_prefix} '{value}' is duplicated.",
                path=path,
            )


def _validate_network(spec: NetworkSpec, issues: list[ValidationIssue]) -> None:
    if not _is_valid_name(spec.name):
        _append_issue(
            issues,
            code="invalid-name",
            message="Network name cannot be empty.",
            path="name",
        )

    _validate_metadata("metadata", spec.metadata, issues)
    _append_duplicate_id_issues(
        (tensor.id for tensor in spec.tensors),
        code="duplicate-tensor-id",
        path="tensors",
        message_prefix="Tensor id",
        issues=issues,
    )
    _append_duplicate_id_issues(
        (edge.id for edge in spec.edges),
        code="duplicate-edge-id",
        path="edges",
        message_prefix="Edge id",
        issues=issues,
    )
    _append_duplicate_id_issues(
        (index.id for tensor in spec.tensors for index in tensor.indices),
        code="duplicate-index-id",
        path="tensors.indices",
        message_prefix="Index id",
        issues=issues,
    )
    _append_duplicate_id_issues(
        (group.id for group in spec.groups),
        code="duplicate-group-id",
        path="groups",
        message_prefix="Group id",
        issues=issues,
    )


def _validate_tensor(
    tensor: TensorSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    if not _is_valid_name(tensor.name):
        _append_issue(
            issues,
            code="invalid-name",
            message=f"Tensor '{tensor.id}' has an empty name.",
            path=f"tensors.{tensor.id}.name",
        )
    _validate_metadata(f"tensors.{tensor.id}.metadata", tensor.metadata, issues)

    if not math.isfinite(tensor.position.x) or not math.isfinite(tensor.position.y):
        _append_issue(
            issues,
            code="invalid-position",
            message=f"Tensor '{tensor.id}' has a non-finite position.",
            path=f"tensors.{tensor.id}.position",
        )
    if (
        not math.isfinite(tensor.size.width)
        or not math.isfinite(tensor.size.height)
        or tensor.size.width <= 0
        or tensor.size.height <= 0
    ):
        _append_issue(
            issues,
            code="invalid-size",
            message=f"Tensor '{tensor.id}' must have a positive finite size.",
            path=f"tensors.{tensor.id}.size",
        )

    index_id_counts = Counter(index.id for index in tensor.indices)
    for index_id, count in index_id_counts.items():
        if count > 1:
            _append_issue(
                issues,
                code="duplicate-index-id",
                message=(
                    f"Tensor '{tensor.id}' contains duplicate index id '{index_id}'."
                ),
                path=f"tensors.{tensor.id}.indices",
            )

    index_name_counts = Counter(
        index.name.strip() for index in tensor.indices if _is_valid_name(index.name)
    )
    for index_name, count in index_name_counts.items():
        if count > 1:
            _append_issue(
                issues,
                code="duplicate-index-name",
                message=(
                    f"Tensor '{tensor.id}' contains duplicate index name "
                    f"'{index_name}'."
                ),
                path=f"tensors.{tensor.id}.indices",
            )

    for index in tensor.indices:
        _validate_index(tensor=tensor, issues=issues, index=index)


def _validate_index(
    *,
    tensor: TensorSpec,
    index: IndexSpec,
    issues: list[ValidationIssue],
) -> None:
    if not _is_valid_name(index.name):
        _append_issue(
            issues,
            code="invalid-name",
            message=f"Index '{index.id}' has an empty name.",
            path=f"tensors.{tensor.id}.indices.{index.id}.name",
        )
    if index.dimension <= 0:
        _append_issue(
            issues,
            code="invalid-dimension",
            message=f"Index '{index.id}' must have a positive dimension.",
            path=f"tensors.{tensor.id}.indices.{index.id}.dimension",
        )
    if not math.isfinite(index.offset.x) or not math.isfinite(index.offset.y):
        _append_issue(
            issues,
            code="invalid-offset",
            message=f"Index '{index.id}' has a non-finite offset.",
            path=f"tensors.{tensor.id}.indices.{index.id}.offset",
        )
    _validate_metadata(
        f"tensors.{tensor.id}.indices.{index.id}.metadata",
        index.metadata,
        issues,
    )


def _validate_group(
    group: GroupSpec,
    *,
    tensor_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    if not _is_valid_name(group.name):
        _append_issue(
            issues,
            code="invalid-name",
            message=f"Group '{group.id}' has an empty name.",
            path=f"groups.{group.id}.name",
        )
    _validate_metadata(f"groups.{group.id}.metadata", group.metadata, issues)
    for tensor_id in group.tensor_ids:
        if tensor_id not in tensor_ids:
            _append_issue(
                issues,
                code="missing-group-tensor",
                message=f"Group '{group.id}' refers to missing tensor '{tensor_id}'.",
                path=f"groups.{group.id}.tensor_ids",
            )


def _validate_edge(
    edge: EdgeSpec,
    *,
    analysis_tensor_map: dict[str, TensorSpec],
    analysis_index_map: dict[str, tuple[TensorSpec, IndexSpec]],
    connected_indices: set[str],
    issues: list[ValidationIssue],
) -> None:
    if not _is_valid_name(edge.name):
        _append_issue(
            issues,
            code="invalid-name",
            message=f"Edge '{edge.id}' has an empty name.",
            path=f"edges.{edge.id}.name",
        )
    _validate_metadata(f"edges.{edge.id}.metadata", edge.metadata, issues)

    left_tensor = analysis_tensor_map.get(edge.left.tensor_id)
    right_tensor = analysis_tensor_map.get(edge.right.tensor_id)
    left_item = analysis_index_map.get(edge.left.index_id)
    right_item = analysis_index_map.get(edge.right.index_id)

    if left_tensor is None or left_item is None:
        _append_issue(
            issues,
            code="missing-endpoint",
            message=f"Edge '{edge.id}' refers to a missing left endpoint.",
            path=f"edges.{edge.id}.left",
        )
        return
    if right_tensor is None or right_item is None:
        _append_issue(
            issues,
            code="missing-endpoint",
            message=f"Edge '{edge.id}' refers to a missing right endpoint.",
            path=f"edges.{edge.id}.right",
        )
        return

    left_owner, left_index = left_item
    right_owner, right_index = right_item
    if left_owner.id != edge.left.tensor_id:
        _append_issue(
            issues,
            code="endpoint-tensor-mismatch",
            message=(
                f"Edge '{edge.id}' left endpoint does not belong to tensor "
                f"'{edge.left.tensor_id}'."
            ),
            path=f"edges.{edge.id}.left",
        )
    if right_owner.id != edge.right.tensor_id:
        _append_issue(
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
            _append_issue(
                issues,
                code="index-already-connected",
                message=f"Index '{index_id}' is connected more than once.",
                path=endpoint_path,
            )
        connected_indices.add(index_id)

    if left_index.dimension != right_index.dimension:
        _append_issue(
            issues,
            code="dimension-mismatch",
            message=(
                f"Edge '{edge.id}' connects dimensions {left_index.dimension} and "
                f"{right_index.dimension}, which must match."
            ),
            path=f"edges.{edge.id}",
        )


def validate_spec(spec: NetworkSpec) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    _validate_network(spec, issues)
    analysis = analyze_network(spec)
    tensor_ids = set(analysis.tensor_map)

    for tensor in spec.tensors:
        _validate_tensor(tensor, issues=issues)

    for group in spec.groups:
        _validate_group(group, tensor_ids=tensor_ids, issues=issues)

    connected_indices: set[str] = set()
    for edge in spec.edges:
        _validate_edge(
            edge,
            analysis_tensor_map=analysis.tensor_map,
            analysis_index_map=analysis.index_map,
            connected_indices=connected_indices,
            issues=issues,
        )

    return issues


def ensure_valid_spec(spec: NetworkSpec) -> NetworkSpec:
    issues = validate_spec(spec)
    if issues:
        raise SpecValidationError(issues)
    return spec

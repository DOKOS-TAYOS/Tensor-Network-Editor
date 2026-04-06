from __future__ import annotations

import math
from collections import Counter

from ._validation_common import (
    append_duplicate_id_issues,
    append_issue,
    is_valid_name,
    validate_metadata,
)
from .models import (
    CanvasNoteSpec,
    GroupSpec,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
    ValidationIssue,
)


def validate_network(spec: NetworkSpec, issues: list[ValidationIssue]) -> None:
    if not is_valid_name(spec.name):
        append_issue(
            issues,
            code="invalid-name",
            message="Network name cannot be empty.",
            path="name",
        )

    validate_metadata("metadata", spec.metadata, issues)
    append_duplicate_id_issues(
        (tensor.id for tensor in spec.tensors),
        code="duplicate-tensor-id",
        path="tensors",
        message_prefix="Tensor id",
        issues=issues,
    )
    append_duplicate_id_issues(
        (edge.id for edge in spec.edges),
        code="duplicate-edge-id",
        path="edges",
        message_prefix="Edge id",
        issues=issues,
    )
    append_duplicate_id_issues(
        (index.id for tensor in spec.tensors for index in tensor.indices),
        code="duplicate-index-id",
        path="tensors.indices",
        message_prefix="Index id",
        issues=issues,
    )
    append_duplicate_id_issues(
        (group.id for group in spec.groups),
        code="duplicate-group-id",
        path="groups",
        message_prefix="Group id",
        issues=issues,
    )
    append_duplicate_id_issues(
        (note.id for note in spec.notes),
        code="duplicate-note-id",
        path="notes",
        message_prefix="Note id",
        issues=issues,
    )


def validate_tensor(
    tensor: TensorSpec,
    *,
    issues: list[ValidationIssue],
) -> None:
    if not is_valid_name(tensor.name):
        append_issue(
            issues,
            code="invalid-name",
            message=f"Tensor '{tensor.id}' has an empty name.",
            path=f"tensors.{tensor.id}.name",
        )
    validate_metadata(f"tensors.{tensor.id}.metadata", tensor.metadata, issues)

    if not math.isfinite(tensor.position.x) or not math.isfinite(tensor.position.y):
        append_issue(
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
        append_issue(
            issues,
            code="invalid-size",
            message=f"Tensor '{tensor.id}' must have a positive finite size.",
            path=f"tensors.{tensor.id}.size",
        )

    index_id_counts = Counter(index.id for index in tensor.indices)
    for index_id, count in index_id_counts.items():
        if count > 1:
            append_issue(
                issues,
                code="duplicate-index-id",
                message=(
                    f"Tensor '{tensor.id}' contains duplicate index id '{index_id}'."
                ),
                path=f"tensors.{tensor.id}.indices",
            )

    index_name_counts = Counter(
        index.name.strip() for index in tensor.indices if is_valid_name(index.name)
    )
    for index_name, count in index_name_counts.items():
        if count > 1:
            append_issue(
                issues,
                code="duplicate-index-name",
                message=(
                    f"Tensor '{tensor.id}' contains duplicate index name "
                    f"'{index_name}'."
                ),
                path=f"tensors.{tensor.id}.indices",
            )

    for index in tensor.indices:
        validate_index(tensor=tensor, index=index, issues=issues)


def validate_index(
    *,
    tensor: TensorSpec,
    index: IndexSpec,
    issues: list[ValidationIssue],
) -> None:
    if not is_valid_name(index.name):
        append_issue(
            issues,
            code="invalid-name",
            message=f"Index '{index.id}' has an empty name.",
            path=f"tensors.{tensor.id}.indices.{index.id}.name",
        )
    if index.dimension <= 0:
        append_issue(
            issues,
            code="invalid-dimension",
            message=f"Index '{index.id}' must have a positive dimension.",
            path=f"tensors.{tensor.id}.indices.{index.id}.dimension",
        )
    if not math.isfinite(index.offset.x) or not math.isfinite(index.offset.y):
        append_issue(
            issues,
            code="invalid-offset",
            message=f"Index '{index.id}' has a non-finite offset.",
            path=f"tensors.{tensor.id}.indices.{index.id}.offset",
        )
    validate_metadata(
        f"tensors.{tensor.id}.indices.{index.id}.metadata",
        index.metadata,
        issues,
    )


def validate_group(
    group: GroupSpec,
    *,
    tensor_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    if not is_valid_name(group.name):
        append_issue(
            issues,
            code="invalid-name",
            message=f"Group '{group.id}' has an empty name.",
            path=f"groups.{group.id}.name",
        )
    validate_metadata(f"groups.{group.id}.metadata", group.metadata, issues)
    for tensor_id in group.tensor_ids:
        if tensor_id not in tensor_ids:
            append_issue(
                issues,
                code="missing-group-tensor",
                message=f"Group '{group.id}' refers to missing tensor '{tensor_id}'.",
                path=f"groups.{group.id}.tensor_ids",
            )


def validate_note(note: CanvasNoteSpec, *, issues: list[ValidationIssue]) -> None:
    if not note.text.strip():
        append_issue(
            issues,
            code="invalid-note-text",
            message=f"Note '{note.id}' must contain non-empty text.",
            path=f"notes.{note.id}.text",
        )
    if not math.isfinite(note.position.x) or not math.isfinite(note.position.y):
        append_issue(
            issues,
            code="invalid-note-position",
            message=f"Note '{note.id}' has a non-finite position.",
            path=f"notes.{note.id}.position",
        )
    validate_metadata(f"notes.{note.id}.metadata", note.metadata, issues)

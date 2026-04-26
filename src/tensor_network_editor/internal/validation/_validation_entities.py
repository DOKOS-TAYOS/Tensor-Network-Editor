"""Validation helpers for network, tensor, group, and note models."""

from __future__ import annotations

import math
from collections import Counter

from ...models import (
    CanvasNoteSpec,
    GroupSpec,
    IndexSpec,
    NetworkSpec,
    TensorDataMode,
    TensorSpec,
    ValidationIssue,
)
from ._validation_common import (
    append_duplicate_id_issues,
    append_issue,
    is_valid_name,
    validate_metadata,
)


def validate_network(spec: NetworkSpec, issues: list[ValidationIssue]) -> None:
    """Validate network-wide names, metadata, and duplicate identifiers."""
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
        (hyperedge.id for hyperedge in spec.hyperedges),
        code="duplicate-hyperedge-id",
        path="hyperedges",
        message_prefix="Hyperedge id",
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
    """Validate one tensor and all of its indices."""
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
    validate_tensor_data(tensor=tensor, issues=issues)


def validate_index(
    *,
    tensor: TensorSpec,
    index: IndexSpec,
    issues: list[ValidationIssue],
) -> None:
    """Validate one tensor index in the context of its owning tensor."""
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


def validate_tensor_data(
    *,
    tensor: TensorSpec,
    issues: list[ValidationIssue],
) -> None:
    """Validate optional tensor initialization data against the tensor shape."""
    tensor_data = tensor.tensor_data
    if tensor_data is None:
        return
    path = f"tensors.{tensor.id}.tensor_data"
    if not isinstance(tensor_data.mode, TensorDataMode):
        append_issue(
            issues,
            code="invalid-tensor-data",
            message=f"Tensor '{tensor.id}' uses an unsupported tensor_data mode.",
            path=path,
        )
        return
    if tensor_data.mode in {
        TensorDataMode.ZEROS,
        TensorDataMode.ONES,
        TensorDataMode.IDENTITY,
        TensorDataMode.COPY,
    }:
        if (
            tensor_data.fill_value is not None
            or tensor_data.values is not None
            or tensor_data.seed is not None
            or tensor_data.distribution is not None
        ):
            append_issue(
                issues,
                code="invalid-tensor-data",
                message=(
                    f"Tensor '{tensor.id}' uses {tensor_data.mode!r} with unexpected "
                    "extra fields."
                ),
                path=path,
            )
            return
        if tensor_data.mode is TensorDataMode.IDENTITY:
            _validate_identity_tensor_data_shape(
                tensor=tensor, path=path, issues=issues
            )
            return
        if tensor_data.mode is TensorDataMode.COPY:
            _validate_copy_tensor_data_shape(tensor=tensor, path=path, issues=issues)
        return
    if tensor_data.mode is TensorDataMode.FILL:
        if (
            tensor_data.fill_value is None
            or not _is_valid_tensor_scalar_literal(tensor_data.fill_value)
            or tensor_data.values is not None
            or tensor_data.seed is not None
            or tensor_data.distribution is not None
        ):
            append_issue(
                issues,
                code="invalid-tensor-data",
                message=(
                    f"Tensor '{tensor.id}' uses TensorDataMode.FILL without one "
                    "finite fill_value."
                ),
                path=path,
            )
        return
    if tensor_data.mode is TensorDataMode.RANDOM:
        if (
            tensor_data.fill_value is not None
            or tensor_data.values is not None
            or tensor_data.seed is None
            or isinstance(tensor_data.seed, bool)
            or not isinstance(tensor_data.seed, int)
            or tensor_data.seed < 0
            or tensor_data.distribution is None
        ):
            append_issue(
                issues,
                code="invalid-tensor-data",
                message=(
                    f"Tensor '{tensor.id}' uses TensorDataMode.RANDOM without one "
                    "non-negative integer seed and supported distribution."
                ),
                path=path,
            )
        return
    if (
        tensor_data.values is None
        or tensor_data.fill_value is not None
        or tensor_data.seed is not None
        or tensor_data.distribution is not None
    ):
        append_issue(
            issues,
            code="invalid-tensor-data",
            message=(
                f"Tensor '{tensor.id}' uses TensorDataMode.LITERAL without one "
                "literal values tree."
            ),
            path=path,
        )
        return
    literal_shape = _tensor_literal_shape(
        tensor_data.values,
        tensor_id=tensor.id,
        path=path,
        issues=issues,
    )
    if literal_shape is None:
        return
    if literal_shape != tensor.shape:
        append_issue(
            issues,
            code="tensor-data-shape-mismatch",
            message=(
                f"Tensor '{tensor.id}' literal data has shape {literal_shape!r}, "
                f"expected {tensor.shape!r}."
            ),
            path=path,
        )


def _validate_identity_tensor_data_shape(
    *,
    tensor: TensorSpec,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    """Require a matrix square shape for an identity initializer."""
    if len(tensor.shape) == 2 and tensor.shape[0] == tensor.shape[1]:
        return
    append_issue(
        issues,
        code="tensor-data-shape-mismatch",
        message=(
            f"Tensor '{tensor.id}' identity initializer requires a square matrix "
            f"shape, received {tensor.shape!r}."
        ),
        path=path,
    )


def _validate_copy_tensor_data_shape(
    *,
    tensor: TensorSpec,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    """Require equal axis dimensions for a generalized copy tensor."""
    if len(tensor.shape) >= 2 and all(
        axis_dimension == tensor.shape[0] for axis_dimension in tensor.shape[1:]
    ):
        return
    append_issue(
        issues,
        code="tensor-data-shape-mismatch",
        message=(
            f"Tensor '{tensor.id}' copy initializer requires at least two axes with "
            f"the same dimension, received {tensor.shape!r}."
        ),
        path=path,
    )


def _is_valid_tensor_scalar_literal(value: object) -> bool:
    """Return whether ``value`` is a finite real or portable complex scalar."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return set(value) == {"real", "imag"} and all(
            isinstance(value[component], (int, float))
            and not isinstance(value[component], bool)
            and math.isfinite(float(value[component]))
            for component in ("real", "imag")
        )
    return False


def _tensor_literal_shape(
    value: object,
    *,
    tensor_id: str,
    path: str,
    issues: list[ValidationIssue],
) -> tuple[int, ...] | None:
    """Return the nested literal shape or append one validation issue."""
    if isinstance(value, dict):
        if not _is_valid_tensor_scalar_literal(value):
            append_issue(
                issues,
                code="invalid-tensor-data",
                message=(
                    f"Tensor '{tensor_id}' literal data must contain only finite "
                    "numeric or portable complex scalars and lists."
                ),
                path=path,
            )
            return None
        return ()
    if isinstance(value, bool) or not isinstance(value, (int, float, list)):
        append_issue(
            issues,
            code="invalid-tensor-data",
            message=(
                f"Tensor '{tensor_id}' literal data must contain only finite "
                "numeric or portable complex scalars and lists."
            ),
            path=path,
        )
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            append_issue(
                issues,
                code="invalid-tensor-data",
                message=f"Tensor '{tensor_id}' literal data must be finite.",
                path=path,
            )
            return None
        return ()
    if not value:
        append_issue(
            issues,
            code="invalid-tensor-data",
            message=f"Tensor '{tensor_id}' literal data cannot contain empty lists.",
            path=path,
        )
        return None
    child_shapes: list[tuple[int, ...]] = []
    for child in value:
        child_shape = _tensor_literal_shape(
            child,
            tensor_id=tensor_id,
            path=path,
            issues=issues,
        )
        if child_shape is None:
            return None
        child_shapes.append(child_shape)
    first_shape = child_shapes[0]
    if any(child_shape != first_shape for child_shape in child_shapes[1:]):
        append_issue(
            issues,
            code="invalid-tensor-data",
            message=f"Tensor '{tensor_id}' literal data must not be ragged.",
            path=path,
        )
        return None
    return (len(value), *first_shape)


def validate_group(
    group: GroupSpec,
    *,
    tensor_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    """Validate a group name, metadata, and referenced tensor ids."""
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
    """Validate a note's text, position, and metadata."""
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

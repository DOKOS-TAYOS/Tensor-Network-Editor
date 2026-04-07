"""Shared helpers for building validation issues."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable

from .models import ValidationIssue


def is_valid_name(value: str) -> bool:
    """Return ``True`` when ``value`` contains non-whitespace characters."""
    return bool(value.strip())


def validate_metadata(
    path: str,
    metadata: object,
    issues: list[ValidationIssue],
) -> None:
    """Append a validation issue when metadata is not JSON serializable."""
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


def append_issue(
    issues: list[ValidationIssue],
    *,
    code: str,
    message: str,
    path: str,
) -> None:
    """Append a single ``ValidationIssue`` to ``issues``."""
    issues.append(ValidationIssue(code=code, message=message, path=path))


def append_duplicate_id_issues(
    values: Iterable[str],
    *,
    code: str,
    path: str,
    message_prefix: str,
    issues: list[ValidationIssue],
) -> None:
    """Append issues for any duplicated identifiers in ``values``."""
    counts = Counter(values)
    for value, count in counts.items():
        if count > 1:
            append_issue(
                issues,
                code=code,
                message=f"{message_prefix} '{value}' is duplicated.",
                path=path,
            )

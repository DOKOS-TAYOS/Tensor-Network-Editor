"""Shared helpers for building validation issues."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable

from ...models import ValidationIssue


def is_valid_name(value: str) -> bool:
    """Return ``True`` when ``value`` contains non-whitespace characters."""
    return bool(value.strip())


def validate_metadata(
    path: str,
    metadata: object,
    issues: list[ValidationIssue],
) -> None:
    """Append a validation issue when metadata is not JSON serializable."""
    if _is_fast_json_metadata(metadata):
        return
    try:
        json.dumps(metadata)
    except (TypeError, ValueError) as exc:
        issues.append(
            ValidationIssue(
                code="metadata-not-serializable",
                message=f"Metadata at {path} is not JSON serializable: {exc}",
                path=path,
            )
        )


def _is_fast_json_metadata(
    value: object,
    *,
    seen_container_ids: set[int] | None = None,
) -> bool:
    """Return ``True`` when metadata is plainly JSON serializable."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, list):
        container_id = id(value)
        seen = seen_container_ids or set()
        if container_id in seen:
            return False
        seen.add(container_id)
        try:
            return all(
                _is_fast_json_metadata(item, seen_container_ids=seen) for item in value
            )
        finally:
            seen.remove(container_id)
    if isinstance(value, dict):
        container_id = id(value)
        seen = seen_container_ids or set()
        if container_id in seen:
            return False
        seen.add(container_id)
        try:
            return all(
                isinstance(item_key, str)
                and _is_fast_json_metadata(item_value, seen_container_ids=seen)
                for item_key, item_value in value.items()
            )
        finally:
            seen.remove(container_id)
    return False


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

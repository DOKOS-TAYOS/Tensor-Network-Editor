from __future__ import annotations

from collections.abc import Sequence

from .models import ValidationIssue


class TensorNetworkEditorError(Exception):
    """Base exception for the package."""


class PackageIOError(TensorNetworkEditorError):
    """Raised when a file operation fails."""


class SerializationError(TensorNetworkEditorError):
    """Raised when serialized editor data cannot be parsed or emitted safely."""


class SpecValidationError(TensorNetworkEditorError):
    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues: list[ValidationIssue] = list(issues)
        first_issue = self.issues[0] if self.issues else None
        message = "Network specification is invalid."
        if first_issue is not None:
            message = f"Network specification is invalid: {first_issue.message}"
        super().__init__(message)

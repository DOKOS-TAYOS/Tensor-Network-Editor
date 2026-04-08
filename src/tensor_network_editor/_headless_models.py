"""Public result models for headless analysis, linting, and diff helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from ._contraction_analysis_types import ContractionAnalysisResult
from .types import JSONValue


@dataclass(slots=True, frozen=True)
class LintIssue:
    """One soft warning or informational finding returned by the network linter."""

    severity: str
    code: str
    message: str
    path: str
    suggestion: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the lint issue to a JSON-compatible mapping."""
        payload: dict[str, JSONValue] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }
        if self.suggestion is not None:
            payload["suggestion"] = self.suggestion
        return payload


@dataclass(slots=True)
class LintReport:
    """Collection of lint issues produced for one specification."""

    issues: list[LintIssue] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        """Return ``True`` when the report contains at least one warning."""
        return any(issue.severity == "warning" for issue in self.issues)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the lint report to a JSON-compatible mapping."""
        return {"issues": cast(JSONValue, [issue.to_dict() for issue in self.issues])}


@dataclass(slots=True, frozen=True)
class DiffEntityChanges:
    """Identifier-level changes for one spec entity family."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the entity changes to a JSON-compatible mapping."""
        return {
            "added": cast(JSONValue, list(self.added)),
            "removed": cast(JSONValue, list(self.removed)),
            "changed": cast(JSONValue, list(self.changed)),
        }


@dataclass(slots=True, frozen=True)
class SpecDiffResult:
    """Structured diff grouped by top-level spec entity type."""

    tensor: DiffEntityChanges = field(default_factory=DiffEntityChanges)
    edge: DiffEntityChanges = field(default_factory=DiffEntityChanges)
    group: DiffEntityChanges = field(default_factory=DiffEntityChanges)
    note: DiffEntityChanges = field(default_factory=DiffEntityChanges)
    plan: DiffEntityChanges = field(default_factory=DiffEntityChanges)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the diff result to a JSON-compatible mapping."""
        return {
            "tensor": self.tensor.to_dict(),
            "edge": self.edge.to_dict(),
            "group": self.group.to_dict(),
            "note": self.note.to_dict(),
            "plan": self.plan.to_dict(),
        }


@dataclass(slots=True, frozen=True)
class NetworkSummary:
    """Basic structural counts derived from a network spec."""

    tensor_count: int
    edge_count: int
    group_count: int
    note_count: int
    open_index_count: int

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the network summary to a JSON-compatible mapping."""
        return {
            "tensor_count": self.tensor_count,
            "edge_count": self.edge_count,
            "group_count": self.group_count,
            "note_count": self.note_count,
            "open_index_count": self.open_index_count,
        }


@dataclass(slots=True, frozen=True)
class SpecAnalysisReport:
    """Top-level headless analysis payload for one specification."""

    network: NetworkSummary
    contraction: ContractionAnalysisResult | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the analysis report to a JSON-compatible mapping."""
        payload: dict[str, JSONValue] = {"network": self.network.to_dict()}
        if self.contraction is not None:
            payload["contraction"] = self.contraction.to_dict()
        else:
            payload["contraction"] = None
        return payload

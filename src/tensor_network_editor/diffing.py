"""Public helpers for comparing two tensor-network specifications."""

from __future__ import annotations

from .internal.diffing._diffing import (
    DiffEntityChanges,
    SemanticDiffEntry,
    SemanticFieldChange,
    SemanticSpecDiffResult,
    SpecDiffResult,
    diff_specs,
    semantic_diff_specs,
)

__all__ = [
    "DiffEntityChanges",
    "SemanticDiffEntry",
    "SemanticFieldChange",
    "SemanticSpecDiffResult",
    "SpecDiffResult",
    "diff_specs",
    "semantic_diff_specs",
]

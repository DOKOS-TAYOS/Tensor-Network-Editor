"""Result and enum models returned by validation, codegen, and sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from .types import MetadataDict

if TYPE_CHECKING:
    from ._model_graph import NetworkSpec


@dataclass(slots=True)
class ValidationIssue:
    """A single validation problem found in a network specification."""

    code: str
    message: str
    path: str


class EngineName(StrEnum):
    """Supported code-generation backends."""

    TENSORNETWORK = "tensornetwork"
    QUIMB = "quimb"
    TENSORKROWCH = "tensorkrowch"
    EINSUM_NUMPY = "einsum_numpy"
    EINSUM_TORCH = "einsum_torch"


class TensorCollectionFormat(StrEnum):
    """Supported container layouts for generated tensor collections."""

    LIST = "list"
    MATRIX = "matrix"
    DICT = "dict"


@dataclass(slots=True)
class CodegenResult:
    """Generated Python code together with metadata about the export."""

    engine: EngineName
    code: str
    warnings: list[str] = field(default_factory=list)
    artifacts: MetadataDict = field(default_factory=dict)


@dataclass(slots=True)
class EditorResult:
    """Final result returned when an editor session finishes."""

    spec: NetworkSpec
    engine: EngineName
    codegen: CodegenResult | None = None
    confirmed: bool = False

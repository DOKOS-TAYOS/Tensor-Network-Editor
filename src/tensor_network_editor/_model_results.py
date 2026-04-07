from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from .types import MetadataDict

if TYPE_CHECKING:
    from ._model_graph import NetworkSpec


@dataclass(slots=True)
class ValidationIssue:
    code: str
    message: str
    path: str


class EngineName(StrEnum):
    TENSORNETWORK = "tensornetwork"
    QUIMB = "quimb"
    TENSORKROWCH = "tensorkrowch"
    EINSUM_NUMPY = "einsum_numpy"
    EINSUM_TORCH = "einsum_torch"


class TensorCollectionFormat(StrEnum):
    LIST = "list"
    MATRIX = "matrix"
    DICT = "dict"


@dataclass(slots=True)
class CodegenResult:
    engine: EngineName
    code: str
    warnings: list[str] = field(default_factory=list)
    artifacts: MetadataDict = field(default_factory=dict)


@dataclass(slots=True)
class EditorResult:
    spec: NetworkSpec
    engine: EngineName
    codegen: CodegenResult | None = None
    confirmed: bool = False

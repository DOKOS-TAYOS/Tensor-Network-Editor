"""Base abstractions shared by all code generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat


class CodeGenerator(ABC):
    """Abstract interface implemented by backend-specific code generators."""

    engine: EngineName

    @abstractmethod
    def generate(
        self,
        spec: NetworkSpec,
        collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    ) -> CodegenResult:
        """Generate Python code for ``spec`` using the given collection layout."""
        raise NotImplementedError

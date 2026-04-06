from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat


class CodeGenerator(ABC):
    engine: EngineName

    @abstractmethod
    def generate(
        self,
        spec: NetworkSpec,
        collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    ) -> CodegenResult:
        raise NotImplementedError

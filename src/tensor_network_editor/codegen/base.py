from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import CodegenResult, EngineName, NetworkSpec


class CodeGenerator(ABC):
    engine: EngineName

    @abstractmethod
    def generate(self, spec: NetworkSpec) -> CodegenResult:
        raise NotImplementedError

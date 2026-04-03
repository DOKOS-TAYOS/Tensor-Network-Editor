from __future__ import annotations

from ..models import CodegenResult, EngineName, NetworkSpec
from .base import CodeGenerator
from .einsum import EinsumCodeGenerator
from .quimb import QuimbCodeGenerator
from .tensorkrowch import TensorKrowchCodeGenerator
from .tensornetwork import TensorNetworkCodeGenerator

_GENERATORS: dict[EngineName, CodeGenerator] = {
    EngineName.TENSORNETWORK: TensorNetworkCodeGenerator(),
    EngineName.QUIMB: QuimbCodeGenerator(),
    EngineName.TENSORKROWCH: TensorKrowchCodeGenerator(),
    EngineName.EINSUM: EinsumCodeGenerator(),
}


def get_generator(engine: EngineName) -> CodeGenerator:
    return _GENERATORS[engine]


def generate_code(spec: NetworkSpec, engine: EngineName) -> CodegenResult:
    return get_generator(engine).generate(spec)

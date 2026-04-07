"""Registry of backend code generators."""

from __future__ import annotations

from ..models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat
from .base import CodeGenerator
from .einsum_numpy import EinsumNumpyCodeGenerator
from .einsum_torch import EinsumTorchCodeGenerator
from .quimb import QuimbCodeGenerator
from .tensorkrowch import TensorKrowchCodeGenerator
from .tensornetwork import TensorNetworkCodeGenerator

_GENERATORS: dict[EngineName, CodeGenerator] = {
    EngineName.TENSORNETWORK: TensorNetworkCodeGenerator(),
    EngineName.QUIMB: QuimbCodeGenerator(),
    EngineName.TENSORKROWCH: TensorKrowchCodeGenerator(),
    EngineName.EINSUM_NUMPY: EinsumNumpyCodeGenerator(),
    EngineName.EINSUM_TORCH: EinsumTorchCodeGenerator(),
}


def get_generator(engine: EngineName) -> CodeGenerator:
    """Return the generator instance registered for ``engine``."""
    return _GENERATORS[engine]


def generate_code(
    spec: NetworkSpec,
    engine: EngineName,
    *,
    collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
) -> CodegenResult:
    """Generate Python code through the registered backend generator."""
    return get_generator(engine).generate(spec, collection_format=collection_format)

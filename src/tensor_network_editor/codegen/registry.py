"""Registry of backend code generators."""

from __future__ import annotations

import inspect
import re

from ..models import (
    CodegenResult,
    EngineIdentifier,
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
)
from .backends.einsum_numpy import EinsumNumpyCodeGenerator
from .backends.einsum_torch import EinsumTorchCodeGenerator
from .backends.quimb import QuimbCodeGenerator
from .backends.tensorkrowch import TensorKrowchCodeGenerator
from .backends.tensornetwork import TensorNetworkCodeGenerator
from .modes.grid_periodic import generate_grid_periodic_code
from .modes.linear_periodic import generate_linear_periodic_code
from .modes.tree_periodic import generate_tree_periodic_code
from .shared.base import CodeGenerator

_GENERATOR_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_GENERATORS: dict[str, CodeGenerator] = {}


def engine_name_to_text(engine: EngineIdentifier) -> str:
    """Return the serialized registry key for ``engine``."""
    if isinstance(engine, EngineName):
        return engine.value
    return str(engine).strip()


def normalize_engine_name(engine: EngineIdentifier) -> EngineIdentifier:
    """Return ``engine`` as a built-in enum when possible."""
    engine_name = engine_name_to_text(engine)
    try:
        return EngineName(engine_name)
    except ValueError:
        return engine_name


def _validate_generator_name(name: EngineIdentifier) -> str:
    """Validate and normalize one generator registration name."""
    engine_name = engine_name_to_text(name)
    if not _GENERATOR_NAME_PATTERN.fullmatch(engine_name):
        raise ValueError(
            "Generator names must start with a lowercase letter and contain only lowercase letters, digits, and underscores."
        )
    return engine_name


def register_generator(
    name: EngineIdentifier,
    generator: CodeGenerator,
    *,
    overwrite: bool = False,
) -> None:
    """Register one backend code generator under ``name``."""
    engine_name = _validate_generator_name(name)
    declared_name = _validate_generator_name(generator.engine)
    if declared_name != engine_name:
        raise ValueError(
            f"Generator declared engine '{declared_name}' does not match registration name '{engine_name}'."
        )
    if engine_name in _GENERATORS and not overwrite:
        raise ValueError(f"Generator '{engine_name}' is already registered.")
    _GENERATORS[engine_name] = generator


def list_generator_names() -> list[str]:
    """Return registered generator names in display order."""
    return list(_GENERATORS)


def resolve_registered_engine(engine: EngineIdentifier) -> EngineIdentifier:
    """Normalize and validate one engine identifier against the registry."""
    normalized_engine = normalize_engine_name(engine)
    engine_name = engine_name_to_text(normalized_engine)
    if engine_name not in _GENERATORS:
        raise ValueError(f"Unsupported engine '{engine_name}'.")
    return normalized_engine


def get_generator(engine: EngineIdentifier) -> CodeGenerator:
    """Return the generator instance registered for ``engine``."""
    engine_name = engine_name_to_text(resolve_registered_engine(engine))
    return _GENERATORS[engine_name]


def _generator_supports_validate(generator: CodeGenerator) -> bool:
    """Return whether ``generator.generate`` accepts the ``validate`` keyword."""
    try:
        signature = inspect.signature(generator.generate)
    except (TypeError, ValueError):
        return True
    return "validate" in signature.parameters


def generate_code(
    spec: NetworkSpec,
    engine: EngineIdentifier,
    *,
    collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    include_roundtrip_metadata: bool = True,
    validate: bool = True,
) -> CodegenResult:
    """Generate Python code through the registered backend generator."""
    normalized_engine = resolve_registered_engine(engine)
    if spec.grid_periodic_grid is not None and isinstance(
        normalized_engine, EngineName
    ):
        return generate_grid_periodic_code(
            spec,
            normalized_engine,
            collection_format=collection_format,
            include_roundtrip_metadata=include_roundtrip_metadata,
            validate=validate,
        )
    if spec.tree_periodic_tree is not None and isinstance(
        normalized_engine, EngineName
    ):
        return generate_tree_periodic_code(
            spec,
            normalized_engine,
            collection_format=collection_format,
            include_roundtrip_metadata=include_roundtrip_metadata,
            validate=validate,
        )
    if spec.linear_periodic_chain is not None and isinstance(
        normalized_engine, EngineName
    ):
        return generate_linear_periodic_code(
            spec,
            normalized_engine,
            collection_format=collection_format,
            include_roundtrip_metadata=include_roundtrip_metadata,
            validate=validate,
        )
    generator = get_generator(normalized_engine)
    if _generator_supports_validate(generator):
        return generator.generate(
            spec,
            collection_format=collection_format,
            validate=validate,
        )
    return generator.generate(spec, collection_format=collection_format)


def _seed_builtin_generators() -> None:
    """Register the built-in generators in their stable order."""
    register_generator(
        EngineName.TENSORNETWORK,
        TensorNetworkCodeGenerator(),
        overwrite=True,
    )
    register_generator(EngineName.QUIMB, QuimbCodeGenerator(), overwrite=True)
    register_generator(
        EngineName.TENSORKROWCH,
        TensorKrowchCodeGenerator(),
        overwrite=True,
    )
    register_generator(
        EngineName.EINSUM_NUMPY,
        EinsumNumpyCodeGenerator(),
        overwrite=True,
    )
    register_generator(
        EngineName.EINSUM_TORCH,
        EinsumTorchCodeGenerator(),
        overwrite=True,
    )


def _reset_generator_registry_for_tests() -> None:
    """Reset the generator registry to its built-in state."""
    _GENERATORS.clear()
    _seed_builtin_generators()


_seed_builtin_generators()

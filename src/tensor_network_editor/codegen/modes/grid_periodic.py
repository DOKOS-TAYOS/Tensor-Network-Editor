"""Code generation helpers for typed bidimensional periodic-grid specifications."""

from __future__ import annotations

from ...errors import CodeGenerationError
from ...models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat
from ..shared.roundtrip import with_roundtrip_spec_marker
from ._grid_periodic_array_renderers import generate_array_grid_periodic_code
from ._grid_periodic_graph_renderers import generate_graph_grid_periodic_code


def generate_grid_periodic_code(
    spec: NetworkSpec,
    engine: EngineName,
    *,
    collection_format: TensorCollectionFormat,
    validate: bool = True,
) -> CodegenResult:
    """Generate helper-based Python code for the bidimensional periodic mode."""
    del validate
    if spec.grid_periodic_grid is None:
        raise CodeGenerationError(
            "Grid periodic code generation requires a grid payload."
        )

    grid = spec.grid_periodic_grid
    if engine in {
        EngineName.QUIMB,
        EngineName.EINSUM_NUMPY,
        EngineName.EINSUM_TORCH,
    }:
        result = generate_array_grid_periodic_code(
            grid=grid,
            engine=engine,
            collection_format=collection_format,
        )
        return with_roundtrip_spec_marker(result, spec=spec)
    if engine not in {EngineName.TENSORNETWORK, EngineName.TENSORKROWCH}:
        raise CodeGenerationError(
            f"The {engine.value} backend does not support grid periodic code generation."
        )
    result = generate_graph_grid_periodic_code(
        grid=grid,
        engine=engine,
        collection_format=collection_format,
    )
    return with_roundtrip_spec_marker(result, spec=spec)

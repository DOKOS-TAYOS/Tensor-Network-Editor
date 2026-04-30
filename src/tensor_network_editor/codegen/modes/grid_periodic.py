"""Code generation helpers for typed bidimensional periodic-grid specifications."""

from __future__ import annotations

from ...models import (
    CodegenResult,
    EngineName,
    GridPeriodicGridSpec,
    NetworkSpec,
    TensorCollectionFormat,
)
from ._grid_periodic_array_renderers import generate_array_grid_periodic_code
from ._grid_periodic_graph_renderers import generate_graph_grid_periodic_code
from ._periodic_codegen import dispatch_periodic_codegen


def generate_grid_periodic_code(
    spec: NetworkSpec,
    engine: EngineName,
    *,
    collection_format: TensorCollectionFormat,
    include_roundtrip_metadata: bool = True,
    validate: bool = True,
) -> CodegenResult:
    """Generate helper-based Python code for the bidimensional periodic mode."""
    del validate
    grid = spec.grid_periodic_grid

    def render_array(resolved_grid: GridPeriodicGridSpec) -> CodegenResult:
        return generate_array_grid_periodic_code(
            grid=resolved_grid,
            engine=engine,
            collection_format=collection_format,
        )

    def render_graph(resolved_grid: GridPeriodicGridSpec) -> CodegenResult:
        return generate_graph_grid_periodic_code(
            grid=resolved_grid,
            engine=engine,
            collection_format=collection_format,
        )

    return dispatch_periodic_codegen(
        spec=spec,
        payload=grid,
        missing_payload_message="Grid periodic code generation requires a grid payload.",
        unsupported_backend_label="grid periodic",
        engine=engine,
        include_roundtrip_metadata=include_roundtrip_metadata,
        array_renderer=render_array,
        graph_renderer=render_graph,
    )

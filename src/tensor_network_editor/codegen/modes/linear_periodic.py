"""Code generation helpers for typed linear periodic-chain specifications."""

from __future__ import annotations

from ...internal.modes._linear_periodic import linear_periodic_chain_uses_carry_mode
from ...models import (
    CodegenResult,
    EngineName,
    LinearPeriodicChainSpec,
    NetworkSpec,
    TensorCollectionFormat,
)
from ._linear_periodic_array_renderers import generate_array_linear_periodic_code
from ._linear_periodic_graph_renderers import generate_graph_linear_periodic_code
from ._periodic_codegen import dispatch_periodic_codegen


def generate_linear_periodic_code(
    spec: NetworkSpec,
    engine: EngineName,
    *,
    collection_format: TensorCollectionFormat,
    include_roundtrip_metadata: bool = True,
    validate: bool = True,
) -> CodegenResult:
    """Generate helper-based Python code for the linear periodic-chain mode."""
    del validate
    chain = spec.linear_periodic_chain

    def render_array(resolved_chain: LinearPeriodicChainSpec) -> CodegenResult:
        return generate_array_linear_periodic_code(
            chain=resolved_chain,
            engine=engine,
            collection_format=collection_format,
            uses_carry_mode=linear_periodic_chain_uses_carry_mode(resolved_chain),
        )

    def render_graph(resolved_chain: LinearPeriodicChainSpec) -> CodegenResult:
        return generate_graph_linear_periodic_code(
            chain=resolved_chain,
            engine=engine,
            collection_format=collection_format,
            uses_carry_mode=linear_periodic_chain_uses_carry_mode(resolved_chain),
        )

    return dispatch_periodic_codegen(
        spec=spec,
        payload=chain,
        missing_payload_message="Linear periodic code generation requires a chain payload.",
        unsupported_backend_label="linear periodic",
        engine=engine,
        include_roundtrip_metadata=include_roundtrip_metadata,
        array_renderer=render_array,
        graph_renderer=render_graph,
    )

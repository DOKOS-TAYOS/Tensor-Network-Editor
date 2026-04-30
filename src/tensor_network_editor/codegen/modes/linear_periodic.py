"""Code generation helpers for typed linear periodic-chain specifications."""

from __future__ import annotations

from ...errors import CodeGenerationError
from ...internal.modes._linear_periodic import linear_periodic_chain_uses_carry_mode
from ...models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat
from ..shared.roundtrip import with_roundtrip_spec_marker
from ._linear_periodic_array_renderers import generate_array_linear_periodic_code
from ._linear_periodic_graph_renderers import generate_graph_linear_periodic_code


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
    if spec.linear_periodic_chain is None:
        raise CodeGenerationError(
            "Linear periodic code generation requires a chain payload."
        )

    chain = spec.linear_periodic_chain
    uses_carry_mode = linear_periodic_chain_uses_carry_mode(chain)
    if engine in {
        EngineName.QUIMB,
        EngineName.EINSUM_NUMPY,
        EngineName.EINSUM_TORCH,
    }:
        result = generate_array_linear_periodic_code(
            chain=chain,
            engine=engine,
            collection_format=collection_format,
            uses_carry_mode=uses_carry_mode,
        )
        return (
            with_roundtrip_spec_marker(result, spec=spec)
            if include_roundtrip_metadata
            else result
        )
    if engine not in {EngineName.TENSORNETWORK, EngineName.TENSORKROWCH}:
        raise CodeGenerationError(
            f"The {engine.value} backend does not support linear periodic code generation."
        )
    result = generate_graph_linear_periodic_code(
        chain=chain,
        engine=engine,
        collection_format=collection_format,
        uses_carry_mode=uses_carry_mode,
    )
    return (
        with_roundtrip_spec_marker(result, spec=spec)
        if include_roundtrip_metadata
        else result
    )

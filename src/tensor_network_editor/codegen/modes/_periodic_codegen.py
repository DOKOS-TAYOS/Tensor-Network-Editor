"""Shared dispatch helpers for periodic code-generation entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from ...errors import CodeGenerationError
from ...models import CodegenResult, EngineName, NetworkSpec
from ..shared.roundtrip import with_roundtrip_spec_marker

_PeriodicPayloadT = TypeVar("_PeriodicPayloadT")
_ARRAY_ENGINES: frozenset[EngineName] = frozenset(
    {
        EngineName.QUIMB,
        EngineName.EINSUM_NUMPY,
        EngineName.EINSUM_TORCH,
    }
)
_GRAPH_ENGINES: frozenset[EngineName] = frozenset(
    {
        EngineName.TENSORNETWORK,
        EngineName.TENSORKROWCH,
    }
)


def dispatch_periodic_codegen(
    *,
    spec: NetworkSpec,
    payload: _PeriodicPayloadT | None,
    missing_payload_message: str,
    unsupported_backend_label: str,
    engine: EngineName,
    include_roundtrip_metadata: bool,
    array_renderer: Callable[[_PeriodicPayloadT], CodegenResult],
    graph_renderer: Callable[[_PeriodicPayloadT], CodegenResult],
) -> CodegenResult:
    """Route periodic code generation to the array or graph backend family."""
    if payload is None:
        raise CodeGenerationError(missing_payload_message)

    if engine in _ARRAY_ENGINES:
        result = array_renderer(payload)
    elif engine in _GRAPH_ENGINES:
        result = graph_renderer(payload)
    else:
        raise CodeGenerationError(
            f"The {engine.value} backend does not support "
            f"{unsupported_backend_label} code generation."
        )

    return (
        with_roundtrip_spec_marker(result, spec=spec)
        if include_roundtrip_metadata
        else result
    )

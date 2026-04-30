"""Code generation helpers for typed tree periodic specifications."""

from __future__ import annotations

from ...errors import CodeGenerationError
from ...models import CodegenResult, EngineName, NetworkSpec, TensorCollectionFormat
from ..shared.roundtrip import with_roundtrip_spec_marker
from ._tree_periodic_array_renderers import generate_array_tree_periodic_code
from ._tree_periodic_graph_renderers import generate_graph_tree_periodic_code


def generate_tree_periodic_code(
    spec: NetworkSpec,
    engine: EngineName,
    *,
    collection_format: TensorCollectionFormat,
    include_roundtrip_metadata: bool = True,
    validate: bool = True,
) -> CodegenResult:
    """Generate helper-based Python code for the tree periodic mode."""
    del validate
    if spec.tree_periodic_tree is None:
        raise CodeGenerationError(
            "Tree periodic code generation requires a tree payload."
        )

    tree = spec.tree_periodic_tree
    if engine in {
        EngineName.QUIMB,
        EngineName.EINSUM_NUMPY,
        EngineName.EINSUM_TORCH,
    }:
        result = generate_array_tree_periodic_code(
            tree=tree,
            engine=engine,
            collection_format=collection_format,
        )
        return (
            with_roundtrip_spec_marker(result, spec=spec)
            if include_roundtrip_metadata
            else result
        )
    if engine not in {EngineName.TENSORNETWORK, EngineName.TENSORKROWCH}:
        raise CodeGenerationError(
            f"The {engine.value} backend does not support tree periodic code generation."
        )
    result = generate_graph_tree_periodic_code(
        tree=tree,
        engine=engine,
        collection_format=collection_format,
    )
    return (
        with_roundtrip_spec_marker(result, spec=spec)
        if include_roundtrip_metadata
        else result
    )

"""Code generation helpers for typed tree periodic specifications."""

from __future__ import annotations

from ...models import (
    CodegenResult,
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
    TreePeriodicTreeSpec,
)
from ._periodic_codegen import dispatch_periodic_codegen
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
    tree = spec.tree_periodic_tree

    def render_array(resolved_tree: TreePeriodicTreeSpec) -> CodegenResult:
        return generate_array_tree_periodic_code(
            tree=resolved_tree,
            engine=engine,
            collection_format=collection_format,
        )

    def render_graph(resolved_tree: TreePeriodicTreeSpec) -> CodegenResult:
        return generate_graph_tree_periodic_code(
            tree=resolved_tree,
            engine=engine,
            collection_format=collection_format,
        )

    return dispatch_periodic_codegen(
        spec=spec,
        payload=tree,
        missing_payload_message="Tree periodic code generation requires a tree payload.",
        unsupported_backend_label="tree periodic",
        engine=engine,
        include_roundtrip_metadata=include_roundtrip_metadata,
        array_renderer=render_array,
        graph_renderer=render_graph,
    )

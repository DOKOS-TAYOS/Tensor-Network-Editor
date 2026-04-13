"""Parse supported generated Python exports back into ``NetworkSpec`` objects."""

from __future__ import annotations

import ast

from ._python_roundtrip_build import (
    _build_edge_specs,
    _build_empty_network_spec,
    _build_network_spec,
    _ParsedTensor,
    _PendingEdge,
)
from ._python_roundtrip_collect import (
    _collect_data_shape,
    _collect_einsum_labels,
    _collect_pending_edge,
    _collect_remaining_einsum_labels,
    _collect_tensor,
)
from .errors import SerializationError
from .models import NetworkSpec


def parse_generated_python_network(code: str) -> NetworkSpec:
    """Reconstruct a ``NetworkSpec`` from supported generated Python source."""
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SerializationError("Could not parse generated Python code.") from exc

    data_shapes: dict[str, tuple[int, ...]] = {}
    tensors_by_reference: dict[str, _ParsedTensor] = {}
    tensor_rows: list[list[str]] = []
    tensor_order: list[str] = []
    pending_edges: list[_PendingEdge] = []
    einsum_labels_by_reference: dict[str, list[str]] = {}
    remaining_einsum_labels_by_reference: dict[str, list[str]] = {}
    saw_supported_tensor_collection = False

    for statement in module.body:
        _collect_data_shape(statement, data_shapes)
        saw_supported_tensor_collection = _collect_tensor(
            statement=statement,
            data_shapes=data_shapes,
            tensors_by_reference=tensors_by_reference,
            tensor_rows=tensor_rows,
            tensor_order=tensor_order,
            saw_supported_tensor_collection=saw_supported_tensor_collection,
        )
        _collect_pending_edge(statement, pending_edges)
        _collect_einsum_labels(statement, einsum_labels_by_reference)
        _collect_remaining_einsum_labels(
            statement, remaining_einsum_labels_by_reference
        )

    if not tensor_order:
        if saw_supported_tensor_collection:
            return _build_empty_network_spec()
        raise SerializationError(
            "Could not reconstruct a tensor network from the generated Python code."
        )

    for reference in tensor_order:
        parsed_tensor = tensors_by_reference[reference]
        if parsed_tensor.index_labels is None:
            labels = einsum_labels_by_reference.get(reference) or (
                remaining_einsum_labels_by_reference.get(reference)
            )
            if labels is None:
                raise SerializationError(
                    "Generated Python code does not follow a supported Tensor Network Editor format."
                )
            parsed_tensor.index_labels = labels

    edge_specs = _build_edge_specs(
        tensors_by_reference=tensors_by_reference,
        tensor_order=tensor_order,
        pending_edges=pending_edges,
    )
    return _build_network_spec(
        tensors_by_reference=tensors_by_reference,
        tensor_rows=tensor_rows or [tensor_order],
        edge_specs=edge_specs,
    )

"""Parse supported generated Python exports back into ``NetworkSpec`` objects."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ...errors import SerializationError
from ...models import NetworkSpec, TensorDataSpec
from ._python_roundtrip_build import (
    _build_edge_specs,
    _build_empty_network_spec,
    _build_network_spec,
    _ManualStepComment,
    _ParsedTensor,
    _PendingEdge,
    _PendingManualStep,
)
from ._python_roundtrip_collect import (
    _collect_copy_tensor_data_update,
    _collect_data_shape,
    _collect_einsum_labels,
    _collect_manual_step,
    _collect_manual_step_comments,
    _collect_pending_edge,
    _collect_remaining_einsum_labels,
    _collect_tensor,
)


@dataclass(slots=True)
class _RoundtripParseState:
    """Mutable parser state while reconstructing one generated Python module."""

    module: ast.Module
    manual_step_comments_by_statement_line: dict[int, _ManualStepComment]
    data_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    tensor_data_by_name: dict[str, TensorDataSpec | None] = field(default_factory=dict)
    tensors_by_reference: dict[str, _ParsedTensor] = field(default_factory=dict)
    tensor_rows: list[list[str]] = field(default_factory=list)
    tensor_order: list[str] = field(default_factory=list)
    pending_edges: list[_PendingEdge] = field(default_factory=list)
    pending_manual_steps: list[_PendingManualStep] = field(default_factory=list)
    einsum_labels_by_reference: dict[str, list[str]] = field(default_factory=dict)
    remaining_einsum_labels_by_reference: dict[str, list[str]] = field(
        default_factory=dict
    )
    preferred_tensor_ids_by_reference: dict[str, str] = field(default_factory=dict)
    step_ids_by_results_list_index: list[str] = field(default_factory=list)
    saw_supported_tensor_collection: bool = False


def _build_roundtrip_parse_state(code: str) -> _RoundtripParseState:
    """Parse source and build the initial roundtrip state container."""
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SerializationError("Could not parse generated Python code.") from exc
    return _RoundtripParseState(
        module=module,
        manual_step_comments_by_statement_line=_collect_manual_step_comments(code),
    )


def _collect_roundtrip_statement(
    statement: ast.stmt,
    state: _RoundtripParseState,
) -> None:
    """Collect all supported roundtrip signals from one module statement."""
    _collect_data_shape(statement, state)
    _collect_copy_tensor_data_update(statement, state)
    _collect_tensor(statement=statement, state=state)
    _collect_pending_edge(statement, state)
    _collect_einsum_labels(statement, state)
    _collect_remaining_einsum_labels(statement, state)
    _collect_manual_step(statement=statement, state=state)


def _resolve_roundtrip_tensor_labels(state: _RoundtripParseState) -> None:
    """Fill missing tensor labels from einsum or remaining-operand metadata."""
    for reference in state.tensor_order:
        parsed_tensor = state.tensors_by_reference[reference]
        if parsed_tensor.index_labels is not None:
            continue
        labels = state.einsum_labels_by_reference.get(reference) or (
            state.remaining_einsum_labels_by_reference.get(reference)
        )
        if labels is None:
            raise SerializationError(
                "Generated Python code does not follow a supported Tensor Network Editor format."
            )
        parsed_tensor.index_labels = labels


def parse_generated_python_network(
    code: str, *, include_manual_plan: bool = True
) -> NetworkSpec:
    """Reconstruct a ``NetworkSpec`` from supported generated Python source."""
    state = _build_roundtrip_parse_state(code)
    for statement in state.module.body:
        _collect_roundtrip_statement(statement, state)

    if not state.tensor_order:
        if state.saw_supported_tensor_collection:
            return _build_empty_network_spec()
        raise SerializationError(
            "Could not reconstruct a tensor network from the generated Python code."
        )

    _resolve_roundtrip_tensor_labels(state)
    edge_specs = _build_edge_specs(
        tensors_by_reference=state.tensors_by_reference,
        tensor_order=state.tensor_order,
        pending_edges=state.pending_edges,
    )
    return _build_network_spec(
        tensors_by_reference=state.tensors_by_reference,
        tensor_rows=state.tensor_rows or [state.tensor_order],
        edge_specs=edge_specs,
        pending_manual_steps=(
            state.pending_manual_steps if include_manual_plan else None
        ),
        preferred_tensor_ids_by_reference=state.preferred_tensor_ids_by_reference,
    )

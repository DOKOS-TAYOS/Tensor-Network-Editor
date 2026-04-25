"""Prepared network helpers shared by analysis and code generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...models import EdgeSpec, IndexSpec, NetworkSpec, TensorSpec
from ._analysis import NetworkAnalysis, analyze_network
from ._hyperedge_lowering import lower_hyperedges_to_pairwise_spec

_NON_IDENTIFIER_PATTERN = re.compile(r"[^0-9a-zA-Z_]+")


def sanitize_identifier(value: str, prefix: str) -> str:
    """Normalize free-form text into a safe lowercase Python identifier."""
    collapsed = _NON_IDENTIFIER_PATTERN.sub("_", value.strip()).strip("_").lower()
    if not collapsed:
        collapsed = prefix
    if collapsed[0].isdigit():
        collapsed = f"{prefix}_{collapsed}"
    return collapsed


def make_unique_identifiers(values: list[str], prefix: str) -> list[str]:
    """Return unique normalized identifiers while preserving the input order."""
    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for value in values:
        candidate = sanitize_identifier(value, prefix)
        count = seen.get(candidate, 0)
        seen[candidate] = count + 1
        unique_names.append(candidate if count == 0 else f"{candidate}_{count + 1}")
    return unique_names


@dataclass(slots=True)
class PreparedIndex:
    """A prepared index annotated with generator-friendly metadata."""

    tensor: TensorSpec
    spec: IndexSpec
    label: str
    is_open: bool


@dataclass(slots=True)
class PreparedTensor:
    """A prepared tensor with generated variable names and layout metadata."""

    spec: TensorSpec
    variable_name: str
    data_variable_name: str
    indices: list[PreparedIndex]
    row_index: int
    column_index: int
    flat_index: int


@dataclass(slots=True)
class PreparedEdge:
    """A prepared edge with generated names and resolved endpoints."""

    spec: EdgeSpec
    variable_name: str
    label: str
    left: PreparedIndex
    right: PreparedIndex


@dataclass(slots=True)
class PreparedNetwork:
    """The normalized network representation consumed by code generators."""

    spec: NetworkSpec
    tensors: list[PreparedTensor]
    tensor_by_id: dict[str, PreparedTensor]
    tensor_rows: list[list[PreparedTensor]]
    edges: list[PreparedEdge]
    open_indices: list[PreparedIndex]


def prepare_network(spec: NetworkSpec, *, validate: bool = True) -> PreparedNetwork:
    """Validate and normalize ``spec`` for shared analysis/codegen work."""
    resolved_spec = spec
    if validate and spec.hyperedges:
        from ...validation import ensure_valid_spec

        resolved_spec = ensure_valid_spec(spec)
    lowered_spec = lower_hyperedges_to_pairwise_spec(resolved_spec)
    analysis = analyze_network(lowered_spec, validate=validate and not spec.hyperedges)
    return prepare_analyzed_network(analysis)


def prepare_analyzed_network(analysis: NetworkAnalysis) -> PreparedNetwork:
    """Normalize an existing analyzed network for analysis and code generation."""
    tensor_rows = group_tensors_by_visual_rows(analysis.spec.tensors)
    ordered_tensors = [tensor for tensor_row in tensor_rows for tensor in tensor_row]
    tensor_names = make_unique_identifiers(
        [tensor.name or tensor.id for tensor in ordered_tensors],
        "tensor",
    )
    tensor_name_by_id = {
        tensor.id: tensor_name
        for tensor, tensor_name in zip(ordered_tensors, tensor_names, strict=True)
    }

    edge_labels = make_unique_identifiers(
        [edge.name or edge.id for edge in analysis.spec.edges],
        "edge",
    )
    edge_variable_names = [f"{label}_edge" for label in edge_labels]
    edge_label_by_id = {
        edge.id: label
        for edge, label in zip(analysis.spec.edges, edge_labels, strict=True)
    }

    connected_index_labels: dict[str, str] = {}
    for edge in analysis.spec.edges:
        connected_index_labels[edge.left.index_id] = edge_label_by_id[edge.id]
        connected_index_labels[edge.right.index_id] = edge_label_by_id[edge.id]

    prepared_tensors: list[PreparedTensor] = []
    prepared_tensor_rows: list[list[PreparedTensor]] = []
    prepared_index_lookup: dict[str, PreparedIndex] = {}
    flat_index = 0
    for row_index, tensor_row in enumerate(tensor_rows):
        prepared_row: list[PreparedTensor] = []
        for column_index, tensor in enumerate(tensor_row):
            variable_name = tensor_name_by_id[tensor.id]
            prepared_indices: list[PreparedIndex] = []
            for index in tensor.indices:
                label = connected_index_labels.get(index.id)
                if label is None:
                    label = sanitize_identifier(
                        f"{variable_name}_{index.name}", "index"
                    )
                prepared_index = PreparedIndex(
                    tensor=tensor,
                    spec=index,
                    label=label,
                    is_open=index.id not in connected_index_labels,
                )
                prepared_indices.append(prepared_index)
                prepared_index_lookup[index.id] = prepared_index

            prepared_tensor = PreparedTensor(
                spec=tensor,
                variable_name=variable_name,
                data_variable_name=f"{variable_name}_data",
                indices=prepared_indices,
                row_index=row_index,
                column_index=column_index,
                flat_index=flat_index,
            )
            flat_index += 1
            prepared_row.append(prepared_tensor)
            prepared_tensors.append(prepared_tensor)
        prepared_tensor_rows.append(prepared_row)

    prepared_edges: list[PreparedEdge] = []
    for edge, variable_name, label in zip(
        analysis.spec.edges, edge_variable_names, edge_labels, strict=True
    ):
        prepared_edges.append(
            PreparedEdge(
                spec=edge,
                variable_name=variable_name,
                label=label,
                left=prepared_index_lookup[edge.left.index_id],
                right=prepared_index_lookup[edge.right.index_id],
            )
        )

    open_indices = [
        prepared_index
        for tensor in prepared_tensors
        for prepared_index in tensor.indices
        if prepared_index.is_open
    ]

    return PreparedNetwork(
        spec=analysis.spec,
        tensors=prepared_tensors,
        tensor_by_id={
            prepared_tensor.spec.id: prepared_tensor
            for prepared_tensor in prepared_tensors
        },
        tensor_rows=prepared_tensor_rows,
        edges=prepared_edges,
        open_indices=open_indices,
    )


def group_tensors_by_visual_rows(tensors: list[TensorSpec]) -> list[list[TensorSpec]]:
    """Group tensors into visual rows based on their canvas positions."""
    if not tensors:
        return []

    row_tolerance = max(
        24.0,
        sum(tensor.size.height for tensor in tensors) / len(tensors) * 0.6,
    )
    ordered_tensors = sorted(
        tensors,
        key=lambda tensor: (tensor.position.y, tensor.position.x, tensor.id),
    )

    rows: list[list[TensorSpec]] = []
    row_y_totals: list[float] = []
    for tensor in ordered_tensors:
        tensor_y = tensor.position.y
        if not rows:
            rows.append([tensor])
            row_y_totals.append(tensor_y)
            continue

        current_row = rows[-1]
        current_row_center = row_y_totals[-1] / len(current_row)
        if abs(tensor_y - current_row_center) <= row_tolerance:
            current_row.append(tensor)
            row_y_totals[-1] += tensor_y
            continue

        rows.append([tensor])
        row_y_totals.append(tensor_y)

    for row in rows:
        row.sort(key=lambda member: (member.position.x, member.position.y, member.id))

    return rows

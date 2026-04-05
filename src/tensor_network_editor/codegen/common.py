from __future__ import annotations

import re
from dataclasses import dataclass

from .._analysis import analyze_network
from ..models import EdgeSpec, IndexSpec, NetworkSpec, TensorSpec

_NON_IDENTIFIER_PATTERN = re.compile(r"[^0-9a-zA-Z_]+")


def sanitize_identifier(value: str, prefix: str) -> str:
    collapsed = _NON_IDENTIFIER_PATTERN.sub("_", value.strip()).strip("_").lower()
    if not collapsed:
        collapsed = prefix
    if collapsed[0].isdigit():
        collapsed = f"{prefix}_{collapsed}"
    return collapsed


def make_unique_identifiers(values: list[str], prefix: str) -> list[str]:
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
    tensor: TensorSpec
    spec: IndexSpec
    label: str
    is_open: bool


@dataclass(slots=True)
class PreparedTensor:
    spec: TensorSpec
    variable_name: str
    data_variable_name: str
    indices: list[PreparedIndex]


@dataclass(slots=True)
class PreparedEdge:
    spec: EdgeSpec
    variable_name: str
    label: str
    left: PreparedIndex
    right: PreparedIndex


@dataclass(slots=True)
class PreparedNetwork:
    spec: NetworkSpec
    tensors: list[PreparedTensor]
    edges: list[PreparedEdge]
    open_indices: list[PreparedIndex]


def prepare_network(spec: NetworkSpec) -> PreparedNetwork:
    analysis = analyze_network(spec, validate=True)
    tensor_names = make_unique_identifiers(
        [tensor.name or tensor.id for tensor in analysis.spec.tensors],
        "tensor",
    )

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
    prepared_index_lookup: dict[str, PreparedIndex] = {}
    for tensor, variable_name in zip(analysis.spec.tensors, tensor_names, strict=True):
        prepared_indices: list[PreparedIndex] = []
        for index in tensor.indices:
            label = connected_index_labels.get(index.id)
            if label is None:
                label = sanitize_identifier(f"{variable_name}_{index.name}", "index")
            prepared_index = PreparedIndex(
                tensor=tensor,
                spec=index,
                label=label,
                is_open=index.id not in connected_index_labels,
            )
            prepared_indices.append(prepared_index)
            prepared_index_lookup[index.id] = prepared_index

        prepared_tensors.append(
            PreparedTensor(
                spec=tensor,
                variable_name=variable_name,
                data_variable_name=f"{variable_name}_data",
                indices=prepared_indices,
            )
        )

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

    open_index_ids = {index.id for _, index in analysis.open_indices}
    open_indices = [
        prepared_index
        for tensor in prepared_tensors
        for prepared_index in tensor.indices
        if prepared_index.spec.id in open_index_ids
    ]

    return PreparedNetwork(
        spec=analysis.spec,
        tensors=prepared_tensors,
        edges=prepared_edges,
        open_indices=open_indices,
    )


def tensor_variable_name(prepared: PreparedNetwork, tensor_id: str) -> str:
    for tensor in prepared.tensors:
        if tensor.spec.id == tensor_id:
            return tensor.variable_name
    raise KeyError(tensor_id)

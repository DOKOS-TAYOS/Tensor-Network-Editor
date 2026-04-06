from __future__ import annotations

import re
from dataclasses import dataclass

from .._analysis import analyze_network
from ..models import (
    EdgeSpec,
    IndexSpec,
    NetworkSpec,
    TensorCollectionFormat,
    TensorSpec,
)

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
    row_index: int
    column_index: int
    flat_index: int


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
    tensor_rows: list[list[PreparedTensor]]
    edges: list[PreparedEdge]
    open_indices: list[PreparedIndex]


def prepare_network(spec: NetworkSpec) -> PreparedNetwork:
    analysis = analyze_network(spec, validate=True)
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
        tensor_rows=prepared_tensor_rows,
        edges=prepared_edges,
        open_indices=open_indices,
    )


def group_tensors_by_visual_rows(tensors: list[TensorSpec]) -> list[list[TensorSpec]]:
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
    for tensor in ordered_tensors:
        if not rows:
            rows.append([tensor])
            continue

        current_row = rows[-1]
        current_row_center = sum(member.position.y for member in current_row) / len(
            current_row
        )
        if abs(tensor.position.y - current_row_center) <= row_tolerance:
            current_row.append(tensor)
            current_row.sort(
                key=lambda member: (member.position.x, member.position.y, member.id)
            )
            continue

        rows.append([tensor])

    return rows


def tensor_variable_name(prepared: PreparedNetwork, tensor_id: str) -> str:
    for tensor in prepared.tensors:
        if tensor.spec.id == tensor_id:
            return tensor.variable_name
    raise KeyError(tensor_id)


def tensor_display_name_by_id(prepared: PreparedNetwork) -> dict[str, str]:
    return {
        tensor.spec.id: (tensor.spec.name or tensor.variable_name or tensor.spec.id)
        for tensor in prepared.tensors
    }


def joined_tensor_display_name(
    source_tensor_ids: tuple[str, ...],
    tensor_names_by_id: dict[str, str],
) -> str:
    return "-".join(
        tensor_names_by_id.get(tensor_id, tensor_id) for tensor_id in source_tensor_ids
    )


def render_results_list_reference(
    result_index: int,
    *,
    latest_result_index: int | None,
) -> str:
    if latest_result_index is not None and result_index == latest_result_index:
        return "results_list[-1]"
    return f"results_list[{result_index}]"


def container_name_for_format(collection_format: TensorCollectionFormat) -> str:
    if collection_format is TensorCollectionFormat.MATRIX:
        return "tensor_rows"
    if collection_format is TensorCollectionFormat.DICT:
        return "tensors_dict"
    return "tensors"


def tensor_collection_reference(
    tensor: PreparedTensor,
    collection_format: TensorCollectionFormat,
    collection_name: str | None = None,
) -> str:
    resolved_collection_name = collection_name or container_name_for_format(
        collection_format
    )
    if collection_format is TensorCollectionFormat.MATRIX:
        return f"{resolved_collection_name}[{tensor.row_index}][{tensor.column_index}]"
    if collection_format is TensorCollectionFormat.DICT:
        return f"{resolved_collection_name}[{tensor.variable_name!r}]"
    return f"{resolved_collection_name}[{tensor.flat_index}]"


def tensor_collection_reference_by_id(
    prepared: PreparedNetwork,
    tensor_id: str,
    collection_format: TensorCollectionFormat,
    collection_name: str | None = None,
) -> str:
    for tensor in prepared.tensors:
        if tensor.spec.id == tensor_id:
            return tensor_collection_reference(
                tensor, collection_format, collection_name
            )
    raise KeyError(tensor_id)


def flattened_tensor_collection_expression(
    collection_format: TensorCollectionFormat,
    collection_name: str | None = None,
) -> str:
    resolved_collection_name = collection_name or container_name_for_format(
        collection_format
    )
    if collection_format is TensorCollectionFormat.MATRIX:
        return f"[tensor for row in {resolved_collection_name} for tensor in row]"
    if collection_format is TensorCollectionFormat.DICT:
        return f"list({resolved_collection_name}.values())"
    return resolved_collection_name


def render_tensor_collection_assignment(
    collection_name: str,
    collection_format: TensorCollectionFormat,
    prepared: PreparedNetwork,
    tensor_value_by_id: dict[str, str],
) -> list[str]:
    if collection_format is TensorCollectionFormat.MATRIX:
        lines = [f"{collection_name} = []"]
        for row_index, tensor_row in enumerate(prepared.tensor_rows):
            lines.append(f"{collection_name}.append([])")
            for tensor in tensor_row:
                lines.append(f"# Tensor {tensor.spec.name}")
                lines.append(
                    f"{collection_name}[{row_index}].append("
                    f"{tensor_value_by_id[tensor.spec.id]})"
                )
        return lines

    if collection_format is TensorCollectionFormat.DICT:
        lines = [f"{collection_name} = {{}}"]
        for tensor in prepared.tensors:
            lines.append(f"# Tensor {tensor.spec.name}")
            lines.append(
                f"{collection_name}[{tensor.variable_name!r}] = "
                f"{tensor_value_by_id[tensor.spec.id]}"
            )
        return lines

    lines = [f"{collection_name} = []"]
    for tensor in prepared.tensors:
        lines.append(f"# Tensor {tensor.spec.name}")
        lines.append(f"{collection_name}.append({tensor_value_by_id[tensor.spec.id]})")
    return lines

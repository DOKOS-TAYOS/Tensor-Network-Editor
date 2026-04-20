"""Reconstruction helpers for generated-Python roundtrips."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ...errors import SerializationError
from ...models import (
    CanvasPosition,
    ContractionPlanSpec,
    ContractionStepSpec,
    EdgeEndpointRef,
    EdgeSpec,
    IndexSpec,
    NetworkSpec,
    TensorSize,
    TensorSpec,
)
from ._python_roundtrip_ast import (
    _call_name,
    _extract_name_from_expression,
    _keyword_value,
    _literal_string,
    _literal_string_sequence,
    _parse_operand_tag_string,
    _parse_zeros_shape,
)
from ._python_roundtrip_helpers import (
    default_tensor_name_from_position,
    recover_index_name,
    recover_tensor_name_from_data_variable,
    synthetic_data_variable_name,
)


@dataclass(slots=True)
class _ParsedTensor:
    """Intermediate tensor data recovered from generated Python source."""

    reference: str
    data_variable_name: str
    shape: tuple[int, ...]
    name: str
    index_labels: list[str] | None
    operand_id: str | None = None


@dataclass(slots=True)
class _PendingEdge:
    """Intermediate edge data recovered before tensor specs are finalized."""

    name: str
    left_reference: str
    left_index_name: str
    right_reference: str
    right_index_name: str


@dataclass(slots=True, frozen=True)
class _ManualStepComment:
    """Structured metadata parsed from a generated manual-step comment."""

    step_id: str
    left_operand_id: str
    right_operand_id: str


@dataclass(slots=True)
class _PendingManualStep:
    """One reconstructed manual contraction step recovered from Python."""

    step_id: str
    left_operand_id: str
    right_operand_id: str


_EdgeDescriptor = tuple[str, int, str, int, str]


def _parse_tensor_expression(
    *,
    expression: ast.expr,
    data_shapes: dict[str, tuple[int, ...]],
    reference: str,
    fallback_name: str | None,
) -> _ParsedTensor:
    """Parse a supported tensor-construction expression."""
    resolved_data = _resolve_tensor_data_expression(
        expression=expression,
        data_shapes=data_shapes,
        reference=reference,
        fallback_name=fallback_name,
    )
    if resolved_data is not None:
        data_variable_name, shape = resolved_data
        return _ParsedTensor(
            reference=reference,
            data_variable_name=data_variable_name,
            shape=shape,
            name=_recover_tensor_name_from_data_variable(
                data_variable_name, fallback_name
            ),
            index_labels=None,
        )

    if not isinstance(expression, ast.Call):
        raise SerializationError(
            "Generated Python code contains an unsupported tensor construction."
        )

    call_name = _call_name(expression.func)
    if call_name.endswith(".Node") or call_name == "Node":
        data_expression = (
            expression.args[0]
            if expression.args
            else _keyword_value(expression, "tensor")
            or _keyword_value(expression, "data")
        )
        resolved_data = _resolve_tensor_data_expression(
            expression=data_expression,
            data_shapes=data_shapes,
            reference=reference,
            fallback_name=fallback_name,
        )
        if resolved_data is None:
            raise SerializationError(
                "Generated Python node construction is missing supported tensor data."
            )
        data_variable_name, shape = resolved_data

        axis_names = _literal_string_sequence(
            _keyword_value(expression, "axis_names")
            or _keyword_value(expression, "axes_names")
        )
        if axis_names is None:
            raise SerializationError(
                "Generated Python node construction is missing supported axis names."
            )

        return _ParsedTensor(
            reference=reference,
            data_variable_name=data_variable_name,
            shape=shape,
            name=(
                _literal_string(_keyword_value(expression, "name"))
                or _recover_tensor_name_from_data_variable(
                    data_variable_name, fallback_name
                )
            ),
            index_labels=axis_names,
        )

    if call_name.endswith(".Tensor") or call_name == "Tensor":
        data_expression = _keyword_value(expression, "data") or (
            expression.args[0] if expression.args else None
        )
        resolved_data = _resolve_tensor_data_expression(
            expression=data_expression,
            data_shapes=data_shapes,
            reference=reference,
            fallback_name=fallback_name,
        )
        if resolved_data is None:
            raise SerializationError(
                "Generated Python tensor construction is missing supported tensor data."
            )
        data_variable_name, shape = resolved_data

        inds = _literal_string_sequence(_keyword_value(expression, "inds"))
        if inds is None:
            raise SerializationError(
                "Generated Python tensor construction is missing supported indices."
            )

        tags = _literal_string_sequence(_keyword_value(expression, "tags")) or []
        tensor_name = tags[0] if tags else None
        return _ParsedTensor(
            reference=reference,
            data_variable_name=data_variable_name,
            shape=shape,
            name=(
                tensor_name
                or _recover_tensor_name_from_data_variable(
                    data_variable_name, fallback_name
                )
            ),
            index_labels=inds,
            operand_id=_parse_operand_tag_string(tags[1] if len(tags) > 1 else None),
        )

    raise SerializationError(
        "Generated Python code does not follow a supported Tensor Network Editor format."
    )


def _resolve_tensor_data_expression(
    *,
    expression: ast.expr | None,
    data_shapes: dict[str, tuple[int, ...]],
    reference: str,
    fallback_name: str | None,
) -> tuple[str, tuple[int, ...]] | None:
    """Resolve a tensor data expression to its variable name and shape."""
    data_variable_name = _extract_name_from_expression(expression)
    if data_variable_name is not None:
        shape = data_shapes.get(data_variable_name)
        if shape is None:
            raise SerializationError(
                "Generated Python code references tensor data without a supported zeros initializer."
            )
        return data_variable_name, shape

    if isinstance(expression, ast.Call):
        shape = _parse_zeros_shape(expression)
        if shape is not None:
            return (
                _synthetic_data_variable_name(reference, fallback_name),
                shape,
            )
    return None


def _synthetic_data_variable_name(reference: str, fallback_name: str | None) -> str:
    """Delegate synthetic data-variable naming to the helper module."""
    return synthetic_data_variable_name(reference, fallback_name)


def _default_tensor_name_from_position(position: int) -> str:
    """Delegate positional tensor naming to the helper module."""
    return default_tensor_name_from_position(position)


def _recover_tensor_name_from_data_variable(
    data_variable_name: str, fallback_name: str | None = None
) -> str:
    """Delegate tensor-name recovery to the helper module."""
    return recover_tensor_name_from_data_variable(data_variable_name, fallback_name)


def _recover_index_name(
    *,
    label: str,
    tensor_name: str,
    data_variable_name: str,
    connected_edge_label: str | None,
) -> str:
    """Delegate index-name recovery to the helper module."""
    return recover_index_name(
        label=label,
        tensor_name=tensor_name,
        data_variable_name=data_variable_name,
        connected_edge_label=connected_edge_label,
    )


def _build_edge_specs(
    *,
    tensors_by_reference: dict[str, _ParsedTensor],
    tensor_order: list[str],
    pending_edges: list[_PendingEdge],
) -> list[_EdgeDescriptor]:
    """Build normalized edge descriptors from parsed tensor references."""
    if pending_edges:
        return _build_edge_specs_from_pending_edges(
            tensors_by_reference=tensors_by_reference,
            pending_edges=pending_edges,
        )
    return _infer_edge_specs_from_shared_labels(
        tensors_by_reference=tensors_by_reference,
        tensor_order=tensor_order,
    )


def _build_edge_specs_from_pending_edges(
    *,
    tensors_by_reference: dict[str, _ParsedTensor],
    pending_edges: list[_PendingEdge],
) -> list[_EdgeDescriptor]:
    """Recover normalized edge descriptors from explicit connect calls."""
    edge_specs: list[_EdgeDescriptor] = []
    for pending_edge in pending_edges:
        left_tensor = tensors_by_reference[pending_edge.left_reference]
        right_tensor = tensors_by_reference[pending_edge.right_reference]
        if left_tensor.index_labels is None or right_tensor.index_labels is None:
            raise SerializationError(
                "Generated Python connect calls require tensor index labels."
            )
        try:
            left_index_position = left_tensor.index_labels.index(
                pending_edge.left_index_name
            )
            right_index_position = right_tensor.index_labels.index(
                pending_edge.right_index_name
            )
        except ValueError as exc:
            raise SerializationError(
                "Generated Python connect calls reference unknown tensor indices."
            ) from exc
        edge_specs.append(
            (
                pending_edge.left_reference,
                left_index_position,
                pending_edge.right_reference,
                right_index_position,
                pending_edge.name,
            )
        )
    return edge_specs


def _infer_edge_specs_from_shared_labels(
    *,
    tensors_by_reference: dict[str, _ParsedTensor],
    tensor_order: list[str],
) -> list[_EdgeDescriptor]:
    """Infer normalized edge descriptors from labels shared by two tensors."""
    label_occurrences: dict[str, list[tuple[str, int]]] = {}
    for reference in tensor_order:
        parsed_tensor = tensors_by_reference[reference]
        if parsed_tensor.index_labels is None:
            raise SerializationError(
                "Generated Python code is missing index information for one or more tensors."
            )
        for index_position, label in enumerate(parsed_tensor.index_labels):
            label_occurrences.setdefault(label, []).append((reference, index_position))

    edge_specs: list[_EdgeDescriptor] = []
    for label, occurrences in label_occurrences.items():
        if len(occurrences) == 2:
            edge_specs.append(
                (
                    occurrences[0][0],
                    occurrences[0][1],
                    occurrences[1][0],
                    occurrences[1][1],
                    label,
                )
            )
            continue
        if len(occurrences) != 1:
            raise SerializationError(
                "Generated Python code contains an unsupported number of shared indices."
            )
    return edge_specs


def _build_network_spec(
    *,
    tensors_by_reference: dict[str, _ParsedTensor],
    tensor_rows: list[list[str]],
    edge_specs: list[_EdgeDescriptor],
    pending_manual_steps: list[_PendingManualStep] | None = None,
    preferred_tensor_ids_by_reference: dict[str, str] | None = None,
) -> NetworkSpec:
    """Convert parsed tensors and edges into a reconstructed ``NetworkSpec``."""
    edge_labels = _build_edge_label_map(edge_specs)
    (
        tensor_specs,
        tensor_id_by_reference,
        index_id_by_reference_and_position,
    ) = _build_tensor_specs(
        tensors_by_reference=tensors_by_reference,
        tensor_rows=tensor_rows,
        edge_labels=edge_labels,
        preferred_tensor_ids_by_reference=preferred_tensor_ids_by_reference,
    )
    edges = _build_imported_edges(
        edge_specs=edge_specs,
        tensor_id_by_reference=tensor_id_by_reference,
        index_id_by_reference_and_position=index_id_by_reference_and_position,
    )
    contraction_plan = _build_imported_contraction_plan(pending_manual_steps)

    return NetworkSpec(
        id="imported_python_network",
        name="Imported Python Network",
        tensors=tensor_specs,
        edges=edges,
        groups=[],
        notes=[],
        contraction_plan=contraction_plan,
    )


def _build_edge_label_map(
    edge_specs: list[_EdgeDescriptor],
) -> dict[tuple[str, int], str]:
    """Map each connected tensor slot to the recovered edge label."""
    edge_labels: dict[tuple[str, int], str] = {}
    for (
        left_reference,
        left_index_position,
        right_reference,
        right_index_position,
        edge_name,
    ) in edge_specs:
        edge_labels[(left_reference, left_index_position)] = edge_name
        edge_labels[(right_reference, right_index_position)] = edge_name
    return edge_labels


def _resolve_imported_tensor_id(
    *,
    reference: str,
    parsed_tensor: _ParsedTensor,
    used_tensor_ids: set[str],
    preferred_tensor_ids_by_reference: dict[str, str] | None,
    tensor_counter: int,
) -> tuple[str, int]:
    """Choose one stable tensor id for a recovered tensor reference."""
    preferred_tensor_id = None
    if preferred_tensor_ids_by_reference is not None:
        preferred_tensor_id = preferred_tensor_ids_by_reference.get(reference)
    if preferred_tensor_id is None:
        preferred_tensor_id = parsed_tensor.operand_id
    if preferred_tensor_id is not None:
        tensor_id = preferred_tensor_id
    else:
        tensor_id = f"tensor_{tensor_counter}"
        tensor_counter += 1
    if tensor_id in used_tensor_ids:
        raise SerializationError(
            "Generated Python code recovers duplicate tensor operand ids."
        )
    used_tensor_ids.add(tensor_id)
    return tensor_id, tensor_counter


def _build_imported_index_specs(
    *,
    reference: str,
    tensor_id: str,
    parsed_tensor: _ParsedTensor,
    edge_labels: dict[tuple[str, int], str],
    index_id_by_reference_and_position: dict[tuple[str, int], str],
) -> list[IndexSpec]:
    """Build recovered index specs for one parsed tensor."""
    if parsed_tensor.index_labels is None:
        raise SerializationError(
            "Generated Python code is missing tensor labels required to rebuild the network."
        )
    index_specs: list[IndexSpec] = []
    for index_position, label in enumerate(parsed_tensor.index_labels):
        index_id = f"{tensor_id}_index_{index_position + 1}"
        index_id_by_reference_and_position[(reference, index_position)] = index_id
        index_specs.append(
            IndexSpec(
                id=index_id,
                name=_recover_index_name(
                    label=label,
                    tensor_name=parsed_tensor.name,
                    data_variable_name=parsed_tensor.data_variable_name,
                    connected_edge_label=edge_labels.get((reference, index_position)),
                ),
                dimension=parsed_tensor.shape[index_position],
            )
        )
    return index_specs


def _build_tensor_specs(
    *,
    tensors_by_reference: dict[str, _ParsedTensor],
    tensor_rows: list[list[str]],
    edge_labels: dict[tuple[str, int], str],
    preferred_tensor_ids_by_reference: dict[str, str] | None,
) -> tuple[
    list[TensorSpec],
    dict[str, str],
    dict[tuple[str, int], str],
]:
    """Build recovered tensors plus the reference maps needed for edges."""
    tensor_specs: list[TensorSpec] = []
    tensor_id_by_reference: dict[str, str] = {}
    index_id_by_reference_and_position: dict[tuple[str, int], str] = {}
    used_tensor_ids: set[str] = set()
    tensor_counter = 1
    for row_index, row_references in enumerate(tensor_rows):
        for column_index, reference in enumerate(row_references):
            parsed_tensor = tensors_by_reference[reference]
            tensor_id, tensor_counter = _resolve_imported_tensor_id(
                reference=reference,
                parsed_tensor=parsed_tensor,
                used_tensor_ids=used_tensor_ids,
                preferred_tensor_ids_by_reference=preferred_tensor_ids_by_reference,
                tensor_counter=tensor_counter,
            )
            tensor_id_by_reference[reference] = tensor_id
            tensor_specs.append(
                TensorSpec(
                    id=tensor_id,
                    name=parsed_tensor.name,
                    position=CanvasPosition(
                        x=120.0 + column_index * 240.0,
                        y=160.0 + row_index * 180.0,
                    ),
                    size=TensorSize(),
                    indices=_build_imported_index_specs(
                        reference=reference,
                        tensor_id=tensor_id,
                        parsed_tensor=parsed_tensor,
                        edge_labels=edge_labels,
                        index_id_by_reference_and_position=(
                            index_id_by_reference_and_position
                        ),
                    ),
                )
            )
    return tensor_specs, tensor_id_by_reference, index_id_by_reference_and_position


def _build_imported_edges(
    *,
    edge_specs: list[_EdgeDescriptor],
    tensor_id_by_reference: dict[str, str],
    index_id_by_reference_and_position: dict[tuple[str, int], str],
) -> list[EdgeSpec]:
    """Build recovered edge specs from normalized edge descriptors."""
    return [
        EdgeSpec(
            id=f"edge_{edge_index + 1}",
            name=edge_name,
            left=EdgeEndpointRef(
                tensor_id=tensor_id_by_reference[left_reference],
                index_id=index_id_by_reference_and_position[
                    (left_reference, left_index_position)
                ],
            ),
            right=EdgeEndpointRef(
                tensor_id=tensor_id_by_reference[right_reference],
                index_id=index_id_by_reference_and_position[
                    (right_reference, right_index_position)
                ],
            ),
        )
        for edge_index, (
            left_reference,
            left_index_position,
            right_reference,
            right_index_position,
            edge_name,
        ) in enumerate(edge_specs)
    ]


def _build_imported_contraction_plan(
    pending_manual_steps: list[_PendingManualStep] | None,
) -> ContractionPlanSpec | None:
    """Build the imported manual contraction plan when one was recovered."""
    if not pending_manual_steps:
        return None
    return ContractionPlanSpec(
        id="imported_contraction_plan",
        name="Imported manual contraction path",
        steps=[
            ContractionStepSpec(
                id=step.step_id,
                left_operand_id=step.left_operand_id,
                right_operand_id=step.right_operand_id,
            )
            for step in pending_manual_steps
        ],
        view_snapshots=[],
        metadata={},
    )


def _build_empty_network_spec() -> NetworkSpec:
    """Return the canonical imported spec for a supported empty network."""
    return NetworkSpec(
        id="imported_python_network",
        name="Imported Python Network",
        tensors=[],
        edges=[],
        groups=[],
        notes=[],
        contraction_plan=None,
    )

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from ._python_roundtrip_helpers import (
    default_tensor_name_from_position,
    recover_index_name,
    recover_tensor_name_from_data_variable,
    sanitize_identifier,
    synthetic_data_variable_name,
)
from .errors import SerializationError
from .models import (
    CanvasPosition,
    EdgeEndpointRef,
    EdgeSpec,
    IndexSpec,
    NetworkSpec,
    TensorSize,
    TensorSpec,
)


@dataclass(slots=True)
class _ParsedTensor:
    reference: str
    data_variable_name: str
    shape: tuple[int, ...]
    name: str
    index_labels: list[str] | None


@dataclass(slots=True)
class _PendingEdge:
    name: str
    left_reference: str
    left_index_name: str
    right_reference: str
    right_index_name: str


def parse_generated_python_network(code: str) -> NetworkSpec:
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

    for statement in module.body:
        _collect_data_shape(statement, data_shapes)
        _collect_tensor(
            statement=statement,
            data_shapes=data_shapes,
            tensors_by_reference=tensors_by_reference,
            tensor_rows=tensor_rows,
            tensor_order=tensor_order,
        )
        _collect_pending_edge(statement, pending_edges)
        _collect_einsum_labels(statement, einsum_labels_by_reference)
        _collect_remaining_einsum_labels(
            statement, remaining_einsum_labels_by_reference
        )

    if not tensor_order:
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


def _collect_data_shape(
    statement: ast.stmt, data_shapes: dict[str, tuple[int, ...]]
) -> None:
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Name)
        or not isinstance(statement.value, ast.Call)
    ):
        return
    shape = _parse_zeros_shape(statement.value)
    if shape is not None:
        data_shapes[statement.targets[0].id] = shape


def _collect_tensor(
    *,
    statement: ast.stmt,
    data_shapes: dict[str, tuple[int, ...]],
    tensors_by_reference: dict[str, _ParsedTensor],
    tensor_rows: list[list[str]],
    tensor_order: list[str],
) -> None:
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        target_name = statement.targets[0].id
        if target_name == "tensor_rows" and isinstance(statement.value, ast.List):
            tensor_rows.clear()
            return
        if target_name in {"tensors", "tensors_dict"}:
            return

    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Subscript)
        ):
            dict_reference = _parse_tensor_reference(statement.targets[0])
            if dict_reference is None or not dict_reference.startswith("dict:"):
                return
            parsed_tensor = _parse_tensor_expression(
                expression=statement.value,
                data_shapes=data_shapes,
                reference=dict_reference,
                fallback_name=dict_reference.removeprefix("dict:"),
            )
            tensors_by_reference[dict_reference] = parsed_tensor
            tensor_order.append(dict_reference)
        return

    call = statement.value
    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "append"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "tensors"
        and len(call.args) == 1
    ):
        reference = f"list:{len(tensor_order)}"
        tensors_by_reference[reference] = _parse_tensor_expression(
            expression=call.args[0],
            data_shapes=data_shapes,
            reference=reference,
            fallback_name=_default_tensor_name_from_position(len(tensor_order)),
        )
        tensor_order.append(reference)
        return

    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "append"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "tensor_rows"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.List)
        and not call.args[0].elts
    ):
        tensor_rows.append([])
        return

    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "append"
        and isinstance(call.func.value, ast.Subscript)
        and len(call.args) == 1
    ):
        row_index = _parse_matrix_row_index(call.func.value)
        if row_index is None:
            return
        while len(tensor_rows) <= row_index:
            tensor_rows.append([])
        reference = f"matrix:{row_index}:{len(tensor_rows[row_index])}"
        tensors_by_reference[reference] = _parse_tensor_expression(
            expression=call.args[0],
            data_shapes=data_shapes,
            reference=reference,
            fallback_name=_default_tensor_name_from_position(len(tensor_order)),
        )
        tensor_rows[row_index].append(reference)
        tensor_order.append(reference)


def _collect_pending_edge(
    statement: ast.stmt, pending_edges: list[_PendingEdge]
) -> None:
    edge_name: str | None = None
    connect_call: ast.Call | None = None

    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Call)
    ):
        connect_call = statement.value
        edge_name = statement.targets[0].id.removesuffix("_edge")
    else:
        appended_value = _parse_list_append_value(statement, "edges_list")
        if appended_value is None:
            return
        if isinstance(appended_value, ast.Call):
            connect_call = appended_value
        elif (
            isinstance(appended_value, (ast.Tuple, ast.List))
            and len(appended_value.elts) == 2
            and isinstance(appended_value.elts[1], ast.Call)
        ):
            edge_name = _literal_string(appended_value.elts[0])
            connect_call = appended_value.elts[1]
        else:
            return

    if connect_call is None:
        return
    call_name = _call_name(connect_call.func)
    if not call_name.endswith(".connect") and call_name != "connect":
        return
    if len(connect_call.args) < 2:
        raise SerializationError("Generated Python connect call is malformed.")
    left_operand = _parse_index_operand(connect_call.args[0])
    right_operand = _parse_index_operand(connect_call.args[1])
    if left_operand is None or right_operand is None:
        raise SerializationError(
            "Generated Python connect calls must target tensor indices."
        )
    keyword_edge_name = _literal_string(_keyword_value(connect_call, "name"))
    if keyword_edge_name is not None:
        edge_name = keyword_edge_name
    if edge_name is None:
        raise SerializationError(
            "Generated Python connect calls must include a recoverable edge name."
        )
    pending_edges.append(
        _PendingEdge(
            name=edge_name,
            left_reference=left_operand[0],
            left_index_name=left_operand[1],
            right_reference=right_operand[0],
            right_index_name=right_operand[1],
        )
    )


def _collect_einsum_labels(
    statement: ast.stmt, einsum_labels_by_reference: dict[str, list[str]]
) -> None:
    einsum_call: ast.Call | None = None
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Call)
    ):
        einsum_call = statement.value
    else:
        appended_value = _parse_list_append_value(statement, "results_list")
        if isinstance(appended_value, ast.Call):
            einsum_call = appended_value
    if einsum_call is None:
        return
    call_name = _call_name(einsum_call.func)
    if not call_name.endswith(".einsum") and call_name != "einsum":
        return
    if einsum_call.args and isinstance(einsum_call.args[0], ast.Constant):
        equation = _literal_string(einsum_call.args[0])
        if equation is not None:
            input_terms = equation.split("->", maxsplit=1)[0].split(",")
            if len(input_terms) != len(einsum_call.args[1:]):
                raise SerializationError(
                    "Generated Python einsum operands do not match the equation."
                )
            for argument, input_term in zip(
                einsum_call.args[1:], input_terms, strict=True
            ):
                reference = _parse_tensor_reference(argument)
                if reference is not None:
                    einsum_labels_by_reference[reference] = list(input_term)
            return

    arguments = einsum_call.args
    if len(arguments) < 3 or len(arguments) % 2 == 0:
        raise SerializationError("Generated Python einsum call is malformed.")
    for argument_index in range(0, len(arguments) - 1, 2):
        reference = _parse_tensor_reference(arguments[argument_index])
        label_values = _literal_int_sequence(arguments[argument_index + 1])
        if label_values is None:
            raise SerializationError("Generated Python einsum sublists are malformed.")
        if reference is not None:
            einsum_labels_by_reference[reference] = [
                f"label_{value}" for value in label_values
            ]


def _collect_remaining_einsum_labels(
    statement: ast.stmt,
    remaining_einsum_labels_by_reference: dict[str, list[str]],
) -> None:
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Name)
        or statement.targets[0].id != "remaining_operand_labels"
        or not isinstance(statement.value, ast.Dict)
    ):
        return

    for key, value in zip(statement.value.keys, statement.value.values, strict=True):
        reference = _parse_tensor_reference_string(_literal_string(key))
        labels = _literal_string_sequence(value)
        if reference is not None and labels is not None:
            remaining_einsum_labels_by_reference[reference] = labels


def _parse_tensor_expression(
    *,
    expression: ast.expr,
    data_shapes: dict[str, tuple[int, ...]],
    reference: str,
    fallback_name: str | None,
) -> _ParsedTensor:
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
        )

    raise SerializationError(
        "Generated Python code does not follow a supported Tensor Network Editor format."
    )


def _parse_zeros_shape(call: ast.Call) -> tuple[int, ...] | None:
    call_name = _call_name(call.func)
    if not call_name.endswith(".zeros") and call_name != "zeros":
        return None

    shape_expression = _keyword_value(call, "shape")
    if shape_expression is None and call.args:
        shape_expression = call.args[0]
    if shape_expression is None:
        return None

    if isinstance(shape_expression, ast.Constant):
        shape_value = _literal_int(shape_expression)
        return (shape_value,) if shape_value is not None else None

    return _literal_int_sequence(shape_expression)


def _resolve_tensor_data_expression(
    *,
    expression: ast.expr | None,
    data_shapes: dict[str, tuple[int, ...]],
    reference: str,
    fallback_name: str | None,
) -> tuple[str, tuple[int, ...]] | None:
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


def _call_name(expression: ast.expr) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        parent_name = _call_name(expression.value)
        return f"{parent_name}.{expression.attr}" if parent_name else expression.attr
    return ""


def _keyword_value(call: ast.Call, keyword_name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    return None


def _literal_string(expression: ast.expr | None) -> str | None:
    if (
        isinstance(expression, ast.Constant)
        and isinstance(expression.value, str)
        and not isinstance(expression.value, bool)
    ):
        return expression.value
    return None


def _literal_string_sequence(expression: ast.expr | None) -> list[str] | None:
    if not isinstance(expression, (ast.List, ast.Tuple)):
        return None
    values: list[str] = []
    for item in expression.elts:
        string_value = _literal_string(item)
        if string_value is None:
            return None
        values.append(string_value)
    return values


def _literal_int(expression: ast.expr | None) -> int | None:
    if (
        isinstance(expression, ast.Constant)
        and isinstance(expression.value, int)
        and not isinstance(expression.value, bool)
    ):
        return expression.value
    return None


def _literal_int_sequence(expression: ast.expr | None) -> tuple[int, ...] | None:
    if not isinstance(expression, (ast.List, ast.Tuple)):
        return None
    values: list[int] = []
    for item in expression.elts:
        int_value = _literal_int(item)
        if int_value is None:
            return None
        values.append(int_value)
    return tuple(values)


def _parse_tensor_reference(expression: ast.expr) -> str | None:
    if not isinstance(expression, ast.Subscript):
        return None

    if isinstance(expression.value, ast.Name):
        if expression.value.id == "tensors":
            index_value = _literal_int(expression.slice)
            return f"list:{index_value}" if index_value is not None else None
        if expression.value.id == "tensors_dict":
            dict_key = _literal_string(expression.slice)
            return f"dict:{dict_key}" if dict_key is not None else None

    if isinstance(expression.value, ast.Subscript):
        row_index = _parse_matrix_row_index(expression.value)
        column_index = _literal_int(expression.slice)
        if row_index is not None and column_index is not None:
            return f"matrix:{row_index}:{column_index}"

    return None


def _parse_tensor_reference_string(expression: str | None) -> str | None:
    if expression is None:
        return None
    list_match = re.fullmatch(r"tensors\[(\d+)\]", expression)
    if list_match is not None:
        return f"list:{list_match.group(1)}"

    matrix_match = re.fullmatch(r"tensor_rows\[(\d+)\]\[(\d+)\]", expression)
    if matrix_match is not None:
        return f"matrix:{matrix_match.group(1)}:{matrix_match.group(2)}"

    dict_match = re.fullmatch(r"""tensors_dict\[(["'])(.+)\1\]""", expression)
    if dict_match is not None:
        return f"dict:{dict_match.group(2)}"

    return None


def _parse_matrix_row_index(expression: ast.expr) -> int | None:
    if (
        isinstance(expression, ast.Subscript)
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "tensor_rows"
    ):
        return _literal_int(expression.slice)
    return None


def _parse_list_append_value(statement: ast.stmt, list_name: str) -> ast.expr | None:
    if (
        not isinstance(statement, ast.Expr)
        or not isinstance(statement.value, ast.Call)
        or not isinstance(statement.value.func, ast.Attribute)
        or statement.value.func.attr != "append"
        or len(statement.value.args) != 1
        or not isinstance(statement.value.func.value, ast.Name)
        or statement.value.func.value.id != list_name
    ):
        return None
    return statement.value.args[0]


def _parse_index_operand(expression: ast.expr) -> tuple[str, str] | None:
    if not isinstance(expression, ast.Subscript):
        return None
    index_name = _literal_string(expression.slice)
    tensor_reference = _parse_tensor_reference(expression.value)
    if tensor_reference is None or index_name is None:
        return None
    return tensor_reference, index_name


def _extract_name_from_expression(expression: ast.expr | None) -> str | None:
    if isinstance(expression, ast.Name):
        return expression.id
    return None


def _synthetic_data_variable_name(reference: str, fallback_name: str | None) -> str:
    return synthetic_data_variable_name(reference, fallback_name)


def _default_tensor_name_from_position(position: int) -> str:
    return default_tensor_name_from_position(position)


def _recover_tensor_name_from_data_variable(
    data_variable_name: str, fallback_name: str | None = None
) -> str:
    return recover_tensor_name_from_data_variable(data_variable_name, fallback_name)


def _recover_index_name(
    *,
    label: str,
    tensor_name: str,
    data_variable_name: str,
    connected_edge_label: str | None,
) -> str:
    return recover_index_name(
        label=label,
        tensor_name=tensor_name,
        data_variable_name=data_variable_name,
        connected_edge_label=connected_edge_label,
    )


def _sanitize_identifier(value: str) -> str:
    return sanitize_identifier(value)


def _build_edge_specs(
    *,
    tensors_by_reference: dict[str, _ParsedTensor],
    tensor_order: list[str],
    pending_edges: list[_PendingEdge],
) -> list[tuple[str, int, str, int, str]]:
    if pending_edges:
        edge_specs: list[tuple[str, int, str, int, str]] = []
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

    label_occurrences: dict[str, list[tuple[str, int]]] = {}
    for reference in tensor_order:
        parsed_tensor = tensors_by_reference[reference]
        if parsed_tensor.index_labels is None:
            raise SerializationError(
                "Generated Python code is missing index information for one or more tensors."
            )
        for index_position, label in enumerate(parsed_tensor.index_labels):
            label_occurrences.setdefault(label, []).append((reference, index_position))

    edge_specs = []
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
    edge_specs: list[tuple[str, int, str, int, str]],
) -> NetworkSpec:
    tensor_specs: list[TensorSpec] = []
    tensor_id_by_reference: dict[str, str] = {}
    index_id_by_reference_and_position: dict[tuple[str, int], str] = {}
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

    tensor_counter = 1
    for row_index, row_references in enumerate(tensor_rows):
        for column_index, reference in enumerate(row_references):
            parsed_tensor = tensors_by_reference[reference]
            if parsed_tensor.index_labels is None:
                raise SerializationError(
                    "Generated Python code is missing tensor labels required to rebuild the network."
                )
            tensor_id = f"tensor_{tensor_counter}"
            tensor_counter += 1
            tensor_id_by_reference[reference] = tensor_id
            index_specs: list[IndexSpec] = []
            for index_position, label in enumerate(parsed_tensor.index_labels):
                index_id = f"{tensor_id}_index_{index_position + 1}"
                index_id_by_reference_and_position[(reference, index_position)] = (
                    index_id
                )
                index_specs.append(
                    IndexSpec(
                        id=index_id,
                        name=_recover_index_name(
                            label=label,
                            tensor_name=parsed_tensor.name,
                            data_variable_name=parsed_tensor.data_variable_name,
                            connected_edge_label=edge_labels.get(
                                (reference, index_position)
                            ),
                        ),
                        dimension=parsed_tensor.shape[index_position],
                    )
                )
            tensor_specs.append(
                TensorSpec(
                    id=tensor_id,
                    name=parsed_tensor.name,
                    position=CanvasPosition(
                        x=120.0 + column_index * 240.0,
                        y=160.0 + row_index * 180.0,
                    ),
                    size=TensorSize(),
                    indices=index_specs,
                )
            )

    edges = [
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

    return NetworkSpec(
        id="imported_python_network",
        name="Imported Python Network",
        tensors=tensor_specs,
        edges=edges,
        groups=[],
        notes=[],
        contraction_plan=None,
    )

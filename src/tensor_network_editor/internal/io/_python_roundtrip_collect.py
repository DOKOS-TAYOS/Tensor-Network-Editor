"""Statement collectors for generated-Python roundtrips."""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING

from ...errors import SerializationError
from ._python_roundtrip_ast import (
    _call_name,
    _keyword_value,
    _literal_int_sequence,
    _literal_string,
    _literal_string_sequence,
    _parse_index_operand,
    _parse_list_append_value,
    _parse_matrix_row_index,
    _parse_operand_tag_string,
    _parse_results_list_reference,
    _parse_tensor_reference,
    _parse_tensor_reference_string,
    _parse_zeros_shape,
)
from ._python_roundtrip_build import (
    _default_tensor_name_from_position,
    _ManualStepComment,
    _parse_tensor_expression,
    _PendingEdge,
    _PendingManualStep,
)

if TYPE_CHECKING:
    from ._python_roundtrip import _RoundtripParseState

_MANUAL_STEP_COMMENT_PATTERN = re.compile(
    r"^\s*# Manual step (?P<step_id>\S+) \| left=(?P<left_operand_id>\S+) \| "
    r"right=(?P<right_operand_id>\S+)\s*$"
)


def _collect_data_shape(statement: ast.stmt, state: _RoundtripParseState) -> None:
    """Collect tensor-data shapes from supported ``zeros(...)`` assignments."""
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Name)
        or not isinstance(statement.value, ast.Call)
    ):
        return
    shape = _parse_zeros_shape(statement.value)
    if shape is not None:
        state.data_shapes[statement.targets[0].id] = shape


def _collect_supported_tensor_collection_initialization(
    statement: ast.stmt,
    state: _RoundtripParseState,
) -> bool:
    """Collect empty supported tensor collections and mark the parser state."""
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Name)
    ):
        return False
    target_name = statement.targets[0].id
    if target_name == "tensor_rows" and isinstance(statement.value, ast.List):
        state.tensor_rows.clear()
        return True
    return target_name in {"tensors", "tensors_dict"}


def _collect_dict_tensor_assignment(
    statement: ast.stmt,
    state: _RoundtripParseState,
) -> bool:
    """Collect one tensor assigned into a supported dict-backed collection."""
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Subscript)
    ):
        return False
    dict_reference = _parse_tensor_reference(statement.targets[0])
    if dict_reference is None or not dict_reference.startswith("dict:"):
        return False
    parsed_tensor = _parse_tensor_expression(
        expression=statement.value,
        data_shapes=state.data_shapes,
        reference=dict_reference,
        fallback_name=dict_reference.removeprefix("dict:"),
    )
    state.tensors_by_reference[dict_reference] = parsed_tensor
    state.tensor_order.append(dict_reference)
    return True


def _collect_list_tensor_append(
    call: ast.Call,
    state: _RoundtripParseState,
) -> bool:
    """Collect one tensor appended to the flat ``tensors`` collection."""
    if (
        not isinstance(call.func, ast.Attribute)
        or call.func.attr != "append"
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "tensors"
        or len(call.args) != 1
    ):
        return False
    reference = f"list:{len(state.tensor_order)}"
    state.tensors_by_reference[reference] = _parse_tensor_expression(
        expression=call.args[0],
        data_shapes=state.data_shapes,
        reference=reference,
        fallback_name=_default_tensor_name_from_position(len(state.tensor_order)),
    )
    state.tensor_order.append(reference)
    return True


def _collect_empty_tensor_row_append(
    call: ast.Call,
    state: _RoundtripParseState,
) -> bool:
    """Collect one empty row appended to the matrix tensor layout."""
    if (
        not isinstance(call.func, ast.Attribute)
        or call.func.attr != "append"
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "tensor_rows"
        or len(call.args) != 1
        or not isinstance(call.args[0], ast.List)
        or call.args[0].elts
    ):
        return False
    state.tensor_rows.append([])
    return True


def _collect_matrix_tensor_append(
    call: ast.Call,
    state: _RoundtripParseState,
) -> bool:
    """Collect one tensor appended to a row inside ``tensor_rows``."""
    if (
        not isinstance(call.func, ast.Attribute)
        or call.func.attr != "append"
        or not isinstance(call.func.value, ast.Subscript)
        or len(call.args) != 1
    ):
        return False
    row_index = _parse_matrix_row_index(call.func.value)
    if row_index is None:
        return False
    while len(state.tensor_rows) <= row_index:
        state.tensor_rows.append([])
    reference = f"matrix:{row_index}:{len(state.tensor_rows[row_index])}"
    state.tensors_by_reference[reference] = _parse_tensor_expression(
        expression=call.args[0],
        data_shapes=state.data_shapes,
        reference=reference,
        fallback_name=_default_tensor_name_from_position(len(state.tensor_order)),
    )
    state.tensor_rows[row_index].append(reference)
    state.tensor_order.append(reference)
    return True


def _collect_tensor(
    *,
    statement: ast.stmt,
    state: _RoundtripParseState,
) -> None:
    """Collect tensor definitions from list, matrix, and dict layouts."""
    if _collect_supported_tensor_collection_initialization(statement, state):
        state.saw_supported_tensor_collection = True
        return
    if _collect_dict_tensor_assignment(statement, state):
        return
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return
    call = statement.value
    if _collect_list_tensor_append(call, state):
        state.saw_supported_tensor_collection = True
        return
    if _collect_empty_tensor_row_append(call, state):
        state.saw_supported_tensor_collection = True
        return
    if _collect_matrix_tensor_append(call, state):
        state.saw_supported_tensor_collection = True


def _collect_pending_edge(
    statement: ast.stmt,
    state: _RoundtripParseState,
) -> None:
    """Collect pending edges from supported ``connect(...)`` calls."""
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
    state.pending_edges.append(
        _PendingEdge(
            name=edge_name,
            left_reference=left_operand[0],
            left_index_name=left_operand[1],
            right_reference=right_operand[0],
            right_index_name=right_operand[1],
        )
    )


def _collect_einsum_labels(
    statement: ast.stmt,
    state: _RoundtripParseState,
) -> None:
    """Collect einsum label sequences emitted by supported generators."""
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
                    state.einsum_labels_by_reference[reference] = list(input_term)
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
            state.einsum_labels_by_reference[reference] = [
                f"label_{value}" for value in label_values
            ]


def _collect_remaining_einsum_labels(
    statement: ast.stmt,
    state: _RoundtripParseState,
) -> None:
    """Collect labels emitted for remaining operands in partial einsum plans."""
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
            state.remaining_einsum_labels_by_reference[reference] = labels


def _collect_manual_step_comments(code: str) -> dict[int, _ManualStepComment]:
    """Collect structured manual-step comments keyed by their next statement."""
    comments_by_statement_line: dict[int, _ManualStepComment] = {}
    source_lines = code.splitlines()
    for line_number, line in enumerate(source_lines, start=1):
        if not line.lstrip().startswith("# Manual step"):
            continue
        match = _MANUAL_STEP_COMMENT_PATTERN.match(line)
        if match is None:
            raise SerializationError(
                "Generated Python manual step comment is malformed."
            )
        next_line_number = line_number + 1
        while next_line_number <= len(source_lines):
            candidate = source_lines[next_line_number - 1].strip()
            if candidate:
                break
            next_line_number += 1
        if next_line_number > len(source_lines):
            raise SerializationError(
                "Generated Python manual step comment must precede a statement."
            )
        comments_by_statement_line[next_line_number] = _ManualStepComment(
            step_id=match.group("step_id"),
            left_operand_id=match.group("left_operand_id"),
            right_operand_id=match.group("right_operand_id"),
        )
    return comments_by_statement_line


def _collect_manual_step(
    *,
    statement: ast.stmt,
    state: _RoundtripParseState,
) -> None:
    """Collect one manual step from supported generated-Python statements."""
    statement_line_number = getattr(statement, "lineno", None)
    if statement_line_number is None:
        return
    comment = state.manual_step_comments_by_statement_line.get(statement_line_number)
    if comment is None:
        return

    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        raise SerializationError(
            "Generated Python manual step comment must precede a supported step statement."
        )

    call = statement.value
    appended_value = _parse_list_append_value(statement, "results_list")
    if isinstance(appended_value, ast.Call):
        if _collect_appended_einsum_manual_step(
            appended_value=appended_value,
            comment=comment,
            state=state,
        ):
            return
        if _collect_appended_contract_manual_step(
            appended_value=appended_value,
            comment=comment,
            state=state,
        ):
            return

    call_name = _call_name(call.func)
    if (
        call_name.endswith(".contract_between") or call_name == "contract_between"
    ) and _collect_tagged_contract_manual_step(
        call=call,
        comment=comment,
        state=state,
    ):
        return

    raise SerializationError(
        "Generated Python manual step comment must precede a supported step statement."
    )


def _record_pending_manual_step(
    comment: _ManualStepComment,
    state: _RoundtripParseState,
) -> None:
    """Append one recovered manual step and track its result index."""
    state.pending_manual_steps.append(
        _PendingManualStep(
            step_id=comment.step_id,
            left_operand_id=comment.left_operand_id,
            right_operand_id=comment.right_operand_id,
        )
    )
    state.step_ids_by_results_list_index.append(comment.step_id)


def _collect_appended_einsum_manual_step(
    *,
    appended_value: ast.Call,
    comment: _ManualStepComment,
    state: _RoundtripParseState,
) -> bool:
    """Collect one manual step emitted as ``results_list.append(einsum(...))``."""
    appended_call_name = _call_name(appended_value.func)
    if not (appended_call_name.endswith(".einsum") or appended_call_name == "einsum"):
        return False
    if len(appended_value.args) < 3:
        raise SerializationError(
            "Generated Python manual step call is missing einsum operands."
        )
    _resolve_manual_operand_id(
        expression=appended_value.args[1],
        expected_operand_id=comment.left_operand_id,
        step_ids_by_results_list_index=state.step_ids_by_results_list_index,
        preferred_tensor_ids_by_reference=state.preferred_tensor_ids_by_reference,
    )
    _resolve_manual_operand_id(
        expression=appended_value.args[2],
        expected_operand_id=comment.right_operand_id,
        step_ids_by_results_list_index=state.step_ids_by_results_list_index,
        preferred_tensor_ids_by_reference=state.preferred_tensor_ids_by_reference,
    )
    _record_pending_manual_step(comment, state)
    return True


def _collect_appended_contract_manual_step(
    *,
    appended_value: ast.Call,
    comment: _ManualStepComment,
    state: _RoundtripParseState,
) -> bool:
    """Collect one manual step emitted as ``results_list.append(contract(...))``."""
    appended_call_name = _call_name(appended_value.func)
    if not (
        appended_call_name.endswith(".contract_between")
        or appended_call_name == "contract_between"
    ):
        return False
    if len(appended_value.args) < 2:
        raise SerializationError(
            "Generated Python manual step call is missing contraction operands."
        )
    _resolve_manual_operand_id(
        expression=appended_value.args[0],
        expected_operand_id=comment.left_operand_id,
        step_ids_by_results_list_index=state.step_ids_by_results_list_index,
        preferred_tensor_ids_by_reference=state.preferred_tensor_ids_by_reference,
    )
    _resolve_manual_operand_id(
        expression=appended_value.args[1],
        expected_operand_id=comment.right_operand_id,
        step_ids_by_results_list_index=state.step_ids_by_results_list_index,
        preferred_tensor_ids_by_reference=state.preferred_tensor_ids_by_reference,
    )
    _record_pending_manual_step(comment, state)
    return True


def _collect_tagged_contract_manual_step(
    *,
    call: ast.Call,
    comment: _ManualStepComment,
    state: _RoundtripParseState,
) -> bool:
    """Collect one graph-backend manual step emitted with operand tags."""
    if len(call.args) < 2:
        raise SerializationError(
            "Generated Python manual step call is missing contraction operands."
        )
    left_operand_id = _parse_operand_tag_string(_literal_string(call.args[0]))
    right_operand_id = _parse_operand_tag_string(_literal_string(call.args[1]))
    if (
        left_operand_id != comment.left_operand_id
        or right_operand_id != comment.right_operand_id
    ):
        raise SerializationError(
            "Generated Python manual step operands conflict with their markup."
        )
    _record_pending_manual_step(comment, state)
    return True


def _resolve_manual_operand_id(
    *,
    expression: ast.expr,
    expected_operand_id: str,
    step_ids_by_results_list_index: list[str],
    preferred_tensor_ids_by_reference: dict[str, str],
) -> str:
    """Resolve one manual-step operand reference and validate it."""
    tensor_reference = _parse_tensor_reference(expression)
    if tensor_reference is not None:
        recovered_tensor_id = preferred_tensor_ids_by_reference.get(tensor_reference)
        if recovered_tensor_id is None:
            preferred_tensor_ids_by_reference[tensor_reference] = expected_operand_id
            return expected_operand_id
        if recovered_tensor_id != expected_operand_id:
            raise SerializationError(
                "Generated Python manual step markup conflicts with tensor references."
            )
        return recovered_tensor_id

    results_list_index = _parse_results_list_reference(expression)
    if results_list_index is not None:
        normalized_index = results_list_index
        if normalized_index < 0:
            normalized_index += len(step_ids_by_results_list_index)
        if normalized_index < 0 or normalized_index >= len(
            step_ids_by_results_list_index
        ):
            raise SerializationError(
                "Generated Python manual step references an unknown intermediate result."
            )
        recovered_step_id = step_ids_by_results_list_index[normalized_index]
        if recovered_step_id != expected_operand_id:
            raise SerializationError(
                "Generated Python manual step markup conflicts with step-result references."
            )
        return recovered_step_id

    raise SerializationError(
        "Generated Python manual step uses an unsupported operand reference."
    )

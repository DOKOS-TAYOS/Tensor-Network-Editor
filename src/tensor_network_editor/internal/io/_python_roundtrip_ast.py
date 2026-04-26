"""AST parsing helpers for generated-Python roundtrips."""

from __future__ import annotations

import ast
import math
import re
from typing import TypeAlias, cast

from ...internal.models._model_tensor_data import TensorNumericLiteral
from ...models import TensorDataMode, TensorDataSpec
from ..analysis._hyperedge_lowering import _build_copy_tensor_values

RealTensorNumericLiteral: TypeAlias = int | float | list["RealTensorNumericLiteral"]


def _parse_zeros_shape(call: ast.Call) -> tuple[int, ...] | None:
    """Parse the shape passed to a supported ``zeros(...)`` call."""
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


def _parse_tensor_data_initializer(
    call: ast.Call,
) -> tuple[tuple[int, ...], TensorDataSpec | None] | None:
    """Parse one supported tensor-data initializer emitted by this package."""
    zeros_shape = _parse_zeros_shape(call)
    if zeros_shape is not None:
        return zeros_shape, None

    call_name = _call_name(call.func)
    if call_name.endswith(".ones") or call_name == "ones":
        shape = _parse_shape_argument(call)
        if shape is None:
            return None
        return shape, TensorDataSpec(mode=TensorDataMode.ONES)

    if call_name.endswith(".full") or call_name == "full":
        shape = _parse_shape_argument(call)
        if shape is None:
            return None
        fill_expression = _keyword_value(call, "fill_value")
        if fill_expression is None and len(call.args) >= 2:
            fill_expression = call.args[1]
        fill_value = _literal_number(fill_expression)
        if fill_value is None:
            return None
        return shape, TensorDataSpec(mode=TensorDataMode.FILL, fill_value=fill_value)

    if (
        call_name.endswith(".array")
        or call_name.endswith(".tensor")
        or call_name
        in {
            "array",
            "tensor",
        }
    ):
        values_expression = _keyword_value(call, "data")
        if values_expression is None and call.args:
            values_expression = call.args[0]
        values = _literal_numeric_tree(values_expression)
        if values is None:
            return None
        shape = _numeric_tree_shape(values)
        if shape is None:
            return None
        return (
            shape,
            TensorDataSpec(
                mode=TensorDataMode.LITERAL,
                values=cast(TensorNumericLiteral, values),
            ),
        )

    return None


def _parse_copy_tensor_data_update(
    statement: ast.stmt,
) -> tuple[str, tuple[int, ...], TensorDataSpec] | None:
    """Parse compact copy-tensor diagonal fills emitted by generated code."""
    parsed_numpy_assignment = _parse_numpy_copy_tensor_assignment(statement)
    if parsed_numpy_assignment is not None:
        return parsed_numpy_assignment
    return _parse_torch_copy_tensor_assignment(statement)


def _parse_numpy_copy_tensor_assignment(
    statement: ast.stmt,
) -> tuple[str, tuple[int, ...], TensorDataSpec] | None:
    """Parse ``array[(arange(d),) * n] = 1`` copy-tensor fills."""
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Subscript)
        or not isinstance(statement.targets[0].value, ast.Name)
    ):
        return None
    fill_value = _literal_number(statement.value)
    if fill_value is None or fill_value != 1:
        return None
    repeated_indices = _parse_repeated_arange_indices(statement.targets[0].slice)
    if repeated_indices is None:
        return None
    data_variable_name = statement.targets[0].value.id
    dimension, rank = repeated_indices
    shape = (dimension,) * rank
    return (
        data_variable_name,
        shape,
        TensorDataSpec(
            mode=TensorDataMode.LITERAL,
            values=_build_copy_tensor_values(dimension, rank),
        ),
    )


def _parse_torch_copy_tensor_assignment(
    statement: ast.stmt,
) -> tuple[str, tuple[int, ...], TensorDataSpec] | None:
    """Parse ``tensor.index_put_((torch.arange(d),) * n, torch.ones(d))``."""
    if (
        not isinstance(statement, ast.Expr)
        or not isinstance(statement.value, ast.Call)
        or not isinstance(statement.value.func, ast.Attribute)
        or statement.value.func.attr != "index_put_"
        or not isinstance(statement.value.func.value, ast.Name)
        or len(statement.value.args) != 2
    ):
        return None
    repeated_indices = _parse_repeated_arange_indices(statement.value.args[0])
    ones_shape = _parse_ones_shape(statement.value.args[1])
    if repeated_indices is None or ones_shape is None or len(ones_shape) != 1:
        return None
    dimension, rank = repeated_indices
    if ones_shape[0] != dimension:
        return None
    data_variable_name = statement.value.func.value.id
    shape = (dimension,) * rank
    return (
        data_variable_name,
        shape,
        TensorDataSpec(
            mode=TensorDataMode.LITERAL,
            values=_build_copy_tensor_values(dimension, rank),
        ),
    )


def _parse_repeated_arange_indices(expression: ast.expr) -> tuple[int, int] | None:
    """Parse ``(module.arange(dimension),) * rank`` index tuples."""
    if (
        not isinstance(expression, ast.BinOp)
        or not isinstance(expression.op, ast.Mult)
        or not isinstance(expression.left, ast.Tuple)
        or len(expression.left.elts) != 1
        or not isinstance(expression.left.elts[0], ast.Call)
    ):
        return None
    call = expression.left.elts[0]
    call_name = _call_name(call.func)
    if not (call_name.endswith(".arange") or call_name == "arange"):
        return None
    if len(call.args) != 1:
        return None
    dimension = _literal_int(call.args[0])
    rank = _literal_int(expression.right)
    if dimension is None or rank is None:
        return None
    return dimension, rank


def _parse_ones_shape(expression: ast.expr) -> tuple[int, ...] | None:
    """Parse the shape passed to a supported ``ones(...)`` call."""
    if not isinstance(expression, ast.Call):
        return None
    call_name = _call_name(expression.func)
    if not (call_name.endswith(".ones") or call_name == "ones"):
        return None
    return _parse_shape_argument(expression)


def _call_name(expression: ast.expr) -> str:
    """Return the dotted call name for a simple AST expression."""
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        parent_name = _call_name(expression.value)
        return f"{parent_name}.{expression.attr}" if parent_name else expression.attr
    return ""


def _keyword_value(call: ast.Call, keyword_name: str) -> ast.expr | None:
    """Return the AST value for the named keyword argument, if present."""
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    return None


def _literal_string(expression: ast.expr | None) -> str | None:
    """Return the literal string value represented by ``expression``."""
    if (
        isinstance(expression, ast.Constant)
        and isinstance(expression.value, str)
        and not isinstance(expression.value, bool)
    ):
        return expression.value
    return None


def _literal_string_sequence(expression: ast.expr | None) -> list[str] | None:
    """Return a list of literal strings or ``None`` if unsupported."""
    if not isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        return None
    values: list[str] = []
    for item in expression.elts:
        string_value = _literal_string(item)
        if string_value is None:
            return None
        values.append(string_value)
    return values


def _literal_number(expression: ast.expr | None) -> int | float | None:
    """Return a finite numeric literal represented by ``expression``."""
    if isinstance(expression, ast.Constant):
        value = expression.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if (
        isinstance(expression, ast.UnaryOp)
        and isinstance(expression.op, ast.USub)
        and isinstance(expression.operand, ast.Constant)
    ):
        operand_value = expression.operand.value
        if isinstance(operand_value, bool) or not isinstance(
            operand_value, (int, float)
        ):
            return None
        numeric_value = -operand_value
        if isinstance(numeric_value, float) and not math.isfinite(numeric_value):
            return None
        return numeric_value
    return None


def _literal_int(expression: ast.expr | None) -> int | None:
    """Return the literal integer value represented by ``expression``."""
    if (
        isinstance(expression, ast.Constant)
        and isinstance(expression.value, int)
        and not isinstance(expression.value, bool)
    ):
        return expression.value
    return None


def _literal_signed_int(expression: ast.expr | None) -> int | None:
    """Return a literal signed integer value represented by ``expression``."""
    literal_value = _literal_int(expression)
    if literal_value is not None:
        return literal_value
    if (
        isinstance(expression, ast.UnaryOp)
        and isinstance(expression.op, ast.USub)
        and isinstance(expression.operand, ast.Constant)
        and isinstance(expression.operand.value, int)
        and not isinstance(expression.operand.value, bool)
    ):
        return -int(expression.operand.value)
    return None


def _literal_int_sequence(expression: ast.expr | None) -> tuple[int, ...] | None:
    """Return a tuple of literal integers or ``None`` if unsupported."""
    if (
        isinstance(expression, ast.BinOp)
        and isinstance(expression.op, ast.Mult)
        and isinstance(expression.left, (ast.List, ast.Tuple))
        and len(expression.left.elts) == 1
    ):
        repeated_value = _literal_int(expression.left.elts[0])
        repeat_count = _literal_int(expression.right)
        if repeated_value is None or repeat_count is None:
            return None
        return (repeated_value,) * repeat_count
    if not isinstance(expression, (ast.List, ast.Tuple)):
        return None
    values: list[int] = []
    for item in expression.elts:
        int_value = _literal_int(item)
        if int_value is None:
            return None
        values.append(int_value)
    return tuple(values)


def _literal_numeric_tree(
    expression: ast.expr | None,
) -> RealTensorNumericLiteral | None:
    """Return a nested numeric literal tree using Python lists."""
    literal_number = _literal_number(expression)
    if literal_number is not None:
        return literal_number
    if not isinstance(expression, (ast.List, ast.Tuple)):
        return None
    values: list[RealTensorNumericLiteral] = []
    for item in expression.elts:
        child_value = _literal_numeric_tree(item)
        if child_value is None:
            return None
        values.append(child_value)
    return values


def _numeric_tree_shape(values: RealTensorNumericLiteral) -> tuple[int, ...] | None:
    """Return the shape of one nested numeric tree or ``None`` if ragged."""
    if isinstance(values, (int, float)):
        return ()
    if not values:
        return None
    child_shapes = [_numeric_tree_shape(child_value) for child_value in values]
    if any(child_shape is None for child_shape in child_shapes):
        return None
    first_shape = child_shapes[0]
    if first_shape is None:
        return None
    if any(child_shape != first_shape for child_shape in child_shapes[1:]):
        return None
    return (len(values), *first_shape)


def _parse_tensor_reference(expression: ast.expr) -> str | None:
    """Parse a tensor reference from supported list, matrix, or dict access."""
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
    """Parse a tensor reference from its generated string representation."""
    if expression is None:
        return None
    list_match = re.fullmatch(r"tensors\[(\d+)\]", expression)
    if list_match is not None:
        return f"list:{list_match.group(1)}"

    matrix_match = re.fullmatch(r"tensor_rows\[(\d+)\]\[(\d+)\]", expression)
    if matrix_match is not None:
        return f"matrix:{matrix_match.group(1)}:{matrix_match.group(2)}"

    dict_match = re.fullmatch(r"""tensors_dict\[(["'])(.*)\1\]""", expression)
    if dict_match is not None:
        return f"dict:{dict_match.group(2)}"

    return None


def _parse_matrix_row_index(expression: ast.expr) -> int | None:
    """Parse the row index from a matrix-layout tensor reference."""
    if (
        isinstance(expression, ast.Subscript)
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "tensor_rows"
    ):
        return _literal_int(expression.slice)
    return None


def _parse_shape_argument(call: ast.Call) -> tuple[int, ...] | None:
    """Parse the shape argument common to zeros, ones, and full initializers."""
    shape_expression = _keyword_value(call, "shape")
    if shape_expression is None and call.args:
        shape_expression = call.args[0]
    if shape_expression is None:
        return None
    if isinstance(shape_expression, ast.Constant):
        shape_value = _literal_int(shape_expression)
        return (shape_value,) if shape_value is not None else None
    return _literal_int_sequence(shape_expression)


def _parse_results_list_reference(expression: ast.expr) -> int | None:
    """Parse a supported ``results_list[...]`` reference."""
    if (
        isinstance(expression, ast.Subscript)
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "results_list"
    ):
        return _literal_signed_int(expression.slice)
    return None


def _parse_list_append_value(statement: ast.stmt, list_name: str) -> ast.expr | None:
    """Return the appended value for a simple ``list.append(...)`` statement."""
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
    """Parse a tensor-index operand like ``tensors[0]['left']``."""
    if not isinstance(expression, ast.Subscript):
        return None
    index_name = _literal_string(expression.slice)
    tensor_reference = _parse_tensor_reference(expression.value)
    if tensor_reference is None or index_name is None:
        return None
    return tensor_reference, index_name


def _extract_name_from_expression(expression: ast.expr | None) -> str | None:
    """Return the referenced variable name for simple name expressions."""
    if isinstance(expression, ast.Name):
        return expression.id
    return None


def _parse_operand_tag_string(value: str | None) -> str | None:
    """Parse one ``__tne_operand_*`` marker and return the operand id."""
    if value is None or not value.startswith("__tne_operand_"):
        return None
    return value.removeprefix("__tne_operand_")

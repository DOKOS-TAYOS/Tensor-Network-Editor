"""Profile-based AST importers for supported external Python network sources."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Literal, cast

from ...errors import SerializationError
from ...models import NetworkSpec, TensorDataSpec
from ._python_import_shared import ExplicitConnection as _ExplicitConnection
from ._python_import_shared import ImportedTensor as _ImportedTensor
from ._python_import_shared import (
    build_network_from_explicit_connections as _build_network_from_explicit_connections,
)
from ._python_import_shared import (
    build_network_from_shared_labels as _build_network_from_shared_labels,
)
from ._python_import_shared import default_connection_name as _default_connection_name
from ._python_roundtrip_ast import (
    _call_name,
    _extract_name_from_expression,
    _keyword_value,
    _literal_string,
    _literal_string_sequence,
    _parse_index_operand,
)
from ._python_roundtrip_build import _resolve_tensor_data_expression
from ._python_roundtrip_helpers import (
    recover_tensor_name_from_data_variable,
)

PythonSourceProfile = Literal["auto", "generated", "quimb", "tensornetwork", "einsum"]
ResolvedPythonSourceProfile = Literal["generated", "quimb", "tensornetwork", "einsum"]

_SUPPORTED_SOURCE_PROFILES = frozenset(
    {"auto", "generated", "quimb", "tensornetwork", "einsum"}
)
_GENERATED_COLLECTION_NAMES = frozenset(
    {"tensors", "tensors_dict", "tensor_rows", "results_list", "remaining_operands"}
)


@dataclass(slots=True)
class _ImportState:
    """Shared mutable state while recovering one external AST profile."""

    module: ast.Module
    data_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    tensor_data_by_name: dict[str, TensorDataSpec | None] = field(default_factory=dict)
    tensors_by_reference: dict[str, _ImportedTensor] = field(default_factory=dict)
    tensor_order: list[str] = field(default_factory=list)
    selected_references: list[str] | None = None
    explicit_connections: list[_ExplicitConnection] = field(default_factory=list)


def normalize_python_source_profile(source_profile: str) -> PythonSourceProfile:
    """Normalize and validate one requested Python source profile."""
    normalized_profile = source_profile.strip().lower()
    if normalized_profile not in _SUPPORTED_SOURCE_PROFILES:
        raise ValueError(
            "Unsupported Python source profile "
            f"{source_profile!r}. Expected one of {sorted(_SUPPORTED_SOURCE_PROFILES)!r}."
        )
    return cast(PythonSourceProfile, normalized_profile)


def detect_python_source_profile(code: str) -> ResolvedPythonSourceProfile:
    """Detect the most likely supported AST importer profile for ``code``."""
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SerializationError("Could not parse generated Python code.") from exc

    if _looks_like_generated_profile(module):
        return "generated"
    if _looks_like_quimb_profile(module):
        return "quimb"
    if _looks_like_tensornetwork_profile(module):
        return "tensornetwork"
    if _looks_like_einsum_profile(module):
        return "einsum"
    raise SerializationError(
        "Could not reconstruct a tensor network from the generated Python code."
    )


def parse_python_source_by_profile(
    code: str, *, source_profile: ResolvedPythonSourceProfile
) -> NetworkSpec:
    """Parse ``code`` using the requested supported AST import profile."""
    if source_profile == "quimb":
        return parse_quimb_python_network(code)
    if source_profile == "tensornetwork":
        return parse_tensornetwork_python_network(code)
    if source_profile == "einsum":
        return parse_einsum_python_network(code)
    raise SerializationError(
        "The generated profile must be handled by the Tensor Network Editor round-trip parser."
    )


def parse_quimb_python_network(code: str) -> NetworkSpec:
    """Parse a supported ``quimb``-style tensor network definition."""
    state = _build_import_state(code)
    for statement in state.module.body:
        _collect_data_initializer(statement, state)
        _collect_quimb_tensor(statement, state)
        _collect_quimb_network_membership(statement, state)
    references = _resolve_selected_references(state)
    return _build_network_from_shared_labels(
        tensors_by_reference=state.tensors_by_reference,
        tensor_order=references,
        allow_hyperedges=True,
    )


def parse_tensornetwork_python_network(code: str) -> NetworkSpec:
    """Parse a supported ``tensornetwork``-style node graph."""
    state = _build_import_state(code)
    for statement in state.module.body:
        _collect_data_initializer(statement, state)
        _collect_tensornetwork_node(statement, state)
        _collect_tensornetwork_connection(statement, state)
    references = _resolve_selected_references(state)
    return _build_network_from_explicit_connections(
        tensors_by_reference=state.tensors_by_reference,
        tensor_order=references,
        explicit_connections=state.explicit_connections,
    )


def parse_einsum_python_network(code: str) -> NetworkSpec:
    """Parse one supported ``einsum`` / ``opt_einsum.contract`` expression."""
    state = _build_import_state(code)
    for statement in state.module.body:
        _collect_data_initializer(statement, state)
    _collect_einsum_operands(state)
    references = _resolve_selected_references(state)
    return _build_network_from_shared_labels(
        tensors_by_reference=state.tensors_by_reference,
        tensor_order=references,
        allow_hyperedges=True,
    )


def _build_import_state(code: str) -> _ImportState:
    """Parse source code and initialize the shared import state."""
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SerializationError("Could not parse generated Python code.") from exc
    return _ImportState(module=module)


def _looks_like_generated_profile(module: ast.Module) -> bool:
    """Return whether ``module`` matches the package's generated source layout."""
    for statement in module.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in _GENERATED_COLLECTION_NAMES
        ):
            return True
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "append"
            and isinstance(statement.value.func.value, ast.Name)
            and statement.value.func.value.id in _GENERATED_COLLECTION_NAMES
        ):
            return True
    return False


def _looks_like_quimb_profile(module: ast.Module) -> bool:
    """Return whether ``module`` contains supported ``quimb`` tensor patterns."""
    for statement in module.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.value, ast.Call)
        ):
            call_name = _call_name(statement.value.func)
            if call_name.endswith(".Tensor") or call_name == "Tensor":
                return True
            if call_name.endswith(".TensorNetwork") or call_name == "TensorNetwork":
                return True
            if _parse_bitand_name_chain(statement.value) is not None:
                return True
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and _parse_bitand_name_chain(statement.value) is not None
        ):
            return True
    return False


def _looks_like_tensornetwork_profile(module: ast.Module) -> bool:
    """Return whether ``module`` contains supported ``tensornetwork`` patterns."""
    for statement in module.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.value, ast.Call)
        ):
            call_name = _call_name(statement.value.func)
            if call_name.endswith(".Node") or call_name == "Node":
                return True
            if call_name.endswith(".connect") or call_name == "connect":
                return True
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and _parse_bitxor_connection(statement.value) is not None
        ):
            return True
    return False


def _looks_like_einsum_profile(module: ast.Module) -> bool:
    """Return whether ``module`` contains supported ``einsum`` expressions."""
    for statement in module.body:
        call = _extract_supported_einsum_call(statement)
        if call is not None:
            return True
    return False


def _collect_data_initializer(statement: ast.stmt, state: _ImportState) -> None:
    """Collect one supported static tensor-data initializer."""
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Name)
        or not isinstance(statement.value, ast.Call)
    ):
        return
    resolved_data = _resolve_tensor_data_expression(
        expression=statement.value,
        data_shapes=state.data_shapes,
        tensor_data_by_name=state.tensor_data_by_name,
        reference=statement.targets[0].id,
        fallback_name=statement.targets[0].id,
    )
    if resolved_data is None:
        return
    _, shape, tensor_data = resolved_data
    state.data_shapes[statement.targets[0].id] = shape
    state.tensor_data_by_name[statement.targets[0].id] = tensor_data


def _collect_quimb_tensor(statement: ast.stmt, state: _ImportState) -> None:
    """Collect one supported ``qtn.Tensor(...)`` assignment."""
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Name)
        or not isinstance(statement.value, ast.Call)
    ):
        return
    call = statement.value
    call_name = _call_name(call.func)
    if not (call_name.endswith(".Tensor") or call_name == "Tensor"):
        return
    reference = statement.targets[0].id
    data_expression = _keyword_value(call, "data") or (
        call.args[0] if call.args else None
    )
    resolved_data = _resolve_tensor_data_expression(
        expression=data_expression,
        data_shapes=state.data_shapes,
        tensor_data_by_name=state.tensor_data_by_name,
        reference=reference,
        fallback_name=reference,
    )
    if resolved_data is None:
        raise SerializationError(
            "The supported quimb importer requires static tensor data initializers."
        )
    data_variable_name, shape, tensor_data = resolved_data
    inds_expression = _keyword_value(call, "inds")
    if inds_expression is None and len(call.args) >= 2:
        inds_expression = call.args[1]
    index_labels = _literal_string_sequence(inds_expression)
    if index_labels is None:
        raise SerializationError(
            "The supported quimb importer requires literal string inds."
        )
    tags_expression = _keyword_value(call, "tags")
    if tags_expression is None and len(call.args) >= 3:
        tags_expression = call.args[2]
    tags = _literal_string_sequence(tags_expression) or []
    tensor_name = (
        tags[0]
        if tags
        else recover_tensor_name_from_data_variable(
            data_variable_name,
            reference,
        )
    )
    state.tensors_by_reference[reference] = _ImportedTensor(
        reference=reference,
        name=tensor_name,
        shape=shape,
        index_labels=tuple(index_labels),
        tensor_data=tensor_data,
    )
    if reference not in state.tensor_order:
        state.tensor_order.append(reference)


def _collect_quimb_network_membership(statement: ast.stmt, state: _ImportState) -> None:
    """Collect one supported ``quimb`` network-membership expression."""
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return
    value = statement.value
    if isinstance(value, ast.Call):
        call_name = _call_name(value.func)
        if call_name.endswith(".TensorNetwork") or call_name == "TensorNetwork":
            sequence_expression = _keyword_value(value, "ts")
            if sequence_expression is None and value.args:
                sequence_expression = value.args[0]
            references = _parse_name_sequence(sequence_expression)
            if references is None:
                raise SerializationError(
                    "The supported quimb importer requires TensorNetwork([...]) with named tensor references."
                )
            state.selected_references = references
            return
    references = _parse_bitand_name_chain(value)
    if references is not None:
        state.selected_references = references


def _collect_tensornetwork_node(statement: ast.stmt, state: _ImportState) -> None:
    """Collect one supported ``tn.Node(...)`` assignment."""
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Name)
        or not isinstance(statement.value, ast.Call)
    ):
        return
    call = statement.value
    call_name = _call_name(call.func)
    if not (call_name.endswith(".Node") or call_name == "Node"):
        return
    reference = statement.targets[0].id
    data_expression = (
        call.args[0]
        if call.args
        else _keyword_value(call, "tensor") or _keyword_value(call, "data")
    )
    resolved_data = _resolve_tensor_data_expression(
        expression=data_expression,
        data_shapes=state.data_shapes,
        tensor_data_by_name=state.tensor_data_by_name,
        reference=reference,
        fallback_name=reference,
    )
    if resolved_data is None:
        raise SerializationError(
            "The supported TensorNetwork importer requires static node tensor initializers."
        )
    _, shape, tensor_data = resolved_data
    axis_names = _literal_string_sequence(
        _keyword_value(call, "axis_names") or _keyword_value(call, "axes_names")
    )
    if axis_names is None:
        raise SerializationError(
            "The supported TensorNetwork importer requires literal axis_names."
        )
    tensor_name = _literal_string(_keyword_value(call, "name")) or reference
    state.tensors_by_reference[reference] = _ImportedTensor(
        reference=reference,
        name=tensor_name,
        shape=shape,
        index_labels=tuple(axis_names),
        tensor_data=tensor_data,
    )
    if reference not in state.tensor_order:
        state.tensor_order.append(reference)


def _collect_tensornetwork_connection(statement: ast.stmt, state: _ImportState) -> None:
    """Collect one supported explicit TensorNetwork binary connection."""
    explicit_connection = _parse_connect_call_connection(statement)
    if explicit_connection is not None:
        state.explicit_connections.append(explicit_connection)
        return
    xor_connection = _parse_bitxor_statement_connection(statement)
    if xor_connection is not None:
        state.explicit_connections.append(xor_connection)


def _collect_einsum_operands(state: _ImportState) -> None:
    """Collect operand tensors from exactly one supported einsum-style call."""
    collected_call: ast.Call | None = None
    for statement in state.module.body:
        maybe_call = _extract_supported_einsum_call(statement)
        if maybe_call is None:
            continue
        if collected_call is not None:
            raise SerializationError(
                "The supported einsum importer currently accepts one top-level contraction expression."
            )
        collected_call = maybe_call
    if collected_call is None:
        raise SerializationError(
            "Could not reconstruct a tensor network from the supported einsum-style Python code."
        )
    equation = _literal_string(collected_call.args[0] if collected_call.args else None)
    if equation is None:
        raise SerializationError(
            "The supported einsum importer requires a literal equation string."
        )
    input_terms = equation.split("->", maxsplit=1)[0].split(",")
    operand_expressions = list(collected_call.args[1:])
    if len(input_terms) != len(operand_expressions):
        raise SerializationError(
            "The supported einsum importer requires one static operand per equation term."
        )
    for operand_index, (input_term, expression) in enumerate(
        zip(input_terms, operand_expressions, strict=True),
        start=1,
    ):
        reference = f"einsum_operand_{operand_index}"
        resolved_data = _resolve_tensor_data_expression(
            expression=expression,
            data_shapes=state.data_shapes,
            tensor_data_by_name=state.tensor_data_by_name,
            reference=reference,
            fallback_name=reference,
        )
        if resolved_data is None:
            raise SerializationError(
                "The supported einsum importer requires operands with static supported initializers."
            )
        data_variable_name, shape, tensor_data = resolved_data
        state.tensors_by_reference[reference] = _ImportedTensor(
            reference=reference,
            name=recover_tensor_name_from_data_variable(
                data_variable_name,
                reference,
            ),
            shape=shape,
            index_labels=tuple(input_term),
            tensor_data=tensor_data,
        )
        state.tensor_order.append(reference)


def _extract_supported_einsum_call(statement: ast.stmt) -> ast.Call | None:
    """Return one supported top-level einsum-style call if present."""
    call: ast.Call | None = None
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Call)
    ):
        call = statement.value
    elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        call = statement.value
    if call is None:
        return None
    call_name = _call_name(call.func)
    if call_name.endswith(".einsum") or call_name == "einsum":
        return call
    if call_name.endswith(".contract") or call_name == "contract":
        return call
    return None


def _parse_name_sequence(expression: ast.expr | None) -> list[str] | None:
    """Parse a flat literal sequence of name references."""
    if not isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        return None
    names: list[str] = []
    for item in expression.elts:
        reference = _extract_name_from_expression(item)
        if reference is None:
            return None
        names.append(reference)
    return names


def _parse_bitand_name_chain(expression: ast.expr) -> list[str] | None:
    """Parse a chain like ``tensor_a & tensor_b & tensor_c`` into references."""
    if isinstance(expression, ast.Name):
        return [expression.id]
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.BitAnd):
        left_references = _parse_bitand_name_chain(expression.left)
        right_references = _parse_bitand_name_chain(expression.right)
        if left_references is None or right_references is None:
            return None
        return [*left_references, *right_references]
    return None


def _parse_connect_call_connection(statement: ast.stmt) -> _ExplicitConnection | None:
    """Parse one explicit ``tn.connect(...)`` assignment or expression."""
    connection_name: str | None = None
    call: ast.Call | None = None
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Call)
    ):
        connection_name = statement.targets[0].id.removesuffix("_edge")
        call = statement.value
    elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        call = statement.value
    if call is None:
        return None
    call_name = _call_name(call.func)
    if not (call_name.endswith(".connect") or call_name == "connect"):
        return None
    if len(call.args) < 2:
        raise SerializationError(
            "The supported TensorNetwork importer requires two operands in tn.connect(...)."
        )
    left_operand = _parse_named_index_operand(call.args[0])
    right_operand = _parse_named_index_operand(call.args[1])
    if left_operand is None or right_operand is None:
        raise SerializationError(
            "The supported TensorNetwork importer requires tn.connect(...) to target named node indices."
        )
    explicit_name = _literal_string(_keyword_value(call, "name"))
    recovered_name = explicit_name or connection_name
    if recovered_name is None:
        recovered_name = _default_connection_name(left_operand[1], right_operand[1])
    return _ExplicitConnection(
        name=recovered_name,
        left_reference=left_operand[0],
        left_index_name=left_operand[1],
        right_reference=right_operand[0],
        right_index_name=right_operand[1],
    )


def _parse_bitxor_statement_connection(
    statement: ast.stmt,
) -> _ExplicitConnection | None:
    """Parse one explicit ``node_a['x'] ^ node_b['x']`` assignment."""
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Name)
    ):
        return None
    parsed_connection = _parse_bitxor_connection(statement.value)
    if parsed_connection is None:
        return None
    connection_name = statement.targets[0].id
    return _ExplicitConnection(
        name=connection_name,
        left_reference=parsed_connection[0],
        left_index_name=parsed_connection[1],
        right_reference=parsed_connection[2],
        right_index_name=parsed_connection[3],
    )


def _parse_bitxor_connection(
    expression: ast.expr,
) -> tuple[str, str, str, str] | None:
    """Parse one ``left_index ^ right_index`` connection expression."""
    if not isinstance(expression, ast.BinOp) or not isinstance(
        expression.op, ast.BitXor
    ):
        return None
    left_operand = _parse_named_index_operand(expression.left)
    right_operand = _parse_named_index_operand(expression.right)
    if left_operand is None or right_operand is None:
        raise SerializationError(
            "The supported TensorNetwork importer requires '^' connections between named node indices."
        )
    return (
        left_operand[0],
        left_operand[1],
        right_operand[0],
        right_operand[1],
    )


def _parse_named_index_operand(expression: ast.expr) -> tuple[str, str] | None:
    """Parse a simple named tensor/node index operand like ``node_a['bond_x']``."""
    parsed_roundtrip_operand = _parse_index_operand(expression)
    if parsed_roundtrip_operand is not None:
        return parsed_roundtrip_operand
    if not isinstance(expression, ast.Subscript):
        return None
    index_name = _literal_string(expression.slice)
    reference = _extract_name_from_expression(expression.value)
    if reference is None or index_name is None:
        return None
    return reference, index_name


def _resolve_selected_references(state: _ImportState) -> list[str]:
    """Resolve the active tensor order for one parsed import state."""
    if not state.tensors_by_reference:
        raise SerializationError(
            "Could not reconstruct a tensor network from the provided Python code."
        )
    selected_references = (
        state.selected_references
        if state.selected_references is not None
        else state.tensor_order
    )
    if not selected_references:
        raise SerializationError(
            "Could not reconstruct a tensor network from the provided Python code."
        )
    resolved_references: list[str] = []
    for reference in selected_references:
        if reference not in state.tensors_by_reference:
            raise SerializationError(
                f"The supported Python importer references an unknown tensor or node '{reference}'."
            )
        if reference not in resolved_references:
            resolved_references.append(reference)
    return resolved_references


__all__ = [
    "PythonSourceProfile",
    "ResolvedPythonSourceProfile",
    "detect_python_source_profile",
    "normalize_python_source_profile",
    "parse_python_source_by_profile",
]

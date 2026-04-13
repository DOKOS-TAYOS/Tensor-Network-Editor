from __future__ import annotations

import ast

from tensor_network_editor._python_roundtrip_helpers import (
    recover_tensor_name_from_data_variable,
    sanitize_identifier,
)


def test_python_roundtrip_internal_helpers_normalize_generated_names() -> None:
    assert recover_tensor_name_from_data_variable("leaf_mid_data") == "Leaf Mid"
    assert recover_tensor_name_from_data_variable("a_data") == "A"
    assert sanitize_identifier(" Leaf Mid ") == "leaf_mid"


def test_python_roundtrip_internal_ast_helpers_parse_supported_references() -> None:
    from tensor_network_editor._python_roundtrip_ast import (
        _parse_tensor_reference,
        _parse_tensor_reference_string,
    )

    list_expression = ast.parse("tensors[1]", mode="eval").body
    matrix_expression = ast.parse("tensor_rows[2][3]", mode="eval").body

    assert _parse_tensor_reference(list_expression) == "list:1"
    assert _parse_tensor_reference(matrix_expression) == "matrix:2:3"
    assert _parse_tensor_reference_string("tensors_dict['tensor_a']") == (
        "dict:tensor_a"
    )


def test_python_roundtrip_internal_build_helpers_resolve_inline_zeros() -> None:
    from tensor_network_editor._python_roundtrip_build import (
        _resolve_tensor_data_expression,
    )

    zeros_expression = ast.parse("np.zeros((2, 3))", mode="eval").body

    assert _resolve_tensor_data_expression(
        expression=zeros_expression,
        data_shapes={},
        reference="list:0",
        fallback_name="A",
    ) == ("a_data", (2, 3))

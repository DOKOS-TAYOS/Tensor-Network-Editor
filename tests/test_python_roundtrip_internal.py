from __future__ import annotations

import ast

import pytest

from tensor_network_editor.errors import SerializationError
from tensor_network_editor.internal.io._python_roundtrip_helpers import (
    recover_tensor_name_from_data_variable,
    sanitize_identifier,
)
from tensor_network_editor.models import TensorDataMode, TensorDataSpec


def test_python_roundtrip_internal_helpers_normalize_generated_names() -> None:
    assert recover_tensor_name_from_data_variable("leaf_mid_data") == "Leaf Mid"
    assert recover_tensor_name_from_data_variable("a_data") == "A"
    assert sanitize_identifier(" Leaf Mid ") == "leaf_mid"


def test_python_roundtrip_internal_ast_helpers_parse_supported_references() -> None:
    from tensor_network_editor.internal.io._python_roundtrip_ast import (
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
    assert _parse_tensor_reference_string('tensors_dict["tensor_a"]') == (
        "dict:tensor_a"
    )
    assert _parse_tensor_reference_string("tensors_dict['']") == "dict:"


def test_python_roundtrip_internal_build_helpers_resolve_inline_zeros() -> None:
    from tensor_network_editor.internal.io._python_roundtrip_build import (
        _resolve_tensor_data_expression,
    )

    zeros_expression = ast.parse("np.zeros((2, 3))", mode="eval").body

    assert _resolve_tensor_data_expression(
        expression=zeros_expression,
        data_shapes={},
        tensor_data_by_name={},
        reference="list:0",
        fallback_name="A",
    ) == ("a_data", (2, 3), None)


def test_python_roundtrip_internal_ast_helpers_parse_literal_tensor_data() -> None:
    from tensor_network_editor.internal.io._python_roundtrip_ast import (
        _parse_tensor_data_initializer,
    )

    literal_expression = ast.parse(
        "np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)",
        mode="eval",
    ).body
    assert isinstance(literal_expression, ast.Call)

    assert _parse_tensor_data_initializer(literal_expression) == (
        (2, 2),
        TensorDataSpec(
            mode=TensorDataMode.LITERAL,
            values=[[1.0, 2.0], [3.0, 4.0]],
        ),
    )


def test_python_roundtrip_internal_manual_step_comments_bind_to_next_statement() -> (
    None
):
    from tensor_network_editor.internal.io._python_roundtrip_collect import (
        _collect_manual_step_comments,
    )

    comments = _collect_manual_step_comments(
        "\n".join(
            [
                "# Manual step step_ab | left=tensor_a | right=tensor_b",
                "",
                "results_list.append(np.einsum('ab,bc->ac', tensor_a, tensor_b))",
            ]
        )
    )

    assert comments[3].step_id == "step_ab"
    assert comments[3].left_operand_id == "tensor_a"
    assert comments[3].right_operand_id == "tensor_b"


def test_python_roundtrip_internal_manual_step_results_list_minus_one_resolves() -> (
    None
):
    from tensor_network_editor.internal.io._python_roundtrip_collect import (
        _resolve_manual_operand_id,
    )

    expression = ast.parse("results_list[-1]", mode="eval").body

    assert (
        _resolve_manual_operand_id(
            expression=expression,
            expected_operand_id="step_abc",
            step_ids_by_results_list_index=["step_ab", "step_abc"],
            preferred_tensor_ids_by_reference={},
        )
        == "step_abc"
    )


def test_python_roundtrip_internal_build_edge_specs_supports_explicit_connect_edges() -> (
    None
):
    from tensor_network_editor.internal.io._python_roundtrip_build import (
        _build_edge_specs,
        _ParsedTensor,
        _PendingEdge,
    )

    edge_specs = _build_edge_specs(
        tensors_by_reference={
            "dict:tensor_a": _ParsedTensor(
                reference="dict:tensor_a",
                data_variable_name="a_data",
                shape=(2, 3),
                name="A",
                index_labels=["left", "bond_x"],
            ),
            "dict:tensor_b": _ParsedTensor(
                reference="dict:tensor_b",
                data_variable_name="b_data",
                shape=(3, 5),
                name="B",
                index_labels=["bond_x", "right"],
            ),
        },
        tensor_order=["dict:tensor_a", "dict:tensor_b"],
        pending_edges=[
            _PendingEdge(
                name="bond_x",
                left_reference="dict:tensor_a",
                left_index_name="bond_x",
                right_reference="dict:tensor_b",
                right_index_name="bond_x",
            )
        ],
    )

    assert edge_specs == [("dict:tensor_a", 1, "dict:tensor_b", 0, "bond_x")]


def test_python_roundtrip_internal_build_edge_specs_rejects_three_shared_labels() -> (
    None
):
    from tensor_network_editor.internal.io._python_roundtrip_build import (
        _build_edge_specs,
        _ParsedTensor,
    )

    with pytest.raises(
        SerializationError,
        match="unsupported number of shared indices",
    ):
        _build_edge_specs(
            tensors_by_reference={
                "list:0": _ParsedTensor(
                    reference="list:0",
                    data_variable_name="a_data",
                    shape=(2,),
                    name="A",
                    index_labels=["shared"],
                ),
                "list:1": _ParsedTensor(
                    reference="list:1",
                    data_variable_name="b_data",
                    shape=(2,),
                    name="B",
                    index_labels=["shared"],
                ),
                "list:2": _ParsedTensor(
                    reference="list:2",
                    data_variable_name="c_data",
                    shape=(2,),
                    name="C",
                    index_labels=["shared"],
                ),
            },
            tensor_order=["list:0", "list:1", "list:2"],
            pending_edges=[],
        )


def test_build_roundtrip_parse_state_collects_initial_context() -> None:
    from tensor_network_editor.internal.io._python_roundtrip import (
        _build_roundtrip_parse_state,
    )

    state = _build_roundtrip_parse_state(
        "\n".join(
            [
                "tensor_a = np.zeros((2, 3))",
                "# Manual step step_ab | left=tensor_a | right=tensor_b",
                "",
                "results_list.append(np.einsum('ab,bc->ac', tensor_a, tensor_b))",
            ]
        )
    )

    assert len(state.module.body) == 2
    assert state.data_shapes == {}
    assert state.tensor_order == []
    assert state.pending_edges == []
    assert state.pending_manual_steps == []
    assert state.manual_step_comments_by_statement_line[4].step_id == "step_ab"
    assert state.saw_supported_tensor_collection is False

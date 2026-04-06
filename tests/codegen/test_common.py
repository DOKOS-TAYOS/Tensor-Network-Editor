from __future__ import annotations

import pytest

from tensor_network_editor.codegen.common import (
    make_unique_identifiers,
    prepare_network,
    sanitize_identifier,
    tensor_variable_name,
)
from tensor_network_editor.models import NetworkSpec


def test_sanitize_identifier_normalizes_empty_and_numeric_names() -> None:
    assert sanitize_identifier("  Tensor A  ", "tensor") == "tensor_a"
    assert sanitize_identifier("123 bond", "edge") == "edge_123_bond"
    assert sanitize_identifier("!!!", "tensor") == "tensor"


def test_make_unique_identifiers_deduplicates_collisions() -> None:
    assert make_unique_identifiers(
        ["Tensor A", "tensor-a", "123", "123"],
        "tensor",
    ) == ["tensor_a", "tensor_a_2", "tensor_123", "tensor_123_2"]


def test_prepare_network_assigns_stable_labels(sample_spec: NetworkSpec) -> None:
    prepared = prepare_network(sample_spec)

    assert [tensor.variable_name for tensor in prepared.tensors] == ["a", "b"]
    assert [tensor.data_variable_name for tensor in prepared.tensors] == [
        "a_data",
        "b_data",
    ]
    assert [edge.label for edge in prepared.edges] == ["bond_x"]
    assert [edge.variable_name for edge in prepared.edges] == ["bond_x_edge"]
    assert [index.label for index in prepared.open_indices] == ["a_i", "b_j"]
    assert prepared.edges[0].left.label == "bond_x"
    assert prepared.edges[0].right.label == "bond_x"


def test_tensor_variable_name_resolves_known_tensors(sample_spec: NetworkSpec) -> None:
    prepared = prepare_network(sample_spec)

    assert tensor_variable_name(prepared, "tensor_a") == "a"
    with pytest.raises(KeyError, match="missing_tensor"):
        tensor_variable_name(prepared, "missing_tensor")

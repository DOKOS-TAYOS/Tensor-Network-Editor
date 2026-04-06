from __future__ import annotations

import pytest

from tensor_network_editor._analysis import analyze_network
from tensor_network_editor.errors import SpecValidationError
from tensor_network_editor.models import IndexSpec, NetworkSpec


def test_analyze_network_builds_lookup_maps_and_open_indices(
    sample_spec: NetworkSpec,
) -> None:
    analysis = analyze_network(sample_spec)

    assert analysis.spec.id == "network_demo"
    assert sorted(analysis.tensor_map) == ["tensor_a", "tensor_b"]
    assert sorted(analysis.index_map) == [
        "tensor_a_i",
        "tensor_a_x",
        "tensor_b_j",
        "tensor_b_x",
    ]
    assert analysis.connected_index_ids == {"tensor_a_x", "tensor_b_x"}
    assert [(tensor.id, index.id) for tensor, index in analysis.open_indices] == [
        ("tensor_a", "tensor_a_i"),
        ("tensor_b", "tensor_b_j"),
    ]

    left_index = analysis.left_index_by_edge_id["edge_x"]
    right_index = analysis.right_index_by_edge_id["edge_x"]

    assert left_index is not None
    assert right_index is not None
    assert left_index.id == "tensor_a_x"
    assert right_index.id == "tensor_b_x"


def test_analyze_network_validates_when_requested(sample_spec: NetworkSpec) -> None:
    sample_spec.tensors[0].indices[0] = IndexSpec(id="tensor_a_i", name="", dimension=2)

    with pytest.raises(SpecValidationError):
        analyze_network(sample_spec, validate=True)

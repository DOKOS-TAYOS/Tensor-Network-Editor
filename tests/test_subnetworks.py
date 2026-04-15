from __future__ import annotations

from math import isclose

import pytest

from tensor_network_editor.models import (
    CanvasPosition,
    GroupSpec,
    NetworkSpec,
    TensorSize,
    TensorSpec,
)
from tensor_network_editor.subnetworks import (
    extract_subnetwork_spec,
    prepare_subnetwork_for_insertion,
)
from tests.factories import build_linear_periodic_chain_spec, build_sample_spec


def _build_grouped_sample_spec() -> NetworkSpec:
    spec = build_sample_spec()
    spec.tensors.append(
        TensorSpec(
            id="tensor_c",
            name="C",
            position=CanvasPosition(x=620.0, y=160.0),
            size=TensorSize(width=180.0, height=108.0),
        )
    )
    spec.groups.extend(
        [
            GroupSpec(
                id="group_partial",
                name="Partial Group",
                tensor_ids=["tensor_b", "tensor_c"],
            ),
            GroupSpec(
                id="group_empty",
                name="Empty Group",
                tensor_ids=[],
            ),
        ]
    )
    return spec


def _bounds_center(spec: NetworkSpec) -> tuple[float, float]:
    left = min(tensor.position.x - tensor.size.width / 2 for tensor in spec.tensors)
    right = max(tensor.position.x + tensor.size.width / 2 for tensor in spec.tensors)
    top = min(tensor.position.y - tensor.size.height / 2 for tensor in spec.tensors)
    bottom = max(tensor.position.y + tensor.size.height / 2 for tensor in spec.tensors)
    return ((left + right) / 2, (top + bottom) / 2)


def test_extract_subnetwork_spec_keeps_internal_edges_and_full_groups() -> None:
    spec = _build_grouped_sample_spec()

    extracted = extract_subnetwork_spec(spec, tensor_ids=["tensor_a", "tensor_b"])

    assert [tensor.id for tensor in extracted.tensors] == ["tensor_a", "tensor_b"]
    assert [edge.id for edge in extracted.edges] == ["edge_x"]
    assert [group.id for group in extracted.groups] == ["group_demo"]
    assert extracted.notes == []
    assert extracted.contraction_plan is None
    assert extracted.linear_periodic_chain is None


def test_extract_subnetwork_spec_rejects_linear_periodic_mode() -> None:
    with pytest.raises(ValueError, match="normal graph mode"):
        extract_subnetwork_spec(
            build_linear_periodic_chain_spec(),
            tensor_ids=["periodic_left_tensor"],
        )


def test_prepare_subnetwork_for_insertion_remaps_ids_and_recenters_fragment() -> None:
    spec = extract_subnetwork_spec(
        build_sample_spec(),
        tensor_ids=["tensor_a", "tensor_b"],
    )
    original_tensor_ids = {tensor.id for tensor in spec.tensors}
    original_index_ids = {
        index.id for tensor in spec.tensors for index in tensor.indices
    }
    original_edge_ids = {edge.id for edge in spec.edges}
    original_group_ids = {group.id for group in spec.groups}

    prepared = prepare_subnetwork_for_insertion(
        spec,
        target_center=CanvasPosition(x=500.0, y=420.0),
    )

    assert prepared.id != spec.id
    assert {tensor.id for tensor in prepared.tensors}.isdisjoint(original_tensor_ids)
    assert {
        index.id for tensor in prepared.tensors for index in tensor.indices
    }.isdisjoint(original_index_ids)
    assert {edge.id for edge in prepared.edges}.isdisjoint(original_edge_ids)
    assert {group.id for group in prepared.groups}.isdisjoint(original_group_ids)
    assert set(prepared.groups[0].tensor_ids) == {
        tensor.id for tensor in prepared.tensors
    }
    assert prepared.notes == []
    assert prepared.contraction_plan is None
    assert prepared.linear_periodic_chain is None
    center_x, center_y = _bounds_center(prepared)
    assert isclose(center_x, 500.0)
    assert isclose(center_y, 420.0)

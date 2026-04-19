from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from tensor_network_editor.codegen.common import (
    group_tensors_by_visual_rows,
    make_unique_identifiers,
    prepare_network,
    sanitize_identifier,
    tensor_collection_reference_by_id,
    tensor_variable_name,
)
from tensor_network_editor.models import (
    CanvasPosition,
    NetworkSpec,
    TensorCollectionFormat,
    TensorSpec,
)


class _CountingPosition:
    def __init__(self, *, x: float, y: float) -> None:
        self.x = x
        self._y = y
        self.y_reads = 0

    @property
    def y(self) -> float:
        self.y_reads += 1
        return self._y


def test_sanitize_identifier_normalizes_empty_and_numeric_names() -> None:
    assert sanitize_identifier("  Tensor A  ", "tensor") == "tensor_a"
    assert sanitize_identifier("123 bond", "edge") == "edge_123_bond"
    assert sanitize_identifier("!!!", "tensor") == "tensor"


def test_make_unique_identifiers_deduplicates_collisions() -> None:
    assert make_unique_identifiers(
        ["Tensor A", "tensor-a", "123", "123"],
        "tensor",
    ) == ["tensor_a", "tensor_a_2", "tensor_123", "tensor_123_2"]


def test_group_tensors_by_visual_rows_preserves_row_grouping_and_x_order() -> None:
    tensors = [
        TensorSpec(id="top_right", position=CanvasPosition(x=300.0, y=100.0)),
        TensorSpec(id="bottom_left", position=CanvasPosition(x=100.0, y=240.0)),
        TensorSpec(id="top_left", position=CanvasPosition(x=100.0, y=104.0)),
        TensorSpec(id="bottom_right", position=CanvasPosition(x=300.0, y=244.0)),
    ]

    rows = group_tensors_by_visual_rows(tensors)

    assert [[tensor.id for tensor in row] for row in rows] == [
        ["top_left", "top_right"],
        ["bottom_left", "bottom_right"],
    ]


def test_group_tensors_by_visual_rows_uses_linear_row_center_work() -> None:
    tensor_count = 150
    positions: list[_CountingPosition] = []
    tensors: list[TensorSpec] = []
    for index in range(tensor_count):
        tensor = TensorSpec(id=f"tensor_{index}", position=CanvasPosition())
        position = _CountingPosition(x=float(index), y=120.0)
        cast(Any, tensor).position = position
        positions.append(position)
        tensors.append(tensor)

    rows = group_tensors_by_visual_rows(tensors)

    assert len(rows) == 1
    assert [tensor.id for tensor in rows[0]] == [
        f"tensor_{index}" for index in range(tensor_count)
    ]
    assert sum(position.y_reads for position in positions) <= tensor_count * 5


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


def test_tensor_collection_reference_by_id_uses_prepared_tensor_lookup() -> None:
    prepared = cast(
        Any,
        SimpleNamespace(
            tensor_by_id={
                "tensor_a": SimpleNamespace(
                    row_index=0,
                    column_index=0,
                    flat_index=0,
                )
            },
            tensors=None,
        ),
    )

    assert (
        tensor_collection_reference_by_id(
            prepared,
            "tensor_a",
            TensorCollectionFormat.LIST,
            "tensors",
        )
        == "tensors[0]"
    )

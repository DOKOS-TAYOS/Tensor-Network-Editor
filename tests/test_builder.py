from __future__ import annotations

import pytest

import tensor_network_editor as tne
from tensor_network_editor.builder import IndexHandle, NetworkBuilder, TensorHandle
from tensor_network_editor.errors import SpecValidationError


def test_network_builder_builds_simple_connected_network() -> None:
    builder = NetworkBuilder("demo", id="network_demo")
    left = builder.tensor("A", id="tensor_a", position=(120.0, 160.0))
    left.index("i", 2, id="tensor_a_i")
    left.index("x", 3, id="tensor_a_x")
    right = builder.tensor("B", id="tensor_b", position=(360.0, 160.0))
    right.index("x", 3, id="tensor_b_x")
    right.index("j", 4, id="tensor_b_j")

    edge = builder.connect(left["x"], right["x"], id="edge_x", name="bond_x")
    spec = builder.build()

    assert spec.id == "network_demo"
    assert spec.name == "demo"
    assert [tensor.name for tensor in spec.tensors] == ["A", "B"]
    assert [tensor.shape for tensor in spec.tensors] == [(2, 3), (3, 4)]
    assert edge is spec.edges[0]
    assert spec.edges[0].left.tensor_id == "tensor_a"
    assert spec.edges[0].right.index_id == "tensor_b_x"


def test_network_builder_index_lookup_rejects_duplicate_names() -> None:
    tensor = NetworkBuilder().tensor("A")
    tensor.index("phys", 2)
    tensor.index("phys", 2)

    with pytest.raises(ValueError, match="more than one index named 'phys'"):
        _ = tensor["phys"]


def test_network_builder_build_runs_validation_by_default() -> None:
    builder = NetworkBuilder("invalid")
    left = builder.tensor("A")
    left.index("x", 2)
    right = builder.tensor("B")
    right.index("x", 3)
    builder.connect(left["x"], right["x"], name="bad_bond")

    with pytest.raises(SpecValidationError, match="dimension"):
        builder.build()


def test_network_builder_build_can_skip_validation() -> None:
    builder = NetworkBuilder("invalid")
    left = builder.tensor("A")
    left.index("x", 2)
    right = builder.tensor("B")
    right.index("x", 3)
    builder.connect(left["x"], right["x"], name="bad_bond")

    spec = builder.build(validate=False)

    assert spec.name == "invalid"
    assert spec.edges[0].name == "bad_bond"


def test_network_builder_creates_hyperedges_groups_notes_and_metadata() -> None:
    builder = NetworkBuilder(
        "annotated",
        id="network_annotated",
        metadata={"tags": ["demo"]},
    )
    left = builder.tensor("A", id="tensor_a", metadata={"role": "state"})
    left.index("h", 3, id="tensor_a_h")
    middle = builder.tensor("B", id="tensor_b")
    middle.index("h", 3, id="tensor_b_h")
    right = builder.tensor("C", id="tensor_c")
    right.index("h", 3, id="tensor_c_h")

    hyperedge = builder.hyperedge(
        [left["h"], middle["h"], right["h"]],
        id="hyperedge_h",
        name="shared_h",
        hub_offset=(12.0, -4.0),
        metadata={"kind": "copy"},
    )
    group = builder.group([left, middle], id="group_pair", name="Pair")
    note = builder.note(
        "Remember boundary choice",
        id="note_boundary",
        position=(40.0, 80.0),
        metadata={"tags": ["note"]},
    )
    spec = builder.build()

    assert spec.metadata == {"tags": ["demo"]}
    assert hyperedge is spec.hyperedges[0]
    assert hyperedge.hub_offset.x == 12.0
    assert hyperedge.metadata == {"kind": "copy"}
    assert group.tensor_ids == ["tensor_a", "tensor_b"]
    assert note.position.y == 80.0
    assert spec.tensors[0].metadata == {"role": "state"}


def test_network_builder_rejects_handles_from_other_builders() -> None:
    first_builder = NetworkBuilder("first")
    left = first_builder.tensor("A")
    left.index("x", 2)
    second_builder = NetworkBuilder("second")
    right = second_builder.tensor("B")
    right.index("x", 2)

    with pytest.raises(ValueError, match="different NetworkBuilder"):
        first_builder.connect(left["x"], right["x"])


def test_network_builder_public_imports_are_available() -> None:
    assert tne.NetworkBuilder is NetworkBuilder
    assert tne.TensorHandle is TensorHandle
    assert tne.IndexHandle is IndexHandle

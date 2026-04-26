from __future__ import annotations

import pytest

from tensor_network_editor.internal.templates._templates import (
    TemplateParameters,
    build_template_spec,
    list_template_names,
)
from tensor_network_editor.linting import lint_spec
from tensor_network_editor.validation import ensure_valid_spec


def test_template_catalog_exposes_expected_names() -> None:
    assert list_template_names() == [
        "mps",
        "mpo",
        "peps_2x2",
        "mera",
        "binary_tree",
        "ttn",
        "pepo",
        "heisenberg_mps",
        "ising_mps",
        "transverse_ising_mpo",
        "tebd_gate_layer",
    ]


@pytest.mark.parametrize("template_name", list_template_names())
def test_all_templates_build_valid_specs(template_name: str) -> None:
    spec = build_template_spec(template_name)

    validated = ensure_valid_spec(spec)

    assert validated.tensors


@pytest.mark.parametrize(
    ("template_name", "min_width", "min_height"),
    [
        ("mps", 900.0, 0.0),
        ("mpo", 930.0, 0.0),
        ("peps_2x2", 300.0, 260.0),
        ("mera", 560.0, 420.0),
        ("binary_tree", 560.0, 420.0),
        ("ttn", 560.0, 420.0),
        ("pepo", 680.0, 560.0),
        ("heisenberg_mps", 900.0, 0.0),
        ("ising_mps", 900.0, 0.0),
        ("transverse_ising_mpo", 930.0, 0.0),
        ("tebd_gate_layer", 900.0, 220.0),
    ],
)
def test_templates_use_generous_spacing(
    template_name: str,
    min_width: float,
    min_height: float,
) -> None:
    spec = build_template_spec(template_name)
    x_positions = [tensor.position.x for tensor in spec.tensors]
    y_positions = [tensor.position.y for tensor in spec.tensors]

    assert max(x_positions) - min(x_positions) >= min_width
    assert max(y_positions) - min(y_positions) >= min_height


@pytest.mark.parametrize(
    ("template_name", "expected_tensors"),
    [
        (
            "mps",
            {
                "A1": {"right", "phys"},
                "A2": {"left", "right", "phys"},
                "A3": {"left", "right", "phys"},
                "A4": {"left", "phys"},
            },
        ),
        (
            "mpo",
            {
                "W1": {"right", "bra", "ket"},
                "W2": {"left", "right", "bra", "ket"},
                "W3": {"left", "right", "bra", "ket"},
                "W4": {"left", "bra", "ket"},
            },
        ),
        (
            "peps_2x2",
            {
                "A1": {"right", "down", "phys"},
                "A2": {"left", "right", "down", "phys"},
                "A3": {"left", "down", "phys"},
                "B1": {"up", "right", "down", "phys"},
                "B2": {"up", "left", "right", "down", "phys"},
                "B3": {"up", "left", "down", "phys"},
                "C1": {"up", "right", "phys"},
                "C2": {"up", "left", "right", "phys"},
                "C3": {"up", "left", "phys"},
            },
        ),
        (
            "mera",
            {
                "Top": {"left", "right"},
                "Mid L": {"up", "left", "down"},
                "Mid R": {"up", "down", "right"},
                "Leaf L": {"up", "phys"},
                "Leaf M": {"left", "right", "phys"},
                "Leaf R": {"up", "phys"},
            },
        ),
        (
            "binary_tree",
            {
                "Root": {"left", "right"},
                "Left": {"up", "left", "right"},
                "Right": {"up", "left", "right"},
                "LL": {"up", "phys"},
                "LR": {"up", "phys"},
                "RL": {"up", "phys"},
                "RR": {"up", "phys"},
            },
        ),
    ],
)
def test_templates_expose_expected_index_sets_per_tensor(
    template_name: str,
    expected_tensors: dict[str, set[str]],
) -> None:
    spec = build_template_spec(template_name)
    actual_tensors = {
        tensor.name: {index.name for index in tensor.indices} for tensor in spec.tensors
    }

    assert actual_tensors == expected_tensors


def test_build_template_spec_rejects_unknown_template() -> None:
    with pytest.raises(ValueError, match="Unknown template"):
        build_template_spec("unknown-template")


def test_mps_template_accepts_custom_length_and_dimensions() -> None:
    spec = build_template_spec(
        "mps",
        TemplateParameters(graph_size=5, bond_dimension=7, physical_dimension=11),
    )

    assert spec.name == "MPS (5 sites)"
    assert len(spec.tensors) == 5
    assert len(spec.edges) == 4
    assert [tensor.name for tensor in spec.tensors] == ["A1", "A2", "A3", "A4", "A5"]
    assert {
        index.dimension
        for tensor in spec.tensors
        for index in tensor.indices
        if index.name in {"left", "right"}
    } == {7}
    assert {
        index.dimension
        for tensor in spec.tensors
        for index in tensor.indices
        if index.name == "phys"
    } == {11}


def test_peps_template_accepts_custom_side_length_and_dimensions() -> None:
    spec = build_template_spec(
        "peps_2x2",
        TemplateParameters(graph_size=3, bond_dimension=5, physical_dimension=2),
    )

    center_tensor = next(tensor for tensor in spec.tensors if tensor.name == "B2")

    assert spec.name == "PEPS 3x3"
    assert len(spec.tensors) == 9
    assert len(spec.edges) == 12
    assert {index.name for index in center_tensor.indices} == {
        "left",
        "right",
        "up",
        "down",
        "phys",
    }
    assert {
        index.dimension
        for tensor in spec.tensors
        for index in tensor.indices
        if index.name in {"left", "right", "up", "down"}
    } == {5}
    assert {
        index.dimension
        for tensor in spec.tensors
        for index in tensor.indices
        if index.name == "phys"
    } == {2}


def test_mera_template_accepts_custom_depth_and_dimensions() -> None:
    spec = build_template_spec(
        "mera",
        TemplateParameters(graph_size=4, bond_dimension=6, physical_dimension=3),
    )
    bottom_layer = [tensor for tensor in spec.tensors if tensor.name.startswith("L4-")]

    assert spec.name == "MERA depth 4"
    assert len(spec.tensors) == 10
    assert len(spec.edges) == 12
    assert len(bottom_layer) == 4
    assert all(
        any(index.name == "phys" for index in tensor.indices) for tensor in bottom_layer
    )
    assert {
        index.dimension
        for tensor in spec.tensors
        for index in tensor.indices
        if index.name != "phys"
    } == {6}
    assert {
        index.dimension
        for tensor in bottom_layer
        for index in tensor.indices
        if index.name == "phys"
    } == {3}


def test_binary_tree_template_accepts_custom_depth_and_dimensions() -> None:
    spec = build_template_spec(
        "binary_tree",
        TemplateParameters(graph_size=4, bond_dimension=8, physical_dimension=5),
    )
    leaves = [tensor for tensor in spec.tensors if tensor.name.startswith("L4-")]

    assert spec.name == "Binary Tree depth 4"
    assert len(spec.tensors) == 15
    assert len(spec.edges) == 14
    assert len(leaves) == 8
    assert all(
        any(index.name == "phys" for index in tensor.indices) for tensor in leaves
    )
    assert {
        index.dimension
        for tensor in spec.tensors
        for index in tensor.indices
        if index.name != "phys"
    } == {8}
    assert {
        index.dimension
        for tensor in leaves
        for index in tensor.indices
        if index.name == "phys"
    } == {5}


def test_ttn_template_accepts_custom_depth_and_dimensions() -> None:
    spec = build_template_spec(
        "ttn",
        TemplateParameters(graph_size=4, bond_dimension=6, physical_dimension=3),
    )
    leaves = [tensor for tensor in spec.tensors if tensor.name.startswith("L4-")]

    assert spec.name == "TTN depth 4"
    assert len(spec.tensors) == 15
    assert len(spec.edges) == 14
    assert len(leaves) == 8
    assert all(
        any(index.name == "phys" for index in tensor.indices) for tensor in leaves
    )
    assert {
        index.dimension
        for tensor in spec.tensors
        for index in tensor.indices
        if index.name != "phys"
    } == {6}
    assert {
        index.dimension
        for tensor in leaves
        for index in tensor.indices
        if index.name == "phys"
    } == {3}


def test_pepo_template_accepts_custom_side_length_and_dimensions() -> None:
    spec = build_template_spec(
        "pepo",
        TemplateParameters(graph_size=3, bond_dimension=5, physical_dimension=2),
    )
    center_tensor = next(tensor for tensor in spec.tensors if tensor.name == "B2")

    assert spec.name == "PEPO 3x3"
    assert len(spec.tensors) == 9
    assert len(spec.edges) == 12
    assert {index.name for index in center_tensor.indices} == {
        "left",
        "right",
        "up",
        "down",
        "bra",
        "ket",
    }
    assert {
        index.dimension
        for tensor in spec.tensors
        for index in tensor.indices
        if index.name in {"left", "right", "up", "down"}
    } == {5}
    assert {
        index.dimension
        for tensor in spec.tensors
        for index in tensor.indices
        if index.name in {"bra", "ket"}
    } == {2}


def test_heisenberg_mps_template_uses_guided_metadata_without_lint_warnings() -> None:
    spec = build_template_spec("heisenberg_mps")
    tensor_metadata = [tensor.metadata for tensor in spec.tensors]
    index_metadata = [
        index.metadata for tensor in spec.tensors for index in tensor.indices
    ]

    assert spec.name == "Heisenberg MPS"
    assert len(spec.tensors) == 4
    assert len(spec.edges) == 3
    assert all(metadata.get("role") == "state" for metadata in tensor_metadata)
    assert all(
        "heisenberg" in str(metadata.get("tags", "")) for metadata in tensor_metadata
    )
    assert {str(metadata.get("leg_kind")) for metadata in index_metadata} <= {
        "bond",
        "physical",
    }
    assert lint_spec(spec).issues == []


def test_ising_mps_template_uses_state_metadata_without_lint_warnings() -> None:
    spec = build_template_spec("ising_mps")

    assert spec.name == "Ising MPS"
    assert len(spec.tensors) == 4
    assert len(spec.edges) == 3
    assert all(tensor.metadata.get("role") == "state" for tensor in spec.tensors)
    assert all(
        "ising" in str(tensor.metadata.get("tags", "")) for tensor in spec.tensors
    )
    assert {
        str(index.metadata.get("leg_kind"))
        for tensor in spec.tensors
        for index in tensor.indices
    } <= {"bond", "physical"}
    assert lint_spec(spec).issues == []


def test_transverse_ising_mpo_template_uses_operator_metadata() -> None:
    spec = build_template_spec(
        "transverse_ising_mpo",
        TemplateParameters(graph_size=5, bond_dimension=4, physical_dimension=2),
    )

    assert spec.name == "Transverse Ising MPO (5 sites)"
    assert len(spec.tensors) == 5
    assert len(spec.edges) == 4
    assert all(tensor.name.startswith("W") for tensor in spec.tensors)
    assert all(tensor.metadata.get("role") == "operator" for tensor in spec.tensors)
    assert {
        index.dimension
        for tensor in spec.tensors
        for index in tensor.indices
        if index.name in {"left", "right"}
    } == {4}
    assert {
        str(index.metadata.get("leg_kind"))
        for tensor in spec.tensors
        for index in tensor.indices
    } <= {"bond", "physical"}


def test_tebd_gate_layer_template_adds_even_two_site_gates() -> None:
    spec = build_template_spec(
        "tebd_gate_layer",
        TemplateParameters(graph_size=5, bond_dimension=7, physical_dimension=3),
    )
    gate_tensors = [
        tensor for tensor in spec.tensors if tensor.metadata.get("role") == "gate"
    ]

    assert spec.name == "TEBD Gate Layer (5 sites)"
    assert len(spec.tensors) == 7
    assert len(spec.edges) == 8
    assert [tensor.name for tensor in gate_tensors] == ["G1-2", "G3-4"]
    assert all(tensor.shape == (3, 3, 3, 3) for tensor in gate_tensors)
    assert {
        index.dimension
        for tensor in spec.tensors
        for index in tensor.indices
        if index.name in {"left", "right"}
    } == {7}

from __future__ import annotations

import importlib

import pytest

from tensor_network_editor.internal.models._model_tensor_data import TensorDataMode
from tensor_network_editor.internal.templates._template_builders import (
    _build_linear_chain_template,
    build_template,
)
from tensor_network_editor.internal.templates._template_catalog import (
    _reset_template_registry_for_tests,
    get_template_builder,
    get_template_definition,
    list_template_names,
    serialize_template_definitions,
)
from tensor_network_editor.internal.templates._templates import (
    TemplateParameters,
    parse_template_parameters,
)


def test_template_catalog_internal_exposes_same_public_metadata() -> None:
    names = list_template_names()
    definitions = serialize_template_definitions()
    mera_definition = get_template_definition("mera")

    assert names == [
        "mps",
        "mpo",
        "peps_2x2",
        "mera",
        "ttn",
        "pepo",
        "tebd_gate_layer",
    ]
    assert list(definitions) == names
    assert mera_definition.display_name == "MERA"
    assert mera_definition.graph_size_label == "Depth"
    assert definitions["mps"]["parameter_fields"] == [
        {
            "name": "graph_size",
            "label": "Graph size (Sites)",
            "kind": "integer",
            "default": 4,
            "minimum": 2,
        },
        {
            "name": "bond_dimension",
            "label": "Bond dimension",
            "kind": "integer",
            "default": 3,
            "minimum": 1,
        },
        {
            "name": "physical_dimension",
            "label": "Physical dimension",
            "kind": "integer",
            "default": 2,
            "minimum": 1,
        },
        {
            "name": "boundary_condition",
            "label": "Boundary condition",
            "kind": "choice",
            "default": "open",
            "options": [
                {"value": "open", "label": "Open"},
                {"value": "periodic", "label": "Periodic"},
            ],
        },
        {
            "name": "symmetry",
            "label": "Symmetry",
            "kind": "choice",
            "default": "none",
            "options": [
                {"value": "none", "label": "None"},
                {"value": "u1", "label": "U1"},
                {"value": "z2", "label": "Z2"},
            ],
        },
        {
            "name": "initial_state",
            "label": "Initial state",
            "kind": "choice",
            "default": "zeros",
            "options": [
                {"value": "zeros", "label": "Zeros"},
                {"value": "random", "label": "Random"},
                {"value": "all_up", "label": "All up"},
                {"value": "all_down", "label": "All down"},
                {"value": "neel", "label": "Neel"},
            ],
        },
    ]


def test_template_builders_internal_dispatches_to_specific_builder() -> None:
    spec = build_template(
        "mps",
        TemplateParameters(
            graph_size=5,
            bond_dimension=7,
            physical_dimension=11,
        ),
    )

    assert spec.name == "MPS (5 sites)"
    assert len(spec.tensors) == 5


def test_template_builder_facade_reexports_family_modules() -> None:
    try:
        linear_module = importlib.import_module(
            "tensor_network_editor.internal.templates._template_builders_linear"
        )
        grid_module = importlib.import_module(
            "tensor_network_editor.internal.templates._template_builders_grid"
        )
        tree_module = importlib.import_module(
            "tensor_network_editor.internal.templates._template_builders_tree"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"Expected split template-builder modules to exist: {exc}")

    _reset_template_registry_for_tests()

    assert _build_linear_chain_template is linear_module._build_linear_chain_template
    assert get_template_builder("mps").__module__ == linear_module.__name__
    assert get_template_builder("peps_2x2").__module__ == grid_module.__name__
    assert get_template_builder("mera").__module__ == tree_module.__name__


def test_template_builder_common_module_exposes_shared_primitives() -> None:
    try:
        common_module = importlib.import_module(
            "tensor_network_editor.internal.templates._template_builders_common"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"Expected shared template-builder primitives module: {exc}")

    left_tensor = common_module._make_tensor(
        "tensor_left",
        "Left",
        10.0,
        20.0,
        [("right", 3, (58.0, 0.0))],
    )
    right_tensor = common_module._make_tensor(
        "tensor_right",
        "Right",
        40.0,
        20.0,
        [("left", 3, (-58.0, 0.0))],
    )
    edge = common_module._make_edge(
        "edge_left_right",
        left_tensor,
        "right",
        right_tensor,
        "left",
    )

    assert left_tensor.indices[0].id == "tensor_left_right"
    assert right_tensor.indices[0].id == "tensor_right_left"
    assert edge.left.tensor_id == "tensor_left"
    assert edge.left.index_id == "tensor_left_right"
    assert edge.right.tensor_id == "tensor_right"
    assert edge.right.index_id == "tensor_right_left"


def test_linear_chain_template_helper_reuses_catalog_metadata() -> None:
    spec = _build_linear_chain_template(
        "mpo",
        TemplateParameters(
            graph_size=5,
            bond_dimension=7,
            physical_dimension=11,
        ),
        tensor_name_prefix="W",
        spacing=330.0,
        site_index_builder=lambda site_index, length, parameters: [
            *(
                [("left", parameters.bond_dimension, (-58.0, 0.0))]
                if site_index > 0
                else []
            ),
            *(
                [("right", parameters.bond_dimension, (58.0, 0.0))]
                if site_index < length - 1
                else []
            ),
            ("bra", parameters.physical_dimension, (0.0, -28.0)),
            ("ket", parameters.physical_dimension, (0.0, 28.0)),
        ],
    )

    assert spec.name == "MPO (5 sites)"
    assert [tensor.name for tensor in spec.tensors] == ["W1", "W2", "W3", "W4", "W5"]
    assert len(spec.edges) == 4


def test_parse_template_parameters_accepts_new_mps_options() -> None:
    parameters = parse_template_parameters(
        "mps",
        {
            "graph_size": 6,
            "bond_dimension": 4,
            "physical_dimension": 2,
            "boundary_condition": "periodic",
            "symmetry": "z2",
            "initial_state": "neel",
        },
    )

    assert parameters == TemplateParameters(
        graph_size=6,
        bond_dimension=4,
        physical_dimension=2,
        boundary_condition="periodic",
        symmetry="z2",
        initial_state="neel",
    )


def test_parse_template_parameters_accepts_new_mpo_and_ttn_options() -> None:
    mpo_parameters = parse_template_parameters(
        "mpo",
        {
            "graph_size": 6,
            "bond_dimension": 4,
            "physical_dimension": 2,
            "boundary_condition": "periodic",
            "j": 1.5,
            "h": 0.25,
        },
    )
    ttn_parameters = parse_template_parameters(
        "ttn",
        {
            "depth": 4,
            "bond_dimension": 6,
            "physical_dimension": 3,
            "leaf_physical_legs": False,
            "root_open_leg": True,
            "isometric": True,
        },
    )

    assert mpo_parameters == TemplateParameters(
        graph_size=6,
        bond_dimension=4,
        physical_dimension=2,
        boundary_condition="periodic",
        j=1.5,
        h=0.25,
    )
    assert ttn_parameters == TemplateParameters(
        depth=4,
        bond_dimension=6,
        physical_dimension=3,
        leaf_physical_legs=False,
        root_open_leg=True,
        isometric=True,
    )


def test_mps_periodic_neel_builder_embeds_requested_configuration() -> None:
    spec = build_template(
        "mps",
        TemplateParameters(
            graph_size=4,
            bond_dimension=3,
            physical_dimension=2,
            boundary_condition="periodic",
            symmetry="z2",
            initial_state="neel",
        ),
    )

    assert spec.metadata["template_name"] == "mps"
    assert spec.metadata["boundary_condition"] == "periodic"
    assert spec.metadata["symmetry"] == "z2"
    assert spec.metadata["initial_state"] == "neel"
    assert len(spec.edges) == 4
    assert {index.name for index in spec.tensors[0].indices} == {
        "left",
        "right",
        "phys",
    }
    assert spec.tensors[0].tensor_data is not None
    assert spec.tensors[0].tensor_data.mode is TensorDataMode.LITERAL
    assert spec.tensors[1].tensor_data is not None
    assert spec.tensors[1].tensor_data.mode is TensorDataMode.LITERAL
    assert spec.tensors[0].metadata["symmetry"] == "z2"
    assert spec.tensors[0].indices[0].metadata["symmetry"] == "z2"


def test_mpo_and_ttn_builders_embed_requested_configuration() -> None:
    mpo_spec = build_template(
        "mpo",
        TemplateParameters(
            graph_size=4,
            bond_dimension=3,
            physical_dimension=2,
            boundary_condition="periodic",
            j=1.5,
            h=0.25,
        ),
    )
    ttn_spec = build_template(
        "ttn",
        TemplateParameters(
            depth=4,
            bond_dimension=3,
            physical_dimension=2,
            leaf_physical_legs=False,
            root_open_leg=True,
            isometric=True,
        ),
    )

    assert mpo_spec.metadata["boundary_condition"] == "periodic"
    assert mpo_spec.metadata["j"] == pytest.approx(1.5)
    assert mpo_spec.metadata["h"] == pytest.approx(0.25)
    assert len(mpo_spec.edges) == 4
    assert ttn_spec.metadata["depth"] == 4
    assert ttn_spec.metadata["leaf_physical_legs"] is False
    assert ttn_spec.metadata["root_open_leg"] is True
    assert ttn_spec.metadata["isometric"] is True


def test_parse_template_parameters_rejects_spin_presets_for_non_spin_dimension() -> (
    None
):
    with pytest.raises(
        ValueError,
        match="physical_dimension",
    ):
        parse_template_parameters(
            "mps",
            {
                "graph_size": 4,
                "bond_dimension": 3,
                "physical_dimension": 3,
                "initial_state": "all_up",
            },
        )


def test_parse_template_parameters_rejects_removed_template_names() -> None:
    with pytest.raises(ValueError, match="Unknown template"):
        parse_template_parameters("binary_tree")

    with pytest.raises(ValueError, match="Unknown template"):
        parse_template_parameters("transverse_ising_mpo")

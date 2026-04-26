from __future__ import annotations

from tensor_network_editor.internal.templates._template_builders import (
    _build_linear_chain_template,
    build_template,
)
from tensor_network_editor.internal.templates._template_catalog import (
    get_template_definition,
    list_template_names,
    serialize_template_definitions,
)
from tensor_network_editor.internal.templates._templates import TemplateParameters


def test_template_catalog_internal_exposes_same_public_metadata() -> None:
    names = list_template_names()
    definitions = serialize_template_definitions()
    mera_definition = get_template_definition("mera")

    assert names == [
        "mps",
        "mpo",
        "peps_2x2",
        "mera",
        "binary_tree",
        "ttn",
        "pepo",
        "heisenberg_mps",
    ]
    assert list(definitions) == names
    assert mera_definition.display_name == "MERA"
    assert mera_definition.graph_size_label == "Depth"


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

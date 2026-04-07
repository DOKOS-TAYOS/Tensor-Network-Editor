from __future__ import annotations

from tensor_network_editor._template_builders import build_template
from tensor_network_editor._template_catalog import (
    get_template_definition,
    list_template_names,
    serialize_template_definitions,
)
from tensor_network_editor._templates import TemplateParameters


def test_template_catalog_internal_exposes_same_public_metadata() -> None:
    names = list_template_names()
    definitions = serialize_template_definitions()
    mera_definition = get_template_definition("mera")

    assert names == ["mps", "mpo", "peps_2x2", "mera", "binary_tree"]
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

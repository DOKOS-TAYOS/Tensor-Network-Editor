from __future__ import annotations

from ._template_builders import build_template
from ._template_catalog import (
    TemplateDefinition,
    TemplateParameters,
    get_template_definition,
    list_template_names,
    parse_template_integer,
    serialize_template_definitions,
    validate_template_parameters,
)
from .models import NetworkSpec


def parse_template_parameters(
    template_name: str, raw_parameters: object | None = None
) -> TemplateParameters:
    definition = get_template_definition(template_name)
    defaults = definition.defaults
    if raw_parameters is None:
        return defaults
    if not isinstance(raw_parameters, dict):
        raise ValueError("Template 'parameters' payload must be an object.")
    return validate_template_parameters(
        template_name,
        TemplateParameters(
            graph_size=parse_template_integer(
                raw_parameters.get("graph_size"),
                field_name="graph_size",
                default=defaults.graph_size,
                minimum=definition.minimum_graph_size,
            ),
            bond_dimension=parse_template_integer(
                raw_parameters.get("bond_dimension"),
                field_name="bond_dimension",
                default=defaults.bond_dimension,
                minimum=definition.minimum_bond_dimension,
            ),
            physical_dimension=parse_template_integer(
                raw_parameters.get("physical_dimension"),
                field_name="physical_dimension",
                default=defaults.physical_dimension,
                minimum=definition.minimum_physical_dimension,
            ),
        ),
    )


def build_template_spec(
    template_name: str, parameters: TemplateParameters | None = None
) -> NetworkSpec:
    return build_template(template_name, parameters)


__all__ = [
    "TemplateDefinition",
    "TemplateParameters",
    "build_template_spec",
    "list_template_names",
    "parse_template_parameters",
    "serialize_template_definitions",
]

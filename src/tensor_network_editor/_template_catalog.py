"""Catalog metadata for the built-in network templates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateParameters:
    """Normalized parameters accepted by a built-in template."""

    graph_size: int
    bond_dimension: int
    physical_dimension: int


@dataclass(frozen=True)
class TemplateDefinition:
    """Metadata shown for one built-in template option."""

    name: str
    display_name: str
    graph_size_label: str
    defaults: TemplateParameters
    minimum_graph_size: int = 2
    minimum_bond_dimension: int = 1
    minimum_physical_dimension: int = 1

    def to_dict(self) -> dict[str, object]:
        """Serialize the template definition for frontend bootstrap payloads."""
        return {
            "display_name": self.display_name,
            "graph_size_label": self.graph_size_label,
            "defaults": {
                "graph_size": self.defaults.graph_size,
                "bond_dimension": self.defaults.bond_dimension,
                "physical_dimension": self.defaults.physical_dimension,
            },
            "minimums": {
                "graph_size": self.minimum_graph_size,
                "bond_dimension": self.minimum_bond_dimension,
                "physical_dimension": self.minimum_physical_dimension,
            },
        }


TEMPLATE_DEFINITIONS: dict[str, TemplateDefinition] = {
    "mps": TemplateDefinition(
        name="mps",
        display_name="MPS",
        graph_size_label="Sites",
        defaults=TemplateParameters(
            graph_size=4,
            bond_dimension=3,
            physical_dimension=2,
        ),
    ),
    "mpo": TemplateDefinition(
        name="mpo",
        display_name="MPO",
        graph_size_label="Sites",
        defaults=TemplateParameters(
            graph_size=4,
            bond_dimension=3,
            physical_dimension=2,
        ),
    ),
    "peps_2x2": TemplateDefinition(
        name="peps_2x2",
        display_name="PEPS",
        graph_size_label="Side length",
        defaults=TemplateParameters(
            graph_size=2,
            bond_dimension=3,
            physical_dimension=2,
        ),
    ),
    "mera": TemplateDefinition(
        name="mera",
        display_name="MERA",
        graph_size_label="Depth",
        defaults=TemplateParameters(
            graph_size=3,
            bond_dimension=3,
            physical_dimension=2,
        ),
    ),
    "binary_tree": TemplateDefinition(
        name="binary_tree",
        display_name="Binary Tree",
        graph_size_label="Depth",
        defaults=TemplateParameters(
            graph_size=3,
            bond_dimension=3,
            physical_dimension=2,
        ),
    ),
}
TEMPLATE_NAMES = list(TEMPLATE_DEFINITIONS)


def list_template_names() -> list[str]:
    """Return the public template names in display order."""
    return list(TEMPLATE_NAMES)


def serialize_template_definitions() -> dict[str, dict[str, object]]:
    """Serialize all template definitions for the browser bootstrap payload."""
    return {
        template_name: definition.to_dict()
        for template_name, definition in TEMPLATE_DEFINITIONS.items()
    }


def get_template_definition(template_name: str) -> TemplateDefinition:
    """Return the catalog entry for ``template_name``."""
    try:
        return TEMPLATE_DEFINITIONS[template_name]
    except KeyError as exc:
        raise ValueError(f"Unknown template '{template_name}'.") from exc


def parse_template_integer(
    value: object, *, field_name: str, default: int, minimum: int
) -> int:
    """Validate one integer template parameter and apply its default."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Template parameter '{field_name}' must be an integer.")
    if value < minimum:
        raise ValueError(
            f"Template parameter '{field_name}' must be greater than or equal to {minimum}."
        )
    return value


def validate_template_parameters(
    template_name: str, parameters: TemplateParameters
) -> TemplateParameters:
    """Normalize template parameters against the rules for ``template_name``."""
    definition = get_template_definition(template_name)
    return TemplateParameters(
        graph_size=parse_template_integer(
            parameters.graph_size,
            field_name="graph_size",
            default=definition.defaults.graph_size,
            minimum=definition.minimum_graph_size,
        ),
        bond_dimension=parse_template_integer(
            parameters.bond_dimension,
            field_name="bond_dimension",
            default=definition.defaults.bond_dimension,
            minimum=definition.minimum_bond_dimension,
        ),
        physical_dimension=parse_template_integer(
            parameters.physical_dimension,
            field_name="physical_dimension",
            default=definition.defaults.physical_dimension,
            minimum=definition.minimum_physical_dimension,
        ),
    )

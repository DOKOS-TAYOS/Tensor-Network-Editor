"""Catalog metadata for the built-in and registered network templates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ...types import JSONValue

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...models import NetworkSpec


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
    supports_parameters: bool = True
    source: Literal["project", "global"] = "global"

    def to_dict(self) -> dict[str, JSONValue]:
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
            "supports_parameters": self.supports_parameters,
            "source": self.source,
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
            graph_size=3,
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
    "ttn": TemplateDefinition(
        name="ttn",
        display_name="TTN",
        graph_size_label="Depth",
        defaults=TemplateParameters(
            graph_size=3,
            bond_dimension=3,
            physical_dimension=2,
        ),
    ),
    "pepo": TemplateDefinition(
        name="pepo",
        display_name="PEPO",
        graph_size_label="Side length",
        defaults=TemplateParameters(
            graph_size=3,
            bond_dimension=3,
            physical_dimension=2,
        ),
    ),
    "heisenberg_mps": TemplateDefinition(
        name="heisenberg_mps",
        display_name="Heisenberg MPS",
        graph_size_label="Sites",
        defaults=TemplateParameters(
            graph_size=4,
            bond_dimension=3,
            physical_dimension=2,
        ),
    ),
}

_TEMPLATE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_REGISTERED_TEMPLATE_DEFINITIONS: dict[str, TemplateDefinition] = {}
_REGISTERED_TEMPLATE_BUILDERS: dict[
    str, Callable[[TemplateParameters], NetworkSpec]
] = {}


def _ensure_template_registry_seeded() -> None:
    """Load the built-in template registrations on first use."""
    if _REGISTERED_TEMPLATE_DEFINITIONS:
        return
    from ._template_builders import register_builtin_templates

    register_builtin_templates()


def _validate_template_name(template_name: str) -> str:
    """Validate and normalize one template registration name."""
    normalized_name = str(template_name).strip()
    if not _TEMPLATE_NAME_PATTERN.fullmatch(normalized_name):
        raise ValueError(
            "Template names must start with a lowercase letter and contain only lowercase letters, digits, and underscores."
        )
    return normalized_name


def register_template(
    template_name: str,
    definition: TemplateDefinition,
    builder: Callable[[TemplateParameters], NetworkSpec],
    *,
    overwrite: bool = False,
) -> None:
    """Register one template definition and its builder."""
    normalized_name = _validate_template_name(template_name)
    if definition.name != normalized_name:
        raise ValueError(
            f"Template definition name '{definition.name}' does not match registration name '{normalized_name}'."
        )
    if normalized_name in _REGISTERED_TEMPLATE_DEFINITIONS and not overwrite:
        raise ValueError(f"Template '{normalized_name}' is already registered.")
    _REGISTERED_TEMPLATE_DEFINITIONS[normalized_name] = definition
    _REGISTERED_TEMPLATE_BUILDERS[normalized_name] = builder


def build_static_template_definition(
    template_name: str,
    display_name: str,
    spec: NetworkSpec,
    *,
    source: Literal["project", "global"] = "global",
) -> TemplateDefinition:
    """Build one non-parametric template definition for a fixed spec."""
    return TemplateDefinition(
        name=template_name,
        display_name=display_name.strip(),
        graph_size_label="Tensors",
        defaults=TemplateParameters(
            graph_size=max(1, len(spec.tensors)),
            bond_dimension=1,
            physical_dimension=1,
        ),
        minimum_graph_size=1,
        supports_parameters=False,
        source=source,
    )


def register_static_template(
    template_name: str,
    display_name: str,
    spec: NetworkSpec,
    *,
    overwrite: bool = False,
) -> None:
    """Register one fixed ``NetworkSpec`` as a reusable static template."""
    from copy import deepcopy

    from ...validation import ensure_valid_spec

    normalized_name = _validate_template_name(template_name)
    normalized_spec = ensure_valid_spec(deepcopy(spec))
    definition = build_static_template_definition(
        normalized_name,
        display_name,
        normalized_spec,
    )

    def build_static_template(_parameters: TemplateParameters) -> NetworkSpec:
        """Return a detached copy of the validated static template spec."""
        return deepcopy(normalized_spec)

    register_template(
        normalized_name,
        definition,
        build_static_template,
        overwrite=overwrite,
    )


def get_template_builder(
    template_name: str,
) -> Callable[[TemplateParameters], NetworkSpec]:
    """Return the registered builder callable for ``template_name``."""
    try:
        normalized_name = _validate_template_name(template_name)
    except ValueError as exc:
        raise ValueError(f"Unknown template '{template_name}'.") from exc
    _ensure_template_registry_seeded()
    try:
        return _REGISTERED_TEMPLATE_BUILDERS[normalized_name]
    except KeyError as exc:
        raise ValueError(f"Unknown template '{template_name}'.") from exc


def list_template_names() -> list[str]:
    """Return the public template names in display order."""
    _ensure_template_registry_seeded()
    return list(_REGISTERED_TEMPLATE_DEFINITIONS)


def serialize_template_definitions() -> dict[str, dict[str, JSONValue]]:
    """Serialize all template definitions for the browser bootstrap payload."""
    _ensure_template_registry_seeded()
    return {
        template_name: definition.to_dict()
        for template_name, definition in _REGISTERED_TEMPLATE_DEFINITIONS.items()
    }


def get_template_definition(template_name: str) -> TemplateDefinition:
    """Return the catalog entry for ``template_name``."""
    try:
        normalized_name = _validate_template_name(template_name)
    except ValueError as exc:
        raise ValueError(f"Unknown template '{template_name}'.") from exc
    _ensure_template_registry_seeded()
    try:
        return _REGISTERED_TEMPLATE_DEFINITIONS[normalized_name]
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


def _reset_template_registry_for_tests() -> None:
    """Reset the live template registry to its built-in state."""
    _REGISTERED_TEMPLATE_DEFINITIONS.clear()
    _REGISTERED_TEMPLATE_BUILDERS.clear()
    from ._template_builders import register_builtin_templates

    register_builtin_templates()

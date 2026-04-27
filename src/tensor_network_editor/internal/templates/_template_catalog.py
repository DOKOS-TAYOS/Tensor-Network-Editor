"""Catalog metadata for the built-in and registered network templates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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
    boundary_condition: Literal["open", "periodic"] = "open"
    symmetry: Literal["none", "u1", "z2"] = "none"
    initial_state: Literal[
        "zeros",
        "random",
        "all_up",
        "all_down",
        "neel",
    ] = "zeros"


@dataclass(frozen=True)
class TemplateParameterOptionDefinition:
    """One select-option entry for a template parameter control."""

    value: str
    label: str

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the option metadata for the frontend bootstrap payload."""
        return {
            "value": self.value,
            "label": self.label,
        }


@dataclass(frozen=True)
class TemplateParameterFieldDefinition:
    """Describe one configurable template parameter."""

    name: str
    label: str
    kind: Literal["integer", "choice"]
    default: int | str
    minimum: int | None = None
    options: tuple[TemplateParameterOptionDefinition, ...] = ()

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize one parameter definition for frontend rendering."""
        payload: dict[str, JSONValue] = {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
        }
        if self.minimum is not None:
            payload["minimum"] = self.minimum
        if self.options:
            payload["options"] = [option.to_dict() for option in self.options]
        return payload


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
    parameter_fields: tuple[TemplateParameterFieldDefinition, ...] = field(
        default_factory=tuple
    )

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the template definition for frontend bootstrap payloads."""
        defaults, minimums = _serialize_template_parameter_payload(
            self,
        )
        return {
            "display_name": self.display_name,
            "graph_size_label": self.graph_size_label,
            "defaults": defaults,
            "minimums": minimums,
            "parameter_fields": [
                parameter_field.to_dict() for parameter_field in self.parameter_fields
            ],
            "supports_parameters": self.supports_parameters,
            "source": self.source,
        }


def _build_standard_parameter_fields(
    *,
    graph_size_label: str,
    defaults: TemplateParameters,
    minimum_graph_size: int = 2,
    minimum_bond_dimension: int = 1,
    minimum_physical_dimension: int = 1,
) -> tuple[TemplateParameterFieldDefinition, ...]:
    """Return the common numeric template-parameter definitions."""
    return (
        TemplateParameterFieldDefinition(
            name="graph_size",
            label=f"Graph size ({graph_size_label})",
            kind="integer",
            default=defaults.graph_size,
            minimum=minimum_graph_size,
        ),
        TemplateParameterFieldDefinition(
            name="bond_dimension",
            label="Bond dimension",
            kind="integer",
            default=defaults.bond_dimension,
            minimum=minimum_bond_dimension,
        ),
        TemplateParameterFieldDefinition(
            name="physical_dimension",
            label="Physical dimension",
            kind="integer",
            default=defaults.physical_dimension,
            minimum=minimum_physical_dimension,
        ),
    )


def _build_mps_parameter_fields(
    defaults: TemplateParameters,
) -> tuple[TemplateParameterFieldDefinition, ...]:
    """Return the extended MPS parameter controls shown in the editor."""
    return (
        *_build_standard_parameter_fields(
            graph_size_label="Sites",
            defaults=defaults,
        ),
        TemplateParameterFieldDefinition(
            name="boundary_condition",
            label="Boundary condition",
            kind="choice",
            default=defaults.boundary_condition,
            options=(
                TemplateParameterOptionDefinition(value="open", label="Open"),
                TemplateParameterOptionDefinition(
                    value="periodic",
                    label="Periodic",
                ),
            ),
        ),
        TemplateParameterFieldDefinition(
            name="symmetry",
            label="Symmetry",
            kind="choice",
            default=defaults.symmetry,
            options=(
                TemplateParameterOptionDefinition(value="none", label="None"),
                TemplateParameterOptionDefinition(value="u1", label="U1"),
                TemplateParameterOptionDefinition(value="z2", label="Z2"),
            ),
        ),
        TemplateParameterFieldDefinition(
            name="initial_state",
            label="Initial state",
            kind="choice",
            default=defaults.initial_state,
            options=(
                TemplateParameterOptionDefinition(value="zeros", label="Zeros"),
                TemplateParameterOptionDefinition(value="random", label="Random"),
                TemplateParameterOptionDefinition(value="all_up", label="All up"),
                TemplateParameterOptionDefinition(
                    value="all_down",
                    label="All down",
                ),
                TemplateParameterOptionDefinition(value="neel", label="Neel"),
            ),
        ),
    )


def _serialize_template_parameter_payload(
    definition: TemplateDefinition,
) -> tuple[dict[str, JSONValue], dict[str, JSONValue]]:
    """Return serialized defaults and minimums for one template definition."""
    parameter_fields = definition.parameter_fields or _build_standard_parameter_fields(
        graph_size_label=definition.graph_size_label,
        defaults=definition.defaults,
        minimum_graph_size=definition.minimum_graph_size,
        minimum_bond_dimension=definition.minimum_bond_dimension,
        minimum_physical_dimension=definition.minimum_physical_dimension,
    )
    defaults: dict[str, JSONValue] = {}
    minimums: dict[str, JSONValue] = {}
    for parameter_field in parameter_fields:
        defaults[parameter_field.name] = parameter_field.default
        if parameter_field.minimum is not None:
            minimums[parameter_field.name] = parameter_field.minimum
    return defaults, minimums


def _get_parameter_field_definition(
    definition: TemplateDefinition,
    field_name: str,
) -> TemplateParameterFieldDefinition | None:
    """Return the parameter metadata entry for one field when it exists."""
    for parameter_field in definition.parameter_fields:
        if parameter_field.name == field_name:
            return parameter_field
    return None


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
        parameter_fields=_build_mps_parameter_fields(
            TemplateParameters(
                graph_size=4,
                bond_dimension=3,
                physical_dimension=2,
            )
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
        parameter_fields=_build_standard_parameter_fields(
            graph_size_label="Sites",
            defaults=TemplateParameters(
                graph_size=4,
                bond_dimension=3,
                physical_dimension=2,
            ),
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
        parameter_fields=_build_standard_parameter_fields(
            graph_size_label="Side length",
            defaults=TemplateParameters(
                graph_size=3,
                bond_dimension=3,
                physical_dimension=2,
            ),
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
        parameter_fields=_build_standard_parameter_fields(
            graph_size_label="Depth",
            defaults=TemplateParameters(
                graph_size=3,
                bond_dimension=3,
                physical_dimension=2,
            ),
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
        parameter_fields=_build_standard_parameter_fields(
            graph_size_label="Depth",
            defaults=TemplateParameters(
                graph_size=3,
                bond_dimension=3,
                physical_dimension=2,
            ),
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
        parameter_fields=_build_standard_parameter_fields(
            graph_size_label="Depth",
            defaults=TemplateParameters(
                graph_size=3,
                bond_dimension=3,
                physical_dimension=2,
            ),
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
        parameter_fields=_build_standard_parameter_fields(
            graph_size_label="Side length",
            defaults=TemplateParameters(
                graph_size=3,
                bond_dimension=3,
                physical_dimension=2,
            ),
        ),
    ),
    "transverse_ising_mpo": TemplateDefinition(
        name="transverse_ising_mpo",
        display_name="Transverse Ising MPO",
        graph_size_label="Sites",
        defaults=TemplateParameters(
            graph_size=4,
            bond_dimension=3,
            physical_dimension=2,
        ),
        parameter_fields=_build_standard_parameter_fields(
            graph_size_label="Sites",
            defaults=TemplateParameters(
                graph_size=4,
                bond_dimension=3,
                physical_dimension=2,
            ),
        ),
    ),
    "tebd_gate_layer": TemplateDefinition(
        name="tebd_gate_layer",
        display_name="TEBD Gate Layer",
        graph_size_label="Sites",
        defaults=TemplateParameters(
            graph_size=4,
            bond_dimension=3,
            physical_dimension=2,
        ),
        parameter_fields=_build_standard_parameter_fields(
            graph_size_label="Sites",
            defaults=TemplateParameters(
                graph_size=4,
                bond_dimension=3,
                physical_dimension=2,
            ),
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
    boundary_condition_definition = _get_parameter_field_definition(
        definition,
        "boundary_condition",
    )
    symmetry_definition = _get_parameter_field_definition(
        definition,
        "symmetry",
    )
    initial_state_definition = _get_parameter_field_definition(
        definition,
        "initial_state",
    )
    boundary_condition = (
        parse_template_choice(
            parameters.boundary_condition,
            field_name="boundary_condition",
            default=definition.defaults.boundary_condition,
            choices=tuple(
                option.value for option in boundary_condition_definition.options
            ),
        )
        if boundary_condition_definition is not None
        else definition.defaults.boundary_condition
    )
    symmetry = (
        parse_template_choice(
            parameters.symmetry,
            field_name="symmetry",
            default=definition.defaults.symmetry,
            choices=tuple(option.value for option in symmetry_definition.options),
        )
        if symmetry_definition is not None
        else definition.defaults.symmetry
    )
    initial_state = (
        parse_template_choice(
            parameters.initial_state,
            field_name="initial_state",
            default=definition.defaults.initial_state,
            choices=tuple(option.value for option in initial_state_definition.options),
        )
        if initial_state_definition is not None
        else definition.defaults.initial_state
    )
    physical_dimension = parse_template_integer(
        parameters.physical_dimension,
        field_name="physical_dimension",
        default=definition.defaults.physical_dimension,
        minimum=definition.minimum_physical_dimension,
    )
    if initial_state in {"all_up", "all_down", "neel"} and physical_dimension != 2:
        raise ValueError(
            "Template parameter 'physical_dimension' must be 2 when "
            f"'initial_state' is '{initial_state}'."
        )
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
        physical_dimension=physical_dimension,
        boundary_condition=boundary_condition,
        symmetry=symmetry,
        initial_state=initial_state,
    )


def parse_template_choice(
    value: object,
    *,
    field_name: str,
    default: str,
    choices: tuple[str, ...],
) -> str:
    """Validate one string template parameter against an allowed choice list."""
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"Template parameter '{field_name}' must be a string.")
    normalized_value = value.strip().lower()
    if normalized_value not in choices:
        raise ValueError(
            f"Template parameter '{field_name}' must be one of: {', '.join(choices)}."
        )
    return normalized_value


def _reset_template_registry_for_tests() -> None:
    """Reset the live template registry to its built-in state."""
    _REGISTERED_TEMPLATE_DEFINITIONS.clear()
    _REGISTERED_TEMPLATE_BUILDERS.clear()
    from ._template_builders import register_builtin_templates

    register_builtin_templates()

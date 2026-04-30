"""Catalog metadata for the built-in and registered network templates."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Literal, cast

from ...types import JSONValue
from .._logging import log_branch, log_operation

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...models import NetworkSpec


@dataclass(frozen=True)
class TemplateParameters:
    """Normalized parameters accepted by built-in templates."""

    graph_size: int | None = None
    bond_dimension: int | None = None
    physical_dimension: int | None = None
    boundary_condition: Literal["open", "periodic"] = "open"
    symmetry: Literal["none", "u1", "z2"] = "none"
    initial_state: Literal[
        "zeros",
        "random",
        "all_up",
        "all_down",
        "neel",
    ] = "zeros"
    depth: int | None = None
    j: float | None = None
    h: float | None = None
    leaf_physical_legs: bool = True
    root_open_leg: bool = False
    isometric: bool = False


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
    kind: Literal["integer", "number", "boolean", "choice"]
    default: int | float | bool | str
    minimum: int | float | None = None
    options: tuple[TemplateParameterOptionDefinition, ...] = ()

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize one parameter definition for frontend rendering."""
        payload: dict[str, JSONValue] = {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": cast(JSONValue, self.default),
        }
        if self.minimum is not None:
            payload["minimum"] = cast(JSONValue, self.minimum)
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
        defaults, minimums = _serialize_template_parameter_payload(self)
        return {
            "display_name": self.display_name,
            "graph_size_label": self.graph_size_label,
            "defaults": defaults,
            "minimums": minimums,
            "parameter_fields": [
                parameter_field.to_dict()
                for parameter_field in _get_template_parameter_fields(self)
            ],
            "supports_parameters": self.supports_parameters,
            "source": self.source,
        }


def _build_standard_parameter_fields(
    *,
    size_field_name: str,
    size_field_label: str,
    defaults: TemplateParameters,
    minimum_graph_size: int = 2,
    minimum_bond_dimension: int = 1,
    minimum_physical_dimension: int = 1,
) -> tuple[TemplateParameterFieldDefinition, ...]:
    """Return the common numeric template-parameter definitions."""
    return (
        TemplateParameterFieldDefinition(
            name=size_field_name,
            label=size_field_label,
            kind="integer",
            default=_require_template_parameter_default(defaults, size_field_name, int),
            minimum=minimum_graph_size,
        ),
        TemplateParameterFieldDefinition(
            name="bond_dimension",
            label="Bond dimension",
            kind="integer",
            default=_require_template_parameter_default(
                defaults,
                "bond_dimension",
                int,
            ),
            minimum=minimum_bond_dimension,
        ),
        TemplateParameterFieldDefinition(
            name="physical_dimension",
            label="Physical dimension",
            kind="integer",
            default=_require_template_parameter_default(
                defaults,
                "physical_dimension",
                int,
            ),
            minimum=minimum_physical_dimension,
        ),
    )


def _build_mps_parameter_fields(
    defaults: TemplateParameters,
) -> tuple[TemplateParameterFieldDefinition, ...]:
    """Return the extended MPS parameter controls shown in the editor."""
    return (
        *_build_standard_parameter_fields(
            size_field_name="graph_size",
            size_field_label="Graph size (Sites)",
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


def _build_mpo_parameter_fields(
    defaults: TemplateParameters,
) -> tuple[TemplateParameterFieldDefinition, ...]:
    """Return the extended MPO parameter controls."""
    return (
        *_build_standard_parameter_fields(
            size_field_name="graph_size",
            size_field_label="Graph size (Sites)",
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
            name="j",
            label="J",
            kind="number",
            default=_require_template_parameter_default(defaults, "j", float),
        ),
        TemplateParameterFieldDefinition(
            name="h",
            label="h",
            kind="number",
            default=_require_template_parameter_default(defaults, "h", float),
        ),
    )


def _build_ttn_parameter_fields(
    defaults: TemplateParameters,
) -> tuple[TemplateParameterFieldDefinition, ...]:
    """Return the extended TTN parameter controls."""
    return (
        *_build_standard_parameter_fields(
            size_field_name="depth",
            size_field_label="Depth",
            defaults=defaults,
        ),
        TemplateParameterFieldDefinition(
            name="leaf_physical_legs",
            label="Leaf physical legs",
            kind="boolean",
            default=defaults.leaf_physical_legs,
        ),
        TemplateParameterFieldDefinition(
            name="root_open_leg",
            label="Root open leg",
            kind="boolean",
            default=defaults.root_open_leg,
        ),
        TemplateParameterFieldDefinition(
            name="isometric",
            label="Isometric",
            kind="boolean",
            default=defaults.isometric,
        ),
    )


def _require_template_parameter_default(
    defaults: TemplateParameters,
    field_name: str,
    expected_type: type[int] | type[float] | type[bool] | type[str],
) -> int | float | bool | str:
    """Return one non-null template default value with the expected type."""
    value = getattr(defaults, field_name)
    if value is None:
        raise ValueError(f"Template default '{field_name}' cannot be None.")
    if expected_type is float and isinstance(value, int | float):
        return float(value)
    if not isinstance(value, expected_type):
        raise ValueError(
            f"Template default '{field_name}' must be a {expected_type.__name__}."
        )
    return value


def _get_template_parameter_fields(
    definition: TemplateDefinition,
) -> tuple[TemplateParameterFieldDefinition, ...]:
    """Return the effective parameter fields for one template definition."""
    if definition.parameter_fields:
        return definition.parameter_fields
    return _build_standard_parameter_fields(
        size_field_name="graph_size",
        size_field_label=f"Graph size ({definition.graph_size_label})",
        defaults=definition.defaults,
        minimum_graph_size=definition.minimum_graph_size,
        minimum_bond_dimension=definition.minimum_bond_dimension,
        minimum_physical_dimension=definition.minimum_physical_dimension,
    )


def _serialize_template_parameter_payload(
    definition: TemplateDefinition,
) -> tuple[dict[str, JSONValue], dict[str, JSONValue]]:
    """Return serialized defaults and minimums for one template definition."""
    defaults: dict[str, JSONValue] = {}
    minimums: dict[str, JSONValue] = {}
    for parameter_field in _get_template_parameter_fields(definition):
        defaults[parameter_field.name] = cast(JSONValue, parameter_field.default)
        if parameter_field.minimum is not None:
            minimums[parameter_field.name] = cast(
                JSONValue,
                parameter_field.minimum,
            )
    return defaults, minimums


_MPS_DEFAULTS = TemplateParameters(
    graph_size=4,
    bond_dimension=3,
    physical_dimension=2,
)
_MPO_DEFAULTS = TemplateParameters(
    graph_size=4,
    bond_dimension=3,
    physical_dimension=2,
    boundary_condition="open",
    j=1.0,
    h=1.0,
)
_PEPS_DEFAULTS = TemplateParameters(
    graph_size=3,
    bond_dimension=3,
    physical_dimension=2,
)
_MERA_DEFAULTS = TemplateParameters(
    graph_size=3,
    bond_dimension=3,
    physical_dimension=2,
)
_TTN_DEFAULTS = TemplateParameters(
    depth=3,
    bond_dimension=3,
    physical_dimension=2,
    leaf_physical_legs=True,
    root_open_leg=False,
    isometric=False,
)
_PEPO_DEFAULTS = TemplateParameters(
    graph_size=3,
    bond_dimension=3,
    physical_dimension=2,
)
_TEBD_GATE_LAYER_DEFAULTS = TemplateParameters(
    graph_size=4,
    bond_dimension=3,
    physical_dimension=2,
)


TEMPLATE_DEFINITIONS: dict[str, TemplateDefinition] = {
    "mps": TemplateDefinition(
        name="mps",
        display_name="MPS",
        graph_size_label="Sites",
        defaults=_MPS_DEFAULTS,
        parameter_fields=_build_mps_parameter_fields(_MPS_DEFAULTS),
    ),
    "mpo": TemplateDefinition(
        name="mpo",
        display_name="MPO",
        graph_size_label="Sites",
        defaults=_MPO_DEFAULTS,
        parameter_fields=_build_mpo_parameter_fields(_MPO_DEFAULTS),
    ),
    "peps_2x2": TemplateDefinition(
        name="peps_2x2",
        display_name="PEPS",
        graph_size_label="Side length",
        defaults=_PEPS_DEFAULTS,
        parameter_fields=_build_standard_parameter_fields(
            size_field_name="graph_size",
            size_field_label="Graph size (Side length)",
            defaults=_PEPS_DEFAULTS,
        ),
    ),
    "mera": TemplateDefinition(
        name="mera",
        display_name="MERA",
        graph_size_label="Depth",
        defaults=_MERA_DEFAULTS,
        parameter_fields=_build_standard_parameter_fields(
            size_field_name="graph_size",
            size_field_label="Graph size (Depth)",
            defaults=_MERA_DEFAULTS,
        ),
    ),
    "ttn": TemplateDefinition(
        name="ttn",
        display_name="TTN",
        graph_size_label="Depth",
        defaults=_TTN_DEFAULTS,
        parameter_fields=_build_ttn_parameter_fields(_TTN_DEFAULTS),
    ),
    "pepo": TemplateDefinition(
        name="pepo",
        display_name="PEPO",
        graph_size_label="Side length",
        defaults=_PEPO_DEFAULTS,
        parameter_fields=_build_standard_parameter_fields(
            size_field_name="graph_size",
            size_field_label="Graph size (Side length)",
            defaults=_PEPO_DEFAULTS,
        ),
    ),
    "tebd_gate_layer": TemplateDefinition(
        name="tebd_gate_layer",
        display_name="TEBD Gate Layer",
        graph_size_label="Sites",
        defaults=_TEBD_GATE_LAYER_DEFAULTS,
        parameter_fields=_build_standard_parameter_fields(
            size_field_name="graph_size",
            size_field_label="Graph size (Sites)",
            defaults=_TEBD_GATE_LAYER_DEFAULTS,
        ),
    ),
}

_TEMPLATE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_REGISTERED_TEMPLATE_DEFINITIONS: dict[str, TemplateDefinition] = {}
_REGISTERED_TEMPLATE_BUILDERS: dict[
    str, Callable[[TemplateParameters], NetworkSpec]
] = {}
LOGGER = logging.getLogger(__name__)


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
    with log_operation(LOGGER, "Template list", context={"command": "template.list"}):
        _ensure_template_registry_seeded()
        names = list(_REGISTERED_TEMPLATE_DEFINITIONS)
        log_branch(LOGGER, "Resolved template names", context={"status": len(names)})
        return names


def serialize_template_definitions() -> dict[str, dict[str, JSONValue]]:
    """Serialize all template definitions for the browser bootstrap payload."""
    with log_operation(
        LOGGER,
        "Template definition serialization",
        context={"command": "template.list"},
    ):
        _ensure_template_registry_seeded()
        definitions = {
            template_name: definition.to_dict()
            for template_name, definition in _REGISTERED_TEMPLATE_DEFINITIONS.items()
        }
        log_branch(
            LOGGER,
            "Serialized template definitions",
            context={"status": len(definitions)},
        )
        return definitions


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
    value: object,
    *,
    field_name: str,
    default: int,
    minimum: int,
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


def parse_template_number(
    value: object,
    *,
    field_name: str,
    default: float,
    minimum: float | None = None,
) -> float:
    """Validate one numeric template parameter and apply its default."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Template parameter '{field_name}' must be a number.")
    numeric_value = float(value)
    if minimum is not None and numeric_value < minimum:
        raise ValueError(
            f"Template parameter '{field_name}' must be greater than or equal to {minimum}."
        )
    return numeric_value


def parse_template_boolean(
    value: object,
    *,
    field_name: str,
    default: bool,
) -> bool:
    """Validate one boolean template parameter and apply its default."""
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"Template parameter '{field_name}' must be a boolean.")
    return value


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


def validate_template_parameters(
    template_name: str,
    parameters: TemplateParameters,
) -> TemplateParameters:
    """Normalize template parameters against the rules for ``template_name``."""
    definition = get_template_definition(template_name)
    resolved_values = {
        parameter_field.name: getattr(definition.defaults, parameter_field.name)
        for parameter_field in fields(TemplateParameters)
    }
    for parameter_field in _get_template_parameter_fields(definition):
        raw_value = getattr(parameters, parameter_field.name)
        if parameter_field.kind == "integer":
            resolved_values[parameter_field.name] = parse_template_integer(
                raw_value,
                field_name=parameter_field.name,
                default=int(parameter_field.default),
                minimum=int(parameter_field.minimum or 1),
            )
            continue
        if parameter_field.kind == "number":
            resolved_values[parameter_field.name] = parse_template_number(
                raw_value,
                field_name=parameter_field.name,
                default=float(parameter_field.default),
                minimum=(
                    float(parameter_field.minimum)
                    if parameter_field.minimum is not None
                    else None
                ),
            )
            continue
        if parameter_field.kind == "boolean":
            resolved_values[parameter_field.name] = parse_template_boolean(
                raw_value,
                field_name=parameter_field.name,
                default=bool(parameter_field.default),
            )
            continue
        resolved_values[parameter_field.name] = parse_template_choice(
            raw_value,
            field_name=parameter_field.name,
            default=str(parameter_field.default),
            choices=tuple(option.value for option in parameter_field.options),
        )

    physical_dimension = cast(int | None, resolved_values.get("physical_dimension"))
    initial_state = cast(str, resolved_values.get("initial_state", "zeros"))
    if initial_state in {"all_up", "all_down", "neel"} and physical_dimension != 2:
        raise ValueError(
            "Template parameter 'physical_dimension' must be 2 when "
            f"'initial_state' is '{initial_state}'."
        )
    return TemplateParameters(**resolved_values)


def _reset_template_registry_for_tests() -> None:
    """Reset the live template registry to its built-in state."""
    _REGISTERED_TEMPLATE_DEFINITIONS.clear()
    _REGISTERED_TEMPLATE_BUILDERS.clear()
    from ._template_builders import register_builtin_templates

    register_builtin_templates()

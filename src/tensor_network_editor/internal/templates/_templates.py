"""Public helpers for working with built-in template definitions and specs."""

from __future__ import annotations

import logging

from ...models import NetworkSpec
from .._logging import log_branch, log_operation, summarize_spec_counts
from ._template_builders import build_template
from ._template_catalog import (
    TemplateDefinition,
    TemplateParameters,
    get_template_definition,
    list_template_names,
    register_static_template,
    register_template,
    serialize_template_definitions,
    validate_template_parameters,
)

LOGGER = logging.getLogger(__name__)


def parse_template_parameters(
    template_name: str, raw_parameters: object | None = None
) -> TemplateParameters:
    """Parse raw template parameters using the defaults for ``template_name``."""
    with log_operation(
        LOGGER,
        "Template parameter parsing",
        context={"command": "template.parse", "path": template_name},
    ):
        definition = get_template_definition(template_name)
        if not definition.supports_parameters:
            log_branch(LOGGER, "Template does not support parameters")
            return definition.defaults
        defaults = definition.defaults
        if raw_parameters is None:
            log_branch(LOGGER, "Using template default parameters")
            return defaults
        if not isinstance(raw_parameters, dict):
            raise ValueError("Template 'parameters' payload must be an object.")
        parameter_values = {
            parameter_field.name: getattr(defaults, parameter_field.name)
            for parameter_field in TemplateParameters.__dataclass_fields__.values()
        }
        for parameter_field in definition.parameter_fields:
            parameter_values[parameter_field.name] = raw_parameters.get(
                parameter_field.name
            )
        return validate_template_parameters(
            template_name,
            TemplateParameters(**parameter_values),
        )


def build_template_spec(
    template_name: str, parameters: TemplateParameters | None = None
) -> NetworkSpec:
    """Build and validate a ``NetworkSpec`` for a built-in template."""
    with log_operation(
        LOGGER,
        "Template build",
        context={"command": "template.build", "path": template_name},
    ):
        spec = build_template(template_name, parameters)
        log_branch(LOGGER, "Built template spec", context=summarize_spec_counts(spec))
        return spec


__all__ = [
    "TemplateDefinition",
    "TemplateParameters",
    "build_template_spec",
    "list_template_names",
    "register_static_template",
    "parse_template_parameters",
    "register_template",
    "serialize_template_definitions",
]

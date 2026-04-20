"""Public template helpers for headless callers."""

from __future__ import annotations

from .internal.templates._template_catalog import TemplateDefinition, TemplateParameters
from .internal.templates._templates import (
    build_template_spec,
    list_template_names,
    parse_template_parameters,
    register_static_template,
    register_template,
    serialize_template_definitions,
)

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

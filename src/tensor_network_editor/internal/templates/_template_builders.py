"""Facade and registry entrypoints for internal template builders."""

from __future__ import annotations

from ...models import NetworkSpec
from ...validation import ensure_valid_spec
from ._template_builders_grid import (
    _build_pepo_template,
    _build_peps_template,
    _build_tebd_gate_layer_template,
)
from ._template_builders_linear import (
    _build_linear_chain_template,
    _build_mpo_template,
    _build_mps_template,
)
from ._template_builders_tree import _build_mera_template, _build_ttn_template
from ._template_catalog import (
    TEMPLATE_DEFINITIONS,
    TemplateParameters,
    get_template_builder,
    get_template_definition,
    register_template,
    validate_template_parameters,
)


def build_template(
    template_name: str,
    parameters: TemplateParameters | None = None,
) -> NetworkSpec:
    """Build and validate the named built-in template."""
    definition = get_template_definition(template_name)
    resolved_parameters = (
        validate_template_parameters(
            template_name,
            parameters or definition.defaults,
        )
        if definition.supports_parameters
        else definition.defaults
    )
    builder = get_template_builder(template_name)
    return ensure_valid_spec(builder(resolved_parameters))


def register_builtin_templates() -> None:
    """Register the built-in templates in their stable display order."""
    register_template(
        "mps",
        TEMPLATE_DEFINITIONS["mps"],
        _build_mps_template,
        overwrite=True,
    )
    register_template(
        "mpo",
        TEMPLATE_DEFINITIONS["mpo"],
        _build_mpo_template,
        overwrite=True,
    )
    register_template(
        "peps_2x2",
        TEMPLATE_DEFINITIONS["peps_2x2"],
        _build_peps_template,
        overwrite=True,
    )
    register_template(
        "mera",
        TEMPLATE_DEFINITIONS["mera"],
        _build_mera_template,
        overwrite=True,
    )
    register_template(
        "ttn",
        TEMPLATE_DEFINITIONS["ttn"],
        _build_ttn_template,
        overwrite=True,
    )
    register_template(
        "pepo",
        TEMPLATE_DEFINITIONS["pepo"],
        _build_pepo_template,
        overwrite=True,
    )
    register_template(
        "tebd_gate_layer",
        TEMPLATE_DEFINITIONS["tebd_gate_layer"],
        _build_tebd_gate_layer_template,
        overwrite=True,
    )


__all__ = [
    "build_template",
    "register_builtin_templates",
    "_build_linear_chain_template",
]

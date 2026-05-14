"""Template catalog helpers for editor routes and sessions."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..internal._logging import log_branch, log_operation, summarize_spec_counts
from ..internal.io._serialization import deserialize_spec
from ..internal.subnetworks._subnetworks import extract_subnetwork_spec
from ..models import NetworkSpec
from ..templates import (
    TemplateParameters,
    build_template_spec,
    parse_template_parameters,
)
from ._bootstrap_payloads import build_template_catalog_payload
from ._limits import enforce_spec_api_limits, enforce_template_api_limits
from ._protocol import JsonDict

if TYPE_CHECKING:
    from .session import EditorSession


LOGGER = logging.getLogger(__name__)


def build_template_from_payload(
    session: EditorSession,
    template_name: str,
    raw_parameters: object | None = None,
) -> NetworkSpec:
    """Build a validated template spec from raw API payload values."""
    context = {"session": session.session_id, "template_name": template_name}
    with log_operation(LOGGER, "Template build", context=context) as success_context:
        if session.has_project_template(template_name):
            log_branch(LOGGER, "Loading template from the project catalog")
            spec = session.build_project_template(template_name)
            enforce_spec_api_limits(spec)
            success_context.update(summarize_spec_counts(spec))
            success_context["status"] = "project"
            return spec
        parameters: TemplateParameters | None = parse_template_parameters(
            template_name,
            raw_parameters,
        )
        enforce_template_api_limits(template_name, parameters)
        spec = build_template_spec(template_name, parameters)
        enforce_spec_api_limits(spec)
        success_context.update(summarize_spec_counts(spec))
        success_context["status"] = "global"
        return spec


def promote_serialized_subnetwork_to_template(
    session: EditorSession,
    serialized_spec: Mapping[str, object],
    *,
    tensor_ids: list[str],
    template_name: str,
    overwrite: bool = False,
) -> JsonDict:
    """Extract one fragment and persist it as a project-local static template."""
    context = {
        "session": session.session_id,
        "template_name": template_name,
        "tensor_id_count": len(tensor_ids),
        "overwrite": overwrite,
    }
    with log_operation(
        LOGGER, "Template promotion", context=context
    ) as success_context:
        spec = deserialize_spec(serialized_spec, validate=False)
        enforce_spec_api_limits(spec)
        promoted_spec = extract_subnetwork_spec(spec, tensor_ids=tensor_ids)
        enforce_spec_api_limits(promoted_spec)
        promoted_spec.name = session.build_project_template_display_name(template_name)
        session.save_project_template(
            template_name,
            promoted_spec,
            overwrite=overwrite,
        )
        payload = build_template_catalog_payload(
            session,
            selected_template=template_name,
        )
        success_context.update(summarize_spec_counts(promoted_spec))
        success_context["selected_template"] = template_name
        return payload


def rename_session_project_template(
    session: EditorSession,
    *,
    template_name: str,
    new_template_name: str,
    overwrite: bool = False,
) -> JsonDict:
    """Rename one project-local template and return the refreshed catalog."""
    context = {
        "session": session.session_id,
        "template_name": template_name,
        "selected_template": new_template_name,
        "overwrite": overwrite,
    }
    with log_operation(LOGGER, "Template rename", context=context) as success_context:
        if session.has_global_template(
            template_name
        ) and not session.has_project_template(template_name):
            raise ValueError(
                f"Template '{template_name}' is registered globally and cannot be renamed."
            )
        session.rename_project_template(
            template_name,
            new_template_name,
            overwrite=overwrite,
        )
        payload = build_template_catalog_payload(
            session,
            selected_template=new_template_name,
        )
        success_context["selected_template"] = new_template_name
        return payload


def delete_session_project_template(
    session: EditorSession,
    *,
    template_name: str,
) -> JsonDict:
    """Delete one project-local template and return the refreshed catalog."""
    context = {
        "session": session.session_id,
        "template_name": template_name,
    }
    with log_operation(LOGGER, "Template delete", context=context) as success_context:
        if session.has_global_template(
            template_name
        ) and not session.has_project_template(template_name):
            raise ValueError(
                f"Template '{template_name}' is registered globally and cannot be deleted."
            )
        previous_project_template_names = list(session.project_template_entries)
        if template_name not in previous_project_template_names:
            raise ValueError(f"Unknown project template '{template_name}'.")
        deleted_template_index = previous_project_template_names.index(template_name)
        session.delete_project_template(template_name)
        selected_template = None
        remaining_project_templates = list(session.project_template_entries)
        if remaining_project_templates:
            selected_template = remaining_project_templates[
                min(deleted_template_index, len(remaining_project_templates) - 1)
            ]
        else:
            available_templates = session.list_available_template_names()
            if available_templates:
                selected_template = available_templates[0]
        success_context["selected_template"] = selected_template
        return build_template_catalog_payload(
            session,
            selected_template=selected_template,
        )

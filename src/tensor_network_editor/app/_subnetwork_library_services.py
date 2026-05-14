"""Reusable-subnetwork catalog helpers for editor routes and sessions."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ..internal._logging import log_operation, summarize_spec_counts
from ..internal.io._serialization import deserialize_spec
from ..internal.subnetworks._subnetworks import (
    extract_subnetwork_spec,
    prepare_subnetwork_for_insertion,
)
from ..models import CanvasPosition, NetworkSpec
from ._bootstrap_payloads import build_subnetwork_catalog_payload
from ._limits import enforce_spec_api_limits
from ._protocol import JsonDict

if TYPE_CHECKING:
    from .session import EditorSession


LOGGER = logging.getLogger(__name__)


def save_serialized_subnetwork_to_library(
    session: EditorSession,
    serialized_spec: Mapping[str, object],
    *,
    tensor_ids: list[str],
    subnetwork_name: str,
    tags: Sequence[str] | None = None,
    overwrite: bool = False,
) -> JsonDict:
    """Extract one fragment and persist it as a reusable project subnetwork."""
    context = {
        "session": session.session_id,
        "subnetwork_name": subnetwork_name,
        "tensor_id_count": len(tensor_ids),
        "tag_count": len(tags or ()),
        "overwrite": overwrite,
    }
    with log_operation(
        LOGGER, "Reusable subnetwork save", context=context
    ) as success_context:
        spec = deserialize_spec(serialized_spec, validate=False)
        enforce_spec_api_limits(spec)
        saved_spec = extract_subnetwork_spec(spec, tensor_ids=tensor_ids)
        enforce_spec_api_limits(saved_spec)
        session.save_project_subnetwork(
            subnetwork_name,
            saved_spec,
            tags=tags,
            overwrite=overwrite,
        )
        payload = build_subnetwork_catalog_payload(
            session,
            selected_subnetwork=subnetwork_name,
        )
        success_context.update(summarize_spec_counts(saved_spec))
        success_context["selected_subnetwork"] = subnetwork_name
        return payload


def rename_session_project_subnetwork(
    session: EditorSession,
    *,
    subnetwork_name: str,
    new_subnetwork_name: str,
    overwrite: bool = False,
) -> JsonDict:
    """Rename one project-local reusable subnetwork and return the catalog."""
    context = {
        "session": session.session_id,
        "subnetwork_name": subnetwork_name,
        "selected_subnetwork": new_subnetwork_name,
        "overwrite": overwrite,
    }
    with log_operation(
        LOGGER, "Reusable subnetwork rename", context=context
    ) as success_context:
        session.rename_project_subnetwork(
            subnetwork_name,
            new_subnetwork_name,
            overwrite=overwrite,
        )
        payload = build_subnetwork_catalog_payload(
            session,
            selected_subnetwork=new_subnetwork_name,
        )
        success_context["selected_subnetwork"] = new_subnetwork_name
        return payload


def delete_session_project_subnetwork(
    session: EditorSession,
    *,
    subnetwork_name: str,
) -> JsonDict:
    """Delete one project-local reusable subnetwork and return the catalog."""
    context = {
        "session": session.session_id,
        "subnetwork_name": subnetwork_name,
    }
    with log_operation(
        LOGGER, "Reusable subnetwork delete", context=context
    ) as success_context:
        previous_names = list(session.project_subnetwork_entries)
        if subnetwork_name not in previous_names:
            raise ValueError(f"Unknown reusable subnetwork '{subnetwork_name}'.")
        deleted_index = previous_names.index(subnetwork_name)
        session.delete_project_subnetwork(subnetwork_name)
        remaining_names = list(session.project_subnetwork_entries)
        selected_subnetwork = (
            remaining_names[min(deleted_index, len(remaining_names) - 1)]
            if remaining_names
            else None
        )
        success_context["selected_subnetwork"] = selected_subnetwork
        return build_subnetwork_catalog_payload(
            session,
            selected_subnetwork=selected_subnetwork,
        )


def prepare_saved_subnetwork_for_insertion(
    session: EditorSession,
    *,
    subnetwork_name: str,
    target_center: CanvasPosition,
) -> NetworkSpec:
    """Build and prepare one saved reusable subnetwork for editor insertion."""
    context = {
        "session": session.session_id,
        "subnetwork_name": subnetwork_name,
    }
    with log_operation(
        LOGGER,
        "Reusable subnetwork insertion preparation",
        context=context,
    ) as success_context:
        spec = session.build_saved_subnetwork(subnetwork_name)
        enforce_spec_api_limits(spec)
        prepared_spec = prepare_subnetwork_for_insertion(
            spec,
            target_center=target_center,
        )
        enforce_spec_api_limits(prepared_spec)
        success_context.update(summarize_spec_counts(prepared_spec))
        return prepared_spec

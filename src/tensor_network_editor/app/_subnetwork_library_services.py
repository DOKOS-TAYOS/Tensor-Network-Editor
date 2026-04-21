"""Reusable-subnetwork catalog helpers for editor routes and sessions."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ..internal.io._serialization import deserialize_spec
from ..internal.subnetworks._subnetworks import (
    extract_subnetwork_spec,
    prepare_subnetwork_for_insertion,
)
from ..models import CanvasPosition, NetworkSpec
from ._bootstrap_payloads import build_subnetwork_catalog_payload
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
    LOGGER.info(
        "[session=%s] Saving reusable subnetwork '%s'",
        session.session_id,
        subnetwork_name,
    )
    spec = deserialize_spec(serialized_spec, validate=False)
    saved_spec = extract_subnetwork_spec(spec, tensor_ids=tensor_ids)
    session.save_project_subnetwork(
        subnetwork_name,
        saved_spec,
        tags=tags,
        overwrite=overwrite,
    )
    return build_subnetwork_catalog_payload(
        session,
        selected_subnetwork=subnetwork_name,
    )


def rename_session_project_subnetwork(
    session: EditorSession,
    *,
    subnetwork_name: str,
    new_subnetwork_name: str,
    overwrite: bool = False,
) -> JsonDict:
    """Rename one project-local reusable subnetwork and return the catalog."""
    LOGGER.info(
        "[session=%s] Renaming reusable subnetwork '%s' to '%s'",
        session.session_id,
        subnetwork_name,
        new_subnetwork_name,
    )
    session.rename_project_subnetwork(
        subnetwork_name,
        new_subnetwork_name,
        overwrite=overwrite,
    )
    return build_subnetwork_catalog_payload(
        session,
        selected_subnetwork=new_subnetwork_name,
    )


def delete_session_project_subnetwork(
    session: EditorSession,
    *,
    subnetwork_name: str,
) -> JsonDict:
    """Delete one project-local reusable subnetwork and return the catalog."""
    previous_names = list(session.project_subnetwork_entries)
    if subnetwork_name not in previous_names:
        raise ValueError(f"Unknown reusable subnetwork '{subnetwork_name}'.")
    deleted_index = previous_names.index(subnetwork_name)
    LOGGER.info(
        "[session=%s] Deleting reusable subnetwork '%s'",
        session.session_id,
        subnetwork_name,
    )
    session.delete_project_subnetwork(subnetwork_name)
    remaining_names = list(session.project_subnetwork_entries)
    selected_subnetwork = (
        remaining_names[min(deleted_index, len(remaining_names) - 1)]
        if remaining_names
        else None
    )
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
    spec = session.build_saved_subnetwork(subnetwork_name)
    return prepare_subnetwork_for_insertion(spec, target_center=target_center)

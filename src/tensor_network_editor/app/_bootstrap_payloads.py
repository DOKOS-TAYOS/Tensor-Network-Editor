"""Bootstrap payload builders for the local editor app."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .._version import __version__
from ..codegen.registry import engine_name_to_text, list_generator_names
from ..internal._logging import (
    build_frontend_logging_payload as build_frontend_log_config,
)
from ..internal.io._serialization import SCHEMA_VERSION
from ..internal.templates._annotation_catalog import serialize_annotation_definitions
from ..models import TensorCollectionFormat
from ..types import JSONValue
from ._protocol import JsonDict

if TYPE_CHECKING:
    from .session import EditorSession


REPOSITORY_URL = "https://github.com/DOKOS-TAYOS/Tensor-Network-Editor"
LICENSE_NAME = "MIT"
AUTHOR_NAME = "Alejandro Mata Ali"


def build_bootstrap_payload(session: EditorSession) -> JsonDict:
    """Build the initial payload used to bootstrap the browser client."""
    return {
        "theme": session.theme,
        "session_id": session.session_id,
        "frontend_logging": build_frontend_logging_payload(session),
        "default_engine": engine_name_to_text(session.default_engine),
        "engines": cast(JSONValue, list_generator_names()),
        "default_collection_format": session.default_collection_format.value,
        "collection_formats": cast(
            JSONValue,
            [collection_format.value for collection_format in TensorCollectionFormat],
        ),
        "schema_version": SCHEMA_VERSION,
        **build_template_catalog_payload(session),
        **build_subnetwork_catalog_payload(session),
        "annotation_definitions": cast(
            JSONValue,
            serialize_annotation_definitions(),
        ),
        "app_metadata": build_app_metadata_payload(),
        "spec": {
            "schema_version": SCHEMA_VERSION,
            "network": session.initial_spec.to_dict(),
        },
    }


def build_frontend_logging_payload(session: EditorSession | None = None) -> JsonDict:
    """Build the browser logging configuration for one editor session."""
    if session is not None and hasattr(session, "frontend_logging_payload"):
        return cast(JsonDict, session.frontend_logging_payload)
    return cast(JsonDict, build_frontend_log_config())


def build_app_metadata_payload() -> JsonDict:
    """Build static application metadata for frontend About dialogs."""
    return {
        "repository_url": REPOSITORY_URL,
        "version": __version__,
        "license_name": LICENSE_NAME,
        "author_name": AUTHOR_NAME,
    }


def build_template_catalog_payload(
    session: EditorSession,
    *,
    selected_template: str | None = None,
) -> JsonDict:
    """Build the merged session template catalog payload for the editor."""
    payload: JsonDict = {
        "templates": cast(JSONValue, session.list_available_template_names()),
        "template_definitions": cast(
            JSONValue,
            session.serialize_available_template_definitions(),
        ),
        "template_catalog_warnings": cast(
            JSONValue,
            session.template_catalog_warnings,
        ),
    }
    if selected_template is not None:
        payload["selected_template"] = selected_template
    return payload


def build_subnetwork_catalog_payload(
    session: EditorSession,
    *,
    selected_subnetwork: str | None = None,
) -> JsonDict:
    """Build the merged reusable-subnetwork catalog payload for the editor."""
    payload: JsonDict = {
        "subnetworks": cast(JSONValue, session.list_available_subnetwork_names()),
        "subnetwork_definitions": cast(
            JSONValue,
            session.serialize_available_subnetwork_definitions(),
        ),
        "subnetwork_catalog_warnings": cast(
            JSONValue,
            session.subnetwork_catalog_warnings,
        ),
    }
    if selected_subnetwork is not None:
        payload["selected_subnetwork"] = selected_subnetwork
    return payload

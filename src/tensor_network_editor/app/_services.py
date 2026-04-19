"""Service-layer helpers shared by the local editor HTTP routes."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from .._annotation_catalog import serialize_annotation_definitions
from .._contraction_analysis import _analyze_validated_contraction
from .._contraction_analysis_types import ContractionAnalysisResult
from .._version import __version__
from ..codegen.registry import (
    engine_name_to_text,
    list_generator_names,
)
from ..codegen.registry import (
    generate_code as generate_code_internal,
)
from ..errors import SpecValidationError
from ..models import (
    CanvasPosition,
    CodegenResult,
    EditorResult,
    EngineIdentifier,
    NetworkSpec,
    TensorCollectionFormat,
)
from ..serialization import SCHEMA_VERSION, deserialize_spec
from ..subnetworks import (
    extract_subnetwork_spec,
    prepare_subnetwork_for_insertion,
)
from ..templates import (
    TemplateParameters,
    build_template_spec,
    parse_template_parameters,
)
from ..types import JSONValue
from ..validation import validate_spec
from ._protocol import JsonDict

if TYPE_CHECKING:
    from .session import EditorSession


LOGGER = logging.getLogger(__name__)
REPOSITORY_URL = "https://github.com/DOKOS-TAYOS/Tensor-Network-Editor"
LICENSE_NAME = "MIT"
AUTHOR_NAME = "Alejandro Mata Ali"


def build_bootstrap_payload(session: EditorSession) -> JsonDict:
    """Build the initial payload used to bootstrap the browser client."""
    return {
        "default_engine": engine_name_to_text(session.default_engine),
        "engines": cast(JSONValue, list_generator_names()),
        "default_collection_format": session.default_collection_format.value,
        "collection_formats": cast(
            JSONValue,
            [collection_format.value for collection_format in TensorCollectionFormat],
        ),
        "schema_version": SCHEMA_VERSION,
        **build_template_catalog_payload(session),
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


def build_app_metadata_payload() -> JsonDict:
    """Build static application metadata for frontend About dialogs."""
    return {
        "repository_url": REPOSITORY_URL,
        "version": __version__,
        "license_name": LICENSE_NAME,
        "author_name": AUTHOR_NAME,
    }


def generate_session_request(
    session: EditorSession,
    serialized_spec: Mapping[str, object],
    engine: EngineIdentifier,
    collection_format: TensorCollectionFormat | None = None,
) -> CodegenResult:
    """Generate preview code for one editor request."""
    LOGGER.debug(
        "[session=%s] Generating preview request for engine '%s'",
        session.session_id,
        engine_name_to_text(engine),
    )
    spec = deserialize_spec(serialized_spec)
    return generate_code_internal(
        spec,
        engine,
        collection_format=_resolve_collection_format(session, collection_format),
        validate=False,
    )


def complete_session_request(
    session: EditorSession,
    serialized_spec: Mapping[str, object],
    engine: EngineIdentifier,
    collection_format: TensorCollectionFormat | None = None,
) -> EditorResult:
    """Finalize a session request and optionally print or save generated code."""
    LOGGER.info(
        "[session=%s] Completing session request for engine '%s'",
        session.session_id,
        engine_name_to_text(engine),
    )
    spec = deserialize_spec(serialized_spec)
    codegen_result = generate_code_internal(
        spec,
        engine,
        collection_format=_resolve_collection_format(session, collection_format),
        validate=False,
    )
    if session.print_code:
        LOGGER.debug(
            "[session=%s] Printing generated code to stdout", session.session_id
        )
        print(codegen_result.code)
    if session.code_path is not None:
        from .._io import write_utf8_text

        LOGGER.debug(
            "[session=%s] Writing generated code to %s",
            session.session_id,
            session.code_path,
        )
        write_utf8_text(
            session.code_path,
            codegen_result.code,
            description="generated Python code",
        )
    return EditorResult(
        spec=spec, engine=engine, codegen=codegen_result, confirmed=True
    )


def build_template_from_payload(
    session: EditorSession,
    template_name: str,
    raw_parameters: object | None = None,
) -> NetworkSpec:
    """Build a validated template spec from raw API payload values."""
    if session.has_project_template(template_name):
        return session.build_project_template(template_name)
    parameters: TemplateParameters | None = parse_template_parameters(
        template_name,
        raw_parameters,
    )
    return build_template_spec(template_name, parameters)


def analyze_serialized_contraction(
    serialized_spec: Mapping[str, object],
) -> ContractionAnalysisResult:
    """Deserialize, validate, and analyze contraction data for one payload."""
    spec = deserialize_spec(serialized_spec, validate=False)
    issues = validate_spec(spec)
    if issues:
        raise SpecValidationError(issues)
    return _analyze_validated_contraction(spec)


def extract_serialized_subnetwork(
    serialized_spec: Mapping[str, object],
    *,
    tensor_ids: list[str],
) -> NetworkSpec:
    """Deserialize one payload and extract its selected tensor fragment."""
    spec = deserialize_spec(serialized_spec, validate=False)
    return extract_subnetwork_spec(spec, tensor_ids=tensor_ids)


def prepare_serialized_subnetwork_for_insertion(
    serialized_spec: Mapping[str, object],
    *,
    target_center: CanvasPosition,
) -> NetworkSpec:
    """Deserialize one payload and prepare it for editor insertion."""
    spec = deserialize_spec(serialized_spec, validate=False)
    return prepare_subnetwork_for_insertion(spec, target_center=target_center)


def promote_serialized_subnetwork_to_template(
    session: EditorSession,
    serialized_spec: Mapping[str, object],
    *,
    tensor_ids: list[str],
    template_name: str,
    overwrite: bool = False,
) -> JsonDict:
    """Extract one fragment and persist it as a project-local static template."""
    LOGGER.info(
        "[session=%s] Promoting selection to project template '%s'",
        session.session_id,
        template_name,
    )
    spec = deserialize_spec(serialized_spec, validate=False)
    promoted_spec = extract_subnetwork_spec(
        spec,
        tensor_ids=tensor_ids,
    )
    promoted_spec.name = session.build_project_template_display_name(template_name)
    session.save_project_template(
        template_name,
        promoted_spec,
        overwrite=overwrite,
    )
    return build_template_catalog_payload(
        session,
        selected_template=template_name,
    )


def rename_session_project_template(
    session: EditorSession,
    *,
    template_name: str,
    new_template_name: str,
    overwrite: bool = False,
) -> JsonDict:
    """Rename one project-local template and return the refreshed catalog."""
    if session.has_global_template(template_name) and not session.has_project_template(
        template_name
    ):
        raise ValueError(
            f"Template '{template_name}' is registered globally and cannot be renamed."
        )
    LOGGER.info(
        "[session=%s] Renaming project template '%s' to '%s'",
        session.session_id,
        template_name,
        new_template_name,
    )
    session.rename_project_template(
        template_name,
        new_template_name,
        overwrite=overwrite,
    )
    return build_template_catalog_payload(
        session,
        selected_template=new_template_name,
    )


def delete_session_project_template(
    session: EditorSession,
    *,
    template_name: str,
) -> JsonDict:
    """Delete one project-local template and return the refreshed catalog."""
    if session.has_global_template(template_name) and not session.has_project_template(
        template_name
    ):
        raise ValueError(
            f"Template '{template_name}' is registered globally and cannot be deleted."
        )
    previous_project_template_names = list(session.project_template_entries)
    if template_name not in previous_project_template_names:
        raise ValueError(f"Unknown project template '{template_name}'.")
    deleted_template_index = previous_project_template_names.index(template_name)
    LOGGER.info(
        "[session=%s] Deleting project template '%s'",
        session.session_id,
        template_name,
    )
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
    return build_template_catalog_payload(
        session,
        selected_template=selected_template,
    )


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


def _resolve_collection_format(
    session: EditorSession,
    collection_format: TensorCollectionFormat | None,
) -> TensorCollectionFormat:
    """Resolve a request collection format against the session default."""
    return collection_format or session.default_collection_format

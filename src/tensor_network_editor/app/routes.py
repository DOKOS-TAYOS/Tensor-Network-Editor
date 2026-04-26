"""HTTP route handlers for the local editor server."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Literal, cast

from ..errors import (
    CodeGenerationError,
    PackageIOError,
    SerializationError,
    SpecValidationError,
)
from ..internal.analysis._contraction_analysis_types import ContractionAnalysisResult
from ..io import serialize_spec
from ..models import CodegenResult, EditorResult
from ..types import JSONValue
from ..validation import validate_spec
from ._drafts import clear_project_draft, load_project_draft, save_project_draft
from ._protocol import (
    JsonDict,
    JsonResponse,
    bad_request_response,
    deserialize_validation_request,
    issues_response,
    ok_response,
    parse_codegen_request,
    parse_subnetwork_library_delete_request,
    parse_subnetwork_library_prepare_insert_request,
    parse_subnetwork_library_rename_request,
    parse_subnetwork_library_save_request,
    parse_subnetwork_prepare_insert_request,
    parse_subnetwork_selection_request,
    parse_template_delete_request,
    parse_template_promote_request,
    parse_template_rename_request,
    require_serialized_spec,
    serialize_codegen_result,
    serialize_editor_result,
    serialize_spec_payload,
)
from ._services import (
    analyze_serialized_contraction,
    build_bootstrap_payload,
    build_template_from_payload,
    delete_session_project_subnetwork,
    delete_session_project_template,
    extract_serialized_subnetwork,
    prepare_saved_subnetwork_for_insertion,
    prepare_serialized_subnetwork_for_insertion,
    promote_serialized_subnetwork_to_template,
    rename_session_project_subnetwork,
    rename_session_project_template,
    save_serialized_subnetwork_to_library,
)
from .session import EditorSession

LOGGER = logging.getLogger(__name__)


def handle_bootstrap(session: EditorSession) -> JsonResponse:
    """Return the bootstrap payload used by the browser client."""
    return HTTPStatus.OK, build_bootstrap_payload(session)


def handle_draft_load(session: EditorSession) -> JsonResponse:
    """Return the active project draft when one has been saved."""
    try:
        draft = load_project_draft(session.draft_path)
    except ValueError as exc:
        return bad_request_response(str(exc))
    return ok_response({"draft": cast(JSONValue, draft)})


def handle_draft_save(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Persist the active browser editor draft for later recovery."""
    try:
        request = parse_codegen_request(
            payload,
            default_engine=session.default_engine,
            default_collection_format=session.default_collection_format,
        )
        draft = save_project_draft(
            session.draft_path,
            serialized_spec=request.serialized_spec,
            engine=request.engine,
            collection_format=request.collection_format,
        )
    except ValueError as exc:
        return bad_request_response(str(exc))
    return ok_response({"draft": cast(JSONValue, draft)})


def handle_draft_clear(session: EditorSession) -> JsonResponse:
    """Clear the active project draft after an explicit user action."""
    clear_project_draft(session.draft_path)
    return ok_response()


def handle_validate(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Validate a serialized spec or supported Python source payload."""
    session_id = session.session_id
    try:
        validation_request = deserialize_validation_request(payload)
        spec = validation_request.spec
    except SerializationError as exc:
        LOGGER.warning(
            "[session=%s] Validation request contained malformed spec payload: %s",
            session_id,
            exc,
        )
        return bad_request_response(str(exc))
    except SpecValidationError as exc:
        return issues_response(exc.issues)
    except ValueError:
        LOGGER.warning(
            "[session=%s] Validation request missing 'spec' or 'python_code' payload.",
            session_id,
        )
        return bad_request_response("Missing 'spec' or 'python_code' payload.")
    issues = validate_spec(spec)
    if issues:
        status, response = issues_response(issues)
        response["spec"] = serialize_spec_payload(spec)
        if validation_request.warnings:
            response["warnings"] = cast(JSONValue, validation_request.warnings)
        return status, response
    response_payload: JsonDict = {
        "issues": [],
        "spec": serialize_spec_payload(spec),
    }
    if validation_request.warnings:
        response_payload["warnings"] = cast(JSONValue, validation_request.warnings)
    return ok_response(response_payload)


def handle_generate(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Generate preview code for the current editor payload."""
    status, response = _handle_session_codegen_request(
        session=session,
        payload=payload,
        operation="generate",
    )
    message = response.get("message")
    if response.get("ok") is False and isinstance(message, str):
        LOGGER.warning(
            "[session=%s] Generate request rejected: %s",
            session.session_id,
            message,
        )
    return status, response


def handle_complete(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Finalize an editor session and return the completion payload."""
    status, response = _handle_session_codegen_request(
        session=session,
        payload=payload,
        operation="complete",
    )
    message = response.get("message")
    if response.get("ok") is False and isinstance(message, str):
        LOGGER.warning(
            "[session=%s] Complete request rejected: %s",
            session.session_id,
            message,
        )
    return status, response


def handle_cancel(session: EditorSession) -> JsonResponse:
    """Cancel the current editor session."""
    session.cancel()
    return ok_response()


def handle_template(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Build a template spec from the requested template payload."""
    template_name = payload.get("template")
    if not isinstance(template_name, str) or not template_name.strip():
        return bad_request_response("Missing 'template' payload.")
    try:
        spec = build_template_from_payload(
            session,
            template_name,
            payload.get("parameters"),
        )
    except ValueError as exc:
        return bad_request_response(str(exc))
    return ok_response({"spec": serialize_spec(spec)})


def handle_template_promote(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Promote a selected subnetwork fragment to the project template catalog."""
    try:
        request = parse_template_promote_request(payload)
        catalog_payload = promote_serialized_subnetwork_to_template(
            session,
            request.serialized_spec,
            tensor_ids=request.tensor_ids,
            template_name=request.template_name,
            overwrite=request.overwrite,
        )
    except (PackageIOError, SerializationError, TypeError, ValueError) as exc:
        return bad_request_response(str(exc))
    return ok_response(catalog_payload)


def handle_template_rename(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Rename one project-local template entry."""
    try:
        request = parse_template_rename_request(payload)
        catalog_payload = rename_session_project_template(
            session,
            template_name=request.template_name,
            new_template_name=request.new_template_name,
            overwrite=request.overwrite,
        )
    except (PackageIOError, ValueError) as exc:
        return bad_request_response(str(exc))
    return ok_response(catalog_payload)


def handle_template_delete(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Delete one project-local template entry."""
    try:
        request = parse_template_delete_request(payload)
        catalog_payload = delete_session_project_template(
            session,
            template_name=request.template_name,
        )
    except (PackageIOError, ValueError) as exc:
        return bad_request_response(str(exc))
    return ok_response(catalog_payload)


def handle_analyze_contraction(
    session: EditorSession, payload: JsonDict
) -> JsonResponse:
    """Analyze contraction information for a validated serialized spec."""
    session_id = session.session_id
    try:
        serialized_spec = require_serialized_spec(payload)
    except ValueError:
        LOGGER.warning(
            "[session=%s] Contraction analysis request missing 'spec' payload.",
            session_id,
        )
        return bad_request_response("Missing 'spec' payload.")

    try:
        result = analyze_serialized_contraction(serialized_spec)
    except SerializationError as exc:
        LOGGER.warning(
            "[session=%s] Contraction analysis request contained malformed spec: %s",
            session_id,
            exc,
        )
        return bad_request_response(str(exc))
    except SpecValidationError as exc:
        return issues_response(exc.issues)

    return ok_response(_serialize_contraction_analysis_result(result))


def handle_subnetwork_extract(
    session: EditorSession, payload: JsonDict
) -> JsonResponse:
    """Extract a reusable subnetwork fragment from the current graph."""
    del session
    try:
        request = parse_subnetwork_selection_request(payload)
        spec = extract_serialized_subnetwork(
            request.serialized_spec,
            tensor_ids=request.tensor_ids,
        )
    except (SerializationError, TypeError, ValueError) as exc:
        return bad_request_response(str(exc))
    return ok_response({"spec": serialize_spec(spec)})


def handle_subnetwork_prepare_insert(
    session: EditorSession, payload: JsonDict
) -> JsonResponse:
    """Prepare one saved fragment for insertion into the current design."""
    del session
    try:
        request = parse_subnetwork_prepare_insert_request(payload)
        spec = prepare_serialized_subnetwork_for_insertion(
            request.serialized_spec,
            target_center=request.target_center,
        )
    except (SerializationError, TypeError, ValueError) as exc:
        return bad_request_response(str(exc))
    return ok_response({"spec": serialize_spec(spec)})


def handle_subnetwork_library_save(
    session: EditorSession,
    payload: JsonDict,
) -> JsonResponse:
    """Save one selected fragment into the reusable-subnetwork catalog."""
    try:
        request = parse_subnetwork_library_save_request(payload)
        catalog_payload = save_serialized_subnetwork_to_library(
            session,
            request.serialized_spec,
            tensor_ids=request.tensor_ids,
            subnetwork_name=request.subnetwork_name,
            tags=request.tags,
            overwrite=request.overwrite,
        )
    except (PackageIOError, SerializationError, TypeError, ValueError) as exc:
        return bad_request_response(str(exc))
    return ok_response(catalog_payload)


def handle_subnetwork_library_rename(
    session: EditorSession,
    payload: JsonDict,
) -> JsonResponse:
    """Rename one project-local reusable-subnetwork catalog entry."""
    try:
        request = parse_subnetwork_library_rename_request(payload)
        catalog_payload = rename_session_project_subnetwork(
            session,
            subnetwork_name=request.subnetwork_name,
            new_subnetwork_name=request.new_subnetwork_name,
            overwrite=request.overwrite,
        )
    except (PackageIOError, ValueError) as exc:
        return bad_request_response(str(exc))
    return ok_response(catalog_payload)


def handle_subnetwork_library_delete(
    session: EditorSession,
    payload: JsonDict,
) -> JsonResponse:
    """Delete one project-local reusable-subnetwork catalog entry."""
    try:
        request = parse_subnetwork_library_delete_request(payload)
        catalog_payload = delete_session_project_subnetwork(
            session,
            subnetwork_name=request.subnetwork_name,
        )
    except (PackageIOError, ValueError) as exc:
        return bad_request_response(str(exc))
    return ok_response(catalog_payload)


def handle_subnetwork_library_prepare_insert(
    session: EditorSession,
    payload: JsonDict,
) -> JsonResponse:
    """Prepare one saved reusable subnetwork for insertion into the graph."""
    try:
        request = parse_subnetwork_library_prepare_insert_request(payload)
        spec = prepare_saved_subnetwork_for_insertion(
            session,
            subnetwork_name=request.subnetwork_name,
            target_center=request.target_center,
        )
    except (SerializationError, TypeError, ValueError) as exc:
        return bad_request_response(str(exc))
    return ok_response({"spec": serialize_spec(spec)})


def _serialize_generate_result(result: CodegenResult) -> JsonDict:
    """Serialize a generate-route code generation result."""
    return serialize_codegen_result(result)


def _serialize_complete_result(result: EditorResult) -> JsonDict:
    """Serialize a complete-route editor result."""
    return serialize_editor_result(result)


def _handle_session_codegen_request(
    *,
    session: EditorSession,
    payload: JsonDict,
    operation: Literal["generate", "complete"],
) -> JsonResponse:
    """Handle shared generate and complete route behavior."""
    try:
        request = parse_codegen_request(
            payload,
            default_engine=session.default_engine,
            default_collection_format=session.default_collection_format,
        )
    except ValueError as exc:
        return bad_request_response(str(exc))

    try:
        if operation == "generate":
            generate_result = session.generate(
                request.serialized_spec,
                request.engine,
                request.collection_format,
            )
            return ok_response(_serialize_generate_result(generate_result))
        if operation == "complete":
            complete_result = session.complete(
                request.serialized_spec,
                request.engine,
                request.collection_format,
            )
            return ok_response(_serialize_complete_result(complete_result))
        raise ValueError(f"Unsupported code generation operation '{operation}'.")
    except SerializationError as exc:
        return bad_request_response(str(exc))
    except CodeGenerationError as exc:
        return bad_request_response(str(exc))
    except PackageIOError as exc:
        return bad_request_response(str(exc))
    except SpecValidationError as exc:
        return issues_response(exc.issues)


def _serialize_contraction_analysis_result(
    result: ContractionAnalysisResult,
) -> JsonDict:
    """Serialize a contraction analysis result for the API."""
    return result.to_dict()

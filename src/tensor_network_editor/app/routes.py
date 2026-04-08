"""HTTP route handlers for the local editor server."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Literal, cast

from .._contraction_analysis_types import ContractionAnalysisResult
from ..analysis import analyze_contraction
from ..errors import CodeGenerationError, SerializationError, SpecValidationError
from ..models import CodegenResult, EditorResult
from ..serialization import serialize_spec
from ..validation import validate_spec
from ._protocol import (
    JsonDict,
    JsonResponse,
    bad_request_response,
    deserialize_spec_with_issues,
    deserialize_validation_payload,
    issues_response,
    ok_response,
    parse_codegen_request,
    require_serialized_spec,
    serialize_codegen_result,
    serialize_editor_result,
    serialize_spec_payload,
)
from ._protocol import read_json as _read_json
from ._services import build_bootstrap_payload, build_template_from_payload
from .session import EditorSession

LOGGER = logging.getLogger(__name__)


def read_json(body: bytes) -> JsonDict:
    """Parse a request body into a JSON object."""
    return _read_json(body)


def handle_bootstrap(session: EditorSession) -> JsonResponse:
    """Return the bootstrap payload used by the browser client."""
    return HTTPStatus.OK, cast(JsonDict, build_bootstrap_payload(session))


def handle_validate(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Validate a serialized spec or supported Python source payload."""
    del session
    try:
        spec = deserialize_validation_payload(payload)
    except SerializationError as exc:
        LOGGER.warning("Validation request contained malformed spec payload: %s", exc)
        return bad_request_response(str(exc))
    except SpecValidationError as exc:
        return issues_response(exc.issues)
    except ValueError:
        LOGGER.warning("Validation request missing 'spec' or 'python_code' payload.")
        return bad_request_response("Missing 'spec' or 'python_code' payload.")
    issues = validate_spec(spec)
    if issues:
        status, response = issues_response(issues)
        response["spec"] = serialize_spec_payload(spec)
        return status, response
    return ok_response({"issues": [], "spec": serialize_spec_payload(spec)})


def handle_generate(session: EditorSession, payload: JsonDict) -> JsonResponse:
    """Generate preview code for the current editor payload."""
    status, response = _handle_session_codegen_request(
        session=session,
        payload=payload,
        operation="generate",
    )
    message = response.get("message")
    if response.get("ok") is False and isinstance(message, str):
        LOGGER.warning("Generate request rejected: %s", message)
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
        LOGGER.warning("Complete request rejected: %s", message)
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


def handle_analyze_contraction(
    session: EditorSession, payload: JsonDict
) -> JsonResponse:
    """Analyze contraction information for a validated serialized spec."""
    del session
    try:
        serialized_spec = require_serialized_spec(payload)
    except ValueError:
        LOGGER.warning("Contraction analysis request missing 'spec' payload.")
        return bad_request_response("Missing 'spec' payload.")

    try:
        spec = deserialize_spec_with_issues(serialized_spec)
    except SerializationError as exc:
        LOGGER.warning("Contraction analysis request contained malformed spec: %s", exc)
        return bad_request_response(str(exc))

    issues = validate_spec(spec)
    if issues:
        return issues_response(issues)

    result = analyze_contraction(spec)
    return ok_response(_serialize_contraction_analysis_result(result))


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
                cast(dict[str, object], request.serialized_spec),
                request.engine,
                request.collection_format,
            )
            return ok_response(_serialize_generate_result(generate_result))
        if operation == "complete":
            complete_result = session.complete(
                cast(dict[str, object], request.serialized_spec),
                request.engine,
                request.collection_format,
            )
            return ok_response(_serialize_complete_result(complete_result))
        raise ValueError(f"Unsupported code generation operation '{operation}'.")
    except SerializationError as exc:
        return bad_request_response(str(exc))
    except CodeGenerationError as exc:
        return bad_request_response(str(exc))
    except SpecValidationError as exc:
        return issues_response(exc.issues)


def _serialize_contraction_analysis_result(
    result: ContractionAnalysisResult,
) -> JsonDict:
    """Serialize a contraction analysis result for the API."""
    return result.to_dict()

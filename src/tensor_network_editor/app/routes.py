from __future__ import annotations

import logging
from http import HTTPStatus
from typing import cast

from .._contraction_analysis import ContractionAnalysisResult, analyze_contraction
from .._templates import parse_template_parameters
from ..errors import CodeGenerationError, SerializationError, SpecValidationError
from ..models import CodegenResult, EditorResult
from ..serialization import serialize_spec
from ..validation import validate_spec
from ._protocol import (
    JsonDict,
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
from .session import EditorSession

LOGGER = logging.getLogger(__name__)


def read_json(body: bytes) -> JsonDict:
    return _read_json(body)


def handle_bootstrap(session: EditorSession) -> tuple[int, JsonDict]:
    return HTTPStatus.OK, session.bootstrap_payload()


def handle_validate(session: EditorSession, payload: JsonDict) -> tuple[int, JsonDict]:
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


def handle_generate(session: EditorSession, payload: JsonDict) -> tuple[int, JsonDict]:
    status, response = _handle_session_codegen_request(
        session=session,
        payload=payload,
        operation="generate",
    )
    if not response["ok"] and "message" in response:
        LOGGER.warning("Generate request rejected: %s", response["message"])
    return status, response


def handle_complete(session: EditorSession, payload: JsonDict) -> tuple[int, JsonDict]:
    status, response = _handle_session_codegen_request(
        session=session,
        payload=payload,
        operation="complete",
    )
    if not response["ok"] and "message" in response:
        LOGGER.warning("Complete request rejected: %s", response["message"])
    return status, response


def handle_cancel(session: EditorSession) -> tuple[int, JsonDict]:
    session.cancel()
    return ok_response()


def handle_template(session: EditorSession, payload: JsonDict) -> tuple[int, JsonDict]:
    template_name = payload.get("template")
    if not isinstance(template_name, str) or not template_name.strip():
        return bad_request_response("Missing 'template' payload.")
    try:
        parameters = parse_template_parameters(
            template_name,
            payload.get("parameters"),
        )
        spec = session.build_template(template_name, parameters)
    except ValueError as exc:
        return bad_request_response(str(exc))
    return ok_response({"spec": serialize_spec(spec)})


def handle_analyze_contraction(
    session: EditorSession, payload: JsonDict
) -> tuple[int, JsonDict]:
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


def _serialize_generate_result(result: object) -> JsonDict:
    if not isinstance(result, CodegenResult):
        raise TypeError("Generate handler expected a code generation result.")
    return serialize_codegen_result(result)


def _serialize_complete_result(result: object) -> JsonDict:
    if not isinstance(result, EditorResult):
        raise TypeError("Complete handler expected an editor result.")
    return serialize_editor_result(result)


def _handle_session_codegen_request(
    *,
    session: EditorSession,
    payload: JsonDict,
    operation: str,
) -> tuple[int, JsonDict]:
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
            result = session.generate(
                request.serialized_spec,
                request.engine,
                request.collection_format,
            )
            return ok_response(_serialize_generate_result(result))
        if operation == "complete":
            result = session.complete(
                request.serialized_spec,
                request.engine,
                request.collection_format,
            )
            return ok_response(_serialize_complete_result(result))
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
    return cast(JsonDict, result.to_dict())

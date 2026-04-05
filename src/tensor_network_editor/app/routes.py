from __future__ import annotations

import logging
from http import HTTPStatus
from typing import cast

from .._contraction_analysis import ContractionAnalysisResult, analyze_contraction
from .._templates import parse_template_parameters
from ..errors import SerializationError
from ..models import CodegenResult, EditorResult
from ..serialization import serialize_spec
from ..validation import validate_spec
from ._protocol import (
    JsonDict,
    bad_request_response,
    deserialize_spec_with_issues,
    handle_codegen_operation,
    issues_response,
    ok_response,
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
        serialized_spec = require_serialized_spec(payload)
    except ValueError:
        LOGGER.warning("Validation request missing 'spec' payload.")
        return bad_request_response("Missing 'spec' payload.")

    try:
        spec = deserialize_spec_with_issues(serialized_spec)
    except SerializationError as exc:
        LOGGER.warning("Validation request contained malformed spec payload: %s", exc)
        return bad_request_response(str(exc))
    issues = validate_spec(spec)
    if issues:
        status, response = issues_response(issues)
        response["spec"] = serialize_spec_payload(spec)
        return status, response
    return ok_response({"issues": [], "spec": serialize_spec_payload(spec)})


def handle_generate(session: EditorSession, payload: JsonDict) -> tuple[int, JsonDict]:
    status, response = handle_codegen_operation(
        payload,
        default_engine=session.default_engine,
        operation=session.generate,
        success_payload_builder=_serialize_generate_result,
    )
    if not response["ok"] and "message" in response:
        LOGGER.warning("Generate request rejected: %s", response["message"])
    return status, response


def handle_complete(session: EditorSession, payload: JsonDict) -> tuple[int, JsonDict]:
    status, response = handle_codegen_operation(
        payload,
        default_engine=session.default_engine,
        operation=session.complete,
        success_payload_builder=_serialize_complete_result,
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


def _serialize_contraction_analysis_result(
    result: ContractionAnalysisResult,
) -> JsonDict:
    return cast(JsonDict, result.to_dict())

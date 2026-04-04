from __future__ import annotations

import logging
from http import HTTPStatus

from ..errors import SerializationError
from ..models import CodegenResult, EditorResult
from ..serialization import SCHEMA_VERSION, serialize_spec
from ..validation import validate_spec
from ._protocol import (
    JsonDict,
    deserialize_spec_with_issues,
    handle_codegen_operation,
    require_serialized_spec,
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
        return HTTPStatus.BAD_REQUEST, {
            "ok": False,
            "message": "Missing 'spec' payload.",
        }

    try:
        spec = deserialize_spec_with_issues(serialized_spec)
    except SerializationError as exc:
        LOGGER.warning("Validation request contained malformed spec payload: %s", exc)
        return HTTPStatus.BAD_REQUEST, {"ok": False, "message": str(exc)}
    issues = validate_spec(spec)
    return (
        HTTPStatus.OK,
        {
            "ok": not issues,
            "issues": [
                {"code": issue.code, "message": issue.message, "path": issue.path}
                for issue in issues
            ],
            "spec": {"schema_version": SCHEMA_VERSION, "network": spec.to_dict()},
        },
    )


def handle_generate(session: EditorSession, payload: JsonDict) -> tuple[int, JsonDict]:
    status, response = handle_codegen_operation(
        payload,
        default_engine=session.default_engine,
        operation=session.generate,
        success_payload_builder=_build_generate_response,
    )
    if status == HTTPStatus.BAD_REQUEST:
        LOGGER.warning("Generate request rejected: %s", response["message"])
    return status, response


def handle_complete(session: EditorSession, payload: JsonDict) -> tuple[int, JsonDict]:
    status, response = handle_codegen_operation(
        payload,
        default_engine=session.default_engine,
        operation=session.complete,
        success_payload_builder=_build_complete_response,
    )
    if status == HTTPStatus.BAD_REQUEST:
        LOGGER.warning("Complete request rejected: %s", response["message"])
    return status, response


def handle_cancel(session: EditorSession) -> tuple[int, JsonDict]:
    session.cancel()
    return HTTPStatus.OK, {"ok": True}


def handle_template(session: EditorSession, payload: JsonDict) -> tuple[int, JsonDict]:
    template_name = payload.get("template")
    if not isinstance(template_name, str) or not template_name.strip():
        return HTTPStatus.BAD_REQUEST, {
            "ok": False,
            "message": "Missing 'template' payload.",
        }
    try:
        spec = session.build_template(template_name)
    except ValueError as exc:
        return HTTPStatus.BAD_REQUEST, {"ok": False, "message": str(exc)}
    return HTTPStatus.OK, {"ok": True, "spec": serialize_spec(spec)}


def _build_generate_response(result: CodegenResult | EditorResult) -> JsonDict:
    if not isinstance(result, CodegenResult):
        raise TypeError("Generate handler expected a code generation result.")
    return {
        "engine": result.engine.value,
        "code": result.code,
        "warnings": result.warnings,
        "artifacts": result.artifacts,
    }


def _build_complete_response(result: CodegenResult | EditorResult) -> JsonDict:
    if not isinstance(result, EditorResult):
        raise TypeError("Complete handler expected an editor result.")
    return {"engine": result.engine.value, "confirmed": result.confirmed}

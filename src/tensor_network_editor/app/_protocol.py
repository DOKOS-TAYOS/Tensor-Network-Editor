from __future__ import annotations

import json
from collections.abc import Callable
from http import HTTPStatus
from typing import Any, TypeAlias

from ..errors import SerializationError, SpecValidationError
from ..models import (
    CodegenResult,
    EditorResult,
    EngineName,
    NetworkSpec,
    ValidationIssue,
)
from ..serialization import SCHEMA_VERSION, deserialize_spec

JsonDict: TypeAlias = dict[str, Any]
CodegenOperation: TypeAlias = Callable[
    [JsonDict, EngineName], CodegenResult | EditorResult
]
JsonResponse: TypeAlias = tuple[int, JsonDict]


def read_json(body: bytes) -> JsonDict:
    if not body:
        return {}
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Request body contains invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object payload.")
    return payload


def require_serialized_spec(payload: JsonDict) -> JsonDict:
    serialized_spec = payload.get("spec")
    if not isinstance(serialized_spec, dict):
        raise ValueError("Missing 'spec' payload.")
    return serialized_spec


def resolve_engine(payload: JsonDict, default_engine: EngineName) -> EngineName:
    engine_value = payload.get("engine", default_engine.value)
    try:
        return EngineName(str(engine_value))
    except ValueError as exc:
        raise ValueError(f"Unsupported engine '{engine_value}'.") from exc


def serialize_issues(issues: list[ValidationIssue]) -> list[JsonDict]:
    return [
        {"code": issue.code, "message": issue.message, "path": issue.path}
        for issue in issues
    ]


def ok_response(payload: JsonDict | None = None) -> JsonResponse:
    body = {"ok": True}
    if payload is not None:
        body.update(payload)
    return HTTPStatus.OK, body


def bad_request_response(message: str) -> JsonResponse:
    return HTTPStatus.BAD_REQUEST, {"ok": False, "message": message}


def not_found_response() -> JsonResponse:
    return HTTPStatus.NOT_FOUND, {"ok": False, "message": "Not found."}


def internal_server_error_response() -> JsonResponse:
    return HTTPStatus.INTERNAL_SERVER_ERROR, {
        "ok": False,
        "message": "Internal server error.",
    }


def issues_response(issues: list[ValidationIssue]) -> JsonResponse:
    return HTTPStatus.OK, {"ok": False, "issues": serialize_issues(issues)}


def serialize_spec_payload(spec: NetworkSpec) -> JsonDict:
    return {"schema_version": SCHEMA_VERSION, "network": spec.to_dict()}


def serialize_codegen_result(result: CodegenResult) -> JsonDict:
    return {
        "engine": result.engine.value,
        "code": result.code,
        "warnings": result.warnings,
        "artifacts": result.artifacts,
    }


def serialize_editor_result(result: EditorResult) -> JsonDict:
    return {"engine": result.engine.value, "confirmed": result.confirmed}


def handle_codegen_operation(
    payload: JsonDict,
    *,
    default_engine: EngineName,
    operation: CodegenOperation,
    success_payload_builder: Callable[[CodegenResult | EditorResult], JsonDict],
) -> JsonResponse:
    try:
        serialized_spec = require_serialized_spec(payload)
    except ValueError:
        return bad_request_response("Missing 'spec' payload.")

    try:
        engine = resolve_engine(payload, default_engine)
    except ValueError as exc:
        return bad_request_response(str(exc))

    try:
        result = operation(serialized_spec, engine)
    except SerializationError as exc:
        return bad_request_response(str(exc))
    except SpecValidationError as exc:
        return issues_response(exc.issues)

    return ok_response(success_payload_builder(result))


def deserialize_spec_with_issues(serialized_spec: JsonDict) -> NetworkSpec:
    return deserialize_spec(serialized_spec, validate=False)

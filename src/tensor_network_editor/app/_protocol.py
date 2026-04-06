from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from http import HTTPStatus
from typing import Any, TypeAlias

from ..errors import CodeGenerationError, SerializationError, SpecValidationError
from ..models import (
    CodegenResult,
    EditorResult,
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
    ValidationIssue,
)
from ..serialization import (
    SCHEMA_VERSION,
    deserialize_spec,
    deserialize_spec_from_python_code,
)

JsonDict: TypeAlias = dict[str, Any]
CodegenOperation: TypeAlias = Callable[..., CodegenResult | EditorResult]
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


def deserialize_validation_payload(payload: JsonDict) -> NetworkSpec:
    serialized_spec = payload.get("spec")
    if isinstance(serialized_spec, dict):
        return deserialize_spec_with_issues(serialized_spec)

    python_code = payload.get("python_code")
    if isinstance(python_code, str):
        if not python_code.strip():
            raise ValueError("Missing 'spec' or 'python_code' payload.")
        return deserialize_spec_from_python_code(python_code, validate=False)

    raise ValueError("Missing 'spec' or 'python_code' payload.")


def resolve_engine(payload: JsonDict, default_engine: EngineName) -> EngineName:
    engine_value = payload.get("engine", default_engine.value)
    try:
        return EngineName(str(engine_value))
    except ValueError as exc:
        raise ValueError(f"Unsupported engine '{engine_value}'.") from exc


def resolve_collection_format(
    payload: JsonDict,
    default_collection_format: TensorCollectionFormat,
) -> TensorCollectionFormat:
    collection_format_value = payload.get(
        "collection_format", default_collection_format.value
    )
    try:
        return TensorCollectionFormat(str(collection_format_value))
    except ValueError as exc:
        raise ValueError(
            f"Unsupported collection format '{collection_format_value}'."
        ) from exc


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
    default_collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
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
        collection_format = resolve_collection_format(
            payload, default_collection_format
        )
    except ValueError as exc:
        return bad_request_response(str(exc))

    try:
        if operation_accepts_collection_format(operation):
            result = operation(serialized_spec, engine, collection_format)
        else:
            result = operation(serialized_spec, engine)
    except SerializationError as exc:
        return bad_request_response(str(exc))
    except CodeGenerationError as exc:
        return bad_request_response(str(exc))
    except SpecValidationError as exc:
        return issues_response(exc.issues)

    return ok_response(success_payload_builder(result))


def deserialize_spec_with_issues(serialized_spec: JsonDict) -> NetworkSpec:
    return deserialize_spec(serialized_spec, validate=False)


def operation_accepts_collection_format(operation: CodegenOperation) -> bool:
    try:
        signature = inspect.signature(operation)
    except (TypeError, ValueError):
        return True

    positional_parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    ):
        return True
    return len(positional_parameters) >= 3

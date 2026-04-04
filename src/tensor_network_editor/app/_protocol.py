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

JsonDict: TypeAlias = dict[str, Any]
CodegenOperation: TypeAlias = Callable[[JsonDict, EngineName], CodegenResult | EditorResult]


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


def handle_codegen_operation(
    payload: JsonDict,
    *,
    default_engine: EngineName,
    operation_name: str,
    operation: CodegenOperation,
    success_payload_builder: Callable[[CodegenResult | EditorResult], JsonDict],
) -> tuple[int, JsonDict]:
    try:
        serialized_spec = require_serialized_spec(payload)
    except ValueError:
        return HTTPStatus.BAD_REQUEST, {"ok": False, "message": "Missing 'spec' payload."}

    try:
        engine = resolve_engine(payload, default_engine)
    except ValueError as exc:
        return HTTPStatus.BAD_REQUEST, {"ok": False, "message": str(exc)}

    try:
        result = operation(serialized_spec, engine)
    except SerializationError as exc:
        return HTTPStatus.BAD_REQUEST, {"ok": False, "message": str(exc)}
    except SpecValidationError as exc:
        return HTTPStatus.OK, {"ok": False, "issues": serialize_issues(exc.issues)}

    response = success_payload_builder(result)
    response["ok"] = True
    return HTTPStatus.OK, response


def deserialize_spec_with_issues(serialized_spec: JsonDict) -> NetworkSpec:
    network_payload = serialized_spec.get("network")
    if not isinstance(network_payload, dict):
        raise SerializationError("Serialized payload must include a network object.")

    try:
        return NetworkSpec.from_dict(network_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise SerializationError(
            "Serialized payload contains a malformed network object."
        ) from exc

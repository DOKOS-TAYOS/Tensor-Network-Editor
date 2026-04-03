from __future__ import annotations

import json
import logging
from http import HTTPStatus
from typing import Any

from ..errors import SerializationError, SpecValidationError
from ..models import EngineName, NetworkSpec
from ..serialization import SCHEMA_VERSION
from ..validation import validate_spec
from .session import EditorSession

LOGGER = logging.getLogger(__name__)


def read_json(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Request body contains invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object payload.")
    return payload


def handle_bootstrap(session: EditorSession) -> tuple[int, dict[str, Any]]:
    return HTTPStatus.OK, session.bootstrap_payload()


def handle_validate(
    session: EditorSession, payload: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    serialized_spec = payload.get("spec")
    if not isinstance(serialized_spec, dict):
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


def handle_generate(
    session: EditorSession, payload: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    serialized_spec = payload.get("spec")
    engine_value = payload.get("engine", session.default_engine.value)
    if not isinstance(serialized_spec, dict):
        LOGGER.warning("Generate request missing 'spec' payload.")
        return HTTPStatus.BAD_REQUEST, {
            "ok": False,
            "message": "Missing 'spec' payload.",
        }
    try:
        engine = EngineName(str(engine_value))
    except ValueError:
        LOGGER.warning("Generate request used unsupported engine '%s'.", engine_value)
        return HTTPStatus.BAD_REQUEST, {
            "ok": False,
            "message": f"Unsupported engine '{engine_value}'.",
        }

    try:
        result = session.generate(serialized_spec=serialized_spec, engine=engine)
    except SerializationError as exc:
        LOGGER.warning("Generate request contained malformed payload: %s", exc)
        return HTTPStatus.BAD_REQUEST, {"ok": False, "message": str(exc)}
    except SpecValidationError as exc:
        return (
            HTTPStatus.OK,
            {
                "ok": False,
                "issues": [
                    {"code": issue.code, "message": issue.message, "path": issue.path}
                    for issue in exc.issues
                ],
            },
        )
    return (
        HTTPStatus.OK,
        {
            "ok": True,
            "engine": result.engine.value,
            "code": result.code,
            "warnings": result.warnings,
            "artifacts": result.artifacts,
        },
    )


def handle_complete(
    session: EditorSession, payload: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    serialized_spec = payload.get("spec")
    engine_value = payload.get("engine", session.default_engine.value)
    if not isinstance(serialized_spec, dict):
        LOGGER.warning("Complete request missing 'spec' payload.")
        return HTTPStatus.BAD_REQUEST, {
            "ok": False,
            "message": "Missing 'spec' payload.",
        }
    try:
        engine = EngineName(str(engine_value))
    except ValueError:
        LOGGER.warning("Complete request used unsupported engine '%s'.", engine_value)
        return HTTPStatus.BAD_REQUEST, {
            "ok": False,
            "message": f"Unsupported engine '{engine_value}'.",
        }

    try:
        result = session.complete(serialized_spec=serialized_spec, engine=engine)
    except SerializationError as exc:
        LOGGER.warning("Complete request contained malformed payload: %s", exc)
        return HTTPStatus.BAD_REQUEST, {"ok": False, "message": str(exc)}
    except SpecValidationError as exc:
        return (
            HTTPStatus.OK,
            {
                "ok": False,
                "issues": [
                    {"code": issue.code, "message": issue.message, "path": issue.path}
                    for issue in exc.issues
                ],
            },
        )
    return (
        HTTPStatus.OK,
        {
            "ok": True,
            "engine": result.engine.value,
            "confirmed": result.confirmed,
        },
    )


def handle_cancel(session: EditorSession) -> tuple[int, dict[str, Any]]:
    session.cancel()
    return HTTPStatus.OK, {"ok": True}


def deserialize_spec_with_issues(serialized_spec: dict[str, Any]) -> NetworkSpec:
    network_payload = serialized_spec.get("network")
    if not isinstance(network_payload, dict):
        raise SerializationError("Serialized payload must include a network object.")

    from ..models import NetworkSpec

    try:
        return NetworkSpec.from_dict(network_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise SerializationError(
            "Serialized payload contains a malformed network object."
        ) from exc

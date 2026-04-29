"""Project-local draft persistence for browser editor sessions."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from ..codegen.registry import engine_name_to_text
from ..internal._logging import log_branch, log_operation
from ..models import EngineIdentifier, TensorCollectionFormat
from ..types import JSONValue, StrPath
from ._protocol import JsonDict

DRAFT_SCHEMA_VERSION = 1
DEFAULT_DRAFT_PATH = Path(".tensor-network-editor") / "drafts" / "active.json"
LOGGER = logging.getLogger(__name__)


def resolve_project_draft_path(draft_path: StrPath | None = None) -> Path:
    """Return the project-local draft path used by one editor session."""
    if draft_path is not None:
        return Path(draft_path)
    return Path.cwd() / DEFAULT_DRAFT_PATH


def load_project_draft(draft_path: StrPath) -> JsonDict | None:
    """Load the active project draft, or ``None`` when no draft exists."""
    resolved_path = Path(draft_path)
    with log_operation(
        LOGGER,
        "Draft load from disk",
        context={"path": resolved_path},
        emit_start=False,
    ) as success_context:
        if not resolved_path.exists():
            success_context["status"] = "missing"
            log_branch(LOGGER, "No project draft found on disk")
            return None
        try:
            raw_payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log_branch(
                LOGGER,
                f"Could not read project draft payload: {exc}",
                level=logging.WARNING,
            )
            raise ValueError(f"Could not read draft from {resolved_path}.") from exc
        if not isinstance(raw_payload, dict):
            log_branch(
                LOGGER,
                "Rejected non-object project draft payload",
                level=logging.WARNING,
            )
            raise ValueError("Draft payload must be a JSON object.")
        draft = _coerce_draft_payload(raw_payload)
        success_context["status"] = "loaded"
        success_context["engine"] = draft["engine"]
        success_context["format"] = draft["collection_format"]
        return draft


def save_project_draft(
    draft_path: StrPath,
    *,
    serialized_spec: JsonDict,
    engine: EngineIdentifier,
    collection_format: TensorCollectionFormat,
) -> JsonDict:
    """Persist and return one editor draft payload."""
    resolved_path = Path(draft_path)
    with log_operation(
        LOGGER,
        "Draft persistence",
        context={"path": resolved_path},
        emit_start=False,
    ) as success_context:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        draft_payload: JsonDict = {
            "schema_version": DRAFT_SCHEMA_VERSION,
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "spec": cast(JSONValue, serialized_spec),
            "engine": engine_name_to_text(engine),
            "collection_format": collection_format.value,
        }
        resolved_path.write_text(
            json.dumps(draft_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        success_context["status"] = "saved"
        success_context["engine"] = draft_payload["engine"]
        success_context["format"] = collection_format.value
        return draft_payload


def clear_project_draft(draft_path: StrPath) -> None:
    """Remove the active project draft when present."""
    resolved_path = Path(draft_path)
    with log_operation(
        LOGGER,
        "Draft clear from disk",
        context={"path": resolved_path},
        emit_start=False,
    ) as success_context:
        try:
            resolved_path.unlink()
        except FileNotFoundError:
            success_context["status"] = "missing"
            log_branch(LOGGER, "Project draft already absent on disk")
            return
        success_context["status"] = "cleared"


def _coerce_draft_payload(payload: dict[str, object]) -> JsonDict:
    """Validate the persisted draft envelope enough for browser recovery."""
    schema_version = payload.get("schema_version")
    if schema_version != DRAFT_SCHEMA_VERSION:
        log_branch(
            LOGGER,
            f"Rejected unsupported project draft schema version '{schema_version}'",
            level=logging.WARNING,
        )
        raise ValueError(f"Unsupported draft schema version '{schema_version}'.")
    spec = payload.get("spec")
    saved_at = payload.get("saved_at")
    engine = payload.get("engine")
    collection_format = payload.get("collection_format")
    if not isinstance(spec, dict):
        log_branch(
            LOGGER,
            "Rejected project draft without spec payload",
            level=logging.WARNING,
        )
        raise ValueError("Draft payload is missing a spec object.")
    if not isinstance(saved_at, str) or not saved_at.strip():
        log_branch(
            LOGGER,
            "Rejected project draft without saved_at value",
            level=logging.WARNING,
        )
        raise ValueError("Draft payload is missing a saved_at value.")
    if not isinstance(engine, str) or not engine.strip():
        log_branch(
            LOGGER,
            "Rejected project draft without engine value",
            level=logging.WARNING,
        )
        raise ValueError("Draft payload is missing an engine value.")
    if not isinstance(collection_format, str) or not collection_format.strip():
        log_branch(
            LOGGER,
            "Rejected project draft without collection_format value",
            level=logging.WARNING,
        )
        raise ValueError("Draft payload is missing a collection_format value.")
    return {
        "schema_version": DRAFT_SCHEMA_VERSION,
        "saved_at": saved_at,
        "spec": cast(JSONValue, spec),
        "engine": engine,
        "collection_format": collection_format,
    }

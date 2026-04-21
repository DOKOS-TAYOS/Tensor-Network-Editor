"""Session code generation request helpers for the local editor."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..codegen.registry import engine_name_to_text
from ..codegen.registry import generate_code as generate_code_internal
from ..internal.io._io import write_utf8_text
from ..internal.io._serialization import deserialize_spec
from ..models import (
    CodegenResult,
    EditorResult,
    EngineIdentifier,
    TensorCollectionFormat,
)

if TYPE_CHECKING:
    from .session import EditorSession


LOGGER = logging.getLogger(__name__)


def generate_session_request(
    session: EditorSession,
    serialized_spec: Mapping[str, object],
    engine: EngineIdentifier,
    collection_format: TensorCollectionFormat | None = None,
) -> CodegenResult:
    """Generate preview code for one editor request."""
    LOGGER.debug(
        "[session=%s] Generating preview request for engine '%s'",
        session.session_id,
        engine_name_to_text(engine),
    )
    spec = deserialize_spec(serialized_spec)
    return generate_code_internal(
        spec,
        engine,
        collection_format=_resolve_collection_format(session, collection_format),
        validate=False,
    )


def complete_session_request(
    session: EditorSession,
    serialized_spec: Mapping[str, object],
    engine: EngineIdentifier,
    collection_format: TensorCollectionFormat | None = None,
) -> EditorResult:
    """Finalize a session request and optionally print or save generated code."""
    LOGGER.info(
        "[session=%s] Completing session request for engine '%s'",
        session.session_id,
        engine_name_to_text(engine),
    )
    spec = deserialize_spec(serialized_spec)
    codegen_result = generate_code_internal(
        spec,
        engine,
        collection_format=_resolve_collection_format(session, collection_format),
        validate=False,
    )
    if session.print_code:
        LOGGER.debug(
            "[session=%s] Printing generated code to stdout", session.session_id
        )
        print(codegen_result.code)
    if session.code_path is not None:
        LOGGER.debug(
            "[session=%s] Writing generated code to %s",
            session.session_id,
            session.code_path,
        )
        write_utf8_text(
            session.code_path,
            codegen_result.code,
            description="generated Python code",
        )
    return EditorResult(
        spec=spec, engine=engine, codegen=codegen_result, confirmed=True
    )


def _resolve_collection_format(
    session: EditorSession,
    collection_format: TensorCollectionFormat | None,
) -> TensorCollectionFormat:
    """Resolve a request collection format against the session default."""
    return collection_format or session.default_collection_format

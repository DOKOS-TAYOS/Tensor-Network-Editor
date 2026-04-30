"""Session code generation request helpers for the local editor."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..codegen.registry import engine_name_to_text
from ..codegen.registry import generate_code as generate_code_internal
from ..internal._logging import log_branch, log_operation, summarize_spec_counts
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
    include_roundtrip_metadata: bool = True,
) -> CodegenResult:
    """Generate preview code for one editor request."""
    with log_operation(
        LOGGER,
        "Preview code generation",
        context={"engine": engine_name_to_text(engine)},
    ):
        spec = deserialize_spec(serialized_spec)
        log_branch(
            LOGGER,
            "Deserialized preview spec",
            context=summarize_spec_counts(spec),
        )
        return generate_code_internal(
            spec,
            engine,
            collection_format=_resolve_collection_format(session, collection_format),
            include_roundtrip_metadata=include_roundtrip_metadata,
            validate=False,
        )


def complete_session_request(
    session: EditorSession,
    serialized_spec: Mapping[str, object],
    engine: EngineIdentifier,
    collection_format: TensorCollectionFormat | None = None,
    include_roundtrip_metadata: bool = True,
) -> EditorResult:
    """Finalize a session request and optionally print or save generated code."""
    with log_operation(
        LOGGER,
        "Session completion request",
        start_level=logging.INFO,
        success_level=logging.INFO,
        context={"engine": engine_name_to_text(engine)},
    ):
        spec = deserialize_spec(serialized_spec)
        log_branch(
            LOGGER,
            "Deserialized completion spec",
            context=summarize_spec_counts(spec),
        )
        codegen_result = generate_code_internal(
            spec,
            engine,
            collection_format=_resolve_collection_format(session, collection_format),
            include_roundtrip_metadata=include_roundtrip_metadata,
            validate=False,
        )
        if session.print_code:
            log_branch(LOGGER, "Printing generated code to stdout")
            print(codegen_result.code)
        if session.code_path is not None:
            log_branch(
                LOGGER,
                "Writing generated code to disk",
                context={"output_path": session.code_path},
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

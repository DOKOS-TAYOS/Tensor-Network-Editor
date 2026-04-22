"""Public code-generation helper with filesystem convenience options."""

from __future__ import annotations

import logging

from .codegen.registry import engine_name_to_text
from .codegen.registry import generate_code as _generate_code
from .internal.io._io import write_utf8_text
from .models import (
    CodegenResult,
    EngineIdentifier,
    NetworkSpec,
    TensorCollectionFormat,
)
from .types import StrPath

LOGGER = logging.getLogger(__name__)


def generate_code(
    spec: NetworkSpec,
    *,
    engine: EngineIdentifier,
    collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    output_path: StrPath | None = None,
    print_code: bool = False,
) -> CodegenResult:
    """Generate Python code for one tensor-network specification."""
    LOGGER.info(
        "Generating %s code for network '%s'",
        engine_name_to_text(engine),
        spec.name,
    )
    result = _generate_code(spec, engine, collection_format=collection_format)
    if print_code:
        LOGGER.debug(
            "Printing generated %s code to stdout", engine_name_to_text(engine)
        )
        print(result.code)
    if output_path is not None:
        LOGGER.debug(
            "Writing generated %s code to %s",
            engine_name_to_text(engine),
            output_path,
        )
        write_utf8_text(
            output_path,
            result.code,
            description="generated Python code",
        )
    return result


__all__ = ["generate_code"]

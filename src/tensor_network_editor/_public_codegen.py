"""Public code-generation helper with filesystem convenience options."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path, PureWindowsPath

from .codegen.registry import engine_name_to_text
from .codegen.registry import generate_code as _generate_code
from .internal._logging import log_branch, log_operation, summarize_spec_counts
from .internal.io._io import write_utf8_text
from .models import (
    CodegenResult,
    EngineIdentifier,
    NetworkSpec,
    TensorCollectionFormat,
    TensorDataMode,
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
    external_data_base_path: StrPath | None = None,
) -> CodegenResult:
    """Generate Python code for one tensor-network specification."""
    context = {
        "engine": engine_name_to_text(engine),
        "output_path": output_path,
        **summarize_spec_counts(spec),
    }
    with log_operation(LOGGER, "Code generation", context=context):
        codegen_spec = _with_resolved_external_data_paths(
            spec,
            external_data_base_path=external_data_base_path,
        )
        result = _generate_code(
            codegen_spec, engine, collection_format=collection_format
        )
        if print_code:
            log_branch(LOGGER, "Printing generated code to stdout")
            print(result.code)
        if output_path is not None:
            log_branch(LOGGER, "Writing generated code to disk")
            write_utf8_text(
                output_path,
                result.code,
                description="generated Python code",
            )
        return result


def _with_resolved_external_data_paths(
    spec: NetworkSpec,
    *,
    external_data_base_path: StrPath | None,
) -> NetworkSpec:
    """Return a copy with relative external-data paths anchored to a base path."""
    if external_data_base_path is None:
        return spec
    resolved_spec = deepcopy(spec)
    base_path = Path(external_data_base_path)
    for tensor in resolved_spec.tensors:
        tensor_data = tensor.tensor_data
        if (
            tensor_data is None
            or tensor_data.mode is not TensorDataMode.EXTERNAL
            or tensor_data.file_path is None
            or _is_absolute_path_text(tensor_data.file_path)
        ):
            continue
        tensor_data.file_path = (base_path / tensor_data.file_path).as_posix()
    return resolved_spec


def _is_absolute_path_text(path_text: str) -> bool:
    """Return whether a path string is absolute on POSIX or Windows."""
    return Path(path_text).is_absolute() or PureWindowsPath(path_text).is_absolute()


__all__ = ["generate_code"]

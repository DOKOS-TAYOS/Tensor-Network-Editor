"""Public API helpers for launching the editor and working with specs."""

from __future__ import annotations

import logging
from collections.abc import Callable

from ._io import write_utf8_text
from .codegen.registry import generate_code as _generate_code
from .models import (
    CodegenResult,
    EditorResult,
    EngineName,
    NetworkSpec,
    TensorCollectionFormat,
)
from .serialization import load_spec as _load_spec
from .serialization import load_spec_from_python_code as _load_spec_from_python_code
from .serialization import save_spec as _save_spec
from .types import StrPath

LOGGER = logging.getLogger(__name__)


def generate_code(
    spec: NetworkSpec,
    engine: EngineName,
    *,
    collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    print_code: bool = False,
    path: StrPath | None = None,
) -> CodegenResult:
    """Generate Python code for ``spec`` using the requested backend and layout."""
    LOGGER.info("Generating %s code for network '%s'", engine.value, spec.name)
    result = _generate_code(spec, engine, collection_format=collection_format)
    if print_code:
        print(result.code)
    if path is not None:
        write_utf8_text(path, result.code, description="generated Python code")
    return result


def save_spec(spec: NetworkSpec, path: StrPath) -> None:
    """Validate and save ``spec`` as a versioned JSON document."""
    _save_spec(spec, path)


def load_spec(path: StrPath) -> NetworkSpec:
    """Load a saved JSON spec or a supported generated Python export."""
    return _load_spec(path)


def load_spec_from_python_code(code: str) -> NetworkSpec:
    """Reconstruct a network specification from supported generated Python code."""
    return _load_spec_from_python_code(code)


def launch_tensor_network_editor(
    initial_spec: NetworkSpec | None = None,
    *,
    default_engine: EngineName = EngineName.TENSORKROWCH,
    default_collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    print_code: bool = False,
    code_path: StrPath | None = None,
    _on_server_ready: Callable[[str], None] | None = None,
) -> EditorResult | None:
    """Launch the local editor session and block until it is confirmed or cancelled."""
    from .app.session import launch_editor_session

    LOGGER.info(
        "Launching tensor network editor with engine '%s'", default_engine.value
    )
    return launch_editor_session(
        initial_spec=initial_spec,
        default_engine=default_engine,
        default_collection_format=default_collection_format,
        open_browser=open_browser,
        host=host,
        port=port,
        print_code=print_code,
        code_path=code_path,
        _on_server_ready=_on_server_ready,
    )

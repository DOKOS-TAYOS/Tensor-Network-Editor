"""Public API helpers for launching the editor and working with specs."""

from __future__ import annotations

import logging
from collections.abc import Callable

from ._io import write_utf8_text
from .codegen.registry import engine_name_to_text
from .codegen.registry import generate_code as _generate_code
from .models import (
    CodegenResult,
    EditorResult,
    EngineIdentifier,
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
    engine: EngineIdentifier,
    *,
    collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    print_code: bool = False,
    path: StrPath | None = None,
) -> CodegenResult:
    """Generate Python code for a tensor-network specification.

    Args:
        spec: Valid network specification to export.
        engine: Target backend for the generated Python code.
        collection_format: Layout used to collect generated tensor variables.
        print_code: Whether to print the generated source to standard output.
        path: Optional destination file for the generated source.

    Returns:
        The generated code bundle, including warnings and backend artifacts.
    """
    LOGGER.info(
        "Generating %s code for network '%s'",
        engine_name_to_text(engine),
        spec.name,
    )
    result = _generate_code(spec, engine, collection_format=collection_format)
    if print_code:
        print(result.code)
    if path is not None:
        write_utf8_text(path, result.code, description="generated Python code")
    return result


def save_spec(spec: NetworkSpec, path: StrPath) -> None:
    """Validate and save a network specification as versioned JSON.

    Args:
        spec: Network specification to validate and serialize.
        path: Destination path for the JSON document.
    """
    _save_spec(spec, path)


def load_spec(path: StrPath) -> NetworkSpec:
    """Load a saved JSON spec or a supported generated Python export.

    Args:
        path: Path to a saved JSON design or supported generated Python file.

    Returns:
        The parsed network specification.
    """
    return _load_spec(path)


def load_spec_from_python_code(code: str) -> NetworkSpec:
    """Reconstruct a network specification from generated Python source.

    Args:
        code: Generated Python source emitted by a supported standard network
            export.

    Returns:
        The reconstructed network specification.
    """
    return _load_spec_from_python_code(code)


def launch_tensor_network_editor(
    initial_spec: NetworkSpec | None = None,
    *,
    default_engine: EngineIdentifier = EngineName.TENSORKROWCH,
    default_collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    print_code: bool = False,
    code_path: StrPath | None = None,
    _on_server_ready: Callable[[str], None] | None = None,
) -> EditorResult | None:
    """Launch the local editor session and wait for it to finish.

    Args:
        initial_spec: Optional network specification to preload in the editor.
        default_engine: Backend initially selected in the editor UI.
        default_collection_format: Initial tensor collection layout for generated
            code.
        open_browser: Whether to open the local editor URL automatically.
        host: Local host interface to bind the editor server.
        port: Local port to bind. Use ``0`` to let the OS choose one.
        print_code: Whether to print generated code after the session is
            confirmed.
        code_path: Optional output path for generated code after confirmation.
        _on_server_ready: Internal callback used by tests once the server URL is
            known.

    Returns:
        ``None`` when the session is cancelled, otherwise the confirmed editor
        result.

    Raises:
        KeyboardInterrupt: If the session is interrupted from the main thread.
    """
    from .app.session import launch_editor_session

    LOGGER.info(
        "Launching tensor network editor with engine '%s'",
        engine_name_to_text(default_engine),
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

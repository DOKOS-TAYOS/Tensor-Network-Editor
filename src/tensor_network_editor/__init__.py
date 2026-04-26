"""Public package exports for Tensor Network Editor."""

from __future__ import annotations

import logging
import sys
from importlib import import_module
from typing import TYPE_CHECKING

if sys.version_info < (3, 11):  # noqa: UP036 - explicit runtime guard for unsupported interpreters
    raise RuntimeError("tensor-network-editor requires Python 3.11 or newer.")

from ._version import __version__

if TYPE_CHECKING:
    from ._public_codegen import generate_code
    from .analysis import analyze_contraction, analyze_spec
    from .builder import IndexHandle, NetworkBuilder, TensorHandle
    from .canonicalization import canonicalize_spec
    from .editor import EditorLaunchOptions, open_editor
    from .internal.diffing._diffing import diff_specs, semantic_diff_specs
    from .io import PythonLoadOptions, load_python_spec, load_spec, save_spec
    from .linting import lint_spec
    from .models import (
        CanvasNoteSpec,
        CanvasPosition,
        CodegenResult,
        ContractionOperandLayoutSpec,
        ContractionPlanSpec,
        ContractionStepSpec,
        ContractionViewSnapshotSpec,
        EdgeEndpointRef,
        EdgeSpec,
        EditorResult,
        EngineName,
        GroupSpec,
        HyperedgeSpec,
        IndexSpec,
        NetworkSpec,
        TensorCollectionFormat,
        TensorDataMode,
        TensorDataSpec,
        TensorSize,
        TensorSpec,
        ValidationIssue,
    )
    from .rendering import SvgRenderOptions, render_spec_png, render_spec_svg
    from .templates import build_template_spec, list_template_names
    from .validation import validate_spec

PACKAGE_LOGGER = logging.getLogger(__name__)
if not any(
    isinstance(handler, logging.NullHandler) for handler in PACKAGE_LOGGER.handlers
):
    PACKAGE_LOGGER.addHandler(logging.NullHandler())

__all__ = [
    "CanvasPosition",
    "CanvasNoteSpec",
    "CodegenResult",
    "ContractionOperandLayoutSpec",
    "ContractionPlanSpec",
    "ContractionStepSpec",
    "ContractionViewSnapshotSpec",
    "EdgeEndpointRef",
    "EdgeSpec",
    "EditorLaunchOptions",
    "EditorResult",
    "EngineName",
    "GroupSpec",
    "HyperedgeSpec",
    "IndexHandle",
    "IndexSpec",
    "NetworkSpec",
    "NetworkBuilder",
    "PythonLoadOptions",
    "SvgRenderOptions",
    "TensorCollectionFormat",
    "TensorDataMode",
    "TensorDataSpec",
    "TensorHandle",
    "TensorSize",
    "TensorSpec",
    "ValidationIssue",
    "__version__",
    "analyze_contraction",
    "analyze_spec",
    "build_template_spec",
    "canonicalize_spec",
    "diff_specs",
    "generate_code",
    "lint_spec",
    "list_template_names",
    "load_python_spec",
    "load_spec",
    "open_editor",
    "render_spec_png",
    "render_spec_svg",
    "save_spec",
    "semantic_diff_specs",
    "validate_spec",
]

_LAZY_EXPORTS: dict[str, str] = {
    "CanvasNoteSpec": ".models",
    "CanvasPosition": ".models",
    "CodegenResult": ".models",
    "ContractionOperandLayoutSpec": ".models",
    "ContractionPlanSpec": ".models",
    "ContractionStepSpec": ".models",
    "ContractionViewSnapshotSpec": ".models",
    "EdgeEndpointRef": ".models",
    "EdgeSpec": ".models",
    "EditorLaunchOptions": ".editor",
    "EditorResult": ".models",
    "EngineName": ".models",
    "GroupSpec": ".models",
    "HyperedgeSpec": ".models",
    "IndexHandle": ".builder",
    "IndexSpec": ".models",
    "NetworkSpec": ".models",
    "NetworkBuilder": ".builder",
    "PythonLoadOptions": ".io",
    "SvgRenderOptions": ".rendering",
    "TensorCollectionFormat": ".models",
    "TensorDataMode": ".models",
    "TensorDataSpec": ".models",
    "TensorHandle": ".builder",
    "TensorSize": ".models",
    "TensorSpec": ".models",
    "ValidationIssue": ".models",
    "analyze_contraction": ".analysis",
    "analyze_spec": ".analysis",
    "build_template_spec": ".templates",
    "canonicalize_spec": ".canonicalization",
    "diff_specs": ".internal.diffing._diffing",
    "generate_code": "._public_codegen",
    "lint_spec": ".linting",
    "list_template_names": ".templates",
    "load_python_spec": ".io",
    "load_spec": ".io",
    "open_editor": ".editor",
    "render_spec_png": ".rendering",
    "render_spec_svg": ".rendering",
    "save_spec": ".io",
    "semantic_diff_specs": ".internal.diffing._diffing",
    "validate_spec": ".validation",
}


def __getattr__(name: str) -> object:
    """Lazily resolve public exports on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the public module surface without triggering lazy imports."""
    return sorted(set(globals()) | set(__all__))

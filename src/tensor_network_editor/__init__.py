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
    from .analysis import analyze_contraction, analyze_spec
    from .api import (
        generate_code,
        launch_tensor_network_editor,
        load_spec,
        load_spec_from_python_code,
        save_spec,
    )
    from .canonicalization import canonicalize_spec
    from .codegen.registry import list_generator_names, register_generator
    from .diffing import (
        SemanticDiffEntry,
        SemanticFieldChange,
        SemanticSpecDiffResult,
        diff_specs,
        semantic_diff_specs,
    )
    from .errors import CodeGenerationError
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
        IndexSpec,
        NetworkSpec,
        TensorCollectionFormat,
        TensorSize,
        TensorSpec,
    )
    from .templates import (
        build_template_spec,
        list_template_names,
        register_static_template,
        register_template,
    )
    from .validation import validate_spec

PACKAGE_LOGGER = logging.getLogger(__name__)
if not any(
    isinstance(handler, logging.NullHandler) for handler in PACKAGE_LOGGER.handlers
):
    PACKAGE_LOGGER.addHandler(logging.NullHandler())

__all__ = [
    "CanvasPosition",
    "CanvasNoteSpec",
    "CodeGenerationError",
    "CodegenResult",
    "ContractionOperandLayoutSpec",
    "ContractionPlanSpec",
    "ContractionStepSpec",
    "ContractionViewSnapshotSpec",
    "EdgeEndpointRef",
    "EdgeSpec",
    "EditorResult",
    "EngineName",
    "GroupSpec",
    "IndexSpec",
    "NetworkSpec",
    "TensorCollectionFormat",
    "TensorSize",
    "TensorSpec",
    "__version__",
    "analyze_contraction",
    "analyze_spec",
    "build_template_spec",
    "canonicalize_spec",
    "diff_specs",
    "generate_code",
    "lint_spec",
    "list_generator_names",
    "launch_tensor_network_editor",
    "list_template_names",
    "load_spec",
    "load_spec_from_python_code",
    "register_generator",
    "register_static_template",
    "register_template",
    "save_spec",
    "SemanticDiffEntry",
    "SemanticFieldChange",
    "SemanticSpecDiffResult",
    "semantic_diff_specs",
    "validate_spec",
]

_LAZY_EXPORTS: dict[str, str] = {
    "CanvasNoteSpec": ".models",
    "CanvasPosition": ".models",
    "CodeGenerationError": ".errors",
    "CodegenResult": ".models",
    "ContractionOperandLayoutSpec": ".models",
    "ContractionPlanSpec": ".models",
    "ContractionStepSpec": ".models",
    "ContractionViewSnapshotSpec": ".models",
    "EdgeEndpointRef": ".models",
    "EdgeSpec": ".models",
    "EditorResult": ".models",
    "EngineName": ".models",
    "GroupSpec": ".models",
    "IndexSpec": ".models",
    "NetworkSpec": ".models",
    "SemanticDiffEntry": ".diffing",
    "SemanticFieldChange": ".diffing",
    "SemanticSpecDiffResult": ".diffing",
    "TensorCollectionFormat": ".models",
    "TensorSize": ".models",
    "TensorSpec": ".models",
    "analyze_contraction": ".analysis",
    "analyze_spec": ".analysis",
    "build_template_spec": ".templates",
    "canonicalize_spec": ".canonicalization",
    "diff_specs": ".diffing",
    "generate_code": ".api",
    "lint_spec": ".linting",
    "list_generator_names": ".codegen.registry",
    "list_template_names": ".templates",
    "launch_tensor_network_editor": ".api",
    "load_spec": ".api",
    "load_spec_from_python_code": ".api",
    "register_generator": ".codegen.registry",
    "register_static_template": ".templates",
    "register_template": ".templates",
    "save_spec": ".api",
    "semantic_diff_specs": ".diffing",
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

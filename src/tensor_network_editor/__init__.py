"""Public package exports for Tensor Network Editor."""

import logging

from ._version import __version__
from .analysis import analyze_contraction, analyze_spec
from .api import (
    generate_code,
    launch_tensor_network_editor,
    load_spec,
    load_spec_from_python_code,
    save_spec,
)
from .canonicalization import canonicalize_spec
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
    "launch_tensor_network_editor",
    "list_template_names",
    "load_spec",
    "load_spec_from_python_code",
    "save_spec",
    "SemanticDiffEntry",
    "SemanticFieldChange",
    "SemanticSpecDiffResult",
    "semantic_diff_specs",
    "validate_spec",
]

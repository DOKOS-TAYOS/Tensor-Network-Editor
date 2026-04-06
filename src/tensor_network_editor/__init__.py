import logging

from .api import (
    generate_code,
    launch_tensor_network_editor,
    load_spec,
    load_spec_from_python_code,
    save_spec,
)
from .models import (
    CanvasNoteSpec,
    CanvasPosition,
    CodegenResult,
    ContractionPlanSpec,
    ContractionStepSpec,
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

PACKAGE_LOGGER = logging.getLogger(__name__)
if not any(
    isinstance(handler, logging.NullHandler) for handler in PACKAGE_LOGGER.handlers
):
    PACKAGE_LOGGER.addHandler(logging.NullHandler())

__version__ = "0.1.2"

__all__ = [
    "CanvasPosition",
    "CanvasNoteSpec",
    "CodegenResult",
    "ContractionPlanSpec",
    "ContractionStepSpec",
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
    "generate_code",
    "launch_tensor_network_editor",
    "load_spec",
    "load_spec_from_python_code",
    "save_spec",
]

import logging

from .api import (
    generate_code,
    launch_tensor_network_editor,
    load_spec,
    save_spec,
)
from .models import (
    CanvasPosition,
    CodegenResult,
    EdgeEndpointRef,
    EdgeSpec,
    EditorResult,
    EngineName,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
)

PACKAGE_LOGGER = logging.getLogger(__name__)
if not any(isinstance(handler, logging.NullHandler) for handler in PACKAGE_LOGGER.handlers):
    PACKAGE_LOGGER.addHandler(logging.NullHandler())

__version__ = "0.1.0"

__all__ = [
    "CanvasPosition",
    "CodegenResult",
    "EdgeEndpointRef",
    "EdgeSpec",
    "EditorResult",
    "EngineName",
    "IndexSpec",
    "NetworkSpec",
    "TensorSpec",
    "__version__",
    "generate_code",
    "launch_tensor_network_editor",
    "load_spec",
    "save_spec",
]

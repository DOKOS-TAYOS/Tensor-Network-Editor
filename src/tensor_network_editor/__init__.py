import logging
from importlib.metadata import PackageNotFoundError, version

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

try:
    __version__ = version("tensor-network-editor")
except PackageNotFoundError:
    __version__ = "0.0.0"

PACKAGE_LOGGER = logging.getLogger(__name__)
if not any(isinstance(handler, logging.NullHandler) for handler in PACKAGE_LOGGER.handlers):
    PACKAGE_LOGGER.addHandler(logging.NullHandler())

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

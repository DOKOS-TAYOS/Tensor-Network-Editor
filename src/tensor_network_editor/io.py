"""Public serialization and Python-import helpers for tensor-network specs."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .internal._logging import log_operation
from .internal.io._python_import_profiles import PythonSourceProfile
from .internal.io._python_live_import import PythonImportMode
from .internal.io._serialization import (
    SCHEMA_VERSION,
    PythonReconstructionLevel,
    deserialize_spec,
    serialize_spec,
)
from .internal.io._serialization import (
    load_spec as _load_spec,
)
from .internal.io._serialization import (
    load_spec_from_python_code as _load_python_spec,
)
from .internal.io._serialization import (
    save_spec as _save_spec,
)
from .models import NetworkSpec
from .types import StrPath

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PythonLoadOptions:
    """Public options for loading a spec from Python or a filesystem path."""

    source_profile: PythonSourceProfile = "auto"
    import_mode: PythonImportMode = "static"
    reconstruction_level: PythonReconstructionLevel = "auto"
    object_name: str | None = None


def load_spec(
    path: StrPath,
    *,
    python: PythonLoadOptions | None = None,
) -> NetworkSpec:
    """Load one saved JSON spec or supported Python source from disk."""
    options = python or PythonLoadOptions()
    context = {
        "path": path,
        "python_import_mode": options.import_mode,
        "source_profile": options.source_profile,
        "python_reconstruction_level": options.reconstruction_level,
    }
    with log_operation(LOGGER, "Spec load", context=context):
        return _load_spec(
            path,
            source_profile=options.source_profile,
            python_import_mode=options.import_mode,
            python_reconstruction_level=options.reconstruction_level,
            python_object_name=options.object_name,
        )


def load_python_spec(
    code: str,
    *,
    python: PythonLoadOptions | None = None,
) -> NetworkSpec:
    """Load one network spec from supported Python source already in memory."""
    options = python or PythonLoadOptions()
    context = {
        "python_import_mode": options.import_mode,
        "source_profile": options.source_profile,
        "python_reconstruction_level": options.reconstruction_level,
    }
    with log_operation(LOGGER, "Python spec load", context=context):
        return _load_python_spec(
            code,
            source_profile=options.source_profile,
            python_import_mode=options.import_mode,
            python_reconstruction_level=options.reconstruction_level,
            python_object_name=options.object_name,
        )


def save_spec(spec: NetworkSpec, *, path: StrPath) -> None:
    """Validate and write one network spec to JSON."""
    with log_operation(LOGGER, "Spec save", context={"path": path}):
        _save_spec(spec, path)


__all__ = [
    "PythonLoadOptions",
    "SCHEMA_VERSION",
    "deserialize_spec",
    "load_python_spec",
    "load_spec",
    "save_spec",
    "serialize_spec",
]

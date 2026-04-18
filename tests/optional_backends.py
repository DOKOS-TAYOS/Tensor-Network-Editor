from __future__ import annotations

import importlib
import os
from collections.abc import Iterable

import pytest

from tensor_network_editor.models import EngineName

_REQUIRE_LIGHT_OPTIONAL_BACKENDS_ENV = "TNE_REQUIRE_OPTIONAL_BACKENDS"

_LIGHT_ENGINE_MODULES: dict[EngineName, tuple[str, ...]] = {
    EngineName.TENSORNETWORK: ("numpy", "tensornetwork"),
    EngineName.QUIMB: ("numpy", "quimb"),
    EngineName.EINSUM_NUMPY: ("numpy",),
}

_HEAVY_ENGINE_MODULES: dict[EngineName, tuple[str, ...]] = {
    EngineName.TENSORKROWCH: ("torch", "tensorkrowch"),
    EngineName.EINSUM_TORCH: ("torch",),
}


def require_engine_backend(engine: EngineName) -> None:
    """Import or skip the runtime modules needed to execute generated code."""
    light_modules = _LIGHT_ENGINE_MODULES.get(engine)
    if light_modules is not None:
        require_light_optional_modules(light_modules)
        return

    heavy_modules = _HEAVY_ENGINE_MODULES.get(engine, ())
    for module_name in heavy_modules:
        pytest.importorskip(module_name)


def require_light_optional_module(module_name: str) -> None:
    """Import a light optional module, failing when CI explicitly requires it."""
    require_light_optional_modules((module_name,))


def require_light_optional_modules(module_names: Iterable[str]) -> None:
    """Import light optional modules or skip them outside the optional CI job."""
    required = os.environ.get(_REQUIRE_LIGHT_OPTIONAL_BACKENDS_ENV) == "1"
    for module_name in module_names:
        if required:
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError:
                pytest.fail(
                    f"Required optional backend module {module_name!r} is missing. "
                    f"Install the balanced optional test dependencies or unset "
                    f"{_REQUIRE_LIGHT_OPTIONAL_BACKENDS_ENV}.",
                    pytrace=False,
                )
        else:
            pytest.importorskip(module_name)

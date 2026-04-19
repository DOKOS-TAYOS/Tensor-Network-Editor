from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.integration


def _import_installed_package() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[1]
    checkout_src = (repo_root / "src").resolve()
    sys.path[:] = [path for path in sys.path if Path(path).resolve() != checkout_src]
    module_names = [
        module_name
        for module_name in sys.modules
        if module_name == "tensor_network_editor"
        or module_name.startswith("tensor_network_editor.")
    ]
    for module_name in module_names:
        del sys.modules[module_name]
    importlib.invalidate_caches()
    return importlib.import_module("tensor_network_editor")


def test_installed_wheel_exposes_runtime_contracts() -> None:
    wheel_path_text = os.environ.get("TNE_WHEEL_UNDER_TEST")
    if not wheel_path_text:
        pytest.skip("Set TNE_WHEEL_UNDER_TEST to run the clean wheel smoke test.")
    wheel_path = Path(wheel_path_text)
    assert wheel_path.is_file()

    tensor_network_editor = _import_installed_package()

    checkout_src = (Path(__file__).resolve().parents[1] / "src").resolve()
    package_file_text = tensor_network_editor.__file__
    assert package_file_text is not None
    package_file = Path(package_file_text).resolve()
    assert not package_file.is_relative_to(checkout_src)
    assert {
        "site-packages",
        "dist-packages",
    } & {part.lower() for part in package_file.parts}
    assert tensor_network_editor.__version__ == importlib.metadata.version(
        "tensor-network-editor"
    )

    distribution = importlib.metadata.distribution("tensor-network-editor")
    installed_files = {str(path) for path in (distribution.files or [])}
    required_suffixes = {
        "tensor_network_editor/app/static/index.html",
        "tensor_network_editor/app/static/app.css",
        "tensor_network_editor/app/static/js/main.js",
        "tensor_network_editor/app/static/vendor/cytoscape.min.js",
        "tensor_network_editor/py.typed",
        "licenses/THIRD_PARTY_LICENSES",
    }
    for suffix in required_suffixes:
        assert any(path.endswith(suffix) for path in installed_files), suffix

    help_result = subprocess.run(
        [sys.executable, "-m", "tensor_network_editor", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stdout + help_result.stderr
    assert "Work with tensor-network specs" in help_result.stdout

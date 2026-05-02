from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import tensor_network_editor
from tests.conftest import distribution_for_checkout_import_or_skip


def test_test_session_imports_package_from_current_checkout_src() -> None:
    package_root = Path(tensor_network_editor.__file__).resolve().parent
    expected_package_root = (
        Path.cwd().resolve() / "src" / "tensor_network_editor"
    ).resolve()

    assert package_root == expected_package_root


def test_installed_distribution_exposes_public_metadata_contracts() -> None:
    distribution = distribution_for_checkout_import_or_skip(tensor_network_editor)
    project_urls = distribution.metadata.get_all("Project-URL") or []

    assert distribution.metadata["Name"] == "tensor-network-editor"
    assert distribution.version == tensor_network_editor.__version__
    assert any(url.startswith("Homepage, https://") for url in project_urls)
    assert any(url.startswith("Repository, https://") for url in project_urls)
    assert any(url.startswith("Issues, https://") for url in project_urls)
    assert any(
        entry_point.group == "console_scripts"
        and entry_point.name == "tensor-network-editor"
        and entry_point.value == "tensor_network_editor.cli:main"
        for entry_point in distribution.entry_points
    )


def test_installed_package_contains_required_frontend_assets() -> None:
    package_root = Path(tensor_network_editor.__file__).resolve().parent

    required_assets = [
        package_root / "app" / "static" / "index.html",
        package_root / "app" / "static" / "app.css",
        package_root / "app" / "static" / "js" / "main.js",
        package_root / "app" / "static" / "vendor" / "cytoscape.min.js",
    ]

    for asset_path in required_assets:
        assert asset_path.is_file()


def test_project_metadata_declares_package_data_and_license_files() -> None:
    pyproject_path = Path.cwd() / "pyproject.toml"
    manifest_path = Path.cwd() / "MANIFEST.in"
    third_party_notices = Path.cwd() / "THIRD_PARTY_LICENSES"

    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    license_files = set(payload["project"]["license-files"])
    package_data = set(
        payload["tool"]["setuptools"]["package-data"]["tensor_network_editor"]
    )
    manifest_text = manifest_path.read_text(encoding="utf-8")

    assert {"LICENSE", "THIRD_PARTY_LICENSES"} <= license_files
    assert {
        "app/static/*.html",
        "app/static/*.css",
        "app/static/js/*.js",
        "app/static/js/actions/*.js",
        "app/static/js/core/*.js",
        "app/static/js/graph/*.js",
        "app/static/js/interactions/*.js",
        "app/static/js/planner/*.js",
        "app/static/js/properties/*.js",
        "app/static/js/services/*.js",
        "app/static/js/session/*.js",
        "app/static/js/shell/*.js",
        "app/static/js/spec/*.js",
        "app/static/js/state/*.js",
        "app/static/js/utils/*.js",
        "app/static/js/views/*.js",
        "app/static/vendor/*.js",
    } <= package_data
    assert "app/static/*.js" not in package_data
    assert "include THIRD_PARTY_LICENSES" in manifest_text
    assert third_party_notices.is_file()
    assert third_party_notices.read_text(encoding="utf-8").strip()


def test_project_metadata_declares_required_matplotlib_dependency_and_backend_extras() -> (
    None
):
    pyproject_path = Path.cwd() / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = payload["project"]["dependencies"]
    optional_dependencies = payload["project"]["optional-dependencies"]

    assert "matplotlib>=3.8" in dependencies
    assert optional_dependencies["numpy"] == ["numpy>=1.24"]
    assert optional_dependencies["torch"] == ["torch>=2.0"]
    assert "png" not in optional_dependencies


def test_docs_do_not_advertise_removed_png_extra() -> None:
    readme_text = (Path.cwd() / "README.md").read_text(encoding="utf-8")
    installation_text = (Path.cwd() / "docs" / "installation.md").read_text(
        encoding="utf-8"
    )

    assert "tensor-network-editor[png]" not in readme_text
    assert "optional `png` extra" not in readme_text
    assert "tensor-network-editor[png]" not in installation_text


def test_manifest_omits_redundant_non_package_exclusions() -> None:
    manifest_text = (Path.cwd() / "MANIFEST.in").read_text(encoding="utf-8")

    assert "docs/images" not in manifest_text
    assert "prune tests" not in manifest_text
    assert "tests" not in manifest_text
    assert "recursive-exclude docs/images *" not in manifest_text
    assert "recursive-exclude tests *" not in manifest_text


def test_third_party_notices_describe_bundled_asset_scope() -> None:
    third_party_text = (Path.cwd() / "THIRD_PARTY_LICENSES").read_text(encoding="utf-8")
    readme_text = (Path.cwd() / "README.md").read_text(encoding="utf-8")

    assert "Cytoscape.js" in third_party_text
    assert "Version: 3.30.2" in third_party_text
    assert "src/tensor_network_editor/app/static/vendor/cytoscape.min.js" in (
        third_party_text
    )
    assert "Runtime pip-installed dependencies are not bundled" in third_party_text
    assert "Package: Matplotlib" in third_party_text
    assert "License: Matplotlib license" in third_party_text
    assert "THIRD_PARTY_LICENSES" in readme_text


def test_readme_uses_singular_operation_cost_labels() -> None:
    readme_text = (Path.cwd() / "README.md").read_text(encoding="utf-8")

    assert "FLOPs" not in readme_text
    assert "MACs" not in readme_text
    assert "FLOP" in readme_text
    assert "MAC" in readme_text


def test_package_root_defers_heavy_public_modules_until_first_access() -> None:
    env = os.environ.copy()
    current_src_path = str((Path.cwd() / "src").resolve())
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        current_src_path
        if not current_pythonpath
        else os.pathsep.join([current_src_path, current_pythonpath])
    )
    script = """
import json
import sys

import tensor_network_editor as tne

before = {
    "analysis": "tensor_network_editor.analysis" in sys.modules,
    "editor": "tensor_network_editor.editor" in sys.modules,
    "io": "tensor_network_editor.io" in sys.modules,
    "public_codegen": "tensor_network_editor._public_codegen" in sys.modules,
    "templates": "tensor_network_editor.templates" in sys.modules,
    "canonicalization": "tensor_network_editor.canonicalization" in sys.modules,
    "linting": "tensor_network_editor.linting" in sys.modules,
}
_ = tne.generate_code
after_generate = {
    "analysis": "tensor_network_editor.analysis" in sys.modules,
    "editor": "tensor_network_editor.editor" in sys.modules,
    "io": "tensor_network_editor.io" in sys.modules,
    "public_codegen": "tensor_network_editor._public_codegen" in sys.modules,
    "templates": "tensor_network_editor.templates" in sys.modules,
    "canonicalization": "tensor_network_editor.canonicalization" in sys.modules,
    "linting": "tensor_network_editor.linting" in sys.modules,
}
_ = tne.open_editor
after_editor = {
    "analysis": "tensor_network_editor.analysis" in sys.modules,
    "editor": "tensor_network_editor.editor" in sys.modules,
    "io": "tensor_network_editor.io" in sys.modules,
    "public_codegen": "tensor_network_editor._public_codegen" in sys.modules,
    "templates": "tensor_network_editor.templates" in sys.modules,
    "canonicalization": "tensor_network_editor.canonicalization" in sys.modules,
    "linting": "tensor_network_editor.linting" in sys.modules,
}
_ = tne.load_spec
after_io = {
    "analysis": "tensor_network_editor.analysis" in sys.modules,
    "editor": "tensor_network_editor.editor" in sys.modules,
    "io": "tensor_network_editor.io" in sys.modules,
    "public_codegen": "tensor_network_editor._public_codegen" in sys.modules,
    "templates": "tensor_network_editor.templates" in sys.modules,
    "canonicalization": "tensor_network_editor.canonicalization" in sys.modules,
    "linting": "tensor_network_editor.linting" in sys.modules,
}
_ = tne.analyze_spec
after_analysis = {
    "analysis": "tensor_network_editor.analysis" in sys.modules,
    "editor": "tensor_network_editor.editor" in sys.modules,
    "io": "tensor_network_editor.io" in sys.modules,
    "public_codegen": "tensor_network_editor._public_codegen" in sys.modules,
    "templates": "tensor_network_editor.templates" in sys.modules,
    "canonicalization": "tensor_network_editor.canonicalization" in sys.modules,
    "linting": "tensor_network_editor.linting" in sys.modules,
}
print(json.dumps({
    "before": before,
    "after_generate": after_generate,
    "after_editor": after_editor,
    "after_io": after_io,
    "after_analysis": after_analysis,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["before"] == {
        "analysis": False,
        "editor": False,
        "io": False,
        "public_codegen": False,
        "templates": False,
        "canonicalization": False,
        "linting": False,
    }
    assert payload["after_generate"]["public_codegen"] is True
    assert payload["after_generate"]["analysis"] is False
    assert payload["after_editor"]["editor"] is True
    assert payload["after_editor"]["analysis"] is False
    assert payload["after_io"]["io"] is True
    assert payload["after_io"]["analysis"] is False
    assert payload["after_analysis"]["analysis"] is True

from __future__ import annotations

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
        "app/static/js/interactions/*.js",
        "app/static/js/planner/*.js",
        "app/static/js/properties/*.js",
        "app/static/js/services/*.js",
        "app/static/js/session/*.js",
        "app/static/js/shell/*.js",
        "app/static/js/spec/*.js",
        "app/static/js/state/*.js",
        "app/static/js/views/*.js",
        "app/static/vendor/*.js",
    } <= package_data
    assert "app/static/*.js" not in package_data
    assert "include THIRD_PARTY_LICENSES" in manifest_text
    assert third_party_notices.is_file()
    assert third_party_notices.read_text(encoding="utf-8").strip()


def test_third_party_notices_describe_bundled_asset_scope() -> None:
    third_party_text = (Path.cwd() / "THIRD_PARTY_LICENSES").read_text(encoding="utf-8")
    readme_text = (Path.cwd() / "README.md").read_text(encoding="utf-8")

    assert "Cytoscape.js" in third_party_text
    assert "Version: 3.30.2" in third_party_text
    assert "src/tensor_network_editor/app/static/vendor/cytoscape.min.js" in (
        third_party_text
    )
    assert "Optional pip-installed dependencies are not bundled" in third_party_text
    assert "THIRD_PARTY_LICENSES" in readme_text


def test_readme_uses_singular_operation_cost_labels() -> None:
    readme_text = (Path.cwd() / "README.md").read_text(encoding="utf-8")

    assert "FLOPs" not in readme_text
    assert "MACs" not in readme_text
    assert "FLOP" in readme_text
    assert "MAC" in readme_text

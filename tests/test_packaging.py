from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

import tensor_network_editor


def test_installed_distribution_exposes_public_metadata_contracts() -> None:
    distribution = importlib.metadata.distribution("tensor-network-editor")
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
        "app/static/*.js",
        "app/static/js/*.js",
        "app/static/vendor/*.js",
    } <= package_data
    assert "include THIRD_PARTY_LICENSES" in manifest_text
    assert third_party_notices.is_file()
    assert third_party_notices.read_text(encoding="utf-8").strip()

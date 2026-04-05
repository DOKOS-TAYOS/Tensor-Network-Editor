from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


class PackagingMetadataTests(unittest.TestCase):
    def test_pyproject_declares_public_project_urls(self) -> None:
        pyproject_path = Path.cwd() / "pyproject.toml"
        self.assertTrue(pyproject_path.is_file(), "pyproject.toml should exist.")

        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        project_urls = payload["project"]["urls"]

        self.assertIn("Homepage", project_urls)
        self.assertIn("Repository", project_urls)
        self.assertIn("Issues", project_urls)

    def test_distribution_declares_third_party_license_notices(self) -> None:
        project_root = Path.cwd()
        notices_path = project_root / "THIRD_PARTY_LICENSES"
        manifest_path = project_root / "MANIFEST.in"
        pyproject_path = project_root / "pyproject.toml"

        self.assertTrue(
            notices_path.is_file(),
            "THIRD_PARTY_LICENSES should exist for bundled third-party assets.",
        )
        self.assertTrue(manifest_path.is_file(), "MANIFEST.in should exist.")

        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        license_files = payload["project"]["license-files"]
        manifest_text = manifest_path.read_text(encoding="utf-8")
        notices_text = notices_path.read_text(encoding="utf-8")

        self.assertIn("THIRD_PARTY_LICENSES", license_files)
        self.assertIn("include THIRD_PARTY_LICENSES", manifest_text)
        self.assertIn("Cytoscape.js", notices_text)
        self.assertIn("MIT License", notices_text)

    def test_package_data_includes_nested_frontend_modules(self) -> None:
        pyproject_path = Path.cwd() / "pyproject.toml"

        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        package_data = payload["tool"]["setuptools"]["package-data"][
            "tensor_network_editor"
        ]

        self.assertIn("app/static/js/*.js", package_data)


if __name__ == "__main__":
    unittest.main()

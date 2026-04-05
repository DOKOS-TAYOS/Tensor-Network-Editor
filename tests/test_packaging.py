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

    def test_pyproject_declares_optional_planner_extra(self) -> None:
        pyproject_path = Path.cwd() / "pyproject.toml"

        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        optional_dependencies = payload["project"]["optional-dependencies"]

        self.assertIn("planner", optional_dependencies)
        self.assertTrue(
            any(
                dependency.startswith("opt_einsum")
                for dependency in optional_dependencies["planner"]
            )
        )

    def test_pyproject_declares_pyright_dev_tooling(self) -> None:
        pyproject_path = Path.cwd() / "pyproject.toml"

        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        dev_dependencies = payload["project"]["optional-dependencies"]["dev"]
        pyright_config = payload["tool"]["pyright"]

        self.assertTrue(
            any(dependency.startswith("pyright") for dependency in dev_dependencies)
        )
        self.assertEqual(pyright_config["venvPath"], ".")
        self.assertEqual(pyright_config["venv"], ".venv")
        self.assertEqual(pyright_config["include"], ["src", "tests"])

    def test_ci_workflow_runs_pyright(self) -> None:
        workflow_path = Path.cwd() / ".github" / "workflows" / "ci.yml"

        workflow_text = workflow_path.read_text(encoding="utf-8")

        self.assertIn("Run Pyright", workflow_text)
        self.assertIn("-m pyright", workflow_text)

    def test_mypy_allows_missing_optional_opt_einsum_dependency(self) -> None:
        pyproject_path = Path.cwd() / "pyproject.toml"

        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        overrides = payload["tool"]["mypy"].get("overrides", [])

        self.assertTrue(
            any(
                (
                    override.get("ignore_missing_imports") is True
                    and "opt_einsum"
                    in (
                        override.get("module")
                        if isinstance(override.get("module"), list)
                        else [override.get("module")]
                    )
                )
                for override in overrides
            ),
            "mypy should ignore missing optional imports for opt_einsum in CI.",
        )

    def test_readme_documents_pyright_check(self) -> None:
        readme_path = Path.cwd() / "README.md"

        readme_text = readme_path.read_text(encoding="utf-8")

        self.assertIn("python -m pyright", readme_text)


if __name__ == "__main__":
    unittest.main()

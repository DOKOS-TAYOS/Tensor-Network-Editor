from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path
from uuid import uuid4


class CleanScriptTests(unittest.TestCase):
    def test_clean_sh_exists_and_targets_standard_artifacts(self) -> None:
        source_script = Path.cwd() / "scripts" / "clean.sh"
        self.assertTrue(source_script.is_file(), "scripts/clean.sh should exist.")

        script_text = source_script.read_text(encoding="utf-8")

        self.assertIn('remove_dir ".pytest_cache"', script_text)
        self.assertIn('remove_dir "build"', script_text)
        self.assertIn('remove_dir "dist"', script_text)
        self.assertIn('remove_glob_dirs_warn "./pytest-cache-files-*"', script_text)
        self.assertIn('remove_named_dirs "./src" "__pycache__"', script_text)
        self.assertIn('remove_named_dirs "./tests" "__pycache__"', script_text)
        self.assertNotIn(".venv", script_text)

    @unittest.skipUnless(os.name == "nt", "clean.bat is a Windows-only helper")
    def test_clean_bat_removes_generated_artifacts_and_preserves_venv(self) -> None:
        source_script = Path.cwd() / "scripts" / "clean.bat"
        self.assertTrue(source_script.is_file(), "scripts/clean.bat should exist.")

        temp_root = Path.cwd() / ".test_output" / f"clean_script_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_root, ignore_errors=True))

        try:
            temp_scripts_dir = temp_root / "scripts"
            temp_scripts_dir.mkdir()
            shutil.copy2(source_script, temp_scripts_dir / "clean.bat")

            directories_to_create = [
                temp_root / "build",
                temp_root / "dist",
                temp_root / ".pytest_cache",
                temp_root / ".mypy_cache",
                temp_root / ".ruff_cache",
                temp_root / ".test_output",
                temp_root / "pytest-cache-files-demo",
                temp_root / "package.egg-info",
                temp_root / "__pycache__",
                temp_root / "src" / "tensor_network_editor" / "__pycache__",
                temp_root / "tests" / "__pycache__",
                temp_root / "examples" / "__pycache__",
                temp_root / ".venv",
            ]
            for directory in directories_to_create:
                directory.mkdir(parents=True, exist_ok=True)

            files_to_create = [
                temp_root / "build" / "artifact.txt",
                temp_root / "dist" / "artifact.whl",
                temp_root / ".pytest_cache" / "cache.txt",
                temp_root / ".mypy_cache" / "cache.txt",
                temp_root / ".ruff_cache" / "cache.txt",
                temp_root / ".test_output" / "output.txt",
                temp_root / "pytest-cache-files-demo" / "temp.txt",
                temp_root / "package.egg-info" / "PKG-INFO",
                temp_root / "__pycache__" / "root.pyc",
                temp_root
                / "src"
                / "tensor_network_editor"
                / "__pycache__"
                / "module.pyc",
                temp_root / "tests" / "__pycache__" / "module.pyc",
                temp_root / "examples" / "__pycache__" / "module.pyc",
                temp_root / ".venv" / "marker.txt",
                temp_root / ".coverage",
                temp_root / ".coverage.unit",
            ]
            for file_path in files_to_create:
                file_path.write_text("temporary", encoding="utf-8")

            for _ in range(2):
                result = subprocess.run(
                    ["cmd", "/c", "scripts\\clean.bat"],
                    cwd=temp_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                )

            removed_paths = [
                temp_root / "build",
                temp_root / "dist",
                temp_root / ".pytest_cache",
                temp_root / ".mypy_cache",
                temp_root / ".ruff_cache",
                temp_root / ".test_output",
                temp_root / "pytest-cache-files-demo",
                temp_root / "package.egg-info",
                temp_root / "__pycache__",
                temp_root / "src" / "tensor_network_editor" / "__pycache__",
                temp_root / "tests" / "__pycache__",
                temp_root / "examples" / "__pycache__",
                temp_root / ".coverage",
                temp_root / ".coverage.unit",
            ]
            for path in removed_paths:
                self.assertFalse(path.exists(), f"{path} should have been removed.")

            self.assertTrue((temp_root / ".venv").exists())
            self.assertTrue((temp_root / ".venv" / "marker.txt").exists())
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

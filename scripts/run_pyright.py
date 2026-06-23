"""Run Pyright against the current checkout using the shared project virtualenv."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

DEFAULT_TARGETS: Final[tuple[str, ...]] = ("src", "tests")


def find_checkout_root(start: Path | None = None) -> Path:
    """Return the current repository checkout root.

    Args:
        start: Optional path inside the checkout. When omitted, the current
            working directory is used.

    Returns:
        The checkout root that contains ``pyproject.toml``, ``src``, and
        ``tests``.

    Raises:
        FileNotFoundError: If no matching checkout root can be found.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src").is_dir()
            and (candidate / "tests").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not locate the checkout root containing pyproject.toml, src, and tests."
    )


def find_shared_venv_root(checkout_root: Path) -> Path:
    """Return the nearest ancestor that provides the shared project ``.venv``.

    Args:
        checkout_root: The checkout that should reuse a shared virtual
            environment.

    Returns:
        The directory that contains the shared ``.venv`` folder.

    Raises:
        FileNotFoundError: If no shared virtual environment can be found.
    """
    for candidate in (checkout_root.resolve(), *checkout_root.resolve().parents):
        if (candidate / ".venv").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate a shared .venv for Pyright.")


def python_executable(shared_root: Path) -> Path:
    """Return the Python executable inside the shared virtual environment.

    Args:
        shared_root: Directory that owns the shared ``.venv``.

    Returns:
        The Python interpreter path for Windows or Linux.

    Raises:
        FileNotFoundError: If the virtual environment does not expose a Python
            interpreter in the expected location.
    """
    venv_root = shared_root / ".venv"
    candidates = (
        venv_root / "Scripts" / "python.exe",
        venv_root / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find the Python executable inside the shared .venv."
    )


def _python_supports_module(python_path: Path, module_name: str) -> bool:
    """Return whether one Python executable can import a module successfully."""
    try:
        result = subprocess.run(
            [
                str(python_path),
                "-c",
                (
                    "import importlib.util, sys; "
                    "raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) "
                    "is not None else 1)"
                ),
                module_name,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def find_bundled_pyright_entrypoint(shared_root: Path) -> Path | None:
    """Return the bundled Pyright CLI entrypoint shipped inside the shared venv."""
    venv_root = shared_root / ".venv"
    windows_entrypoint = (
        venv_root / "Lib" / "site-packages" / "pyright" / "dist" / "index.js"
    )
    if windows_entrypoint.is_file():
        return windows_entrypoint
    for candidate in venv_root.glob("lib/python*/site-packages/pyright/dist/index.js"):
        if candidate.is_file():
            return candidate
    return None


def find_node_executable() -> Path | None:
    """Return the global Node.js executable when it is available."""
    executable_name = "node.exe" if os.name == "nt" else "node"
    resolved = shutil.which(executable_name) or shutil.which("node")
    if resolved is None:
        return None
    return Path(resolved).resolve()


def resolve_pyright_python(shared_root: Path) -> Path:
    """Return a working Python executable that can run the ``pyright`` module."""
    shared_python = python_executable(shared_root)
    if _python_supports_module(shared_python, "pyright"):
        return shared_python
    current_python = Path(sys.executable).resolve()
    if _python_supports_module(current_python, "pyright"):
        return current_python
    raise RuntimeError(
        "Could not find a working Python interpreter with the 'pyright' module "
        "available. Repair the shared .venv or install the development "
        "dependencies in the active interpreter."
    )


def resolve_pyright_command(shared_root: Path) -> list[str]:
    """Return the command prefix used to invoke Pyright for this checkout."""
    node_executable = find_node_executable()
    bundled_entrypoint = find_bundled_pyright_entrypoint(shared_root)
    if node_executable is not None and bundled_entrypoint is not None:
        return [str(node_executable), str(bundled_entrypoint)]
    return [str(resolve_pyright_python(shared_root)), "-m", "pyright"]


def build_pyright_config(checkout_root: Path, shared_root: Path) -> dict[str, object]:
    """Build a temporary Pyright config for the current checkout.

    Args:
        checkout_root: Current checkout root. Present for API symmetry with the
            caller.
        shared_root: Directory that provides the shared virtual environment.

    Returns:
        A JSON-serializable Pyright configuration dictionary.
    """
    del checkout_root
    return {
        "venvPath": str(shared_root.resolve()),
        "venv": ".venv",
        "include": list(DEFAULT_TARGETS),
        "extraPaths": ["src", "."],
        "typeCheckingMode": "standard",
    }


def resolve_targets(checkout_root: Path, args: Sequence[str]) -> list[str]:
    """Resolve CLI targets relative to the current checkout.

    Args:
        checkout_root: Root directory of the active checkout.
        args: Raw CLI target arguments.

    Returns:
        A list of absolute paths to pass to Pyright.
    """
    if not args:
        return [str((checkout_root / target).resolve()) for target in DEFAULT_TARGETS]
    return [str((checkout_root / argument).resolve()) for argument in args]


def run_pyright(
    checkout_root: Path,
    shared_root: Path,
    args: Sequence[str],
) -> int:
    """Execute Pyright for the current checkout.

    Args:
        checkout_root: Root directory of the active checkout.
        shared_root: Directory that provides the shared virtual environment.
        args: Raw CLI target arguments.

    Returns:
        Pyright's process exit code.
    """
    config = build_pyright_config(checkout_root, shared_root)
    pyright_command = resolve_pyright_command(shared_root)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
            dir=checkout_root,
            prefix=".pyright-",
        ) as handle:
            json.dump(config, handle, indent=2)
            handle.flush()
            temp_path = Path(handle.name)
        result = subprocess.run(
            [
                *pyright_command,
                "-p",
                str(temp_path),
                *resolve_targets(checkout_root, args),
            ],
            cwd=checkout_root,
            check=False,
        )
        return int(result.returncode)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Pyright against the current checkout root or a subpath of it.

    Args:
        argv: Optional CLI arguments. When omitted, ``sys.argv[1:]`` is used.

    Returns:
        Pyright's process exit code.
    """
    checkout_root = find_checkout_root()
    shared_root = find_shared_venv_root(checkout_root)
    return run_pyright(checkout_root, shared_root, list(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())

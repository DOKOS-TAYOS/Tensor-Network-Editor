# Installation

This page covers normal installation, optional extras, source installs, and
development setup.

## Contents

- [Requirements](#requirements)
- [Install From PyPI](#install-from-pypi)
- [Optional Extras](#optional-extras)
- [Install From Source](#install-from-source)
- [Development Setup](#development-setup)
- [Check the Installation](#check-the-installation)
- [Cleanup Scripts](#cleanup-scripts)

## Requirements

- Python `3.11` or newer
- A virtual environment, usually named `.venv`
- Windows PowerShell or a Linux shell

The distribution name is `tensor-network-editor`. The import package name is
`tensor_network_editor`.

## Install From PyPI

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install tensor-network-editor
```

Bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install tensor-network-editor
```

After activation, `python` and `pip` should point to the virtual environment.
That keeps project dependencies separate from the rest of your machine.

The published source distribution and wheel include `LICENSE` and
`THIRD_PARTY_LICENSES`. `THIRD_PARTY_LICENSES` documents bundled assets shipped
inside this package. Optional extras installed from PyPI are not vendored into
this package and keep their own upstream licenses.

## Optional Extras

Install extras only when you need them.

Backend extras:

```bash
python -m pip install "tensor-network-editor[quimb]"
python -m pip install "tensor-network-editor[tensornetwork]"
python -m pip install "tensor-network-editor[tensorkrowch]"
```

Planner extra:

```bash
python -m pip install "tensor-network-editor[planner]"
```

The `planner` extra installs `opt_einsum`, which enables automatic greedy
contraction suggestions.

Desktop extra:

```bash
python -m pip install "tensor-network-editor[desktop]"
```

The `desktop` extra installs `pywebview` for environments that want a desktop
webview dependency available. The standard documented workflow is still the
local browser editor.

You can combine extras:

```bash
python -m pip install "tensor-network-editor[quimb,planner]"
```

Use backend extras when you want generated code to run in the same environment.
If you only want to generate source text, the editor can still do that without
installing the backend package.

## Install From Source

From a local checkout:

```bash
python -m pip install .
```

Use this when you want the repository version instead of the published PyPI
version.

## Development Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Useful development checks:

```bash
python -m ruff check . --fix
python -m ruff format .
python -m mypy
python -m pyright
python -m pytest
python -m build
python -m twine check dist/*
```

If you change package metadata such as the version, rerun the editable install
in the active `.venv` so the installed metadata stays aligned with the checkout:

```bash
python -m pip install -e ".[dev]"
```

## Check the Installation

Check that the CLI is available:

```bash
tensor-network-editor --help
```

Check that Python can import the package:

```bash
python -c "import tensor_network_editor; print(tensor_network_editor.__version__)"
```

Launch the editor:

```bash
tensor-network-editor edit
```

If the browser does not open, see
[troubleshooting.md#the-browser-did-not-open-automatically](troubleshooting.md#the-browser-did-not-open-automatically).

## Cleanup Scripts

The repository includes cleanup scripts for generated local artifacts:

- Windows: `.\scripts\clean.bat`
- Linux: `./scripts/clean.sh`

They are useful during development, not required for normal package use.

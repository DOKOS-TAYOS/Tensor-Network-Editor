# Tensor Network Editor

`tensor-network-editor` is a local, offline-friendly Python package for building tensor networks interactively and generating Python code for a chosen backend.

This `0.1.1` release is aimed at researchers and developers who want a lightweight visual editor that still returns plain Python data structures and code.

## Highlights

- Build tensor-network diagrams in a local browser session.
- Save and load versioned JSON designs.
- Generate readable Python code for `tensornetwork`, `quimb`, `tensorkrowch`, and `einsum`.
- Use the package as a library or from the `tensor-network-editor` CLI.
- Run on Windows and Linux with Python `3.11+`.

## Installation

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install .
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install .
```

Optional extras:

```bash
python -m pip install ".[quimb]"
python -m pip install ".[tensornetwork]"
python -m pip install ".[tensorkrowch]"
python -m pip install ".[desktop]"
```

## Quick Start

Python API:

```python
from tensor_network_editor import EngineName, launch_tensor_network_editor

result = launch_tensor_network_editor(default_engine=EngineName.EINSUM)

if result is not None:
    print(result.spec.name)
    if result.codegen is not None:
        print(result.codegen.code)
```

CLI:

```bash
tensor-network-editor --engine einsum
```

Load an existing design:

```bash
tensor-network-editor --load my_network.json --engine quimb
```

## Public API

- `launch_tensor_network_editor(...) -> EditorResult | None`
- `generate_code(spec, engine=...) -> CodegenResult`
- `save_spec(spec, path) -> None`
- `load_spec(path) -> NetworkSpec`

The editor blocks until the user presses `Done` or `Cancel`. On `Done`, it returns the abstract `NetworkSpec` and the generated code for the selected engine.

## Architecture

The package is split into small layers:

- `models.py`: abstract source-of-truth objects like `NetworkSpec`, `TensorSpec`, `IndexSpec`, and `EdgeSpec`
- `validation.py`: checks names, dimensions, endpoints, duplicate connections, and serialization safety
- `serialization.py`: JSON save/load with a versioned wire format
- `codegen/`: one generator per engine plus a shared normalization layer
- `app/`: local HTTP server, blocking session management, and the offline frontend assets

The frontend uses **Cytoscape.js**, bundled locally in the package, so the editor does not depend on a CDN or a Node runtime for end users.

## Development

Create the virtual environment and install the project in editable mode with the development tools:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Useful checks:

```powershell
python -m pytest
python -m ruff check .
python -m mypy
python -m build
python -m twine check dist/*
```

To remove local caches and generated build artifacts on Windows without touching `.venv`, run:

```powershell
.\scripts\clean.bat
```

On Linux, use:

```bash
./scripts/clean.sh
```

Bundled third-party notices for redistributed frontend assets are tracked in `THIRD_PARTY_LICENSES`.

## Save Format

Designs are stored as plain JSON with a schema wrapper:

```json
{
  "schema_version": 1,
  "network": {
    "...": "..."
  }
}
```

## Current Limits

- No hyperedges
- No real tensor data editing: generated tensors are initialized to zeros
- No TenPy backend in `0.1.1`
- No desktop wrapper by default

## Future Extension Points

The project is already prepared for a future optional `pywebview` wrapper because the browser launcher is separate from:

- `EditorSession`, which owns validation and return values
- `EditorServer`, which serves the same app locally

That means a desktop extra can reuse the same UI and backend flow without changing the core model or the generators.

# Tensor Network Editor

[![CI](https://img.shields.io/github/actions/workflow/status/DOKOS-TAYOS/Tensor-Network-Editor/ci.yml?branch=main&label=CI)](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor)
[![Windows%20%7C%20Linux](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-0A7BBB)](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor/blob/main/LICENSE)

`tensor-network-editor` is a local-first Python package for building tensor networks visually, saving them as versioned JSON, and generating readable Python code for multiple backends.

It is meant for research, teaching, and experimentation workflows where you want a friendly editor without giving up plain Python objects, reproducible files, or offline use. The UI opens in your browser, but the whole session is served locally from your own machine.

## Why this project

- Build tensor-network diagrams interactively in a local browser session.
- Save and reload designs as versioned JSON files.
- Generate Python code for `tensornetwork`, `quimb`, `tensorkrowch`, `einsum_numpy`, and `einsum_torch`.
- Reconstruct a `NetworkSpec` from supported generated Python exports when you need a code-to-spec round trip.
- Insert built-in templates for MPS, MPO, PEPS, MERA, and binary-tree layouts.
- Tune template size, bond dimension, and physical dimension before inserting them.
- Add notes and groups so larger diagrams stay easier to understand.
- Inspect and edit manual contraction paths, with optional automatic suggestions through the planner extra.
- Use the project either as a Python library or from the `tensor-network-editor` CLI.

## Why it is useful in practice

- **Local and offline-friendly.** No cloud dependency, no CDN requirement, and no Node runtime needed for end users.
- **Python-native output.** The editor returns `NetworkSpec` data structures and generated Python code you can inspect, save, or post-process.
- **Backend-flexible.** You can keep one abstract network design and target different Python ecosystems from it.
- **Cross-platform by default.** The project is tested on Windows and Linux with Python `3.11+`.

## Installation

The distribution name is `tensor-network-editor`. The import package is `tensor_network_editor`.

### Install from PyPI

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

Optional extras:

```bash
python -m pip install "tensor-network-editor[quimb]"
python -m pip install "tensor-network-editor[tensornetwork]"
python -m pip install "tensor-network-editor[tensorkrowch]"
python -m pip install "tensor-network-editor[planner]"
```

Use backend extras when you want the generated code to run in the same environment without installing those libraries separately. The `planner` extra installs `opt_einsum` for automatic greedy contraction suggestions inside the editor.

You can combine extras when needed, for example:

```bash
python -m pip install "tensor-network-editor[quimb,planner]"
```

### Install from source

If you want the current repository version instead of the published package:

```bash
python -m pip install .
```

For development work:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

### Launch the editor from the CLI

Start a local editing session:

```bash
tensor-network-editor --engine einsum_numpy
```

Open an existing design and write generated code to a file when you confirm the session:

```bash
tensor-network-editor --load my_network.json --engine quimb --save-code generated_network.py
```

Useful flags:

- `--print-code` prints the generated code to standard output.
- `--no-browser` starts the local server without opening the browser automatically.

### Launch the editor from Python

```python
from tensor_network_editor import EngineName, launch_tensor_network_editor

result = launch_tensor_network_editor(default_engine=EngineName.EINSUM_NUMPY)

if result is None:
    print("Editor cancelled.")
else:
    print(f"Design name: {result.spec.name}")
    if result.codegen is not None:
        print(result.codegen.code)
```

The editor blocks until the user presses `Done` or `Cancel`. On `Done`, it returns the abstract `NetworkSpec` together with the generated code for the selected engine.

## Use it as a library

You can also skip the UI and work directly with saved network designs:

```python
from tensor_network_editor import (
    CodeGenerationError,
    EngineName,
    generate_code,
    load_spec,
)

spec = load_spec("my_network.json")
try:
    result = generate_code(
        spec,
        engine=EngineName.QUIMB,
        path="generated_network.py",
    )
except CodeGenerationError as exc:
    print(f"Cannot generate this backend: {exc}")
else:
    print(result.code)
```

Main public entry points:

- `launch_tensor_network_editor(...) -> EditorResult | None`
- `generate_code(spec, engine=..., collection_format=..., path=...) -> CodegenResult`
- `save_spec(spec, path) -> None`
- `load_spec(path) -> NetworkSpec`
- `load_spec_from_python_code(code) -> NetworkSpec`

If the `NetworkSpec` includes a saved manual `contraction_plan`, generated code
now follows that plan directly instead of collapsing everything into one final
contraction. Complete plans emit a final `result`. Partial plans emit named
intermediates and a `remaining_operands` mapping so you can continue from that
state manually.

For `einsum_numpy` and `einsum_torch`, partial plans also emit
`remaining_operand_labels`, which makes the surviving index labels easier to
inspect when you continue the contraction manually.

`load_spec(path)` accepts saved JSON designs and supported generated `.py`
exports. If you already have the generated source code in memory, use
`load_spec_from_python_code(code)` directly.

The package also exposes the main data structures at the top level, including
`NetworkSpec`, `TensorSpec`, `IndexSpec`, `EdgeSpec`, `GroupSpec`,
`CanvasNoteSpec`, `ContractionPlanSpec`, `ContractionStepSpec`,
`ContractionOperandLayoutSpec`, `ContractionViewSnapshotSpec`, `EngineName`,
`TensorCollectionFormat`, `CodegenResult`, and `EditorResult`.

## Templates and planner

Built-in templates:

- `MPS`
- `MPO`
- `PEPS`
- `MERA`
- `Binary Tree`

Template controls let you adjust:

- graph size
- bond dimension
- physical dimension

The graph-size control label depends on the template:

- `MPS` and `MPO` use `Sites`
- `PEPS` uses `Side length`
- `MERA` and `Binary Tree` use `Depth`

The planner tools help with contraction-order work:

- Manual contraction paths are available directly in the editor.
- Automatic greedy suggestions are available when the optional `planner` extra is installed.
- Contraction summaries include useful size and cost estimates such as FLOP, MAC, and intermediate sizes.
- Generated code respects the saved manual plan when one is present.

## Supported code-generation targets

Available engine names:

- `tensornetwork`
- `quimb`
- `tensorkrowch`
- `einsum_numpy`
- `einsum_torch`

Generated code can organize created tensors in three collection formats:

- `list`
- `matrix`
- `dict`

In practice:

- `einsum_numpy` and `einsum_torch` are useful when you want lightweight generated code.
- `tensornetwork`, `quimb`, and `tensorkrowch` are good fits when you already work in those ecosystems.
- The abstract JSON save format stays backend-agnostic, so you can reopen the same design and generate for a different engine later.
- `tensornetwork` and `quimb` can export manual contraction plans step by step, including outer products.
- `einsum_numpy` and `einsum_torch` export manual plans as several `einsum(...)` calls with intermediate tensors.
- `tensorkrowch` also exports manual plans step by step, but manual outer-product steps are rejected because `contract_between(...)` cannot represent them safely there.

From Python, choose the layout with
`TensorCollectionFormat.LIST`, `TensorCollectionFormat.MATRIX`, or
`TensorCollectionFormat.DICT` through `generate_code(...)` or
`launch_tensor_network_editor(...)`.

## Save format

Designs are stored as plain JSON with a schema wrapper:

```json
{
  "schema_version": 3,
  "network": {
    "...": "..."
  }
}
```

That makes saved files easy to version, inspect, and exchange inside a normal Python workflow.

When a saved design contains a manual contraction plan, the plan can also carry
`view_snapshots`. Those snapshots store operand positions and sizes for the
contraction-scene UI through `ContractionOperandLayoutSpec` and
`ContractionViewSnapshotSpec`.

## Development

Set up the project in editable mode:

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

Useful checks:

```bash
python -m ruff check .
python -m ruff format .
python -m mypy
python -m pyright
python -m pytest
python -m build
python -m twine check dist/*
```

Cleanup scripts:

- Windows: `.\scripts\clean.bat`
- Linux: `./scripts/clean.sh`

Bundled third-party notices for redistributed frontend assets are tracked in `THIRD_PARTY_LICENSES`.

## Current limits

- Hyperedges are not supported yet.
- Real tensor data editing is not supported yet; generated tensors are initialized to zeros.
- TenPy code generation is not included in the current release.
- The main supported experience today is the local browser editor.
- Manual outer-product steps cannot be exported to `tensorkrowch`; `generate_code(...)` raises `CodeGenerationError` for that backend-specific case.

## Project links

- Documentation: [docs/README.md](docs/README.md)
- Source code: [github.com/DOKOS-TAYOS/Tensor-Network-Editor](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor)
- Changelog: [CHANGELOG.md](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor/blob/main/CHANGELOG.md)
- Example script: [examples/basic_usage.py](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor/blob/main/examples/basic_usage.py)
- Issue tracker: [github.com/DOKOS-TAYOS/Tensor-Network-Editor/issues](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor/issues)

## License

This project is distributed under the MIT License. See [LICENSE](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor/blob/main/LICENSE) for details.

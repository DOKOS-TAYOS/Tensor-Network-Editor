<p align="center">
  <img
    src="https://raw.githubusercontent.com/DOKOS-TAYOS/Tensor-Network-Editor/main/docs/images/Tensor_Network_Editor_logo.png"
    alt="Tensor Network Editor logo"
    width="880"
  />
</p>

# Tensor Network Editor

[![CI](https://img.shields.io/github/actions/workflow/status/DOKOS-TAYOS/Tensor-Network-Editor/ci.yml?branch=main&label=CI)](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor)
[![Windows%20%7C%20Linux](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-0A7BBB)](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`tensor-network-editor` is a local Python package for drawing tensor networks,
saving them as versioned JSON, and generating readable Python code for several
backends.

It is useful when you want a visual editor without losing the things that make
scientific Python workflows practical: plain data objects, files you can version,
offline use, and generated code you can inspect.

## Why This Project

- Draw tensor-network diagrams in a local browser session.
- Save and reload backend-independent JSON designs.
- Generate code for `tensornetwork`, `quimb`, `tensorkrowch`, `einsum_numpy`,
  and `einsum_torch`.
- Import supported Python network layouts from generated exports plus simple
  `quimb`, `tensornetwork`, and `einsum` / `opt_einsum` source files, or run
  explicit live imports for `quimb` and `tensornetwork` objects in a
  subprocess.
- Edit tensor initializers in the sidebar with generated zeros, ones, fill
  values, or explicit numeric JSON literals that round-trip through saved
  designs and supported generated Python.
- Create first-class hyperedges in normal mode; exports lower them
  automatically into copy tensors plus binary edges for backend code.
- Use built-in templates for MPS, MPO, PEPS, MERA, and binary-tree layouts.
- Save reusable subnetworks into project or shared catalogs and reinsert them
  later with fresh ids, tags, and quick previews.
- Work with linear, grid, and tree periodic modes and export them with any
  bundled backend.
- Reflow the current selection or the whole graph with `Auto layout` when
  imported or irregular networks need a cleaner arrangement.
- Inspect manual contraction paths and optional planner suggestions.
- Benchmark manual and automatic contraction variants from the editor or the
  CLI, with reproducible CSV/TXT/LaTeX-style tables.
- Get structural analysis with FLOP and MAC cost summaries.
- Use the package from the CLI or directly from Python.

The editor opens in your browser, but the server runs locally on your own
machine. No Node runtime or cloud service is needed for normal use.

## Minimal Installation

The PyPI package name is `tensor-network-editor`. The Python import package is
`tensor_network_editor`.

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

For backend extras, planner support, source installs, and development setup,
read [docs/installation.md](docs/installation.md).

## Basic Use

Launch the visual editor:

```bash
tensor-network-editor edit
```

This command starts a local server and waits until you press `Done` or
`Cancel` in the browser session.

Open an existing design and save generated code when the session is confirmed:

```bash
tensor-network-editor edit --load my_network.json --engine quimb --save-code generated_network.py
```

Generate a reproducible benchmark table from one saved design:

```bash
tensor-network-editor benchmark my_network.json
tensor-network-editor benchmark my_network.json --dtype float32 --format csv --output benchmark.csv
```

Use the editor from Python:

```python
from tensor_network_editor import launch_tensor_network_editor


def main() -> None:
    result = launch_tensor_network_editor()
    if result is None:
        print("Editor cancelled.")
        return

    print(f"Design name: {result.spec.name}")
    if result.codegen is not None:
        print(result.codegen.code)


if __name__ == "__main__":
    main()
```

Generate code without opening the editor:

```python
from tensor_network_editor import EngineName, generate_code, load_spec


spec = load_spec("my_network.json")
result = generate_code(spec, engine=EngineName.EINSUM_NUMPY)
print(result.code)
```

Load a live `quimb` or `tensornetwork` object from Python source:

```python
from tensor_network_editor import load_spec_from_python_code


spec = load_spec_from_python_code(
    python_source,
    python_import_mode="live",
    python_object_name="network",
)
```

This live mode executes the source in a subprocess with the active Python
interpreter from your `.venv`, auto-detects one supported runtime object when
possible, and falls back to `python_object_name` when several compatible
globals exist.

Python imports also expose an explicit reconstruction contract through
`python_reconstruction_level="auto" | "simple" | "best_available"`:

- `auto` keeps the richest supported result for the selected profile
- `generated` resolves `auto` to `best_available`, which preserves supported
  manual contraction steps
- external static profiles and live imports resolve `auto` to `simple`, which
  rebuilds only the portable network structure

## Documentation

- [Documentation index](docs/README.md): where to go for each topic.
- [Installation](docs/installation.md): full setup instructions.
- [Getting started](docs/getting-started.md): first useful workflow.
- [User guide](docs/user-guide.md): editor workflow, templates, reusable
  subnetworks, auto layout, planner, tips, benchmark mode, periodic modes, and
  limits.
- [Python API](docs/api.md): public functions and practical examples.
- [Data models](docs/data-models.md): `NetworkSpec`, tensors, edges,
  hyperedges, groups, notes, contraction plans, and periodic-mode payloads.
- [CLI](docs/cli.md): terminal commands, subnetwork catalogs,
  benchmark/export workflows, and JSON output.
- [Troubleshooting](docs/troubleshooting.md): common problems and fixes.

## Current Limits

- Hyperedges are supported only in normal mode. They are lowered to generated
  copy tensors for export, re-imported generated Python stays in that lowered
  binary form, and planner/manual contraction editing plus benchmark mode are
  disabled while hyperedges exist in the design.
- Python import is intentionally conservative. It supports the package's own
  generated exports plus static AST patterns for simple `quimb`,
  `tensornetwork`, and `einsum` / `opt_einsum` sources. It also offers an
  explicit live-import mode for `quimb` and `tensornetwork`, but that mode
  still does not recover editor layout/groups/notes, rebuild manual
  contraction plans, or load periodic-mode Python back into editable specs.
- `python_reconstruction_level="best_available"` is currently only supported
  for the package's own `generated` Python profile. External static profiles
  and live imports use the portable `simple` reconstruction contract instead.
- Browser-based live import from the editor works best for self-contained
  scripts or imports already resolvable from the active `.venv`. If a Python
  file depends on sibling modules or path-sensitive imports, prefer the Python
  API or CLI with the real file path.
- Tensor values in the visual editor are currently limited to generated zeros,
  ones, fill values, and explicit numeric JSON literals. Symbolic
  initializers, random initializers, and direct `.npy` / `.pt` imports are not
  supported yet.
- TenPy code generation is not included.
- Linear, grid, and tree periodic code generation work with all bundled
  backends.
- Manual outer-product steps still cannot be exported safely to `tensorkrowch`.
- Planner/manual contraction editing is more limited in `For bidimensional`
  and `For Tree` than in normal or linear-periodic workflows.

## Project Links

- Source code: [github.com/DOKOS-TAYOS/Tensor-Network-Editor](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Example script: [examples/basic_usage.py](examples/basic_usage.py)
- Issue tracker: [github.com/DOKOS-TAYOS/Tensor-Network-Editor/issues](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor/issues)
- License: [LICENSE](LICENSE)
- Third-party notices: [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES)

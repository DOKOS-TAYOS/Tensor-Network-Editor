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
- Use built-in templates for MPS, MPO, PEPS, MERA, and binary-tree layouts.
- Build repeated chains with For mode and export them with any bundled backend.
- Inspect manual contraction paths and optional planner suggestions.
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

## Documentation

- [Documentation index](docs/README.md): where to go for each topic.
- [Installation](docs/installation.md): full setup instructions.
- [Getting started](docs/getting-started.md): first useful workflow.
- [User guide](docs/user-guide.md): editor workflow, templates, planner, tips,
  and limits.
- [Python API](docs/api.md): public functions and practical examples.
- [Data models](docs/data-models.md): `NetworkSpec`, tensors, edges, groups,
  notes, and contraction plans.
- [CLI](docs/cli.md): terminal commands, JSON output, and template workflows.
- [Troubleshooting](docs/troubleshooting.md): common problems and fixes.

## Current Limits

- Hyperedges are not supported yet.
- Real tensor values are not edited in the visual editor; generated tensors are
  initialized by the generated backend code.
- TenPy code generation is not included.
- For mode works with all bundled backends. Manual outer-product steps still
  cannot be exported safely to `tensorkrowch`.

## Project Links

- Source code: [github.com/DOKOS-TAYOS/Tensor-Network-Editor](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Example script: [examples/basic_usage.py](examples/basic_usage.py)
- Issue tracker: [github.com/DOKOS-TAYOS/Tensor-Network-Editor/issues](https://github.com/DOKOS-TAYOS/Tensor-Network-Editor/issues)
- License: [LICENSE](LICENSE)
- Third-party notices: [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES)

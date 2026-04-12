# Troubleshooting

This page collects common problems, likely causes, and practical fixes.

## Contents

- [The Browser Did Not Open Automatically](#the-browser-did-not-open-automatically)
- [PowerShell Will Not Activate the Virtual Environment](#powershell-will-not-activate-the-virtual-environment)
- [The Command Is Not Found](#the-command-is-not-found)
- [Python Cannot Import the Package](#python-cannot-import-the-package)
- [Generated Backend Code Does Not Run](#generated-backend-code-does-not-run)
- [Schema Version Errors](#schema-version-errors)
- [Validation Errors](#validation-errors)
- [Generated Python Round Trip Fails](#generated-python-round-trip-fails)
- [Planner Suggestions Are Unavailable](#planner-suggestions-are-unavailable)
- [TensorKrowch Rejects a Manual Plan](#tensorkrowch-rejects-a-manual-plan)
- [What Should I Save?](#what-should-i-save)
- [Current Unsupported Features](#current-unsupported-features)

## The Browser Did Not Open Automatically

The local server may still be running correctly.

Try:

```bash
tensor-network-editor --no-browser
```

Then open the printed local URL manually.

From Python, browser opening is controlled with:

```python
launch_tensor_network_editor(open_browser=True)
```

Some terminals, remote sessions, and locked-down environments block automatic
browser opening even when the server is fine.

## PowerShell Will Not Activate the Virtual Environment

If PowerShell blocks `.venv` activation, you can still run commands through the
environment's Python directly:

```powershell
.\.venv\Scripts\python -m pip install tensor-network-editor
.\.venv\Scripts\python -m pytest
```

You can also ask your system administrator or Windows documentation about your
PowerShell execution policy. The exact policy depends on your machine.

## The Command Is Not Found

If `tensor-network-editor` is not recognized, first check that the virtual
environment is active.

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip show tensor-network-editor
```

Bash:

```bash
source .venv/bin/activate
python -m pip show tensor-network-editor
```

If the package is installed but the script is still not found, reinstall in the
active `.venv`:

```bash
python -m pip install --force-reinstall tensor-network-editor
```

## Python Cannot Import the Package

Check that you are using the same Python where the package was installed:

```bash
python -c "import sys; print(sys.executable)"
python -m pip show tensor-network-editor
```

In a source checkout, install editable mode for development:

```bash
python -m pip install -e ".[dev]"
```

## Generated Backend Code Does Not Run

The editor can generate code for a backend even if that backend package is not
installed. Running the generated code may need extra dependencies.

Install the matching extra:

```bash
python -m pip install "tensor-network-editor[quimb]"
python -m pip install "tensor-network-editor[tensornetwork]"
python -m pip install "tensor-network-editor[tensorkrowch]"
```

For lightweight generated code, try:

- `einsum_numpy`
- `einsum_torch`

## Schema Version Errors

Saved designs use this wrapper:

```json
{
  "schema_version": 4,
  "network": {
    "...": "..."
  }
}
```

If the schema version is different, `load_spec(...)` rejects the file clearly.
This is safer than guessing how to interpret an unknown file shape.

## Validation Errors

Common validation problems:

- empty names
- duplicated ids
- non-positive index dimensions
- missing edge endpoints
- connecting indices with different dimensions
- invalid manual contraction-plan operand ids
- invalid linear periodic carry ordering

If you build specs by hand, start by checking tensor ids, index ids, and edge
endpoint references.

Run:

```bash
tensor-network-editor validate my_network.json
tensor-network-editor lint my_network.json
```

Validation catches hard errors. Linting reports softer warnings and suggestions.

## Generated Python Round Trip Fails

The package can load supported generated Python exports:

```python
from tensor_network_editor import load_spec, load_spec_from_python_code


spec_from_file = load_spec("generated_network.py")
spec_from_text = load_spec_from_python_code(generated_source)
```

This is intentionally limited to source produced by this package. It is not a
general importer for arbitrary Python tensor-network code.

## Planner Suggestions Are Unavailable

Automatic greedy suggestions require the `planner` extra:

```bash
python -m pip install "tensor-network-editor[planner]"
```

Manual contraction plans still work without that extra.

If a network is invalid or partially inconsistent, analysis may also be
unavailable until validation issues are fixed.

## TensorKrowch Rejects a Manual Plan

TensorKrowch exports manual plans through `contract_between(...)`. That works
for ordinary shared-index contractions, but it cannot safely represent manual
outer-product steps.

If the saved plan contains an outer-product step, `generate_code(...)` raises
`CodeGenerationError` for `tensorkrowch`.

Practical fixes:

- generate code for `tensornetwork` or `quimb`
- change the manual plan so each TensorKrowch step contracts shared indices
- use `einsum_numpy` or `einsum_torch` if step-by-step einsum output fits your
  workflow

## What Should I Save?

Good practical rule:

- save JSON when the abstract network is the important artifact
- save generated Python when you want a runnable backend implementation
- keep both when you want reproducibility and code you can run immediately

If you must choose one, keep the JSON design. It can be reopened and generated
for another backend later.

## Current Unsupported Features

These are current limits, not installation problems:

- hyperedges are not supported
- real tensor values are not edited in the visual editor
- generated tensors are initialized by generated backend code
- TenPy code generation is not included
- manual outer-product plans cannot be exported to `tensorkrowch`

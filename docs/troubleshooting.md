# Troubleshooting

This page collects common problems, likely causes, and practical fixes.

## Contents

- [The Browser Did Not Open Automatically](#the-browser-did-not-open-automatically)
- [PowerShell Will Not Activate the Virtual Environment](#powershell-will-not-activate-the-virtual-environment)
- [The Command Is Not Found](#the-command-is-not-found)
- [Python Cannot Import the Package](#python-cannot-import-the-package)
- [Installed Version Does Not Match the Checkout](#installed-version-does-not-match-the-checkout)
- [The Editable Install Points to Another Worktree](#the-editable-install-points-to-another-worktree)
- [How Do I Turn On Debug Logs](#how-do-i-turn-on-debug-logs)
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
tensor-network-editor edit --no-browser
```

Then open the printed local URL manually.

From Python, browser opening is controlled with:

```python
from tensor_network_editor.editor import EditorLaunchOptions, open_editor


open_editor(options=EditorLaunchOptions(open_browser=True))
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
python -c "from pathlib import Path; import tensor_network_editor; print(Path(tensor_network_editor.__file__).resolve())"
```

In a source checkout, install editable mode for development:

```bash
python -m pip install -e ".[dev]"
```

## Installed Version Does Not Match the Checkout

If you are working from a source checkout, the package metadata installed in
the active `.venv` can lag behind the source tree after a version change.

Refresh the editable install:

```bash
python -m pip install -e ".[dev]"
```

That realigns `importlib.metadata.version("tensor-network-editor")` with
`tensor_network_editor.__version__`.

## The Editable Install Points to Another Worktree

This usually happens when one shared `.venv` has an editable install that still
points at an older checkout or a different git worktree.

Check what Python is importing:

```bash
python -c "from pathlib import Path; import tensor_network_editor; print(Path(tensor_network_editor.__file__).resolve())"
```

If that path points somewhere else, reinstall from the checkout you actually
want to use:

```bash
python -m pip install -e ".[dev]"
python -c "from pathlib import Path; import tensor_network_editor; print(Path(tensor_network_editor.__file__).resolve())"
```

If you are unsure which environment the CLI is using, enable logging once and
read the runtime diagnostics:

```bash
tensor-network-editor --log-level info template list
```

That summary includes the Python executable, current working directory,
imported package path, version, and editable-install root when available.

## How Do I Turn On Debug Logs

The package stays quiet unless you ask for logs explicitly.

Use the CLI flag for one command:

```bash
tensor-network-editor --log-level debug edit --no-browser
```

Or use the environment variable for a short session:

PowerShell:

```powershell
$env:TNE_LOG_LEVEL = "debug"
tensor-network-editor validate my_network.json
Remove-Item Env:\TNE_LOG_LEVEL
```

Bash:

```bash
TNE_LOG_LEVEL=debug tensor-network-editor validate my_network.json
```

The CLI flag takes priority over `TNE_LOG_LEVEL`.

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
  "schema_version": 2,
  "network": {
    "...": "..."
  }
}
```

Saved designs now use schema version `2`. Schema version `1` is still accepted
for older saved designs. Older compatibility-only schema numbers such as `4`,
`5`, and `6` are rejected on purpose so the loader does not silently guess how
to interpret an outdated file shape.

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
from tensor_network_editor import PythonLoadOptions, load_python_spec, load_spec


spec_from_file = load_spec("generated_network.py")
spec_from_text = load_python_spec(
    generated_source,
    python=PythonLoadOptions(),
)
```

This is intentionally limited to source produced by this package. It is not a
general importer for arbitrary Python tensor-network code.

For supported standard exports, saved manual contraction steps are recovered on
round-trip. Supported tensor initializer modes (`ones`, `fill`, and explicit
numeric literals) are also recovered. Editor-only contraction `view_snapshots`
are still dropped because the generated source does not encode that layout
state. Hyperedges are exported through autogenerated copy tensors, so the
round-trip result keeps that lowered binary network instead of reconstructing
the original hyperedge.

If you request live import for a generated file and the generated backend
package is not installed in the active `.venv`, the loader falls back to the
static generated-source parser and includes a warning that explains what
happened.

Periodic generated Python is still best treated as an output artifact rather
than a reloadable source format. Linear periodic generated Python is rejected
explicitly by the round-trip parser, and grid/tree periodic exports do not
rebuild editable periodic-mode payloads.

## Planner Suggestions Are Unavailable

Automatic greedy suggestions require the `planner` extra:

```bash
python -m pip install "tensor-network-editor[planner]"
```

Manual contraction plans still work without that extra.

If a network is invalid or partially inconsistent, analysis may also be
unavailable until validation issues are fixed.
If the design contains hyperedges, planner/manual contraction editing is
intentionally disabled in this first release.
Benchmark mode follows the same limitation for hyperedge designs.

## TensorKrowch Rejects a Manual Plan

For mode and normal network exports work with all bundled backends. The main
backend-specific restriction that still remains is TensorKrowch manual
outer-product export.

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

## Current Limits

These are current limits, not installation problems:

- hyperedges work only in normal mode, disable planner/manual contraction
  editing plus benchmark mode while present, and round-trip from generated
  Python in their lowered binary form
- tensor values support portable built-in initializers and complex scalars, but
  not symbolic expressions or direct `.npy` / `.pt` imports
- TenPy code generation is not included
- linear, grid, and tree periodic code generation work with every bundled
  backend; grid/tree manual plans can leave partial outputs in
  `remaining_operands` when a virtual boundary is intentionally kept alive
- manual outer-product plans cannot be exported to `tensorkrowch`

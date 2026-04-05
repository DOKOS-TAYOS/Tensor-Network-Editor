# Getting Started

This guide is the fastest way to get a first working result with
`tensor-network-editor`.

## What you will do

By the end of this page, you will know how to:

- install the package
- launch the local editor
- create or load a network
- confirm the session
- save the design and generate Python code

## 1. Install the package

The PyPI package name is `tensor-network-editor`. The import package name is
`tensor_network_editor`.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install tensor-network-editor
```

### Linux or macOS shell

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install tensor-network-editor
```

### Optional extras

Install extras only when you need them:

- `planner` adds automatic greedy contraction suggestions through
  `opt_einsum`
- backend extras help you run generated code in the same environment

Examples:

```bash
python -m pip install "tensor-network-editor[planner]"
python -m pip install "tensor-network-editor[quimb]"
python -m pip install "tensor-network-editor[tensornetwork]"
python -m pip install "tensor-network-editor[tensorkrowch]"
python -m pip install "tensor-network-editor[quimb,planner]"
```

## 2. Launch the editor from the command line

The quickest way to try the library is the CLI.

```bash
tensor-network-editor --engine einsum_numpy
```

What happens next:

- the package starts a local server on your machine
- it opens a browser tab unless you disable that behavior
- you edit the tensor network in the browser
- the command finishes when you press `Done` or `Cancel`

### Useful CLI options

Load an existing design:

```bash
tensor-network-editor --load my_network.json --engine quimb
```

Write the generated code to a file after confirming the session:

```bash
tensor-network-editor --load my_network.json --engine quimb --save-code generated_network.py
```

Print the generated code to the terminal:

```bash
tensor-network-editor --engine einsum_numpy --print-code
```

Start the local server without opening the browser automatically:

```bash
tensor-network-editor --engine einsum_numpy --no-browser
```

## 3. Launch the editor from Python

If you prefer to control the workflow from a script or notebook, use the public
Python API.

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

Important detail:

- when the user cancels the editor, the function returns `None`
- when the user confirms, it returns an `EditorResult`

## 4. Save and reload a network design

The package stores designs as plain JSON with a schema wrapper. That makes the
files easy to version and inspect.

```python
from tensor_network_editor import EngineName, generate_code, load_spec, save_spec

spec = load_spec("my_network.json")
save_spec(spec, "copy_of_my_network.json")

result = generate_code(
    spec,
    engine=EngineName.EINSUM_NUMPY,
    path="generated_network.py",
)

print(result.code)
```

## 5. Understand the save format at a glance

Saved files look like this at the top level:

```json
{
  "schema_version": 3,
  "network": {
    "...": "..."
  }
}
```

In practice, that means:

- files are backend-agnostic
- the same design can be reopened later and regenerated for another engine
- normal version control tools can track changes in saved networks

## 6. First workflow to try

If you want a simple first session, this is a good path:

1. Start with `einsum_numpy`.
2. Create a small network with two tensors.
3. Connect one pair of matching indices.
4. Press `Done`.
5. Save the generated code.
6. Save the JSON design so you can reopen it later.

After that, continue with [user-guide.md](user-guide.md) to understand the
editor workflow more clearly.


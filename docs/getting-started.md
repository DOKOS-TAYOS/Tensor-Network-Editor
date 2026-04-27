# Getting Started

This page gives you the fastest path to a useful first result with
`tensor-network-editor`.

## Contents

- [1. Install](#1-install)
- [2. Launch the Editor](#2-launch-the-editor)
- [3. Create a Small Network](#3-create-a-small-network)
- [4. Save JSON and Generated Code](#4-save-json-and-generated-code)
- [5. Use the Same Flow From Python](#5-use-the-same-flow-from-python)
- [Next Steps](#next-steps)

## 1. Install

Create and activate a virtual environment, then install the package.

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

For source installs, development setup, and optional extras, see
[installation.md](installation.md).

## 2. Launch the Editor

Start a local editor session:

```bash
tensor-network-editor edit
```

What happens:

- a local server starts on your machine
- a browser tab opens by default
- the command waits until you press `Done` or `Cancel`
- `Done` returns the final design and generated code for the selected engine

If your environment cannot open a browser automatically, use:

```bash
tensor-network-editor edit --no-browser
```

Then open the local URL printed in the terminal.

You can also choose the editor colors when the session starts:

```bash
tensor-network-editor edit --theme light
```

Available themes are `dark`, `light`, `contrast`, `colorblind`, and `shiny`.
The default is `dark`, and the choice only changes the browser appearance.

## 3. Create a Small Network

A simple first design is enough:

1. Keep the default engine for the first run.
2. Add two tensors with `N` or the toolbar.
3. Add one shared index dimension to each tensor. You can use `I` after
   selecting a tensor, or use the tensor controls in the sidebar.
4. Connect the matching indices.
5. Press `Done`.

If you select three or more compatible open indices, you can also press `H` to
create one hyperedge instead of several pairwise edges. The same action is
available from the `Selection` sidebar.

If the drawing looks messy after edits or imports, press `R` to open Reflow and
try `Auto layout`.

If you want a prebuilt shape, insert one of the templates instead of drawing
everything manually. Templates are explained in [user-guide.md](user-guide.md).

## 4. Save JSON and Generated Code

Open an existing JSON design:

```bash
tensor-network-editor edit --load my_network.json
```

Choose a backend in the editor, or set the initial backend from the command
line:

```bash
tensor-network-editor edit --load my_network.json --engine quimb
```

Save generated Python code when you confirm the session:

```bash
tensor-network-editor edit --load my_network.json --engine quimb --save-code generated_network.py
```

You can also print the generated code to the terminal:

```bash
tensor-network-editor edit --print-code
```

The JSON design and the generated Python code serve different purposes:

- save JSON when you want to reopen or version the abstract network
- save Python code when you want a concrete backend implementation

## 5. Use the Same Flow From Python

The editor can be launched from Python:

```python
from tensor_network_editor import open_editor


def main() -> None:
    result = open_editor()
    if result is None:
        print("Editor cancelled.")
        return

    print(result.spec.name)
    if result.codegen is not None:
        print(result.codegen.code)


if __name__ == "__main__":
    main()
```

You can also work without opening the editor:

```python
from tensor_network_editor import EngineName, generate_code, load_spec, save_spec


spec = load_spec("my_network.json")
save_spec(spec, path="copy_of_my_network.json")

result = generate_code(spec, engine=EngineName.EINSUM_NUMPY)
print(result.code)
```

## Next Steps

- Read [user-guide.md](user-guide.md) for normal editor workflow, templates,
  subnetworks, metadata filters, benchmark mode, shortcuts, planner behavior,
  and practical tips.
- Read [extended_guide.md](extended_guide.md) when you want the complete
  practical manual with deeper examples, CLI recipes, Python recipes, exports,
  modes, and current limits in one place.
- Read [api.md](api.md) if you want to integrate the package into Python code.
- Read [cli.md](cli.md) if you want validation, linting, analysis, export, and
  diff commands from the terminal.
- Read [troubleshooting.md](troubleshooting.md) if something does not behave as
  expected.

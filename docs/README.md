# Documentation

This folder contains the practical documentation for `tensor-network-editor`.
The goal is simple: help you find the right page quickly, use the library, and
avoid reading one huge document from top to bottom.

## Start Here

- New to the project: read [getting-started.md](getting-started.md).
- Installing or setting up extras: read [installation.md](installation.md).
- Using the visual editor regularly: read [user-guide.md](user-guide.md).
- Calling the package from Python: read [api.md](api.md).
- Extending templates or code generators: read [api.md](api.md).
- Building specs by hand: read [data-models.md](data-models.md).
- Working from the terminal or CI: read [cli.md](cli.md).
- Something is not working: read [troubleshooting.md](troubleshooting.md).

## What This Library Does

`tensor-network-editor` lets you:

- draw tensor-network structures in a local browser editor
- save those structures as backend-independent JSON
- reload the same design later
- generate Python code for several tensor-network backends
- create first-class hyperedges with a saved draggable hub position in normal
  mode
- keep reusable subnetworks in project or shared catalogs
- auto-layout the current selection or the whole graph when needed
- benchmark contraction variants from the editor or the CLI
- use linear, grid, and tree periodic editor modes
- inspect validation, linting, diff, and analysis results from Python or the CLI

The browser interface is local to your machine. The package starts a local
server, opens a browser tab by default, and waits until you confirm or cancel
the session.

For Python imports, prefer the documented public package surface such as
`tensor_network_editor`, `tensor_network_editor.editor`,
`tensor_network_editor.io`, `tensor_network_editor.models`,
`tensor_network_editor.validation`, and `tensor_network_editor.linting`.
Modules under `tensor_network_editor.internal` are implementation details and
are not part of the stable user-facing API.

## Pages By Need

| Need | Read |
| --- | --- |
| Install the package quickly | [installation.md](installation.md) |
| Try the editor for the first time | [getting-started.md](getting-started.md) |
| Choose a backend or collection format | [user-guide.md](user-guide.md) |
| Use templates, reusable subnetworks, auto layout, benchmark mode, or periodic modes | [user-guide.md](user-guide.md) |
| Generate code from Python | [api.md](api.md) |
| Understand `NetworkSpec` and related models | [data-models.md](data-models.md) |
| Validate, lint, analyze, benchmark, export, or diff from the terminal | [cli.md](cli.md) |
| Fix install, backend, schema, or validation problems | [troubleshooting.md](troubleshooting.md) |

## Typical Workflow

1. Install the package in a `.venv`.
2. Launch the editor from the CLI or Python.
3. Draw or load a tensor network.
4. Save the JSON design.
5. Generate backend Python code when you need a runnable implementation.
6. Reopen the JSON later if you want to edit the design or target another
   backend.

The JSON design is usually the durable artifact. Generated Python code is useful
when you want to run, inspect, or adapt a concrete backend implementation.

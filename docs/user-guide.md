# User Guide

This guide explains how to use `tensor-network-editor` comfortably after the
first launch. It focuses on practical workflow, choices, examples, and current
limits.

## Contents

- [Core Idea](#core-idea)
- [Normal Editor Workflow](#normal-editor-workflow)
- [Choosing a Backend](#choosing-a-backend)
- [Choosing a Collection Format](#choosing-a-collection-format)
- [Templates](#templates)
- [Saving and Loading](#saving-and-loading)
- [Manual Contraction Plans](#manual-contraction-plans)
- [Planner Extra](#planner-extra)
- [Linear Periodic Mode](#linear-periodic-mode)
- [Useful Tips](#useful-tips)
- [Current Limits](#current-limits)

## Core Idea

The package separates two artifacts:

- the abstract tensor-network design, stored as a `NetworkSpec`
- generated Python code for one target backend

That separation is important. You can keep one JSON design, reopen it later,
and generate code for another backend without redrawing the network.

It helps to think in these objects:

- a tensor is a node on the canvas
- an index belongs to one tensor and has a dimension
- an edge connects two indices with matching dimensions
- a group organizes several tensors visually
- a note stores text on the canvas
- a contraction plan stores a manual contraction order

## Normal Editor Workflow

A typical session looks like this:

1. Launch the editor with a default backend.
2. Create tensors and indices.
3. Connect matching indices.
4. Add groups or notes when the network becomes hard to read.
5. Optionally inspect or edit the contraction plan.
6. Confirm the session with `Done`.
7. Save the JSON design and, when useful, generated Python code.

The editor is local. Closing the browser tab does not upload data anywhere.

## Choosing a Backend

Supported engine names are:

- `tensorkrowch`
- `einsum_torch`
- `einsum_numpy`
- `quimb`
- `tensornetwork`

Simple rule of thumb:

- choose `einsum_numpy` for lightweight generated NumPy code
- choose `einsum_torch` for a PyTorch-style workflow
- choose `quimb`, `tensornetwork`, or `tensorkrowch` when you already use that
  ecosystem

The editor starts with `TensorKrowch` selected unless you pass another default
engine from the CLI or Python API.

Backend extras only help you run generated code in the same environment. The
editor can still generate source text without those backend libraries installed.
See [installation.md](installation.md#optional-extras).

## Choosing a Collection Format

Generated code can organize created tensors in three layouts:

- `list`
- `matrix`
- `dict`

This only changes how generated Python stores the tensor variables. It does not
change the abstract network stored in JSON.

Use:

- `list` for a simple ordered container
- `matrix` when the visual row layout matters
- `dict` when stable names are more convenient than positions

From Python, pass `TensorCollectionFormat.LIST`,
`TensorCollectionFormat.MATRIX`, or `TensorCollectionFormat.DICT` to
`generate_code(...)` or `launch_tensor_network_editor(...)`.

## Templates

Templates help you start from common tensor-network shapes instead of placing
each tensor by hand.

Built-in templates:

- `MPS`
- `MPO`
- `PEPS`
- `MERA`
- `Binary Tree`

Template controls include:

- graph size
- bond dimension
- physical dimension

The graph-size label depends on the selected template:

- `MPS` and `MPO` use `Sites`
- `PEPS` uses `Side length`
- `MERA` and `Binary Tree` use `Depth`

You can also build templates from Python or the CLI. See [api.md](api.md) and
[cli.md](cli.md#template-commands).

## Saving and Loading

The JSON design is the durable part of your work. Generated code is a useful
implementation artifact, but the JSON remains backend-independent.

Practical rule:

- save JSON if you want to reopen, version, compare, or regenerate the network
- save generated Python if you want to run or adapt a concrete backend script
- keep both when you want reproducibility and immediately runnable code

Saved files use a schema wrapper:

```json
{
  "schema_version": 4,
  "network": {
    "...": "..."
  }
}
```

The package validates saved designs when loading or saving.

## Manual Contraction Plans

Manual contraction plans are stored with `ContractionPlanSpec` and
`ContractionStepSpec`.

In practice:

- a plan is a named list of contraction steps
- each step consumes two operands and creates a new intermediate operand
- consumed operands cannot be reused by later steps
- a complete plan ends with one final result
- a partial plan leaves surviving operands in `remaining_operands`

When `generate_code(...)` sees a saved manual plan, generated code follows that
plan instead of using the backend's usual one-shot export.

Backend notes:

- `tensornetwork` and `quimb` can export step-by-step manual plans, including
  outer products
- `einsum_numpy` and `einsum_torch` export one `einsum(...)` call per manual
  step
- `tensorkrowch` exports normal manual contractions, but rejects manual
  outer-product steps with `CodeGenerationError`

Manual plans may also store contraction-scene snapshots. Those snapshots keep
UI layout state for operands and survive JSON round trips.

## Planner Extra

The optional `planner` extra installs `opt_einsum` and enables automatic greedy
contraction suggestions.

```bash
python -m pip install "tensor-network-editor[planner]"
```

The planner can compare manual and automatic paths using metrics such as:

- operation cost
- peak intermediate size
- estimated peak bytes for the selected dtype
- the step where the peak appears

The package still works without the `planner` extra. You only lose automatic
suggestions.

## Linear Periodic Mode

Linear periodic mode is for repeated one-dimensional structures with an
initial cell, a periodic cell, and a final cell.

Useful ideas:

- each cell can have its own tensors, edges, notes, groups, and contraction
  plan
- `Previous cell` and `Next cell` are special carry operands in manual plans
- `Next cell` must be the last contraction step in a carry plan
- generated code forwards the chosen carry operand to the next cell

This mode is more specialized than normal free drawing. Start with the regular
editor workflow unless your network really is a repeated chain.

## Useful Tips

- Keep tensor and index names meaningful. Generated Python is easier to read.
- Save the JSON design early, not only the generated code.
- Use groups for larger diagrams so visual organization does not depend only on
  tensor positions.
- Use notes to store assumptions, boundary choices, or experiment context.
- If a backend export fails, try `einsum_numpy` to inspect a simpler generated
  representation.
- Run `tensor-network-editor lint my_network.json` when a network loads but
  looks suspicious.
- Run `tensor-network-editor analyze my_network.json --dtype float32` when
  memory estimates should match your intended element width.

## Current Limits

- Hyperedges are not supported.
- The editor does not edit real tensor values.
- Generated tensors are initialized by generated backend code.
- TenPy code generation is not included.
- Manual outer-product steps cannot be exported to `tensorkrowch`.

For common fixes, see [troubleshooting.md](troubleshooting.md).

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
- [Subnetwork Library](#subnetwork-library)
- [Layout Tools](#layout-tools)
- [Metadata and Filters](#metadata-and-filters)
- [Saving and Loading](#saving-and-loading)
- [Manual Contraction Plans](#manual-contraction-plans)
- [Planner Extra](#planner-extra)
- [Benchmark Mode](#benchmark-mode)
- [Periodic Modes](#periodic-modes)
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

## Subnetwork Library

The subnetwork library is for reusable building blocks that are smaller than a
full template. It keeps the same extraction and insertion behavior used inside
the editor, but wraps that workflow in a persistent catalog.

Useful ideas:

- `Save to library` stores the current tensor selection as a reusable
  subnetwork
- the default project catalog lives next to your design at
  `.tensor-network-editor/subnetworks.json`
- a session can also point to an optional shared catalog, merged at runtime
  with the project catalog
- project entries win on name conflicts, and the editor warns when they shadow
  a shared entry
- each entry keeps a stable name, a display name, optional tags, and the saved
  subnetwork spec
- the library dialog lets you browse, search, filter by tag, inspect a small
  generated preview, and insert the saved block back into the canvas
- inserted subnetworks always receive fresh ids, so you can reuse the same
  block many times safely

This is especially useful when you repeat boundary gadgets, local motifs, or
hand-tuned fragments across several designs.

## Layout Tools

The `Reflow` popover groups the layout actions that help clean up a network
after imports, large edits, or repeated insertions.

Useful ideas:

- `Auto layout` chooses a layout for the current tensor selection
- when nothing is selected, `Auto layout` arranges the whole graph instead
- chain- and tree-like structures still use those specialized layouts
- irregular or cyclic structures use a layered placement with overlap-safe
  spacing instead of falling back immediately to a coarse grid
- the other manual actions remain available for explicit control: `Chain`,
  `Tree`, `Grid`, and `Snap to Grid`

When the active editor mode already disables reflow tools, `Auto layout`
follows the same rule instead of bypassing those restrictions.

## Metadata and Filters

The editor now exposes metadata in three layers instead of leaving everything
inside raw JSON:

- `Tags` writes `metadata.tags`
- `Suggested annotations` writes a small guided set of tensor or index keys
- `Custom metadata (JSON)` keeps the rest of your free-form metadata

The guided keys are:

- tensor: `role`, `state`, `provenance`, `symmetry`
- index: `leg_kind`, `symmetry`, `observable`

These are still plain text annotations, not locked enums. The editor offers
suggestions to get you started, but you can type any value that fits your
workflow. Clearing a guided field removes that key.

The JSON editor remains available for everything else. To avoid duplicate
editing surfaces, it hides the guided keys and `metadata.tags` while those
dedicated controls are present.

The Selection tab also includes `Metadata filters`. They are meant for visual
inspection, not structural edits:

- choose `Tensor` or `Index` scope
- filter by tag
- optionally filter by one guided key plus value
- use `Clear` to return to the normal view

Filters only change emphasis on the canvas and the minimap. They do not hide
elements, change the current selection, modify saved metadata, or enter
undo/redo history. The filter state is local to the current session.

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
  "schema_version": 5,
  "network": {
    "...": "..."
  }
}
```

The package validates saved designs when loading or saving. New saves use
schema version `5`, while older schema version `4` files are still accepted on
load.

## Tensor Values

The tensor sidebar can now store simple real tensor values directly in the
design instead of treating every tensor as an implicit backend-side zero array.

Available modes:

- `Generated zeros`: no explicit payload is stored; generated backend code
  initializes the tensor with zeros
- `Ones`: generated backend code uses a backend-native `ones(...)`
  initializer
- `Fill value`: one scalar is repeated across the whole tensor shape
- `Explicit values`: you provide JSON numbers that exactly match the tensor
  shape

Useful rules:

- explicit values must be valid JSON and must match the tensor shape exactly
- invalid JSON or ragged lists are rejected before they overwrite the saved
  design
- supported generated Python round-trips can recover these initializer modes
- symbolic expressions, random initializers, and direct `.npy` / `.pt` imports
  are still out of scope for the editor

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

## Benchmark Mode

Benchmark mode is the editor workflow for comparing several contraction schemes
on the same network without permanently changing the saved manual path until
you leave the benchmark session.

Useful ideas:

- `Benchmark` starts from the current tensor network view and lets you move
  through saved comparison schemes
- `Compare` opens a summary table with `Name`, `FLOP`, `MAC`, `Peak`, and
  `Peak Memory`
- the compare dialog can export the current table as `CSV` or `TXT`, and can
  copy a `LaTeX` table for papers or notes
- the CLI mirrors this workflow with
  `tensor-network-editor benchmark my_network.json`
- `--format csv`, `--format latex`, and `--output ...` are useful when you
  want reproducible tables outside the browser
- benchmark scheme views keep template, subnetwork-library, and reflow actions
  disabled so comparison sessions do not drift into normal editing by mistake

If the active `.venv` does not include the `planner` extra, manual benchmark
rows still work and the automatic rows are reported as unavailable.

## Periodic Modes

Periodic modes are the specialized editor workflows for repeated structures.
They are more constrained than free drawing, but they let you keep a reusable
typed payload instead of treating repetition as a visual convention only.

### Linear periodic (For unidimensional)

Linear periodic mode is for repeated one-dimensional structures with an
initial cell, a periodic cell, and a final cell.

Useful ideas:

- each cell can have its own tensors, edges, notes, groups, and contraction
  plan
- For mode can generate code with `tensornetwork`, `quimb`, `tensorkrowch`,
  `einsum_numpy`, and `einsum_torch`
- `Previous cell` and `Next cell` are special carry operands in manual plans
- `Next cell` must be the last contraction step in a carry plan
- generated code forwards the chosen carry operand to the next cell
- `quimb` exports `network_tensors`, `network`, `open_inds`, and `result` when
  the repeated chain finishes in one contracted object
- `einsum_numpy` and `einsum_torch` export `result`, plus
  `remaining_operands` when a carry/manual path leaves extra operands alive
- `tensorkrowch` still rejects manual outer-product steps, including in For
  mode

This mode is more specialized than normal free drawing. Start with the regular
editor workflow unless your network really is a repeated chain.

### Grid periodic (For bidimensional)

Grid periodic mode is for repeated two-dimensional layouts represented by a
nine-cell neighborhood around the active center cell.

Useful ideas:

- the saved payload uses `GridPeriodicGridSpec` with one representative cell
  for each position in the `3x3` neighborhood
- boundary tensors describe how bonds continue toward neighboring cells
- export works with the bundled backends, so you can keep a reusable 2D design
  instead of flattening it into one large hand-drawn graph
- planner/manual contraction editing is intentionally limited here; the mode is
  focused on modeling, validation, serialization, and code generation

This is a good fit for repeated PEPS-style neighborhoods or other local 2D
motifs where center, edge, and corner cells differ.

### Tree periodic (For Tree)

Tree periodic mode is for repeated rooted tree structures with representative
`root`, `branch`, and `leaf` cells plus a configurable branching factor.

Useful ideas:

- the saved payload uses `TreePeriodicTreeSpec`
- the editor keeps one representative cell for the root, one for internal
  branches, and one for leaves
- export works with the bundled backends, so the mode is officially supported
  for modeling, serialization, and code generation
- planner/manual contraction editing is still disabled in `For Tree`; this
  iteration only productizes the mode as a modeling and export workflow

This mode is useful when the repeated structure is genuinely hierarchical and a
linear or grid neighborhood would hide that intent.

## Useful Tips

- Keep tensor and index names meaningful. Generated Python is easier to read.
- Save the JSON design early, not only the generated code.
- Use groups for larger diagrams so visual organization does not depend only on
  tensor positions.
- Save repeated motifs into the subnetwork library instead of copying them by
  hand between files.
- Use notes to store assumptions, boundary choices, or experiment context.
- Try `Auto layout` after importing a design or inserting several reusable
  blocks, then finish with a manual layout action only if you want a specific
  visual style.
- If a backend export fails, try `einsum_numpy` to inspect a simpler generated
  representation.
- Run `tensor-network-editor lint my_network.json` when a network loads but
  looks suspicious.
- Run `tensor-network-editor analyze my_network.json --dtype float32` when
  memory estimates should match your intended element width.
- Run `tensor-network-editor benchmark my_network.json --format csv --output benchmark.csv`
  when you want a stable comparison table for experiments or papers.

## Current Limits

- Hyperedges are not supported.
- Tensor values are limited to generated zeros, ones, fill values, and
  explicit numeric JSON literals.
- TenPy code generation is not included.
- Linear, grid, and tree periodic code generation work with every bundled
  backend.
- Manual outer-product steps cannot be exported to `tensorkrowch`.
- `For bidimensional` and `For Tree` do not expose the same planner/manual
  contraction workflow as normal or linear-periodic editing.

For common fixes, see [troubleshooting.md](troubleshooting.md).

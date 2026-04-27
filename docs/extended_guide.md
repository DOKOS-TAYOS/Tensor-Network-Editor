# Extended Guide

This guide is the long practical manual for `tensor-network-editor`. It is
written for users who want to understand the available workflows in one place:
the browser editor, saved JSON designs, generated backend code, command-line
checks, Python recipes, templates, subnetworks, tensor values, metadata,
planner tools, benchmark mode, and periodic modes.

If you only want a quick first run, start with [getting-started.md](getting-started.md).
If you want the short everyday editor guide, use [user-guide.md](user-guide.md).
This page intentionally goes deeper.

## Contents

- [The Big Picture](#the-big-picture)
- [Choosing The Right Tool](#choosing-the-right-tool)
- [Installation And Extras](#installation-and-extras)
- [Launching The Editor](#launching-the-editor)
- [Files, Drafts, And Reproducibility](#files-drafts-and-reproducibility)
- [Normal Editor Workflow](#normal-editor-workflow)
- [Selection, Menus, And Shortcuts](#selection-menus-and-shortcuts)
- [Backends And Collection Formats](#backends-and-collection-formats)
- [Tensor Values And External Data](#tensor-values-and-external-data)
- [Templates](#templates)
- [Reusable Subnetworks](#reusable-subnetworks)
- [Layout And Reflow](#layout-and-reflow)
- [Metadata And Filters](#metadata-and-filters)
- [Hyperedges](#hyperedges)
- [Manual Contraction Plans](#manual-contraction-plans)
- [Planner Extra And Benchmark Mode](#planner-extra-and-benchmark-mode)
- [Periodic Modes](#periodic-modes)
- [Saving, Loading, And Python Imports](#saving-loading-and-python-imports)
- [Headless CLI Workflows](#headless-cli-workflows)
- [Python API Recipes](#python-api-recipes)
- [Rendering And Academic Exports](#rendering-and-academic-exports)
- [Recommended Workflows](#recommended-workflows)
- [Current Limits](#current-limits)

## The Big Picture

`tensor-network-editor` separates the model you design from the backend code
you generate.

The durable artifact is usually a JSON design. It stores a `NetworkSpec` inside
a small schema wrapper, including tensors, indices, edges, hyperedges, groups,
notes, metadata, tensor initializers, periodic-mode payloads, and manual
contraction plans when present.

Generated Python code is a backend-specific artifact. It is useful when you
want to run, inspect, benchmark, or adapt a concrete implementation, but it is
not the best long-term editing format. Keep the JSON when you want to reopen
the design and target another backend later.

The editor itself runs locally. The package starts a Python HTTP server on your
machine, opens a browser tab by default, and waits until you press `Done` or
`Cancel`. Normal use does not require Node.js or a cloud service.

## Choosing The Right Tool

| Goal | Best tool |
| --- | --- |
| Draw or edit a network visually | `tensor-network-editor edit` |
| Try the project for the first time | [getting-started.md](getting-started.md) |
| Use common tensor-network shapes | Templates in the editor, CLI, or Python API |
| Reuse a smaller motif across files | Subnetwork library |
| Generate backend Python | `export`, the editor code panel, or `generate_code(...)` |
| Check whether a saved design is valid | `validate` or `validate_spec(...)` |
| Get softer modeling suggestions | `lint` or `lint_spec(...)` |
| Estimate contraction cost and memory | `analyze` or `analyze_spec(...)` |
| Compare manual and automatic paths | Benchmark mode or `benchmark` |
| Diagnose a confusing file or environment | `doctor` |
| Produce a paper-friendly diagram | `render --format tikz` or editor `.tex` export |
| Produce a Graphviz graph | `render --format dot` or editor `.dot` export |
| Normalize JSON for Git diffs | `canonicalize` |
| Compare two versions of a design | `diff` or `diff --semantic` |
| Build a normal-mode network from Python | `NetworkBuilder` |

## Installation And Extras

Use a virtual environment. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install tensor-network-editor
```

On Linux Bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install tensor-network-editor
```

The distribution name is `tensor-network-editor`. The Python import package is
`tensor_network_editor`.

The base package requires Python `3.11+` and `Pillow>=10`.
Install extras only when you need them:

```bash
python -m pip install "tensor-network-editor[numpy]"
python -m pip install "tensor-network-editor[torch]"
python -m pip install "tensor-network-editor[quimb]"
python -m pip install "tensor-network-editor[tensornetwork]"
python -m pip install "tensor-network-editor[tensorkrowch]"
python -m pip install "tensor-network-editor[planner]"
```

Backend extras let you run generated code in the same environment. The editor
can still generate source text when a backend package is not installed.

The `planner` extra installs `opt_einsum`, which enables automatic greedy
contraction suggestions and automatic benchmark rows. Pillow is now part of the
base installation because academic PNG/PDF rendering depends on it.

## Launching The Editor

The main command is:

```bash
tensor-network-editor edit
```

Open an existing design:

```bash
tensor-network-editor edit --load my_network.json
```

Start with a different backend selected:

```bash
tensor-network-editor edit --engine quimb
```

Start with a different color theme:

```bash
tensor-network-editor edit --theme colorblind
```

The available themes are:

- `dark`: the default dark editor palette
- `light`: a neutral light palette
- `contrast`: high-contrast black, white, yellow, and cyan
- `colorblind`: a palette based on distinguishable Okabe-Ito style colors
- `shiny`: a dark palette with brighter cyan, pink, green, and yellow accents

Themes are chosen only when the editor starts. They do not change the saved
JSON design, generated Python code, or recoverable draft data.

Save generated code when the session is confirmed:

```bash
tensor-network-editor edit --load my_network.json --engine quimb --save-code generated_network.py
```

Print generated code to standard output after confirmation:

```bash
tensor-network-editor edit --print-code
```

Start the server without opening the browser automatically:

```bash
tensor-network-editor edit --no-browser
```

Use `--no-browser` when automatic browser opening is blocked, when you work
over SSH, or when you want to copy the printed local URL into a browser
manually.

From Python:

```python
from tensor_network_editor import EngineName
from tensor_network_editor.editor import EditorLaunchOptions, open_editor


def main() -> None:
    result = open_editor(
        options=EditorLaunchOptions(
            default_engine=EngineName.EINSUM_NUMPY,
            theme="light",
            open_browser=True,
        )
    )
    if result is None:
        print("Editor cancelled.")
        return

    print(result.spec.name)
    if result.codegen is not None:
        print(result.codegen.code)


if __name__ == "__main__":
    main()
```

## Files, Drafts, And Reproducibility

There are three useful file types to think about:

- JSON design files: the main editable, backend-independent artifact.
- Generated Python files: runnable backend-specific code.
- Static figure files: SVG, PNG, TikZ, or DOT outputs for reports and papers.

Save JSON early. Generated code can usually be reproduced from JSON, but the
reverse is intentionally more limited.

The browser editor also keeps a project-local recoverable draft under
`.tensor-network-editor/drafts/`. If a previous session draft exists when the
editor starts, it asks whether you want to restore it. Drafts help when a tab
is closed before you save. They are cleared after explicit JSON save, `Done`,
`Cancel`, or `Start fresh` actions.

From Python, `EditorLaunchOptions(draft_path="...")` can override where that
recoverable draft is stored. Most users can ignore this option and use the
project-local default.

For reproducible work, a good pattern is:

1. Save the JSON design.
2. Run `validate` and `lint`.
3. Generate backend code into a separate `.py` file.
4. Save benchmark tables or rendered figures only when you need them.
5. Keep JSON and important generated outputs in version control.

## Normal Editor Workflow

Normal mode is the free drawing mode. It is the best starting point unless
your network is truly periodic.

A typical workflow:

1. Add tensors.
2. Add indices to tensors.
3. Set index dimensions.
4. Connect compatible open indices.
5. Add groups or notes when the diagram needs structure.
6. Add tensor values if you want generated code to include initializers.
7. Add metadata or tags when you want better filtering and lint suggestions.
8. Create hyperedges for one logical bond shared by three or more indices.
9. Build or inspect a manual contraction plan when order matters.
10. Save JSON and optionally generate backend code.

Think of the canvas objects like this:

- Tensor: a node in the tensor network.
- Index: one leg of one tensor, with a dimension.
- Edge: a pairwise connection between two indices.
- Hyperedge: one logical connection between three or more indices.
- Group: visual organization for several tensors.
- Note: text stored on the canvas.
- Contraction plan: a saved manual contraction order.

The editor validates the model as you work and again when saving or generating
code. If a backend cannot represent a specific request, code generation raises
a clear package error instead of producing misleading source.

## Selection, Menus, And Shortcuts

Most actions are available in more than one place:

- the top toolbar
- the Selection sidebar
- right-click menus on tensors, indices, hyperedges, or selections
- keyboard shortcuts

Common shortcuts:

| Shortcut | Action |
| --- | --- |
| `N` | Add a tensor |
| `I` | Add one index to each selected editable tensor |
| `H` | Create a hyperedge from selected compatible open indices |
| `G` | Group the selected tensors |
| `R` | Open the Reflow popover |
| `Shift+E` | Save the selected subnetwork as JSON |
| `Alt+ArrowLeft` / `Alt+ArrowRight` | Move to previous or next periodic cell/item |
| `Alt+ArrowUp` / `Alt+ArrowDown` | Move vertically in grid periodic mode |
| `Ctrl/Cmd+S` | Save the current design |
| `Ctrl/Cmd+L` | Load a design file |
| `Ctrl/Cmd+Enter` | Finish the editor session |

Text fields keep normal typing behavior. Native text copy still works inside
side panels and editable fields.

When several indices are selected, the sidebar and compact right-click menu
show one shared `Dimension` field. Changing it updates all selected editable
indices and synchronizes connected partners where the normal rules allow it.

## Backends And Collection Formats

Supported backend engine names:

- `tensorkrowch`
- `einsum_torch`
- `einsum_numpy`
- `quimb`
- `tensornetwork`

Simple choices:

- Use `einsum_numpy` for small, readable NumPy code.
- Use `einsum_torch` when your downstream workflow uses PyTorch.
- Use `quimb`, `tensornetwork`, or `tensorkrowch` when you already work in one
  of those ecosystems.

Generated code can organize tensors in three collection formats:

- `list`: a simple ordered container.
- `matrix`: useful when the visual row/column layout matters.
- `dict`: useful when stable names are more convenient than positions.

The collection format changes only the generated Python storage layout. It
does not change the saved JSON design.

CLI example:

```bash
tensor-network-editor export my_network.json --engine einsum_numpy --collection-format dict
```

Python example:

```python
from tensor_network_editor import EngineName, TensorCollectionFormat, generate_code, load_spec


spec = load_spec("my_network.json")
result = generate_code(
    spec,
    engine=EngineName.EINSUM_NUMPY,
    collection_format=TensorCollectionFormat.DICT,
)
print(result.code)
```

## Tensor Values And External Data

Tensors can store portable initializer information in the design. This is
useful when generated code should do more than create placeholder arrays.

Available modes:

- Generated zeros
- Ones
- Fill value
- Identity / delta
- Copy tensor
- Random normal or uniform values with a seed
- Explicit values
- External `.npy`, `.npz`, or `.pt` data

Useful details:

- Portable dtypes include `float32`, `float64`, `complex64`, and `complex128`.
- Complex scalars are stored in JSON as objects with `real` and `imag` fields.
- The sidebar accepts friendlier complex text such as `1+2j`.
- Explicit values must be JSON values matching the tensor shape exactly.
- `.npy` references load the whole array.
- `.npz` references require an array key.
- `.pt` references can load a tensor directly or use a key when the file stores
  a mapping.
- CLI exports resolve relative external data paths relative to the input JSON
  design path.

External data is best when the tensor values are too large or too practical to
store directly in the JSON design. Keep the external files next to the design
or use stable project-relative paths.

## Templates

Templates create valid starting networks without placing every tensor by hand.

Built-in template names:

- `mps`
- `mpo`
- `peps_2x2`
- `mera`
- `binary_tree`
- `ttn`
- `pepo`
- `heisenberg_mps`
- `ising_mps`
- `transverse_ising_mpo`
- `tebd_gate_layer`

Template parameters:

- graph size
- bond dimension
- physical dimension

The graph-size label depends on the template:

- MPS-like and MPO-like templates use `Sites`.
- PEPS and PEPO use `Side length`.
- MERA, Binary Tree, and TTN use `Depth`.

CLI examples:

```bash
tensor-network-editor template list
tensor-network-editor template list --format json
tensor-network-editor template build mps --graph-size 6 --bond-dimension 4 --physical-dimension 2 --output mps.json
```

Python example:

```python
from tensor_network_editor.templates import build_template_spec, parse_template_parameters


parameters = parse_template_parameters(
    "mps",
    {
        "graph_size": 6,
        "bond_dimension": 4,
        "physical_dimension": 2,
    },
)
spec = build_template_spec("mps", parameters=parameters)
```

Templates are a starting point. After inserting or building one, you can edit
the resulting design normally.

## Reusable Subnetworks

Subnetworks are reusable fragments smaller than a full template. They are
normal `NetworkSpec` fragments extracted from selected tensors.

Use them for:

- repeated local motifs
- boundary gadgets
- hand-tuned blocks
- reusable cells for experiments
- shared project building blocks

Important behavior:

- Extraction works in normal graph mode.
- Selected tensors are copied into the fragment.
- Pairwise edges and hyperedges are copied only when fully inside the selected
  tensor set.
- Groups inside the selected tensor set are preserved.
- Notes and contraction plans are not copied into fragments.
- Insertion gives tensors, indices, edges, hyperedges, and groups fresh ids.

The project catalog normally lives next to a design at:

```text
.tensor-network-editor/subnetworks.json
```

You can also pass a shared catalog path. Project entries win when project and
shared catalogs contain the same subnetwork name.

CLI examples:

```bash
tensor-network-editor subnetwork list my_network.json
tensor-network-editor subnetwork list my_network.json --shared-catalog-path shared-subnetworks.json
tensor-network-editor subnetwork save my_network.json --tensor-ids tensor_a tensor_b --name local_pair --tags reusable boundary
tensor-network-editor subnetwork export my_network.json local_pair --output local_pair.json
```

Python example:

```python
from tensor_network_editor import CanvasPosition, load_spec
from tensor_network_editor.subnetworks import (
    extract_subnetwork_spec,
    prepare_subnetwork_for_insertion,
)


source = load_spec("my_network.json")
fragment = extract_subnetwork_spec(
    source,
    tensor_ids=["tensor_a", "tensor_b"],
    name="local_pair",
)
prepared = prepare_subnetwork_for_insertion(
    fragment,
    target_center=CanvasPosition(x=400.0, y=260.0),
)
```

## Layout And Reflow

The `Reflow` popover helps clean up diagrams after imports, large edits, or
subnetwork insertion.

Available layout ideas:

- `Auto layout`: choose a suitable layout for the selected tensors.
- Whole-graph auto layout: run `Auto layout` with nothing selected.
- `Chain`: arrange a chain-like selection.
- `Tree`: arrange a tree-like selection.
- `Grid`: place tensors on a grid.
- `Snap to Grid`: align positions without changing the graph structure.

Auto layout recognizes simple chain-like and tree-like structures, while
irregular or cyclic structures use a layered placement with overlap-safe
spacing.

Layout tools change canvas positions, not tensor-network semantics. They are
useful before saving figures or after importing Python-generated structures
that did not carry editor layout.

## Metadata And Filters

Metadata has three user-facing layers:

- Tags stored in `metadata.tags`.
- Guided annotations for common tensor and index meanings.
- Custom JSON metadata for everything else.

Guided tensor keys:

- `role`
- `state`
- `provenance`
- `symmetry`

Guided index keys:

- `leg_kind`
- `symmetry`
- `observable`

These are suggestions, not locked enums. You can type values that fit your
workflow.

Metadata filters are visual inspection tools:

- filter tensors or indices by tag
- optionally filter by guided key and value
- emphasize matches on the canvas and minimap
- clear filters to return to the normal view

Filters do not change saved metadata, selection, undo/redo history, or graph
structure. They are session-local.

Metadata-aware linting can use guided keys to produce better warnings. For
example, it can flag suspicious open bond legs or observable annotations on
connected bonds.

## Hyperedges

Hyperedges represent one logical connection shared by three or more indices.
They are useful when you would otherwise draw an auxiliary copy tensor only to
make the diagram express a multiway bond.

How to create one:

1. Select three or more open indices.
2. Make sure they all have the same dimension.
3. Press `H`, use the Selection action, or use the multi-index right-click
   menu.

Useful rules:

- Hyperedges are available only in normal mode.
- Every endpoint must be an open index.
- All endpoint dimensions must match.
- The visible hub is a virtual UI node, not a tensor in the design.
- You can drag the hub to improve the drawing.
- The saved `hub_offset` stores that manual hub displacement.
- Generated backend code lowers each hyperedge to a generated copy tensor plus
  ordinary binary edges.
- Supported generated Python round-trips can reconstruct the original
  `HyperedgeSpec` from structured copy-tensor markers.

Planner and benchmark analysis lower hyperedges to internal generated copy
tensors for cost analysis while keeping the saved visual model unchanged.

## Manual Contraction Plans

Manual plans store an explicit contraction order.

Conceptually:

- A plan contains steps.
- Each step consumes two operands.
- Each step creates one new intermediate operand.
- Consumed operands cannot be reused.
- A complete plan ends with one final result.
- A partial plan leaves surviving operands in `remaining_operands`.

When generated code sees a saved manual plan, it follows that plan instead of
using a backend's normal one-shot contraction.

Backend notes:

- `tensornetwork` and `quimb` can export step-by-step manual plans, including
  outer products.
- `einsum_numpy` and `einsum_torch` emit one `einsum(...)` call per manual
  step.
- `tensorkrowch` supports normal manual contractions but rejects manual
  outer-product steps.

Manual plans can also keep contraction-scene snapshots. Those snapshots store
editor layout state for planner operands and survive JSON round trips.

Partial plans are useful when you intentionally want a frontier or intermediate
network instead of one final scalar/tensor.

## Planner Extra And Benchmark Mode

The optional planner extra enables automatic greedy contraction suggestions:

```bash
python -m pip install "tensor-network-editor[planner]"
```

Planner metrics include:

- FLOP cost
- MAC cost
- peak intermediate size
- estimated peak memory for the selected dtype
- the step where peak memory appears

Without the `planner` extra, manual analysis still works. Automatic suggestions
and automatic benchmark rows are marked unavailable.

Benchmark mode compares contraction variants:

- Manual
- Auto full
- Auto future
- Auto past

The comparison table uses stable columns:

- Name
- FLOP
- MAC
- Peak
- Peak Memory

CLI examples:

```bash
tensor-network-editor benchmark my_network.json
tensor-network-editor benchmark my_network.json --dtype float32
tensor-network-editor benchmark my_network.json --format csv --output benchmark.csv
tensor-network-editor benchmark my_network.json --format latex --output benchmark.tex
```

Use benchmark mode when you want to compare contraction choices without
permanently replacing your saved manual path during the comparison session.

## Periodic Modes

Periodic modes are specialized workflows for repeated structures. They are
more constrained than normal drawing because they store a typed repeated
payload instead of only a large flattened diagram.

Use normal mode unless repetition is part of the model.

### Linear Periodic Mode

Linear periodic mode is for one-dimensional repeated structures. It stores:

- an initial cell
- a periodic cell
- a final cell

Useful ideas:

- Each cell can have its own tensors, edges, groups, notes, and contraction
  plan.
- `Alt+ArrowLeft` and `Alt+ArrowRight` move between cells/items.
- Manual plans can use `Previous cell` and `Next cell` carry operands.
- `Next cell` must be the last contraction step in a carry plan.
- Generated code forwards the chosen carry operand to the next cell.
- Partial plans can leave extra operands alive in `remaining_operands`.

Linear periodic exports work with all bundled backends.

### Grid Periodic Mode

Grid periodic mode is for repeated two-dimensional neighborhoods. It stores a
`3x3` representative grid around an active center cell.

Useful ideas:

- Corner, edge, and center cells can differ.
- Boundary tensors describe how bonds continue toward neighboring cells.
- Toolbar arrows or `Alt+Arrow` shortcuts move through the neighborhood.
- Manual planner mode exposes virtual boundary operands:
  - `Upper cell`
  - `Right cell`
  - `Lower cell`
  - `Left cell`
- Generated code visits cells row by row from the upper-left corner.
- Partial plans keep surviving values in `remaining_operands`.

Grid periodic mode is a good fit for PEPS-style local neighborhoods or other
2D motifs where edge and corner behavior matters.

### Tree Periodic Mode

Tree periodic mode is for repeated rooted tree structures. It stores:

- one root representative
- one internal branch representative
- one leaf representative
- a branching factor

Useful ideas:

- Manual planner mode exposes virtual boundary operands:
  - `Parent cell`
  - `Child N` for each child branch
- Generated code contracts bottom-up.
- Leaves contract into parents first.
- Internal levels contract upward until the root is reached.
- Partial plans can leave frontier operands in `remaining_operands`.

Tree periodic mode is useful when the repeated structure is genuinely
hierarchical.

Hyperedges are currently normal-mode only, so they are not available inside
periodic-cell payloads.

## Saving, Loading, And Python Imports

JSON save/load is the richest editing workflow:

```bash
tensor-network-editor edit --load my_network.json
```

Python API:

```python
from tensor_network_editor import load_spec, save_spec


spec = load_spec("my_network.json")
save_spec(spec, path="copy_of_my_network.json")
```

Saved JSON uses:

```json
{
  "schema_version": 2,
  "network": {
    "...": "..."
  }
}
```

Schema version `1` files are still accepted for older saved designs. New saves
use schema version `2`.

The package can also load supported Python source profiles:

- generated exports from this package
- conservative static `quimb` patterns
- conservative static `tensornetwork` patterns
- conservative static `einsum` / `opt_einsum` patterns
- live `quimb` or `tensornetwork` runtime objects in a subprocess

Static import is the safer default because it does not execute user code. Live
import is useful when a runtime object is easier to inspect than source text.

CLI examples:

```bash
tensor-network-editor validate generated_network.py
tensor-network-editor --python-import-mode live validate runtime_network.py
tensor-network-editor --python-import-mode live --python-object network validate runtime_network.py
tensor-network-editor --python-reconstruction-level simple validate external_network.py
```

Python example:

```python
from tensor_network_editor import PythonLoadOptions, load_spec


spec = load_spec(
    "runtime_network.py",
    python=PythonLoadOptions(
        source_profile="quimb",
        import_mode="live",
        reconstruction_level="simple",
        object_name="network",
    ),
)
```

Reconstruction levels:

- `auto`: choose the richest supported result for the detected profile.
- `simple`: rebuild portable network structure.
- `best_available`: currently for this package's own generated exports.

Generated exports provide the richest round-trip. External static profiles and
live imports are intentionally conservative and do not recover editor layout,
groups, notes, or manual contraction plans.

If live import is requested for generated source and the backend import fails
because the backend package is missing, the loader can fall back to the static
generated-source parser and report a warning.

## Headless CLI Workflows

Headless commands do not open the editor.

Validation:

```bash
tensor-network-editor validate my_network.json
tensor-network-editor validate my_network.json --format json
```

Linting:

```bash
tensor-network-editor lint my_network.json
tensor-network-editor lint my_network.json --max-tensor-rank 8 --max-tensor-cardinality 50000
tensor-network-editor lint my_network.json --fail-on warning
```

Analysis:

```bash
tensor-network-editor analyze my_network.json
tensor-network-editor analyze my_network.json --dtype float32
tensor-network-editor analyze my_network.json --format json
```

Benchmark:

```bash
tensor-network-editor benchmark my_network.json
tensor-network-editor benchmark my_network.json --format csv --output benchmark.csv
```

Doctor:

```bash
tensor-network-editor doctor my_network.json
tensor-network-editor doctor my_network.json --format json
```

Export:

```bash
tensor-network-editor export my_network.json --engine quimb --output generated_network.py
tensor-network-editor export my_network.json --engine einsum_numpy --collection-format dict
```

Render:

```bash
tensor-network-editor render my_network.json --format svg --output figure.svg
tensor-network-editor render my_network.json --format tikz --output figure.tex
tensor-network-editor render my_network.json --format dot --output graph.dot
tensor-network-editor render my_network.json --format png --output figure.png
```

Canonicalize:

```bash
tensor-network-editor canonicalize my_network.json --output my_network.canonical.json
tensor-network-editor canonicalize my_network.json --deterministic-ids
```

Diff:

```bash
tensor-network-editor diff before.json after.json
tensor-network-editor diff before.json after.json --semantic
tensor-network-editor diff before.json after.json --semantic --format json
```

Template and subnetwork commands:

```bash
tensor-network-editor template list
tensor-network-editor template build mps --graph-size 6 --bond-dimension 4 --physical-dimension 2 --output mps.json
tensor-network-editor subnetwork list my_network.json
tensor-network-editor subnetwork save my_network.json --tensor-ids tensor_a tensor_b --name local_pair
```

Common exit codes:

- `0`: command completed successfully.
- `1`: validation failed, doctor found validation errors, or lint failed with
  `--fail-on warning`.
- `2`: expected package error such as code generation, serialization, IO, or
  invalid option value.
- `130`: interrupted with Ctrl+C.

## Python API Recipes

### Open The Editor

```python
from tensor_network_editor import open_editor


result = open_editor()
if result is None:
    print("Cancelled")
else:
    print(result.spec.name)
```

### Build A Small Network

```python
from tensor_network_editor import NetworkBuilder


builder = NetworkBuilder("chain")
left = builder.tensor("A", position=(120.0, 160.0))
left.index("i", 2)
left.index("x", 3)

right = builder.tensor("B", position=(360.0, 160.0))
right.index("x", 3)
right.index("j", 4)

builder.connect(left["x"], right["x"], name="bond_x")
spec = builder.build()
```

`builder.build()` validates by default. Use `validate=False` only when you
intentionally need an in-progress invalid design.

### Generate Code

```python
from tensor_network_editor import EngineName, generate_code, load_spec


spec = load_spec("my_network.json")
result = generate_code(
    spec,
    engine=EngineName.EINSUM_NUMPY,
    output_path="generated_network.py",
)
print(result.warnings)
```

### Render Figures

```python
from tensor_network_editor import (
    DotRenderOptions,
    SvgRenderOptions,
    TikzRenderOptions,
    load_spec,
    render_spec_dot,
    render_spec_png,
    render_spec_svg,
    render_spec_tikz,
)


spec = load_spec("my_network.json")
render_spec_svg(spec, options=SvgRenderOptions(padding=48.0), output_path="figure.svg")
render_spec_tikz(spec, options=TikzRenderOptions(scale=0.02), output_path="figure.tex")
render_spec_dot(spec, options=DotRenderOptions(include_open_indices=True), output_path="graph.dot")
render_spec_png(spec, output_path="figure.png")
```

### Validate, Lint, Analyze, Canonicalize, And Diff

```python
from tensor_network_editor import (
    analyze_spec,
    canonicalize_spec,
    diff_specs,
    lint_spec,
    load_spec,
    semantic_diff_specs,
    validate_spec,
)


spec = load_spec("my_network.json")
validation_issues = validate_spec(spec)
lint_report = lint_spec(spec)
analysis = analyze_spec(spec, memory_dtype="float32")
canonical = canonicalize_spec(spec, deterministic_ids=False)
diff = diff_specs(spec, canonical)
semantic = semantic_diff_specs(spec, canonical)
```

### Use Templates

```python
from tensor_network_editor import list_template_names
from tensor_network_editor.templates import build_template_spec, parse_template_parameters


print(list_template_names())
parameters = parse_template_parameters(
    "peps_2x2",
    {"graph_size": 3, "bond_dimension": 3, "physical_dimension": 2},
)
spec = build_template_spec("peps_2x2", parameters=parameters)
```

### Use Subnetworks

```python
from tensor_network_editor import CanvasPosition, load_spec
from tensor_network_editor.subnetworks import (
    extract_subnetwork_spec,
    prepare_subnetwork_for_insertion,
)


source = load_spec("my_network.json")
fragment = extract_subnetwork_spec(source, tensor_ids=["tensor_a", "tensor_b"])
prepared = prepare_subnetwork_for_insertion(
    fragment,
    target_center=CanvasPosition(x=500.0, y=300.0),
)
```

## Rendering And Academic Exports

Rendering creates static diagrams from saved canvas geometry.

Supported render targets:

- SVG
- PNG with the `png` extra
- TikZ/LaTeX
- Graphviz/DOT

SVG, TikZ, and DOT renderers are pure Python. They do not require a browser,
Node.js, LaTeX, or Graphviz to produce text output. PNG rendering requires
Pillow.

TikZ output is a `tikzpicture`, not a complete LaTeX document. DOT output is a
Graphviz `graph`.

Use these outputs for:

- papers
- notes
- visual regression artifacts
- documentation
- graph inspection outside the editor

The editor File menu can also export the current canvas directly as `.tex` or
`.dot` using the same renderers.

## Recommended Workflows

For exploratory modeling:

1. Start with `tensor-network-editor edit`.
2. Use templates or subnetworks to avoid repetitive drawing.
3. Save JSON frequently.
4. Use metadata and notes for assumptions.
5. Generate `einsum_numpy` first if you want easy-to-read code.

For reproducible experiments:

1. Save JSON under a stable path.
2. Run `validate`, `lint`, and `doctor`.
3. Generate backend code with explicit `--engine`.
4. Run `benchmark --format csv` or `--format latex`.
5. Commit JSON, generated code if needed, and benchmark output if it is part of
   the result.

For paper figures:

1. Arrange the graph with Reflow and manual cleanup.
2. Save JSON.
3. Render SVG for quick inspection.
4. Render TikZ when the figure belongs in a LaTeX document.
5. Keep the JSON so the figure can be regenerated.

For reusable model families:

1. Use a built-in template when one matches the family.
2. Use subnetworks for repeated local motifs.
3. Use periodic modes when repetition is structural, not just visual.
4. Keep normal mode for custom networks that do not have a strict repeated
   pattern.

## Current Limits

- Hyperedges are available only in normal mode.
- Hyperedges are lowered to generated copy tensors for export and analysis.
- Tensor values support portable initializers and external data references, but
  not symbolic expressions.
- Python import is intentionally conservative. It is not a general Python
  program-to-network converter.
- External static and live Python imports do not recover editor layout, groups,
  notes, or manual contraction plans.
- `best_available` reconstruction is currently for this package's own
  generated Python profile.
- Browser-based live import works best for self-contained files whose imports
  resolve in the active `.venv`.
- TenPy code generation is not included.
- Linear, grid, and tree periodic code generation work with every bundled
  backend.
- Manual outer-product steps cannot be exported to `tensorkrowch`.
- In grid and tree periodic modes, virtual boundary operands represent
  payload/frontier interfaces for partial contractions. They are not physical
  tensors edited directly.

For common problems and fixes, see [troubleshooting.md](troubleshooting.md).

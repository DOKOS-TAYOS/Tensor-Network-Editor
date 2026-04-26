# CLI

This page covers the `tensor-network-editor` command. The CLI can launch the
visual editor or run headless commands for validation, linting, analysis, code
generation, benchmark comparisons, diffs, templates, and reusable-subnetwork
catalogs.

## Contents

- [Launch the Editor](#launch-the-editor)
- [Debug Logging](#debug-logging)
- [Headless Commands](#headless-commands)
- [Validate](#validate)
- [Lint](#lint)
- [Analyze](#analyze)
- [Benchmark](#benchmark)
- [Export](#export)
- [Render](#render)
- [Canonicalize](#canonicalize)
- [Diff](#diff)
- [Template Commands](#template-commands)
- [Subnetwork Commands](#subnetwork-commands)
- [JSON Output](#json-output)
- [Exit Codes](#exit-codes)

## Launch the Editor

Start the local browser editor:

```bash
tensor-network-editor edit
```

The `edit` subcommand launches the visual editor.
The command keeps the local server running until you confirm with `Done` or
stop the session with `Cancel`.

Useful options:

```bash
tensor-network-editor edit --load my_network.json
tensor-network-editor edit --engine quimb
tensor-network-editor edit --save-code generated_network.py
tensor-network-editor edit --print-code
tensor-network-editor edit --no-browser
```

You can combine them:

```bash
tensor-network-editor edit --load my_network.json --engine quimb --save-code generated_network.py
```

Use `--no-browser` when you want to start the local server but open the printed
URL manually.

## Debug Logging

The CLI stays quiet by default. When you need more context for debugging, turn
package logs on explicitly:

```bash
tensor-network-editor --log-level info edit --no-browser
tensor-network-editor --log-level debug validate my_network.json
```

You can also use the environment variable fallback:

PowerShell:

```powershell
$env:TNE_LOG_LEVEL = "debug"
tensor-network-editor template list
Remove-Item Env:\TNE_LOG_LEVEL
```

Bash:

```bash
TNE_LOG_LEVEL=debug tensor-network-editor template list
```

The CLI flag takes priority over `TNE_LOG_LEVEL`. When logging is enabled, the
package prints a short runtime diagnostic summary with the active Python
executable, current working directory, imported package path, version, and any
editable-install root that may point to a different checkout or worktree.

Python import flags are also global and must appear before the subcommand:

```bash
tensor-network-editor --python-import-mode live validate runtime_network.py
tensor-network-editor --python-reconstruction-level simple validate external_network.py
tensor-network-editor --python-import-mode live --python-object network edit --load runtime_network.py
```

The reconstruction flag controls how much editor metadata the Python importer
tries to rebuild:

- `--python-reconstruction-level auto`: choose the best supported level for the
  detected profile
- `--python-reconstruction-level simple`: portable tensors plus inferred
  connections only
- `--python-reconstruction-level best_available`: currently only supported for
  the package's own `generated` Python exports

If `--python-import-mode live` is used on generated source and the live import
fails because the backend package is missing, the loader falls back to the
static generated-source parser and reports the fallback as a warning.

## Headless Commands

Headless commands work without opening the visual editor:

```bash
tensor-network-editor validate my_network.json
tensor-network-editor lint my_network.json
tensor-network-editor analyze my_network.json
tensor-network-editor benchmark my_network.json
tensor-network-editor export my_network.json --engine quimb --output generated_network.py
tensor-network-editor render my_network.json --format svg --output figure.svg
tensor-network-editor render my_network.json --format png --output figure.png
tensor-network-editor canonicalize my_network.json
tensor-network-editor diff before.json after.json
tensor-network-editor template list
tensor-network-editor template build mps --graph-size 6 --bond-dimension 4 --physical-dimension 2
tensor-network-editor subnetwork list my_network.json
tensor-network-editor subnetwork save my_network.json --tensor-ids tensor_a tensor_b --name local_pair --tags reusable boundary
```

These are useful for scripts, quick checks, and CI.

<p align="center">
  <img
    src="images/cli-validation-workflow.png"
    alt="Editor shortcuts dialog for interactive workflows"
    width="900"
  />
</p>

## Validate

Validate a saved JSON design or a supported Python import profile:

```bash
tensor-network-editor validate my_network.json
```

Validation checks hard consistency rules such as missing endpoints, duplicated
ids, invalid dimensions, and schema problems.

For Python files, the default mode is the conservative static parser. When you
want to execute the source and import a live `quimb` or `tensornetwork`
object instead, use:

```bash
tensor-network-editor --python-import-mode live validate runtime_network.py
tensor-network-editor --python-import-mode live --python-object network validate runtime_network.py
tensor-network-editor --python-reconstruction-level simple validate runtime_network.py
```

JSON output:

```bash
tensor-network-editor validate my_network.json --format json
```

## Lint

Run softer diagnostics:

```bash
tensor-network-editor lint my_network.json
```

Useful options:

```bash
tensor-network-editor lint my_network.json --max-tensor-rank 8
tensor-network-editor lint my_network.json --max-tensor-cardinality 50000
tensor-network-editor lint my_network.json --fail-on warning
```

Linting reports things that may be suspicious even if the spec is valid:

- disconnected components
- suspicious open indices
- very large tensor shapes
- empty groups
- uninformative names
- incomplete manual plans
- guided metadata conflicts such as open `bond` legs, observable annotations on
  connected bonds, and mismatched `symmetry` metadata

For `.py` inputs, the CLI autodetects the same supported Python import
profiles as the library API: generated exports plus conservative static AST
imports for simple `quimb`, `tensornetwork`, and `einsum` / `opt_einsum`
sources. You can switch to subprocess execution for live `quimb` or
`tensornetwork` objects with the same global `--python-import-mode live`
option shown above. The global `--python-reconstruction-level` flag uses
`auto` by default, which resolves to `best_available` for `generated` exports
and to `simple` for external static or live imports.

## Analyze

Analyze structure and contraction metadata:

```bash
tensor-network-editor analyze my_network.json
```

Choose the dtype used for memory estimates:

```bash
tensor-network-editor analyze my_network.json --dtype float32
```

Supported dtypes:

- `float16`
- `float32`
- `float64`
- `complex64`
- `complex128`

Analysis can include manual, automatic full, automatic future, and automatic
past contraction summaries when the design supports those comparisons.

## Benchmark

Benchmark one saved design and compare the stable variants:

```bash
tensor-network-editor benchmark my_network.json
```

The benchmark command compares these rows when they are available:

- `Manual`
- `Auto full`
- `Auto future`
- `Auto past`

Useful options:

```bash
tensor-network-editor benchmark my_network.json --dtype float32
tensor-network-editor benchmark my_network.json --format json
tensor-network-editor benchmark my_network.json --format csv --output benchmark.csv
tensor-network-editor benchmark my_network.json --format latex --output benchmark.tex
```

The table always uses the same columns:

- `Name`
- `FLOP`
- `MAC`
- `Peak`
- `Peak Memory`

When the `planner` extra is not installed in the active `.venv`, the manual row
still works and the automatic rows are marked unavailable. Text, CSV, and LaTeX
show `-` for those metrics, while JSON preserves each row `status` and
`message`.

Rows whose analysis is incomplete can still show the partial metrics available
in their summary. Treat the row `status` as the source of truth when a script
needs to distinguish complete and incomplete comparisons.

For periodic specs, benchmark uses the same active-cell normalization as
`analyze`, so it operates on the active linear/grid/tree representative cell.

## Export

Generate backend Python code without opening the editor:

```bash
tensor-network-editor export my_network.json --engine quimb --output generated_network.py
```

Choose a tensor collection format:

```bash
tensor-network-editor export my_network.json --engine einsum_numpy --collection-format dict
```

Supported engines:

- `tensorkrowch`
- `einsum_torch`
- `einsum_numpy`
- `quimb`
- `tensornetwork`

Supported collection formats:

- `list`
- `matrix`
- `dict`

If `--output` is omitted, generated code is printed to standard output.

Relative external tensor-data paths stored in the JSON are resolved relative to
the input spec path before they are emitted into generated code.

## Render

Render a saved design to a static image without opening the browser editor:

```bash
tensor-network-editor render my_network.json --format svg --output figure.svg
tensor-network-editor render my_network.json --format png --output figure.png
```

If `--output` is omitted, the SVG is printed to standard output:

```bash
tensor-network-editor render my_network.json --format svg
```

SVG rendering is pure Python and uses the saved canvas positions. PNG rendering
uses the same geometry and requires the optional `png` extra; PNG output must
be written to a file. Both formats draw tensors, indices, pairwise edges,
hyperedges, groups, and notes.

## Canonicalize

Canonicalize a spec for cleaner Git history and stable JSON ordering:

```bash
tensor-network-editor canonicalize my_network.json
```

Write the canonicalized JSON to a file:

```bash
tensor-network-editor canonicalize my_network.json --output my_network.canonical.json
```

Rewrite ids deterministically in canonical order:

```bash
tensor-network-editor canonicalize my_network.json --deterministic-ids
```

Canonicalization preserves semantics, keeps existing ids by default, preserves
manual contraction step order, normalizes `metadata.tags`, and sorts the main
graph entities deterministically.

## Diff

Compare two specs by stable entity ids:

```bash
tensor-network-editor diff before.json after.json
```

JSON output is often easiest to consume:

```bash
tensor-network-editor diff before.json after.json --format json
```

Request the richer semantic diff:

```bash
tensor-network-editor diff before.json after.json --semantic
tensor-network-editor diff before.json after.json --semantic --format json
```

The basic diff groups changes by tensor, edge, group, note, and plan. The
semantic diff reports field-level tensor, index, edge, group, note, plan, and
step changes, plus step reordering and opaque `linear_periodic_chain` changes.

When either input is a `.py` file, `diff` uses the same global Python import
mode and reconstruction level for both files. Mixed static/live or
mixed simple/best-available settings per file are not supported in one `diff`
command.

## Template Commands

List built-in templates:

```bash
tensor-network-editor template list
```

Build a template:

```bash
tensor-network-editor template build mps --graph-size 6 --bond-dimension 4 --physical-dimension 2 --output mps.json
```

Print the built template JSON instead of writing a file:

```bash
tensor-network-editor template build peps_2x2 --graph-size 3
```

`template build` accepts the same `--format` option shape as other template
commands, but the built spec itself is serialized as JSON when `--output` is
omitted.

Built-in templates include MPS, MPO, PEPS (`peps_2x2`), MERA, and Binary Tree.
When `--output` is omitted, `template build` prints the serialized spec JSON to
standard output.

## Subnetwork Commands

Reusable subnetworks live in the project catalog resolved from the spec path:
`.tensor-network-editor/subnetworks.json` next to the saved design. Some
commands can also merge an explicit shared catalog path.

List the reusable subnetworks available for one project:

```bash
tensor-network-editor subnetwork list my_network.json
```

Include a shared catalog in the merged view:

```bash
tensor-network-editor subnetwork list my_network.json --shared-catalog-path /path/to/shared-subnetworks.json
```

Save selected tensors from one spec into the project catalog:

```bash
tensor-network-editor subnetwork save my_network.json --tensor-ids tensor_a tensor_b --name local_pair --tags reusable boundary
```

Overwrite an existing project entry with the same name:

```bash
tensor-network-editor subnetwork save my_network.json --tensor-ids tensor_a tensor_b --name local_pair --overwrite
```

Export one reusable subnetwork back to a normal spec file:

```bash
tensor-network-editor subnetwork export my_network.json local_pair --output local_pair.json
```

Useful details:

- `subnetwork save` always writes to the project catalog next to the design
- `subnetwork list` and `subnetwork export` can merge a shared catalog through
  `--shared-catalog-path`
- when project and shared catalogs contain the same subnetwork name, the
  project entry is used
- `subnetwork list --format json` includes merged definitions and catalog
  warnings

## JSON Output

These commands support `--format json`:

- `validate`
- `lint`
- `analyze`
- `benchmark`
- `diff`
- `subnetwork list`
- `template list`

`template build` emits the built spec JSON when you omit `--output`.
`canonicalize` already emits canonical JSON when you omit `--output`.
Use JSON output when another script should consume the result.

## Exit Codes

Common exit codes:

- `0`: command completed successfully
- `1`: validation failed, or lint found warnings with `--fail-on warning`
- `2`: expected package error such as code generation, serialization, IO, or
  invalid option value
- `130`: interrupted with Ctrl+C

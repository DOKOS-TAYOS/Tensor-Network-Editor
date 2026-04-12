# CLI

This page covers the `tensor-network-editor` command. The CLI can launch the
visual editor or run headless commands for validation, linting, analysis, code
generation, diffs, and templates.

## Contents

- [Launch the Editor](#launch-the-editor)
- [Headless Commands](#headless-commands)
- [Validate](#validate)
- [Lint](#lint)
- [Analyze](#analyze)
- [Export](#export)
- [Diff](#diff)
- [Template Commands](#template-commands)
- [JSON Output](#json-output)
- [Exit Codes](#exit-codes)

## Launch the Editor

Start the local browser editor:

```bash
tensor-network-editor
```

The no-subcommand form launches the editor directly. The explicit `edit`
subcommand does the same thing:

```bash
tensor-network-editor edit
```

Useful options:

```bash
tensor-network-editor --load my_network.json
tensor-network-editor --engine quimb
tensor-network-editor --save-code generated_network.py
tensor-network-editor --print-code
tensor-network-editor --no-browser
```

You can combine them:

```bash
tensor-network-editor --load my_network.json --engine quimb --save-code generated_network.py
```

Use `--no-browser` when you want to start the local server but open the printed
URL manually.

## Headless Commands

Headless commands work without opening the visual editor:

```bash
tensor-network-editor validate my_network.json
tensor-network-editor lint my_network.json
tensor-network-editor analyze my_network.json
tensor-network-editor export my_network.json --engine quimb --output generated_network.py
tensor-network-editor diff before.json after.json
tensor-network-editor template list
tensor-network-editor template build mps --graph-size 6 --bond-dimension 4 --physical-dimension 2
```

These are useful for scripts, quick checks, and CI.

## Validate

Validate a saved JSON design or supported generated Python export:

```bash
tensor-network-editor validate my_network.json
```

Validation checks hard consistency rules such as missing endpoints, duplicated
ids, invalid dimensions, and schema problems.

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

## Diff

Compare two specs by stable entity ids:

```bash
tensor-network-editor diff before.json after.json
```

JSON output is often easiest to consume:

```bash
tensor-network-editor diff before.json after.json --format json
```

The diff groups changes by tensor, edge, group, note, and plan.

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
tensor-network-editor template build peps_2x2 --graph-size 3 --format json
```

Built-in templates include MPS, MPO, PEPS (`peps_2x2`), MERA, and Binary Tree.

## JSON Output

These commands support `--format json`:

- `validate`
- `lint`
- `analyze`
- `diff`
- `template list`
- `template build`

Use JSON output when another script should consume the result.

## Exit Codes

Common exit codes:

- `0`: command completed successfully
- `1`: validation failed, or lint found warnings with `--fail-on warning`
- `2`: expected package error such as code generation, serialization, IO, or
  invalid option value
- `130`: interrupted with Ctrl+C

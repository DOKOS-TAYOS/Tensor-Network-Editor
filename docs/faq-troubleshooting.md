# FAQ and Troubleshooting

This page collects common questions, practical fixes, and current limits of the
library.

## The browser did not open automatically

The local server may still be running correctly. Try one of these:

- launch the CLI again without `--no-browser`
- if you used `--no-browser`, open the reported local URL manually
- check whether your environment blocks automatic browser opening

From Python, the same behavior is controlled by `open_browser=True` or
`open_browser=False`.

## The editor opens, but I cannot get code for my preferred backend

The package can generate code for several engines, but running that generated
code may require additional libraries in your environment.

Examples:

```bash
python -m pip install "tensor-network-editor[quimb]"
python -m pip install "tensor-network-editor[tensornetwork]"
python -m pip install "tensor-network-editor[tensorkrowch]"
```

If you want fewer external dependencies, start with:

- `einsum_numpy`
- `einsum_torch`

## I loaded a JSON file and got a schema-version error

Saved designs use a schema wrapper. The current package expects:

```json
{
  "schema_version": 3,
  "network": {
    "...": "..."
  }
}
```

If your file uses an older or different structure, `load_spec(...)` will reject
it instead of guessing.

## Why is my network rejected as invalid?

The package validates several consistency rules. Common problems are:

- empty names
- duplicated ids
- non-positive index dimensions
- missing edge endpoints
- connecting indices with different dimensions
- invalid contraction-plan operand ids

If you build `NetworkSpec` objects by hand, it is worth checking these parts
first.

## Can I use the library without the visual editor?

Yes. You can use the package directly from Python:

- load and save `NetworkSpec` objects
- create network objects yourself
- generate code without opening the browser editor

The editor is convenient, but it is not the only way to use the library.

## Can I keep one saved network and generate code for several backends?

Yes. That is one of the main design goals.

The saved JSON is backend-agnostic, so you can:

1. save the design once
2. reopen it later
3. generate code for another engine

## Does the package support automatic contraction planning?

Partially.

- manual contraction paths are supported directly
- automatic greedy suggestions are available with the optional `planner` extra

Install it with:

```bash
python -m pip install "tensor-network-editor[planner]"
```

## Does the package support hyperedges?

Not in the current release.

The present data model is based on pairwise edges between two index endpoints.

## Can I edit real tensor values in the editor?

Not in the current release.

Today the package focuses on the network structure and code generation workflow.
Generated tensors are initialized to zeros.

## Is TenPy supported?

Not in the current release.

Current code-generation targets are:

- `tensornetwork`
- `quimb`
- `tensorkrowch`
- `einsum_numpy`
- `einsum_torch`

## What should I save in a real project?

A good practical rule is:

- save the JSON design if you care about the abstract tensor network
- save generated Python code if you want a concrete implementation artifact
- keep both if you want reproducibility plus an immediately runnable script


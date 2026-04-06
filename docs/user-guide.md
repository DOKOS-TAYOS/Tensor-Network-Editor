# User Guide

This page explains the normal day-to-day workflow with `tensor-network-editor`.

It is not a full internal reference. The goal is to help you use the package
comfortably and understand what the main concepts are for.

## Core idea

The library separates two things:

- the abstract tensor-network design, stored as a `NetworkSpec`
- the generated Python code for a specific backend

That separation is useful because you can keep one design and generate code for
different targets later.

## A practical mental model

When you work with the editor, it helps to think in these objects:

- a **tensor** is a node on the canvas
- an **index** belongs to a tensor and has a dimension
- an **edge** connects two indices with matching dimension
- a **group** helps you organize several tensors visually
- a **note** stores extra information on the canvas
- a **contraction plan** describes a manual contraction order

## Normal workflow in the editor

A typical user session looks like this:

1. Launch the editor with a default engine.
2. Create tensors and indices.
3. Connect matching indices.
4. Add notes or groups if the network becomes larger.
5. Optionally inspect or edit the contraction plan.
6. Confirm the session.
7. Save the JSON design and, if needed, the generated Python code.

## Choosing a code-generation engine

The package currently supports these engine names:

- `tensornetwork`
- `quimb`
- `tensorkrowch`
- `einsum_numpy`
- `einsum_torch`

### Simple rule of thumb

- Choose `einsum_numpy` if you want lightweight generated code without an extra
  tensor-network framework.
- Choose `einsum_torch` if you want the same style but in a PyTorch-based
  workflow.
- Choose `quimb`, `tensornetwork`, or `tensorkrowch` if you already work in
  those ecosystems and want generated code that fits them more naturally.

## Templates

The editor includes built-in layout templates for common tensor-network shapes:

- `MPS`
- `MPO`
- `PEPS`
- `MERA`
- `Binary Tree`

These templates are useful when you do not want to place every tensor manually.

Template controls let you adjust:

- graph size
- bond dimension
- physical dimension

## Saving and loading

The JSON design is the durable part of your work.

This is usually the best way to think about persistence:

- save the JSON if you want to reopen or version the network later
- save generated Python code if you want to run or adapt a concrete backend
  implementation

Because the JSON is backend-agnostic, you can reopen the same design and
generate code for a different engine later.

## Manual contraction plans

The package supports manual contraction plans through `ContractionPlanSpec` and
`ContractionStepSpec`.

In practical terms:

- a plan is a named sequence of contraction steps
- each step consumes two operand ids and produces a new intermediate id
- operand ids must exist and cannot be reused after they have been consumed

This is useful when you want explicit control over contraction order instead of
accepting an automatic heuristic.

When you generate code from a saved design:

- if there is no saved `contraction_plan`, the package keeps the usual
  one-shot backend-specific export
- if there is a saved `contraction_plan`, the generated code follows those
  manual steps exactly
- complete plans emit a final `result`
- partial plans emit explicit intermediate variables and a
  `remaining_operands` mapping with the operands that are still alive after the
  exported prefix

For the `einsum_numpy` and `einsum_torch` engines, partial plans also emit
`remaining_operand_labels` so the surviving index labels stay easy to inspect.

### Backend-specific note for manual plans

- `tensornetwork` exports manual steps with `contract_between(...)`
- `quimb` exports manual steps with its tensor-network contraction helpers
- `tensorkrowch` exports manual steps with `contract_between(...)`, but manual
  outer-product steps are not supported there and are rejected during code
  generation
- `einsum_numpy` and `einsum_torch` export one `einsum(...)` call per manual
  step instead of a single global einsum

## Planner extra

If you install the `planner` extra, the editor can also offer automatic greedy
contraction suggestions.

```bash
python -m pip install "tensor-network-editor[planner]"
```

The planner extra is optional. The library still works without it.

## What the package validates for you

When a network is serialized or loaded, the package validates important parts of
the design. For example:

- names must not be empty
- tensor sizes must be positive
- index dimensions must be positive
- edge endpoints must exist
- connected indices must have matching dimensions
- ids must be unique where required

This helps catch broken or inconsistent network descriptions early.

## Useful limits to know

The current release has some intentional limits:

- hyperedges are not supported
- real tensor data editing is not supported
- generated tensors are initialized to zeros
- TenPy code generation is not part of the current release

These are worth keeping in mind before you design a workflow around the package.

## When to use the library directly from Python

The CLI is great for interactive sessions, but direct Python use is often
better when:

- you want to integrate the editor into a larger script
- you want to save and load designs programmatically
- you want to generate code in a batch process
- you want to inspect open indices or tensor maps from the abstract model

For that workflow, continue with [python-api.md](python-api.md).


# Data Models

This page explains the main public data models. You do not need to memorize
every field, but knowing the shape of the model helps when you build or inspect
specs from Python.

## Contents

- [Saved File Shape](#saved-file-shape)
- [NetworkSpec](#networkspec)
- [TensorSpec and IndexSpec](#tensorspec-and-indexspec)
- [EdgeSpec](#edgespec)
- [GroupSpec and CanvasNoteSpec](#groupspec-and-canvasnotespec)
- [Contraction Plans](#contraction-plans)
- [Linear Periodic Models](#linear-periodic-models)
- [Result Models and Enums](#result-models-and-enums)
- [Practical Advice](#practical-advice)

## Saved File Shape

Saved JSON files use a schema wrapper:

```json
{
  "schema_version": 4,
  "network": {
    "...": "..."
  }
}
```

The wrapper lets the package reject unsupported file versions clearly instead
of guessing how to load them.

## NetworkSpec

`NetworkSpec` is the root object for one abstract tensor-network design.

It stores:

- `id`
- `name`
- `tensors`
- `groups`
- `edges`
- `notes`
- `contraction_plan`
- `linear_periodic_chain`
- `metadata`

Useful helper methods:

- `tensor_map()`: map tensor ids to tensors
- `index_map()`: map index ids to their owning tensor and index
- `connected_index_ids()`: ids of indices used by edges
- `open_indices()`: tensor/index pairs that are not connected

Example:

```python
from tensor_network_editor import NetworkSpec


spec = NetworkSpec(id="network_empty", name="empty example")
print(spec.open_indices())
```

## TensorSpec and IndexSpec

`TensorSpec` represents one tensor node on the canvas. `IndexSpec` represents
one named index on that tensor.

```python
from tensor_network_editor import CanvasPosition, IndexSpec, TensorSpec


tensor = TensorSpec(
    id="tensor_a",
    name="A",
    position=CanvasPosition(x=120.0, y=160.0),
    indices=[
        IndexSpec(id="tensor_a_i", name="i", dimension=2),
        IndexSpec(id="tensor_a_x", name="x", dimension=3),
    ],
)

print(tensor.shape)
```

`tensor.shape` is derived from the dimensions of its indices. In the example
above, it is `(2, 3)`.

Each tensor also stores canvas `position`, visual `size`, optional metadata,
and an optional linear-periodic role used by the specialized periodic editor
mode.

## EdgeSpec

`EdgeSpec` connects two tensor indices. Each side is an `EdgeEndpointRef`.

```python
from tensor_network_editor import EdgeEndpointRef, EdgeSpec


edge = EdgeSpec(
    id="edge_x",
    name="bond_x",
    left=EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_x"),
    right=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_x"),
)
```

For a valid edge:

- both tensors must exist
- both indices must exist
- each index must belong to the referenced tensor
- connected dimensions must match

## GroupSpec and CanvasNoteSpec

`GroupSpec` is a visual grouping of tensor ids:

```python
from tensor_network_editor import GroupSpec


group = GroupSpec(
    id="group_left_block",
    name="Left block",
    tensor_ids=["tensor_a", "tensor_b"],
)
```

`CanvasNoteSpec` stores free-form text on the canvas:

```python
from tensor_network_editor import CanvasNoteSpec, CanvasPosition


note = CanvasNoteSpec(
    id="note_boundary",
    text="Open indices are physical legs.",
    position=CanvasPosition(x=80.0, y=60.0),
)
```

Groups and notes do not change the mathematical connectivity. They are there
to make larger diagrams easier to understand.

## Contraction Plans

Manual contraction plans are stored with:

- `ContractionPlanSpec`
- `ContractionStepSpec`
- `ContractionOperandLayoutSpec`
- `ContractionViewSnapshotSpec`

Basic example:

```python
from tensor_network_editor import ContractionPlanSpec, ContractionStepSpec


plan = ContractionPlanSpec(
    id="plan_manual",
    name="Manual path",
    steps=[
        ContractionStepSpec(
            id="step_contract_ab",
            left_operand_id="tensor_a",
            right_operand_id="tensor_b",
        )
    ],
)
```

A step consumes two operands. Later steps should refer to operands that still
exist at that point in the plan.

View snapshots preserve contraction-scene layout state:

- `ContractionOperandLayoutSpec` stores one operand id, position, and size
- `ContractionViewSnapshotSpec` stores operand layouts after a given number of
  applied steps

Snapshots are mainly for the editor UI. They are still part of the saved design
and round-trip through JSON.

## Linear Periodic Models

Linear periodic mode uses:

- `LinearPeriodicChainSpec`
- `LinearPeriodicCellSpec`
- `LinearPeriodicCellName`
- `LinearPeriodicTensorRole`

Import these from `tensor_network_editor.models` when you need them directly.

This mode stores an initial cell, periodic cell, and final cell. Each cell can
have tensors, edges, groups, notes, metadata, and its own contraction plan.

Most users can start with normal `NetworkSpec` fields and only use these models
when working with repeated one-dimensional structures.

## Result Models and Enums

Important enums:

- `EngineName`: `tensornetwork`, `quimb`, `tensorkrowch`, `einsum_numpy`,
  `einsum_torch`
- `TensorCollectionFormat`: `list`, `matrix`, `dict`

Important result models:

- `EditorResult`: returned by a confirmed editor session
- `CodegenResult`: returned by `generate_code(...)`
- `ValidationIssue`: returned by `validate_spec(...)`
- `LintReport`: returned by `lint_spec(...)`
- `SpecAnalysisReport`: returned by `analyze_spec(...)`
- `SpecDiffResult`: returned by `diff_specs(...)`

Most result models provide `to_dict()` when they are intended for structured
headless output.

## Practical Advice

- Keep ids stable if you plan to diff or post-process saved files.
- Use meaningful names because generated code is easier to read.
- Let `save_spec(...)` validate before writing JSON.
- Use `open_indices()` when you want to inspect dangling legs.
- Use `metadata` for your own small annotations, not for core connectivity.
- Prefer JSON as the long-term design artifact and generated code as the
  backend-specific artifact.

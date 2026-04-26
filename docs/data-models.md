# Data Models

This page explains the main public data models. You do not need to memorize
every field, but knowing the shape of the model helps when you build or inspect
specs from Python.

## Contents

- [Saved File Shape](#saved-file-shape)
- [NetworkSpec](#networkspec)
- [TensorSpec and IndexSpec](#tensorspec-and-indexspec)
- [EdgeSpec](#edgespec)
- [HyperedgeSpec](#hyperedgespec)
- [GroupSpec and CanvasNoteSpec](#groupspec-and-canvasnotespec)
- [Contraction Plans](#contraction-plans)
- [Linear Periodic Models](#linear-periodic-models)
- [Grid Periodic Models](#grid-periodic-models)
- [Tree Periodic Models](#tree-periodic-models)
- [Result Models and Enums](#result-models-and-enums)
- [Practical Advice](#practical-advice)

## Saved File Shape

Saved JSON files use a schema wrapper:

```json
{
  "schema_version": 2,
  "network": {
    "...": "..."
  }
}
```

The wrapper lets the package reject unsupported file versions clearly instead
of guessing how to load them. New saves use schema version `2`, and schema
version `1` remains loadable for older saved designs.

The in-memory object is a `NetworkSpec`. Use
`tensor_network_editor.io.serialize_spec(...)` and
`tensor_network_editor.io.deserialize_spec(...)` when you need to move between
the object and the schema-wrapped JSON payload. Use `load_spec(...)` and
`save_spec(...)` for files.

## NetworkSpec

`NetworkSpec` is the root object for one abstract tensor-network design.

It stores:

- `id`
- `name`
- `tensors`
- `groups`
- `edges`
- `hyperedges`
- `notes`
- `contraction_plan`
- `linear_periodic_chain`
- `grid_periodic_grid`
- `tree_periodic_tree`
- `metadata`

`metadata` is the place for lightweight annotations. The stable tag convention
is `metadata.tags`, which should be a small list of strings on network, tensor,
index, edge, group, or note entities.

For tensors and indices, the editor also understands a small guided convention
inside the existing `metadata` mapping. This is a documented convention, not a
new saved-file schema:

- tensor keys: `role`, `state`, `provenance`, `symmetry`
- index keys: `leg_kind`, `symmetry`, `observable`

These values stay free-form text. You can still keep any extra keys you want in
the same `metadata` object.

`lint_spec(...)` understands these guided keys more deeply than generic custom
metadata. For example, it can warn about open indices marked as `leg_kind="bond"`
or conflicting `symmetry` annotations across connected legs while still keeping
those checks as soft lint findings rather than hard validation errors.

Useful helper methods:

- `tensor_map()`: map tensor ids to tensors
- `index_map()`: map index ids to their owning tensor and index
- `connected_index_ids()`: ids of indices used by edges or hyperedges
- `open_indices()`: tensor/index pairs that are not connected by either kind
  of connection

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
optional `tensor_data`, and optional periodic-mode roles used by the
specialized editors.

`tensor_data` is the portable place for tensor initializers. It is
not stored in `metadata`, because it directly affects generated backend code.

Example:

```python
from tensor_network_editor import TensorDataMode, TensorDataSpec


tensor.tensor_data = TensorDataSpec(
    mode=TensorDataMode.FILL,
    fill_value=0.5,
)
```

Supported tensor-data modes are:

- `None`: no explicit payload, so generated backend code initializes zeros
- `TensorDataMode.ZEROS`: explicit zero initializer, useful when a dtype is set
- `TensorDataMode.ONES`: initialize the whole tensor with ones
- `TensorDataMode.FILL`: repeat one scalar value across the tensor shape
- `TensorDataMode.IDENTITY`: create a square rank-2 identity/delta tensor
- `TensorDataMode.COPY`: create a generalized diagonal copy tensor where every
  axis has the same dimension
- `TensorDataMode.RANDOM`: create deterministic seeded normal or uniform data
- `TensorDataMode.LITERAL`: store nested Python lists of finite real or complex
  values that exactly match `tensor.shape`
- `TensorDataMode.EXTERNAL`: load tensor values from a `.npy` or `.npz` file in
  generated code, with a runtime shape check

`TensorDataSpec.dtype` can be `float32`, `float64`, `complex64`, or
`complex128`. Complex scalars are JSON objects such as
`{"real": 1.0, "imag": -0.5}`.
For external data, `file_path` is required. `.npz` files also require
`array_key`. The optional `dtype` field asks generated code to convert the
loaded array.

Serialized tensor-data payloads are small JSON objects:

```json
{"mode": "ones"}
{"mode": "fill", "fill_value": {"real": 1.0, "imag": -0.5}, "dtype": "complex128"}
{"mode": "identity", "dtype": "float64"}
{"mode": "copy"}
{"mode": "random", "seed": 123, "distribution": "uniform", "dtype": "float32"}
{"mode": "literal", "values": [[1.0, 0.0], [0.0, 1.0]]}
{"mode": "external", "file_path": "data/tensor_a.npz", "array_key": "a", "dtype": "float64"}
```

When generated code is written from the CLI, relative external `file_path`
values are resolved relative to the input JSON file. In the Python API, pass
`external_data_base_path=...` to `generate_code(...)` when you want the same
anchoring behavior. Without that argument, the path is emitted exactly as
stored.

Generated hyperedge copy tensors are an implementation detail of exports.
Supported generated Python reloads use structured comments around those copy
tensors to reconstruct the original `HyperedgeSpec`.

In the editor sidebar, tensor and index properties expose:

- `Tags` for `metadata.tags`
- `Suggested annotations` for the guided keys above
- `Custom metadata (JSON)` for the remaining free-form keys

The guided fields and the JSON editor both write into `metadata`, but the JSON
editor hides the guided keys so you do not edit the same value in two places.
Leaving a guided field empty removes that key from `metadata`.

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

## HyperedgeSpec

`HyperedgeSpec` connects three or more indices that all share the same
dimension. Its endpoints are also stored as `EdgeEndpointRef` values.

```python
from tensor_network_editor import CanvasPosition, EdgeEndpointRef, HyperedgeSpec


hyperedge = HyperedgeSpec(
    id="hyperedge_shared",
    name="shared_bond",
    endpoints=[
        EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_x"),
        EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_x"),
        EdgeEndpointRef(tensor_id="tensor_c", index_id="tensor_c_x"),
    ],
    hub_offset=CanvasPosition(x=24.0, y=-12.0),
)
```

`hub_offset` stores the editor-side visual displacement of the hyperedge hub
relative to the automatic center computed from the endpoints.
`CanvasPosition(x=0.0, y=0.0)` means "keep the hub centered".

Serialized shape:

```json
{
  "id": "hyperedge_shared",
  "name": "shared_bond",
  "endpoints": [
    {"tensor_id": "tensor_a", "index_id": "tensor_a_x"},
    {"tensor_id": "tensor_b", "index_id": "tensor_b_x"},
    {"tensor_id": "tensor_c", "index_id": "tensor_c_x"}
  ],
  "hub_offset": {"x": 24.0, "y": -12.0},
  "metadata": {}
}
```

Older payloads that do not include `hub_offset` still load and default to
`{"x": 0.0, "y": 0.0}` for backward compatibility.

For a valid hyperedge:

- it must have at least three endpoints
- every referenced tensor and index must exist
- each index must belong to the referenced tensor
- endpoint indices must be unique inside the hyperedge
- all endpoint dimensions must match
- an index cannot be reused by another edge or hyperedge

In the editor, hyperedges are first-class saved objects in normal mode. The
visible hub is still not a real tensor node; only the relative `hub_offset` is
saved alongside the endpoints. Generated backend code lowers hyperedges to
autogenerated copy tensors plus binary edges, because the target backends still
consume pairwise connectivity.

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

When you re-import supported generated Python, manual contraction steps can be
recovered, but `view_snapshots` are reset because generated code does not carry
editor layout state.

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

## Grid Periodic Models

Grid periodic mode uses:

- `GridPeriodicGridSpec`
- `LinearPeriodicCellSpec`
- `GridPeriodicCellName`
- `GridPeriodicTensorRole`

Import these from `tensor_network_editor.models` when you need them directly.

This mode stores nine representative cells around a center cell:

- `top_left`
- `top`
- `top_right`
- `left`
- `center`
- `right`
- `bottom_left`
- `bottom`
- `bottom_right`

Each cell can store tensors, edges, groups, notes, metadata, and its own
contraction plan. The typed boundary roles (`up`, `right`, `down`, `left`)
describe how open bonds continue between neighboring cells.

Grid periodic payloads are mainly for repeated two-dimensional structures.
Manual plans inside a grid cell can refer to virtual boundary operands as if
they were clickable neighbors:

- `__grid_up__`
- `__grid_right__`
- `__grid_down__`
- `__grid_left__`

These operands represent already-built payloads or surviving frontiers, not
physical tensors stored in the cell. Generated code folds the grid in row-major
order: it starts at the upper-left cell, moves left-to-right through each row,
then carries the current partial result into the next row. If the plan leaves
more than one operand alive, the export keeps those values in
`remaining_operands` instead of forcing a final scalar or tensor.
Hyperedges are also intentionally not stored inside these cells in v1.

## Tree Periodic Models

Tree periodic mode uses:

- `TreePeriodicTreeSpec`
- `LinearPeriodicCellSpec`
- `TreePeriodicCellName`
- `TreePeriodicTensorRole`

Import these from `tensor_network_editor.models` when you need them directly.

This mode stores three representative cells:

- `root_cell`
- `branch_cell`
- `leaf_cell`

It also stores a `branching_factor` and the active representative cell. Parent
and child boundary tensors describe how the local graph continues upward or
downward in the repeated tree.

Tree periodic payloads are for hierarchical repeated structures. Each tree cell
can store a manual contraction plan, and those plans can refer to virtual tree
boundaries:

- `__tree_parent__`
- `__tree_child_<index>__`

These operands represent the parent payload or one child payload at the current
cell boundary. Generated code contracts from the leaves toward the root, level
by level. That bottom-up direction keeps the live frontier bounded and lets a
manual plan preserve a partial tree network in `remaining_operands` whenever
the user intentionally leaves several operands alive.
Hyperedges are also intentionally not stored inside these cells in v1.

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
- `SemanticFieldChange`, `SemanticDiffEntry`, `SemanticSpecDiffResult`:
  returned by `semantic_diff_specs(...)`

Most result models provide `to_dict()` when they are intended for structured
headless output.

## Practical Advice

- Keep ids stable if you plan to diff or post-process saved files.
- Use meaningful names because generated code is easier to read.
- Let `save_spec(...)` validate before writing JSON.
- Use `open_indices()` when you want to inspect dangling legs.
- Use `metadata` for your own small annotations, not for core connectivity.
- Use `tensor_data` for deterministic tensor values that should affect
  generated backend code.
- Prefer `metadata.tags` for quick labels that you want to reuse in filters.
- Prefer guided tensor keys like `role`, `state`, `provenance`, and `symmetry`
  when they fit what you want to describe.
- Prefer guided index keys like `leg_kind`, `symmetry`, and `observable` for
  leg semantics.
- Expect `lint_spec(...)` to treat those guided keys specially and surface
  higher-signal modeling warnings when they conflict with the graph structure.
- Keep free-form metadata reasonably small so the editor stays responsive.
- Prefer JSON as the long-term design artifact and generated code as the
  backend-specific artifact.

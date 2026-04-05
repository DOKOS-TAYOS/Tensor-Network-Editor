# Python API

This page describes the public Python API you are most likely to use.

The emphasis is practical: how to call the package from your own code and what
kind of objects you get back.

## Public entry points

The package exposes these main functions at the top level:

- `launch_tensor_network_editor(...) -> EditorResult | None`
- `generate_code(spec, engine=..., path=...) -> CodegenResult`
- `save_spec(spec, path) -> None`
- `load_spec(path) -> NetworkSpec`

Main imports:

```python
from tensor_network_editor import (
    CanvasNoteSpec,
    CanvasPosition,
    EdgeEndpointRef,
    EdgeSpec,
    EngineName,
    GroupSpec,
    IndexSpec,
    NetworkSpec,
    TensorSize,
    TensorSpec,
    generate_code,
    launch_tensor_network_editor,
    load_spec,
    save_spec,
)
```

## `launch_tensor_network_editor`

Use this when you want to open the local browser editor from Python.

```python
from tensor_network_editor import EngineName, launch_tensor_network_editor

result = launch_tensor_network_editor(
    default_engine=EngineName.EINSUM_NUMPY,
    open_browser=True,
)

if result is None:
    print("Editor cancelled.")
else:
    print(result.engine.value)
    print(result.spec.name)
```

### Main parameters

- `initial_spec`: preload an existing `NetworkSpec`
- `default_engine`: initial target engine shown in the editor
- `open_browser`: whether the browser should open automatically
- `host`: local host address, default `127.0.0.1`
- `port`: local port, default `0` so the OS chooses one
- `print_code`: print generated code after confirmation
- `code_path`: write generated code to a file after confirmation

### Return value

- `None` if the user cancels
- `EditorResult` if the user confirms

`EditorResult` contains:

- `spec`: the final `NetworkSpec`
- `engine`: the selected `EngineName`
- `codegen`: a `CodegenResult` or `None`
- `confirmed`: whether the session was confirmed

## `generate_code`

Use this when you already have a `NetworkSpec` and want code for a specific
backend.

```python
from tensor_network_editor import EngineName, generate_code, load_spec

spec = load_spec("my_network.json")
result = generate_code(spec, engine=EngineName.QUIMB)

print(result.engine.value)
print(result.code)
print(result.warnings)
```

### Useful behavior

- if `print_code=True`, the generated code is printed
- if `path="..."` is provided, the code is also written to that file

`CodegenResult` contains:

- `engine`: selected backend
- `code`: generated Python source code
- `warnings`: a list of warnings, if any
- `artifacts`: extra metadata returned by the generator

## `save_spec` and `load_spec`

Use these functions when you want persistence for the abstract network design.

```python
from tensor_network_editor import load_spec, save_spec

spec = load_spec("my_network.json")
save_spec(spec, "copy_of_my_network.json")
```

Important detail:

- `save_spec` validates the specification before writing it
- `load_spec` checks the schema wrapper and validates the loaded design

## Main data models

You do not need to memorize every field, but it helps to know the main types.

### `NetworkSpec`

This is the root object. It contains:

- `tensors`
- `groups`
- `edges`
- `notes`
- `contraction_plan`
- `metadata`

It also provides a few helper methods:

- `tensor_map()`
- `index_map()`
- `connected_index_ids()`
- `open_indices()`

These are useful when you want to inspect the structure without rebuilding the
relationships yourself.

### `TensorSpec` and `IndexSpec`

`TensorSpec` represents a tensor node and `IndexSpec` represents one of its
indices.

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

Useful idea:

- the tensor shape is derived from the dimensions of its indices

### `EdgeSpec`

`EdgeSpec` connects two tensor indices through `EdgeEndpointRef`.

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

- both endpoints must exist
- the referenced index must belong to the referenced tensor
- the two connected dimensions must match

### `GroupSpec` and `CanvasNoteSpec`

These objects are mostly about organization and readability.

- `GroupSpec` lets you collect several tensor ids
- `CanvasNoteSpec` lets you place textual notes on the canvas

### `ContractionPlanSpec`

Use this when you want to represent a manual contraction path directly in the
saved design.

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

## Small complete example

This example builds a small abstract network, saves it, and generates NumPy
einsum code.

```python
from tensor_network_editor import (
    CanvasPosition,
    EdgeEndpointRef,
    EdgeSpec,
    EngineName,
    IndexSpec,
    NetworkSpec,
    TensorSpec,
    generate_code,
    save_spec,
)

spec = NetworkSpec(
    id="network_demo",
    name="demo",
    tensors=[
        TensorSpec(
            id="tensor_a",
            name="A",
            position=CanvasPosition(x=120.0, y=160.0),
            indices=[
                IndexSpec(id="tensor_a_i", name="i", dimension=2),
                IndexSpec(id="tensor_a_x", name="x", dimension=3),
            ],
        ),
        TensorSpec(
            id="tensor_b",
            name="B",
            position=CanvasPosition(x=360.0, y=160.0),
            indices=[
                IndexSpec(id="tensor_b_x", name="x", dimension=3),
                IndexSpec(id="tensor_b_j", name="j", dimension=4),
            ],
        ),
    ],
    edges=[
        EdgeSpec(
            id="edge_x",
            name="bond_x",
            left=EdgeEndpointRef(tensor_id="tensor_a", index_id="tensor_a_x"),
            right=EdgeEndpointRef(tensor_id="tensor_b", index_id="tensor_b_x"),
        )
    ],
)

save_spec(spec, "demo_network.json")

result = generate_code(
    spec,
    engine=EngineName.EINSUM_NUMPY,
    path="generated_network.py",
)

print(result.code)
```

## Practical advice

- Store the JSON design if the network is the real artifact you care about.
- Store generated Python code if you want a concrete backend implementation to
  run or adapt manually.
- Keep ids stable if you plan to version or post-process saved network files.
- Use `open_indices()` when you want to inspect dangling legs programmatically.


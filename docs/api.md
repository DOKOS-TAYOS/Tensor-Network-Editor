# Python API

This page describes the public Python API you are most likely to use. For the
data model fields themselves, see [data-models.md](data-models.md).

## Contents

- [Main Imports](#main-imports)
- [Launch the Editor](#launch-the-editor)
- [Generate Code](#generate-code)
- [Save and Load Designs](#save-and-load-designs)
- [Validate, Lint, Analyze, Canonicalize, and Diff](#validate-lint-analyze-canonicalize-and-diff)
- [Templates](#templates)
- [Extension Hooks](#extension-hooks)
- [Errors](#errors)
- [Small Complete Example](#small-complete-example)

## Main Imports

The package exposes the main functions and models at the top level:

```python
from tensor_network_editor import (
    CodeGenerationError,
    EngineName,
    NetworkSpec,
    SemanticDiffEntry,
    SemanticFieldChange,
    SemanticSpecDiffResult,
    TensorCollectionFormat,
    analyze_contraction,
    analyze_spec,
    build_template_spec,
    canonicalize_spec,
    diff_specs,
    generate_code,
    launch_tensor_network_editor,
    lint_spec,
    list_template_names,
    load_spec,
    load_spec_from_python_code,
    save_spec,
    semantic_diff_specs,
    validate_spec,
)
```

Most workflows only need a few of these imports.

The documented imports above are the stable public surface. The package also
contains implementation modules under `tensor_network_editor.internal`, plus a
few public compatibility wrappers such as `tensor_network_editor.serialization`
and `tensor_network_editor.diffing`. Those public modules are safe to import.
The `internal` tree is not a stable API and may be reorganized between
releases.

## Launch the Editor

Use `launch_tensor_network_editor(...)` when you want a local browser editing
session from Python.

```python
from tensor_network_editor import EngineName, launch_tensor_network_editor


def main() -> None:
    result = launch_tensor_network_editor(
        default_engine=EngineName.EINSUM_NUMPY,
        open_browser=True,
    )

    if result is None:
        print("Editor cancelled.")
        return

    print(result.engine.value)
    print(result.spec.name)
    if result.codegen is not None:
        print(result.codegen.code)
```

Main parameters:

- `initial_spec`: preload an existing `NetworkSpec`
- `default_engine`: initial target backend shown in the editor
- `default_collection_format`: initial tensor collection layout
- `open_browser`: open the browser automatically
- `host`: local host address, default `127.0.0.1`
- `port`: local port, default `0` so the OS chooses one
- `print_code`: print generated code after confirmation
- `code_path`: write generated code after confirmation
- `template_catalog_path`: optional per-project static template catalog path
- `subnetwork_catalog_path`: optional per-project reusable-subnetwork catalog
  path
- `shared_subnetwork_catalog_path`: optional shared reusable-subnetwork
  catalog path merged with the project catalog at runtime

Return value:

- `None` when the user cancels
- `EditorResult` when the user confirms

`EditorResult` contains `spec`, `engine`, `codegen`, and `confirmed`.

Practical note:

- if project and shared reusable-subnetwork catalogs define the same entry
  name, the project entry shadows the shared one

## Generate Code

Use `generate_code(...)` when you already have a `NetworkSpec`.

```python
from tensor_network_editor import (
    EngineName,
    TensorCollectionFormat,
    generate_code,
    load_spec,
)


spec = load_spec("my_network.json")
result = generate_code(
    spec,
    engine=EngineName.QUIMB,
    collection_format=TensorCollectionFormat.DICT,
    path="generated_network.py",
)

print(result.engine.value)
print(result.code)
print(result.warnings)
```

Useful behavior:

- `print_code=True` prints the generated source
- `path="..."` writes the generated source to a file
- `collection_format` can be `LIST`, `MATRIX`, or `DICT`
- a backend-specific export problem raises `CodeGenerationError`

If a saved `contraction_plan` exists, generated code follows that manual plan.
Complete plans emit a final `result`. Partial plans emit intermediate values
and a `remaining_operands` mapping.

## Save and Load Designs

Use `save_spec(...)` and `load_spec(...)` for the abstract JSON design.

```python
from tensor_network_editor import load_spec, save_spec


spec = load_spec("my_network.json")
save_spec(spec, "copy_of_my_network.json")
```

Important details:

- `save_spec(...)` validates before writing
- `load_spec(...)` accepts saved JSON designs
- `load_spec(...)` also accepts supported generated `.py` exports from the
  standard network workflow
- `load_spec_from_python_code(...)` works when generated source is already in
  memory for the same supported standard exports

Round-trip from generated source:

```python
from tensor_network_editor import (
    EngineName,
    generate_code,
    load_spec_from_python_code,
)


result = generate_code(spec, engine=EngineName.EINSUM_NUMPY)
round_tripped_spec = load_spec_from_python_code(result.code)
print(round_tripped_spec.name)
```

This parser is intentionally limited to standard source layouts emitted by this
package. For supported standard exports, manual contraction steps now round-trip
back into `ContractionPlanSpec.steps`. Editor-only `view_snapshots` are still
reset to an empty list because generated Python does not carry scene layout.
Linear periodic generated Python remains export-only for now, and this is still
not a general Python-to-network importer.

## Validate, Lint, Analyze, Canonicalize, and Diff

These helpers are useful in scripts and automated checks.

```python
from tensor_network_editor import (
    analyze_contraction,
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
lint_report = lint_spec(spec, max_tensor_rank=8, max_tensor_cardinality=50_000)
analysis = analyze_spec(spec, memory_dtype="float32")
contraction = analyze_contraction(spec, memory_dtype="float32")
canonical_spec = canonicalize_spec(spec, deterministic_ids=False)
diff = diff_specs(spec, spec)
semantic_diff = semantic_diff_specs(spec, canonical_spec)

print(validation_issues)
print(lint_report.to_dict())
print(analysis.to_dict())
print(contraction.to_dict())
print(diff.to_dict())
print(semantic_diff.to_dict())
```

Use:

- `validate_spec(...)` for hard consistency rules
- `lint_spec(...)` for softer warnings and suggestions
- `analyze_spec(...)` for structural counts and contraction summaries
- `analyze_contraction(...)` when you only need contraction analysis
- `canonicalize_spec(...)` for stable ordering, recursive metadata key ordering,
  normalized `metadata.tags`, and optional deterministic ids
- `diff_specs(...)` to compare entities by stable ids
- `semantic_diff_specs(...)` to report field-level tensor/index/edge/plan/step
  changes after the same normalization used by canonicalization

Supported memory dtypes for analysis are `float16`, `float32`, `float64`,
`complex64`, and `complex128`.

## Templates

List built-in templates:

```python
from tensor_network_editor import list_template_names


print(list_template_names())
```

Build a template spec:

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
spec = build_template_spec(
    "mps",
    parameters=parameters,
)
```

The advanced template helpers live under `tensor_network_editor.templates`.
The package root only re-exports `build_template_spec(...)` and
`list_template_names()`.

Templates are useful when you want a valid starting network without placing
every tensor manually.

## Extension Hooks

The package also exposes a few registration hooks when you want to extend the
editor instead of only consuming it.

Inspect registered generator names:

```python
from tensor_network_editor import list_generator_names


print(list_generator_names())
```

Register a fixed project template from an existing `NetworkSpec`:

```python
from tensor_network_editor import NetworkSpec, register_static_template


spec = NetworkSpec(name="Reference cell")
register_static_template(
    "reference_cell",
    "Reference Cell",
    spec,
    overwrite=True,
)
```

Register a parameterized template builder:

```python
from tensor_network_editor import NetworkSpec
from tensor_network_editor.templates import (
    TemplateDefinition,
    TemplateParameters,
    register_template,
)


def build_demo_template(parameters: TemplateParameters) -> NetworkSpec:
    return NetworkSpec(
        name=f"Demo ({parameters.graph_size})",
    )


register_template(
    "demo_template",
    TemplateDefinition(
        name="demo_template",
        display_name="Demo Template",
        graph_size_label="Sites",
        defaults=TemplateParameters(
            graph_size=4,
            bond_dimension=2,
            physical_dimension=2,
        ),
    ),
    build_demo_template,
    overwrite=True,
)
```

Register a custom backend code generator:

```python
from tensor_network_editor import (
    CodegenResult,
    NetworkSpec,
    TensorCollectionFormat,
    register_generator,
)
from tensor_network_editor.codegen.base import CodeGenerator


class MyGenerator(CodeGenerator):
    @property
    def engine(self) -> str:
        return "my_backend"

    def generate(
        self,
        spec: NetworkSpec,
        *,
        collection_format: TensorCollectionFormat = TensorCollectionFormat.LIST,
    ) -> CodegenResult:
        del spec, collection_format
        return CodegenResult(engine="my_backend", code="# custom backend")


register_generator("my_backend", MyGenerator(), overwrite=True)
```

Registration names must use lowercase letters, digits, and underscores. Use
`overwrite=True` only when you intentionally want to replace an existing entry.

## Errors

Common public errors include:

- `CodeGenerationError`: the requested backend cannot represent the export
- serialization and validation errors from loading malformed designs
- normal `ValueError` for unsupported option values

For backend-specific fixes, see [troubleshooting.md](troubleshooting.md).

## Small Complete Example

This example builds a small network by hand and generates NumPy einsum code.

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


def main() -> None:
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
                left=EdgeEndpointRef(
                    tensor_id="tensor_a",
                    index_id="tensor_a_x",
                ),
                right=EdgeEndpointRef(
                    tensor_id="tensor_b",
                    index_id="tensor_b_x",
                ),
            )
        ],
    )

    save_spec(spec, "demo_network.json")
    result = generate_code(spec, engine=EngineName.EINSUM_NUMPY)
    print(result.code)


if __name__ == "__main__":
    main()
```

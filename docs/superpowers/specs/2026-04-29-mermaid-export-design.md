# Mermaid Export Design

## Summary

Add a new static export format, `mermaid`, for tensor-network diagrams.
The goal is to produce a portable text representation that users can paste
directly into GitHub, Markdown documents, or Mermaid-enabled documentation
tools.

This export is documentation-oriented. It should preserve graph structure and
labels, not editor geometry or visual styling.

## Goals

- Add a first-class `mermaid` export alongside `svg`, `png`, `pdf`, `tikz`,
  and `dot`.
- Keep the API, CLI, and browser editor export flows consistent.
- Reuse the existing export label toggles for tensor names, index names, and
  bond names.
- Generate Mermaid that is robust and easy to paste into Markdown.

## Non-Goals

- Preserve canvas positions, exact layout, colors, or node sizes.
- Reproduce the editor appearance inside Mermaid.
- Add a separate `markdown` export format in v1.
- Fail the export because Mermaid cannot express some detail exactly.

## Recommended Approach

Implement a new renderer function:

- `render_spec_mermaid(spec: NetworkSpec, *, options: DotRenderOptions | None = None, output_path: StrPath | None = None) -> str`

`DotRenderOptions` is the best fit for v1 because Mermaid is also a
text-oriented graph export and needs the same label visibility controls as
`dot`.

The renderer should emit a complete Mermaid diagram using:

```text
flowchart LR
```

This direction matches the current left-to-right mental model already used in
`dot`.

## Representation Rules

### Tensors

- Each tensor becomes one Mermaid node.
- If `show_tensor_labels` is true, use the tensor name as the visible label.
- If `show_tensor_labels` is false, keep the node but use a minimal fallback
  label based on the tensor id so the graph remains valid and readable.

### Pairwise edges

- Each standard edge becomes one Mermaid connection between the two tensor
  nodes.
- The edge label should follow the current `dot` behavior:
  - show bond name and index label when both are enabled
  - show only bond name when only bond names are enabled
  - show only index name and dimension when only index names are enabled
  - show no label when both are disabled

### Open indices

- Each open index becomes a terminal Mermaid node connected to its tensor.
- The node label should reuse the same label logic already used by `dot` for
  open indices.

### Hyperedges

- Each hyperedge becomes a synthetic hub node connected to all endpoint
  tensors.
- The hub label should use the hyperedge name when bond labels are enabled.
- Endpoint edge labels should reuse the current `dot` hyperedge endpoint label
  logic when index labels are enabled.

### Groups

- Each group should become a Mermaid `subgraph`.
- The renderer should place the member tensor nodes inside that `subgraph`.
- If Mermaid cannot faithfully reflect complex overlap or crossing semantics,
  the export should still succeed with a simple `subgraph` structure.

### Notes

- Notes should not become positioned visual nodes in v1.
- Export each note as a Mermaid comment line:

```text
%% Note: Check the contraction order
```

This keeps note content available for documentation without forcing awkward
diagram geometry.

## Escaping and Identifiers

- Mermaid node ids must be generated from safe internal identifiers, not from
  raw labels.
- Visible labels must be escaped conservatively so quotes, brackets, newlines,
  and punctuation do not break the diagram.
- Reuse existing conservative escaping ideas from `dot` and `tikz`, but keep
  Mermaid-specific syntax rules separate in small helper functions.

## API and Integration

### Python API

- Export `render_spec_mermaid` from `tensor_network_editor.rendering`.
- Re-export it from `tensor_network_editor.__init__`.

### CLI

- Extend `render --format` with `mermaid`.
- Print to stdout when `--output` is omitted.
- Use `.mmd` as the recommended output extension.
- Label the success message as `Mermaid`.

### Browser editor

- Add `Mermaid` to the export menu and the export format selector.
- Route it through the same `/api/render` flow as `tikz` and `dot`.
- Download it as text with the `.mmd` extension.

### Backend route

- Extend `/api/render` to accept `format == "mermaid"`.
- Return:
  - `format: "mermaid"`
  - `text: <diagram>`
  - `content_type: "text/plain;charset=utf-8"`

## Error Handling

- If the spec is valid, Mermaid export should succeed.
- Unsupported visual details must degrade gracefully instead of raising.
- Rendering should only fail for the same categories already used elsewhere,
  such as invalid payloads or invalid specs.

## Testing

Add focused tests for:

- `render_spec_mermaid` basic output for a normal network
- label toggle behavior for tensor, index, and bond labels
- open indices and hyperedges
- group and note emission
- escaping of special characters
- API route `/api/render` with `format="mermaid"`
- CLI `render --format mermaid`
- editor menu and export selector wiring
- frontend download flow and output filename extension

## Documentation Updates

- Add Mermaid export to `README.md`.
- Add Mermaid export to the editor help text if that text enumerates supported
  export formats.
- Add a short `CHANGELOG.md` entry when implementation lands.

## Rollout Notes

The first version should stay intentionally simple:

- structure first
- labels second
- visual fidelity out of scope

This keeps the renderer predictable, testable, and useful for documentation
without turning Mermaid into a second layout engine.

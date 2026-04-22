# Hyperedge Context Menu Design

Date: 2026-04-22

## Summary

Add a dedicated right-click mini menu for hyperedges in the normal editor.
The menu should open when the user right-clicks either the hyperedge hub or
any spoke, and it should expose the same quick-edit capabilities already
available in the sidebar properties panel:

- `Name`
- `Color`
- `Tags`
- `Custom metadata`
- `Delete`

The implementation should reuse the existing context-menu architecture and the
existing hyperedge property commands rather than creating a parallel edit path.

## Problem

Hyperedges are already first-class editable entities:

- they can be selected from the canvas
- they have dedicated sidebar properties
- they support rename, color, metadata, and deletion

However, the right-click mini-menu system currently only covers tensors,
indices, edges, groups, and selection contexts. Hyperedges therefore require a
sidebar round trip for edits that comparable entities can do directly from the
canvas.

## Goals

- Add a hyperedge-specific context menu in the canvas.
- Open the same menu from both the hyperedge hub and any hyperedge spoke.
- Normalize all hyperedge right-clicks to one canonical selected id so the
  menu, selection, and commands stay consistent.
- Reuse existing `renameHyperedge`, `updateTargetColor`, metadata binding, and
  `deleteHyperedge` flows.
- Keep the behavior visually and structurally aligned with the existing edge
  and group context menus.

## Non-Goals

- No changes to the hyperedge data model or serialization.
- No changes to the sidebar property panel for hyperedges.
- No new hyperedge actions beyond quick editing of existing fields.
- No redesign of the generic context-menu layout system.

## Current Context

The current implementation already provides most of the pieces needed:

- `graphRender.js` handles `cxttap` for `tensor`, `index`, and `edge`, but not
  for `hyperedge-hub` or `hyperedge-spoke`.
- Normal click selection already treats a hyperedge as one logical entity:
  right now a spoke click is normalized to the hub selection id, which is the
  correct canonical behavior to preserve.
- `canvasContextMenuTargets.js` resolves typed context targets and already has
  distinct target kinds such as `selection`, `index-selection`, `edge`, and
  `group`.
- `canvasContextMenuMarkup.js` and `canvasContextMenuBindings.js` already
  separate rendering from behavior, which makes adding one more target kind
  straightforward.
- `actions/propertyCommands.js` already exposes `renameHyperedge(...)` and
  `deleteHyperedge(...)`.
- Hyperedge color and metadata editing already exist in the sidebar property
  bindings, so the context menu can follow the same command/invalidation
  pattern.

## Chosen Design

### 1. Accept Hyperedges in the Right-Click Entry Point

The Cytoscape `cxttap` handler in `graphRender.js` will start accepting:

- `hyperedge-hub`
- `hyperedge-spoke`

This keeps hyperedges inside the same entry point already used by other
graph-backed entities.

### 2. Canonicalize Hyperedge Menu Identity

All hyperedge right-clicks will normalize to the hyperedge hub selection id.

Rules:

- right-click on the hub: use the hub id directly
- right-click on a spoke: resolve `baseHyperedgeId`, then convert it to the
  hub node id

This matches the existing left-click selection behavior and avoids subtle bugs
where the same hyperedge could open under different ids depending on which
visual fragment received the event.

### 3. Add a `hyperedge` Context Target

`canvasContextMenuTargets.js` will gain a new target resolver for
`kind: "hyperedge"`.

That target will contain:

- `id`: canonical hub selection id
- `kind: "hyperedge"`
- `target`: the resolved `HyperedgeSpec`
- `hyperedgeColor`: the current display color derived from metadata

The resolver will use the existing hyperedge lookup path, so it works whether
the incoming id is already the base hyperedge id or a hub/spoke-derived
selection id.

### 4. Render a Dedicated Hyperedge Mini Menu

`canvasContextMenuMarkup.js` will gain a dedicated hyperedge renderer.

The menu will include:

- one inline text input for `Name`
- one inline color input for `Color`
- one inline metadata editor block for `Tags` and `Custom metadata`
- one delete button

This will intentionally mirror the existing edge and group menu style rather
than invent a new visual variant.

### 5. Bind Hyperedge Actions Through Existing Commands

`canvasContextMenuBindings.js` will gain `bindHyperedgeContextTarget(...)`.

Bindings:

- `Name` commits through `renameHyperedge(...)`
- `Color` commits through `updateTargetColor(...)`
- `Tags` and `Custom metadata` bind through `bindInlineMetadataEditor(...)`
- `Delete` commits through `deleteHyperedge(...)`

The metadata binding will keep the current hyperedge convention already used in
the sidebar:

- `annotationScope: "edge"`
- keys under `hyperedge:<id>:...`

That preserves current metadata semantics while still presenting the entity to
the user as a hyperedge.

## User-Facing Behavior

- Right-clicking a hyperedge hub opens the hyperedge mini menu.
- Right-clicking any hyperedge spoke opens the same hyperedge mini menu.
- Opening the menu selects the hyperedge first, just like other entity menus.
- The menu does not create a new editing surface with different rules; it is
  only a quick-access version of the existing hyperedge property edits.

## Error Handling

- If a right-click resolves to a stale or missing hyperedge, the menu resolver
  returns `null` and the context menu does not open.
- Empty names still fail through the existing `renameHyperedge(...)` validation
  and show the current status error instead of introducing a new rule in the
  menu layer.
- Delete behavior keeps the current invalidation path for graph, lookups,
  planner, minimap, and analysis state.

## Testing Plan

Add regression coverage in the frontend architecture/runtime tests for:

- opening the context menu from a hyperedge hub
- opening the same context menu from a hyperedge spoke
- verifying the rendered menu contains the hyperedge name input, color input,
  metadata inputs, and delete button
- verifying rename dispatches `renameHyperedge(...)`
- verifying color updates dispatch `updateTargetColor(...)`
- verifying delete dispatches `deleteHyperedge(...)`
- verifying right-clicking a spoke still preserves canonical hub-based
  selection/menu identity

Add or update asset-level tests for the new menu markup and bindings so the
new hyperedge context-menu entry points are protected against regressions.

## Acceptance Criteria

- Hyperedges expose a right-click mini menu in normal mode.
- Hub and spoke right-clicks open the same hyperedge menu.
- The menu supports `Name`, `Color`, `Tags`, `Custom metadata`, and `Delete`.
- The implementation reuses existing hyperedge commands and metadata behavior.
- Existing non-hyperedge context menus continue to work unchanged.

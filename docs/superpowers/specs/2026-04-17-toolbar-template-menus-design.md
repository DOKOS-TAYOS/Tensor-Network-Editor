# Toolbar Template Menus Design

## Goal

Reorganize the editor chrome so that:

- the top application toolbar behaves like a desktop app menubar for `Load`, `Rename`, and `Export`
- the template area on the canvas toolbar only keeps the template selector, a `...` settings button, the `+` insert button, the existing delete button, and `Reflow`
- template parameters move into a toggleable settings popover opened from the `...` button

## Confirmed UX

### Top toolbar

- `Load` becomes a visible menu button.
- Clicking `Load` opens a dropdown with:
  - `Load tensor network`
  - `Load subnetwork`
- Clicking `Load` again closes the dropdown.
- `Rename` becomes a visible top-toolbar button next to the main app actions.
- `Export` becomes a visible menu button.
- Clicking `Export` opens a dropdown with:
  - `Export as Python`
  - `Export as PNG`
  - `Export as SVG`
- Clicking an export option immediately runs that export.
- Clicking outside an open top-toolbar menu closes it.

### Template toolbar

- The template strip keeps only:
  - the template `<select>`
  - a `...` button
  - the existing `+` insert-template button
  - the existing delete-template button in the same place
  - the existing `Reflow` button
- `+ Subnetwork` is removed from this strip and only exposed through `Load -> Load subnetwork`.
- The current inline `Rename` button is removed from this strip and only exposed through the visible top-toolbar `Rename` button.

### Template settings popover

- Clicking the `...` button opens a compact anchored panel below the button.
- Clicking the same `...` button again closes the panel.
- The panel contains editable controls for:
  - `Graph size`
  - `Bond dimension`
  - `Physical dimension`
- Clicking outside the popover closes it.
- Pressing `Escape` closes it.
- Existing template parameter enable/disable rules remain the same as today.
- Existing validation and persistence logic for template parameter edits remains the same as today.

## Interaction Model

The change should reuse existing editor actions instead of introducing new backend endpoints or changing template catalog data.

- `Load tensor network` still uses the existing `load-input`.
- `Load subnetwork` still uses the existing `subnetwork-load-input`.
- `Rename` still calls the existing `renameSelectedTemplate()` flow and keeps its current enable/disable rules.
- `Export as Python/PNG/SVG` should dispatch through the existing export flow, with the selected format driving the existing implementation.
- The template settings popover should reuse the current parameter inputs so that the existing change handlers keep working.

## Affected Files

### Markup and styling

- `src/tensor_network_editor/app/static/index.html`
  - Replace the current `Load` and `Export` controls with menu-button markup.
  - Move `Rename` to the main top toolbar.
  - Restructure the template strip to remove inline parameter fields and inline rename/subnetwork buttons.
  - Add markup for the template settings toggle and anchored popover.
- `src/tensor_network_editor/app/static/app.css`
  - Add desktop-style toolbar menu styles.
  - Add anchored popover styles for the template settings panel.
  - Adjust the template strip layout so it remains stable on wide and narrow screens.

### DOM wiring and shell bindings

- `src/tensor_network_editor/app/static/js/dom.js`
  - Replace removed DOM references.
  - Add references for toolbar menu buttons, menu panels, menu items, template settings toggle, and template settings popover.
- `src/tensor_network_editor/app/static/js/shell/editorShellBindings.js`
  - Bind menu open/close behavior.
  - Bind menu actions to the existing editor actions.
  - Bind outside-click and `Escape` handling for the new menus/popover.
  - Rebind `Rename` from the template strip to the top toolbar button.

### Runtime/UI state

- `src/tensor_network_editor/app/static/js/utilitiesUi.js`
  - Keep enable/disable logic correct for moved controls.
  - Update button titles for the new top-toolbar `Rename`.
  - Keep `Reflow` and delete-template state logic unchanged.
- `src/tensor_network_editor/app/static/js/interactionsShortcuts.js`
  - Preserve current keyboard shortcuts such as `Ctrl/Cmd+L`.
  - Ensure `Escape` also closes the new menus/popover before falling back to the existing editor escape behavior.
- `src/tensor_network_editor/app/static/js/bootstrap.js`
  - Only needs updates if the shell action surface changes; otherwise keep existing action contracts.

### Existing business logic to reuse

- `src/tensor_network_editor/app/static/js/session/sessionTemplateFlows.js`
  - Reuse `openSubnetworkPicker()` and `renameSelectedTemplate()`.
- `src/tensor_network_editor/app/static/js/session/sessionEditorFlows.js`
  - Reuse the existing export logic.

## Behavioral Constraints

- No backend API changes.
- No template catalog data-model changes.
- No change to the current delete-template location or semantics.
- No change to the current template parameter semantics.
- No change to existing shortcuts unless required for menu dismissal behavior.
- Menus and popovers should feel like application chrome, not like large floating cards.

## Testing Strategy

Add or update focused frontend regressions that cover:

- asset-level markup expectations for the new top-toolbar menu buttons and the simplified template strip
- shell binding expectations for the moved `Rename` button and menu item handlers
- runtime behavior for:
  - `Load -> Load tensor network`
  - `Load -> Load subnetwork`
  - `Export -> Python/PNG/SVG`
  - template settings popover toggle/open/close behavior
  - `Escape` closing the active menu or popover
  - toolbar state updates for the moved `Rename` control

Likely test files:

- `tests/test_app_assets.py`
- `tests/test_frontend_architecture.py`
- `tests/test_frontend_runtime.py`

## Non-Goals

- Building a full `File / Edit / View` menubar system
- Changing the backend export/template APIs
- Redesigning unrelated canvas controls

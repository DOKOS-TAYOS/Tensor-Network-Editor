from __future__ import annotations

import pytest

from tensor_network_editor.app.server import EditorServer
from tests.app_support import request_text, request_with_headers


def test_root_serves_editor_shell_with_versioned_module_entry(
    editor_server: EditorServer,
) -> None:
    html, headers = request_with_headers(f"{editor_server.base_url}/")

    assert "Tensor Network Editor" in html
    assert 'type="module"' in html
    assert "/js/main.js?v=" in html
    assert 'id="collection-format-select"' in html
    assert 'id="sidebar-toggle-button"' in html
    assert 'id="minimap-shell"' in html
    assert '<div class="code-header-layout">' in html
    assert ">Format<" not in html
    assert "<strong>Ctrl+Y</strong><span>Select NumPy einsum</span>" in html
    assert "<strong>S</strong><span>Toggle sidebar</span>" in html
    assert "<strong>Shift+M</strong><span>Toggle minimap</span>" in html
    assert "<strong>Shift+R</strong><span>Reset contraction path</span>" in html
    assert headers["Content-Type"].startswith("text/html")


def test_root_places_editor_title_in_toolbar_and_keeps_canvas_controls_in_requested_order(
    editor_server: EditorServer,
) -> None:
    html = request_text(f"{editor_server.base_url}/")

    assert '<h1 class="toolbar-title">' in html
    assert 'href="https://github.com/DOKOS-TAYOS/Tensor-Network-Editor"' in html
    assert 'class="toolbar-title-link"' in html
    assert '<div class="title-main">' not in html
    assert 'class="title-control-divider"' in html
    assert 'class="title-control-group title-control-group-template"' in html
    assert html.index('class="toolbar-title-link"') < html.index(
        'id="new-design-button"'
    )

    add_index = html.index('id="add-tensor-button"')
    delete_index = html.index('id="delete-button"')
    undo_index = html.index('id="undo-button"')
    redo_index = html.index('id="redo-button"')
    connect_index = html.index('id="connect-button"')
    group_index = html.index('id="create-group-button"')
    note_index = html.index('id="add-note-button"')
    template_index = html.index('id="template-select"')
    insert_template_index = html.index('id="insert-template-button"')

    assert add_index < delete_index < undo_index < redo_index
    assert redo_index < connect_index < group_index < note_index
    assert note_index < template_index < insert_template_index
    assert ">Insert Template<" in html


def test_main_module_is_served_from_static_directory(
    editor_server: EditorServer,
) -> None:
    body, headers = request_with_headers(f"{editor_server.base_url}/js/main.js")

    assert body.strip()
    assert "startEditor" in body
    assert headers["Content-Type"].startswith("application/javascript")


def test_notes_planner_uses_singular_operation_labels(
    editor_server: EditorServer,
) -> None:
    body = request_text(f"{editor_server.base_url}/js/planner.js")

    assert '"FLOPs"' not in body
    assert '"MACs"' not in body
    assert '"FLOP"' in body
    assert '"MAC"' in body


def test_notes_and_planner_feature_modules_are_served(
    editor_server: EditorServer,
) -> None:
    notes_body = request_text(f"{editor_server.base_url}/js/notes.js")
    planner_body = request_text(f"{editor_server.base_url}/js/planner.js")
    registrar_body = request_text(f"{editor_server.base_url}/js/notesPlanner.js")

    assert "registerNotesFeature" in notes_body
    assert "registerPlannerFeature" in planner_body
    assert 'from "./notes.js"' in registrar_body
    assert 'from "./planner.js"' in registrar_body


def test_vendor_asset_is_served_locally(editor_server: EditorServer) -> None:
    body, headers = request_with_headers(
        f"{editor_server.base_url}/vendor/cytoscape.min.js"
    )

    assert "cytoscape" in body
    assert headers["Content-Type"].startswith("application/javascript")


def test_interactions_asset_exposes_updated_keyboard_shortcuts(
    editor_server: EditorServer,
) -> None:
    body = request_text(f"{editor_server.base_url}/js/interactions.js")

    assert "ctx.isTextInput(event.target) || ctx.isTextInput(activeElement)" in body
    assert 'if (hasModifier && lowerKey === "y") {' in body
    assert 'setSelectedEngine("einsum_numpy");' in body
    assert 'if (hasModifier && lowerKey === "n") {' not in body
    assert 'if (lowerKey === "s") {' in body
    assert "toggleSidebarCollapsed();" in body
    assert 'if (event.shiftKey && lowerKey === "m") {' in body
    assert "toggleMinimapVisibility();" in body
    assert 'if (event.shiftKey && lowerKey === "r") {' in body
    assert "trimContractionPlan(0);" in body


def test_css_asset_exposes_explicit_canvas_layer_ordering(
    editor_server: EditorServer,
) -> None:
    body = request_text(f"{editor_server.base_url}/app.css")

    assert ".canvas-panel {" in body
    assert "isolation: isolate;" in body
    assert ".canvas-shell {" in body
    assert "overflow: hidden;" in body
    assert ".toolbar {" in body
    assert "z-index: 20;" in body
    assert ".sidebar {" in body
    assert "z-index: 10;" in body
    assert "#canvas {" in body
    assert "z-index: 0;" in body
    assert "#group-layer {" in body
    assert "z-index: 10;" in body
    assert "#notes-layer {" in body
    assert "z-index: 30;" in body


def test_css_asset_aligns_template_controls_apart_from_main_canvas_actions(
    editor_server: EditorServer,
) -> None:
    body = request_text(f"{editor_server.base_url}/app.css")

    assert "--canvas-control-height:" in body
    assert ".toolbar-title-link {" in body
    assert ".title-control-divider {" in body
    assert ".title-control-group-template {" in body
    assert "margin-left: auto;" in body
    assert ".title-button-row {" in body
    assert "align-items: flex-end;" in body
    assert ".title-button-row button {" in body
    assert "height: var(--canvas-control-height);" in body
    assert ".template-parameter-panel select," in body
    assert "height: var(--canvas-control-height);" in body
    assert ".template-select-field select {" in body
    assert "min-width: 9rem;" in body
    assert "min-width: 10.5rem;" not in body


def test_properties_asset_exposes_total_element_summaries_and_icon_delete_controls(
    editor_server: EditorServer,
) -> None:
    body = request_text(f"{editor_server.base_url}/js/properties.js")

    assert "Total elements" in body
    assert "Delete Selected" not in body
    assert "Delete Connection" not in body
    assert "Delete Note" not in body
    assert 'aria-label="Delete selection"' in body
    assert 'aria-label="Delete connection"' in body
    assert 'aria-label="Delete note"' in body
    assert "function getSelectionTotalElementCount(" in body
    assert "function getTensorTotalElementCount(" in body


def test_note_assets_move_note_editing_into_canvas(
    editor_server: EditorServer,
) -> None:
    notes_body = request_text(f"{editor_server.base_url}/js/notes.js")
    properties_body = request_text(f"{editor_server.base_url}/js/properties.js")
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert 'textarea.addEventListener("keydown", (event) => {' in notes_body
    assert "event.stopPropagation();" in notes_body
    assert 'className = "canvas-note-color-button"' in notes_body
    assert 'colorInput.type = "color";' in notes_body
    assert "ctx.bindDebouncedAutosave(" in notes_body
    assert '<label for="note-text-input">Note text</label>' in properties_body
    assert 'id="note-color-input"' in properties_body
    assert "Edit this note directly on the canvas." not in properties_body
    assert ".canvas-note-color-button {" in css_body


def test_interaction_assets_support_latest_contraction_scene_editing(
    editor_server: EditorServer,
) -> None:
    interactions_body = request_text(f"{editor_server.base_url}/js/interactions.js")
    planner_body = request_text(f"{editor_server.base_url}/js/planner.js")
    graph_body = request_text(f"{editor_server.base_url}/js/graphRender.js")
    utilities_body = request_text(f"{editor_server.base_url}/js/utilities.js")

    assert (
        "Connect mode is only available in the base tensor view."
        not in interactions_body
    )
    assert "Selection cleared." in planner_body
    assert (
        "Choose a different tensor or intermediate; both selections refer to the same contracted operand."
        not in planner_body
    )
    assert "const indexNodesInteractive = !readOnlyScene;" in graph_body
    assert "selectable: !readOnlyScene," in graph_body
    assert "ctx.ensureContractionViewSnapshots();" in utilities_body


def test_performance_sensitive_assets_use_lightweight_analysis_paths(
    editor_server: EditorServer,
) -> None:
    planner_body = request_text(f"{editor_server.base_url}/js/planner.js")
    interactions_body = request_text(f"{editor_server.base_url}/js/interactions.js")
    utilities_body = request_text(f"{editor_server.base_url}/js/utilities.js")
    minimap_body = request_text(f"{editor_server.base_url}/js/exportMinimap.js")
    overlays_body = request_text(
        f"{editor_server.base_url}/js/overlaysLayoutTemplates.js"
    )

    assert "function serializeCurrentSpec(options = {})" in utilities_body
    assert "persistViewSnapshots = false" in utilities_body
    assert "ctx.serializeCurrentSpec({ persistViewSnapshots: false })" in planner_body
    assert (
        "ctx.serializeCurrentSpec({ persistViewSnapshots: false })" in interactions_body
    )
    assert (
        "ctx.serializeCurrentSpec({ persistViewSnapshots: true })" in interactions_body
    )
    assert "ANALYSIS_REFRESH_DELAY_MS = 200" in planner_body
    assert "requestAnimationFrame" in minimap_body
    assert "requestAnimationFrame" in overlays_body


def test_planner_assets_expose_total_elements_and_step_spacing(
    editor_server: EditorServer,
) -> None:
    planner_body = request_text(f"{editor_server.base_url}/js/planner.js")
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert "Total elements" in planner_body
    assert "function getShapeElementCount(" in planner_body
    assert "planner-manual-step-list" in planner_body
    assert ".planner-manual-step-list {" in css_body
    assert "border-top:" in css_body


def test_graph_assets_expose_fixed_tensor_edge_port_layers_and_selection_border(
    editor_server: EditorServer,
) -> None:
    graph_body = request_text(f"{editor_server.base_url}/js/graphRender.js")
    utilities_body = request_text(f"{editor_server.base_url}/js/utilities.js")

    assert "const TENSOR_BASE_Z_INDEX = 10;" in graph_body
    assert "const EDGE_Z_INDEX = 100;" in graph_body
    assert "const PORT_BASE_Z_INDEX = 200;" in graph_body
    assert "selector: \"node[kind = 'tensor']:selected\"" in graph_body
    assert '"border-width": 4' in graph_body
    assert '"border-color": "#8bc2ff"' in graph_body
    assert '"overlay-opacity": 0' in graph_body
    assert (
        'tensorElement.data("zIndex", TENSOR_BASE_Z_INDEX + tensorRank);'
        in utilities_body
    )
    assert (
        'indexElement.data("zIndex", PORT_BASE_Z_INDEX + tensorRank * 10 + indexPosition);'
        in utilities_body
    )
    assert 'edgeElement.data("zIndex", EDGE_Z_INDEX);' in utilities_body


@pytest.mark.parametrize("path", ["/", "/js/main.js", "/vendor/cytoscape.min.js"])
def test_static_assets_disable_browser_cache(
    editor_server: EditorServer,
    path: str,
) -> None:
    _, headers = request_with_headers(f"{editor_server.base_url}{path}")

    assert "no-store" in headers["Cache-Control"]
    assert headers["Pragma"] == "no-cache"
    assert headers["Expires"] == "0"

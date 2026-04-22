from __future__ import annotations

import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from tensor_network_editor.app.server import EditorServer
from tests.app_support import request_text, request_with_headers


def request_runtime_bundle(editor_server: EditorServer, *relative_paths: str) -> str:
    return "\n".join(
        request_text(f"{editor_server.base_url}/{relative_path}")
        for relative_path in relative_paths
    )


def request_utilities_runtime_bundle(editor_server: EditorServer) -> str:
    return request_runtime_bundle(
        editor_server,
        "js/utils/utilities.js",
        "js/utils/utilitiesBase.js",
        "js/utils/utilitiesGeometry.js",
        "js/utils/utilitiesGridPeriodic.js",
        "js/utils/utilitiesGridPeriodicState.js",
        "js/utils/utilitiesGridPeriodicBoundaries.js",
        "js/utils/utilitiesGridPeriodicFlow.js",
        "js/utils/utilitiesLayout.js",
        "js/utils/utilitiesLayoutAlgorithms.js",
        "js/utils/utilitiesLayoutAlgorithmsGraph.js",
        "js/utils/utilitiesLayoutAlgorithmsPositions.js",
        "js/utils/utilitiesLinearPeriodic.js",
        "js/utils/utilitiesLinearPeriodicState.js",
        "js/utils/utilitiesLinearPeriodicBoundaries.js",
        "js/utils/utilitiesLinearPeriodicFlow.js",
        "js/utils/utilitiesSpec.js",
        "js/utils/utilitiesTreePeriodic.js",
        "js/utils/utilitiesTreePeriodicState.js",
        "js/utils/utilitiesTreePeriodicBoundaries.js",
        "js/utils/utilitiesTreePeriodicFlow.js",
        "js/utils/utilitiesUi.js",
        "js/utils/utilitiesUiDom.js",
        "js/utils/utilitiesUiPanels.js",
        "js/utils/utilitiesUiGeneratedCode.js",
        "js/utils/utilitiesUiToolbar.js",
        "js/utils/utilitiesUiToolbarWarnings.js",
        "js/utils/utilitiesUiToolbarDerivedState.js",
        "js/utils/utilitiesUiToolbarModeControls.js",
        "js/utils/utilitiesUiToolbarActionState.js",
        "js/utils/utilitiesUiStatus.js",
    )


def request_interactions_runtime_bundle(editor_server: EditorServer) -> str:
    return request_runtime_bundle(
        editor_server,
        "js/actions/sessionCommands.js",
        "js/interactions/interactions.js",
        "js/interactions/interactionsCanvas.js",
        "js/interactions/interactionsEditor.js",
        "js/interactions/interactionsSession.js",
        "js/interactions/interactionsShortcuts.js",
        "js/session/sessionEditorFlows.js",
        "js/session/sessionTemplateFlows.js",
        "js/session/sessionUiAdapters.js",
        "js/services/editorSessionService.js",
        "js/services/subnetworkService.js",
        "js/services/templateCatalogService.js",
        "js/state/editorSelectors.js",
        "js/state/editorStore.js",
    )


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
    assert "<strong>Shift+S</strong><span>Switch to Single mode</span>" in html
    assert "<strong>Shift+M</strong><span>Toggle minimap</span>" in html
    assert "<strong>Shift+R</strong><span>Reset contraction path</span>" in html
    assert "<strong>F</strong><span>Toggle For unidimensional mode</span>" in html
    assert "<strong>D</strong><span>Switch to For bidimensional mode</span>" in html
    assert "<strong>E</strong><span>Toggle For Tree mode</span>" in html
    assert "<strong>B</strong><span>Switch to Benchmark mode</span>" in html
    assert "<strong>Ctrl/Cmd+F</strong><span>Open Search</span>" in html
    assert "<strong>Ctrl/Cmd+Shift+F</strong><span>Open Filters</span>" in html
    assert "<strong>L</strong><span>Load templates from JSON</span>" in html
    assert "<strong>Shift+E</strong><span>Export the selected template</span>" in html
    assert "Ctrl/Cmd+N" not in html
    assert headers["Content-Type"].startswith("text/html")


def test_root_places_editor_title_in_toolbar_and_keeps_canvas_controls_in_requested_order(
    editor_server: EditorServer,
) -> None:
    html = request_text(f"{editor_server.base_url}/")

    assert '<h1 class="toolbar-title">' in html
    assert 'href="https://github.com/DOKOS-TAYOS/Tensor-Network-Editor"' in html
    assert 'class="toolbar-title-link"' in html
    assert 'class="toolbar-scroll-shell"' in html
    assert 'class="toolbar-menubar"' in html
    assert 'id="file-menu-button"' in html
    assert 'id="modes-menu-button"' in html
    assert 'id="templates-menu-button"' in html
    assert 'id="help-menu-button"' in html
    assert 'id="file-menu-panel"' in html
    assert 'id="modes-menu-panel"' in html
    assert 'id="templates-menu-panel"' in html
    assert 'id="help-menu-panel"' in html
    assert 'class="toolbar-menu-item-content"' in html
    assert 'class="toolbar-menu-item-header"' in html
    assert 'class="toolbar-menu-item-description"' in html
    assert "Write the current design to disk." in html
    assert 'id="load-button"' not in html
    assert 'id="export-button"' not in html
    assert 'id="help-button"' not in html
    assert 'class="title-control-divider"' in html
    assert 'class="title-control-group title-control-group-template"' in html
    assert html.index('class="toolbar-title-link"') < html.index(
        'id="file-menu-button"'
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
    assert ">+<" in html


def test_root_groups_export_actions_and_code_generation_controls_as_requested(
    editor_server: EditorServer,
) -> None:
    html = request_text(f"{editor_server.base_url}/")

    assert 'id="file-menu-panel"' in html
    assert 'id="load-design-menu-item"' in html
    assert 'id="export-python-menu-item"' in html
    assert 'id="export-png-menu-item"' in html
    assert 'id="export-svg-menu-item"' in html
    assert html.index('id="file-menu-button"') < html.index('id="file-menu-panel"')

    code_pane_index = html.index('id="sidebar-pane-code"')
    engine_index = html.index('id="engine-select"')
    collection_index = html.index('id="collection-format-select"')
    generate_index = html.index('id="generate-button"')
    copy_index = html.index('id="copy-code-button"')
    expand_index = html.index('id="expand-generated-code-button"')
    warning_index = html.index('id="code-generation-warning"')

    assert (
        code_pane_index
        < engine_index
        < collection_index
        < generate_index
        < copy_index
        < expand_index
        < warning_index
    )
    assert 'id="copy-code-button"' in html
    assert 'id="expand-generated-code-button"' in html
    assert 'id="generated-code-view"' in html
    assert 'id="generated-code"' in html
    assert "/vendor/prism-core.min.js?v=" not in html
    assert "/vendor/prism-python.min.js?v=" not in html
    assert "window.__TNE_ASSET_VERSION__" in html


def test_root_renders_done_and_cancel_as_icon_toolbar_actions(
    editor_server: EditorServer,
) -> None:
    html = request_text(f"{editor_server.base_url}/")

    assert 'id="done-button"' in html
    assert 'id="cancel-button"' in html
    assert 'class="icon-button toolbar-icon-button danger button-close-static"' in html
    assert 'aria-label="Done"' in html
    assert 'aria-label="Cancel"' in html
    assert ">Done<" not in html
    assert ">Cancel<" not in html


def test_root_exposes_linear_periodic_toolbar_controls(
    editor_server: EditorServer,
) -> None:
    html = request_text(f"{editor_server.base_url}/")

    assert 'id="modes-menu-panel"' in html
    assert 'id="single-mode-menu-item"' in html
    assert 'id="linear-periodic-mode-menu-item"' in html
    assert 'id="grid-periodic-mode-menu-item"' in html
    assert 'id="tree-mode-menu-item"' in html
    assert 'id="benchmark-mode-menu-item"' in html
    assert 'id="linear-periodic-previous-cell-button"' in html
    assert 'id="linear-periodic-cell-label"' in html
    assert 'id="grid-periodic-up-cell-button"' in html
    assert 'id="grid-periodic-down-cell-button"' in html
    assert 'id="linear-periodic-next-cell-button"' in html
    assert ">For unidimensional<" in html
    assert ">For bidimensional<" in html
    assert ">For Tree<" in html
    assert ">Benchmark<" in html
    assert "Work with the root, branch, and leaf tree cells." in html
    assert 'title="Bidimensional periodic mode is not available yet."' not in html
    assert 'title="Benchmark mode is not available yet."' not in html
    assert 'id="benchmark-compare-button"' in html
    assert 'id="benchmark-scheme-name-input"' in html
    assert html.index('class="title-button-row"') < html.index(
        'class="toolbar-mode-controls"'
    )


def test_root_exposes_benchmark_compare_modal(editor_server: EditorServer) -> None:
    html = request_text(f"{editor_server.base_url}/")

    assert 'id="benchmark-compare-modal"' in html
    assert 'id="benchmark-compare-close-button"' in html
    assert 'data-tooltip-enabled="true"' in html
    assert 'data-shortcut-label="Close benchmark comparison"' in html
    assert 'id="benchmark-compare-export-csv-button"' in html
    assert 'id="benchmark-compare-export-text-button"' in html
    assert 'id="benchmark-compare-copy-latex-button"' in html
    assert 'id="benchmark-compare-table-body"' in html
    assert ">Peak Memory<" in html
    assert 'title="Close"' not in html
    assert html.index('class="toolbar-mode-controls"') < html.index(
        'id="template-select-field"'
    )


def test_root_exposes_generated_code_modal(editor_server: EditorServer) -> None:
    html = request_text(f"{editor_server.base_url}/")
    shell_bindings_body = request_text(
        f"{editor_server.base_url}/js/shell/editorShellBindings.js"
    )

    assert 'id="generated-code-modal"' in html
    assert 'id="generated-code-modal-backdrop"' in html
    assert 'id="generated-code-modal-close-button"' in html
    assert 'id="generated-code-modal-view"' in html
    assert ">Generated code<" in html
    assert 'title="Close generated code"' not in html
    assert '"generated-code-modal-close-button"' not in shell_bindings_body


def test_main_module_is_served_from_static_directory(
    editor_server: EditorServer,
) -> None:
    body, headers = request_with_headers(f"{editor_server.base_url}/js/main.js")

    assert body.strip()
    assert "startEditor" in body
    assert headers["Content-Type"].startswith("application/javascript")


def test_legacy_app_shim_is_not_served(editor_server: EditorServer) -> None:
    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"{editor_server.base_url}/app.js", timeout=5)

    assert exc_info.value.code == 404


def test_static_server_rejects_parent_directory_traversal(
    editor_server: EditorServer,
) -> None:
    app_directory = (
        Path(__file__).resolve().parents[1] / "src" / "tensor_network_editor" / "app"
    )
    sibling_directory = app_directory / "static_backup"
    secret_path = sibling_directory / "secret.txt"
    sibling_directory.mkdir(exist_ok=True)
    secret_path.write_text("secret", encoding="utf-8")

    try:
        with pytest.raises(HTTPError) as exc_info:
            urlopen(
                f"{editor_server.base_url}/../static_backup/secret.txt",
                timeout=5,
            )
    finally:
        secret_path.unlink(missing_ok=True)
        sibling_directory.rmdir()

    assert exc_info.value.code == 404


def test_notes_planner_uses_singular_operation_labels(
    editor_server: EditorServer,
) -> None:
    body = request_runtime_bundle(
        editor_server,
        "js/planner/plannerRenderers.js",
        "js/planner/plannerRenderersAutomatic.js",
        "js/planner/plannerRenderersManual.js",
    )

    assert '"FLOPs"' not in body
    assert '"MACs"' not in body
    assert '"FLOP"' in body
    assert '"MAC"' in body


def test_notes_and_planner_feature_modules_are_served(
    editor_server: EditorServer,
) -> None:
    notes_body = request_text(f"{editor_server.base_url}/js/graph/notes.js")
    planner_body = request_text(f"{editor_server.base_url}/js/planner/planner.js")
    planner_support_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerSupport.js"
    )
    planner_renderers_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerRenderers.js"
    )
    planner_support_guards_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerSupportGuards.js"
    )
    planner_support_operands_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerSupportOperands.js"
    )
    planner_support_analysis_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerSupportAnalysis.js"
    )
    planner_support_actions_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerSupportActions.js"
    )
    planner_renderers_common_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerRenderersCommon.js"
    )
    planner_renderers_automatic_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerRenderersAutomatic.js"
    )
    planner_renderers_manual_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerRenderersManual.js"
    )
    planner_renderers_panel_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerRenderersPanel.js"
    )
    planner_selectors_body = request_text(
        f"{editor_server.base_url}/js/state/plannerSelectors.js"
    )
    planner_commands_body = request_text(
        f"{editor_server.base_url}/js/actions/plannerCommands.js"
    )
    planner_service_body = request_text(
        f"{editor_server.base_url}/js/services/plannerAnalysisService.js"
    )
    registrar_body = request_text(
        f"{editor_server.base_url}/js/planner/notesPlanner.js"
    )
    utilities_body = request_text(f"{editor_server.base_url}/js/utils/utilities.js")
    utilities_templates_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesTemplates.js"
    )

    assert "registerNotesFeature" in notes_body
    assert "registerPlannerFeature" in planner_body
    assert 'from "./plannerSupport.js"' in planner_body
    assert 'from "./plannerRenderers.js"' in planner_body
    assert 'from "../state/plannerSelectors.js"' in planner_support_body
    assert 'from "./plannerSupportGuards.js"' in planner_support_body
    assert 'from "./plannerSupportOperands.js"' in planner_support_body
    assert 'from "./plannerSupportAnalysis.js"' in planner_support_body
    assert 'from "./plannerSupportActions.js"' in planner_support_body
    assert 'from "../actions/plannerCommands.js"' in planner_support_actions_body
    assert (
        'from "../services/plannerAnalysisService.js"' in planner_support_analysis_body
    )
    assert "createPlannerSupport" in planner_support_body
    assert "createPlannerRenderers" in planner_renderers_body
    assert 'from "./plannerRenderersCommon.js"' in planner_renderers_body
    assert 'from "./plannerRenderersAutomatic.js"' in planner_renderers_body
    assert 'from "./plannerRenderersManual.js"' in planner_renderers_body
    assert 'from "./plannerRenderersPanel.js"' in planner_renderers_body
    assert "createPlannerGuardSupport" in planner_support_guards_body
    assert "createPlannerOperandSupport" in planner_support_operands_body
    assert "createPlannerAnalysisSupport" in planner_support_analysis_body
    assert "createPlannerActionSupport" in planner_support_actions_body
    assert "createPlannerRendererCommonSupport" in planner_renderers_common_body
    assert "createPlannerAutomaticRendererSupport" in planner_renderers_automatic_body
    assert "createPlannerManualRendererSupport" in planner_renderers_manual_body
    assert "createPlannerPanelRendererSupport" in planner_renderers_panel_body
    assert "buildPlannerOperandState" in planner_selectors_body
    assert "createPlannerCommands" in planner_commands_body
    assert "createPlannerAnalysisService" in planner_service_body
    assert 'from "../graph/notes.js"' in registrar_body
    assert 'from "./planner.js"' in registrar_body
    assert 'from "./utilitiesTemplates.js"' in utilities_body
    assert "createTemplateOptionHelpers" in utilities_templates_body


def test_notes_and_shell_assets_delegate_to_split_helper_modules(
    editor_server: EditorServer,
) -> None:
    notes_body = request_text(f"{editor_server.base_url}/js/graph/notes.js")
    notes_support_body = request_text(
        f"{editor_server.base_url}/js/graph/notesSupport.js"
    )
    notes_clipboard_body = request_text(
        f"{editor_server.base_url}/js/graph/notesClipboard.js"
    )
    bootstrap_body = request_text(f"{editor_server.base_url}/js/bootstrap.js")
    shell_actions_body = request_text(
        f"{editor_server.base_url}/js/shell/shellActions.js"
    )

    assert 'from "./notesSupport.js"' in notes_body
    assert 'from "./notesClipboard.js"' in notes_body
    assert "createNotesSupport" in notes_support_body
    assert "createNotesClipboardActions" in notes_clipboard_body
    assert 'from "./shell/shellActions.js"' in bootstrap_body
    assert "createShellActions" in shell_actions_body


def test_vendor_asset_is_served_locally(editor_server: EditorServer) -> None:
    body, headers = request_with_headers(
        f"{editor_server.base_url}/vendor/cytoscape.min.js"
    )

    assert "cytoscape" in body
    assert headers["Content-Type"].startswith("application/javascript")


def test_prism_vendor_assets_are_served_locally(editor_server: EditorServer) -> None:
    core_body, core_headers = request_with_headers(
        f"{editor_server.base_url}/vendor/prism-core.min.js"
    )
    python_body, python_headers = request_with_headers(
        f"{editor_server.base_url}/vendor/prism-python.min.js"
    )

    assert "Prism" in core_body
    assert "python" in python_body
    assert core_headers["Content-Type"].startswith("application/javascript")
    assert python_headers["Content-Type"].startswith("application/javascript")


def test_root_defers_prism_vendor_loading_until_code_preview_is_needed(
    editor_server: EditorServer,
) -> None:
    html = request_text(f"{editor_server.base_url}/")

    assert "/vendor/prism-core.min.js" not in html
    assert "/vendor/prism-python.min.js" not in html
    assert "window.__TNE_ASSET_VERSION__" in html


def test_interactions_asset_exposes_updated_keyboard_shortcuts(
    editor_server: EditorServer,
) -> None:
    body = request_interactions_runtime_bundle(editor_server)
    html = request_text(f"{editor_server.base_url}/")

    assert "ctx.isTextInput(event.target) || ctx.isTextInput(activeElement)" in body
    assert 'if (hasSystemModifier && lowerKey === "y") {' in body
    assert 'setSelectedEngine("einsum_numpy");' in body
    assert 'if (hasSystemModifier && lowerKey === "n") {' not in body
    assert 'if (!hasAnyModifier && lowerKey === "s") {' in body
    assert "toggleSidebarCollapsed();" in body
    assert (
        'if (!hasSystemModifier && !event.altKey && event.shiftKey && lowerKey === "m") {'
        in body
    )
    assert "toggleMinimapVisibility();" in body
    assert (
        'if (!hasSystemModifier && !event.altKey && event.shiftKey && lowerKey === "r") {'
        in body
    )
    assert "trimContractionPlan(0);" in body
    assert 'if (!hasAnyModifier && lowerKey === "f") {' in body
    assert "toggleLinearPeriodicMode();" in body
    assert 'if (hasSystemModifier && event.altKey && lowerKey === "a") {' in body
    assert 'if (!hasSystemModifier && event.altKey && lowerKey === "a") {' in body
    assert 'if (hasSystemModifier && lowerKey === "a") {' in body
    assert "selectAllTensors();" in body
    assert (
        'if (!hasSystemModifier && !event.altKey && event.shiftKey && lowerKey === "g") {'
        in body
    )
    assert (
        'if (!hasSystemModifier && !event.altKey && event.shiftKey && lowerKey === "s") {'
        in body
    )
    assert 'if (!hasAnyModifier && lowerKey === "h") {' in body
    assert "createHyperedgeFromSelection();" in body
    assert (
        'if (!hasSystemModifier && !event.altKey && event.shiftKey && lowerKey === "e") {'
        in body
    )
    assert 'if (hasSystemModifier && event.shiftKey && lowerKey === "f") {' in body
    assert 'if (hasSystemModifier && lowerKey === "f") {' in body
    assert "openCanvasMetadataFilter();" in body
    assert "openCanvasNameSearch();" in body
    assert 'if (!hasAnyModifier && lowerKey === "d") {' in body
    assert 'if (!hasAnyModifier && lowerKey === "b") {' in body
    assert 'if (!hasAnyModifier && lowerKey === "l") {' in body
    assert 'if (!hasAnyModifier && lowerKey === "e") {' in body
    assert "setGridPeriodicMode(true);" in body
    assert "setBenchmarkMode(true);" in body
    assert "openSessionTemplatePicker();" in body
    assert "exportSelectedTemplateSpec();" in body
    assert "Alt+A" in html
    assert "Ctrl/Cmd+A" in html
    assert "Ctrl/Cmd+Alt+A" in html
    assert "Ctrl/Cmd+F" in html
    assert "Ctrl/Cmd+Shift+F" in html
    assert "Shift+S" in html
    assert "Shift+E" in html
    assert ">D<" in html
    assert ">B<" in html
    assert ">L<" in html


def test_overlays_asset_reuses_shared_tensor_size_helpers(
    editor_server: EditorServer,
) -> None:
    body = request_text(f"{editor_server.base_url}/js/graph/overlaysLayoutTemplates.js")

    assert "ctx.tensorWidth(" in body
    assert "ctx.tensorHeight(" in body
    assert "function tensorWidth(tensor)" not in body
    assert "function tensorHeight(tensor)" not in body


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
    assert "--toolbar-height:" in body
    assert ".toolbar-title-link {" in body
    assert ".title-control-divider {" in body
    assert ".title-control-group[hidden] {" in body
    assert ".title-control-divider[hidden] {" in body
    assert ".title-control-group-mode {" in body
    assert ".title-control-group-template {" in body
    assert "margin-left: auto;" in body
    assert "margin-right: auto;" in body
    assert ".title-button-row {" in body
    assert "align-items: flex-end;" in body
    assert ".title-button-row button {" in body
    assert "height: var(--canvas-control-height);" in body
    assert ".template-settings-shell {" in body
    assert ".template-settings-popover {" in body
    assert "top: var(--template-settings-popover-top, 0px);" in body
    assert "left: var(--template-settings-popover-left, 0px);" in body
    assert "position: fixed;" in body
    assert ".select-chevron-field::after {" in body
    assert 'content: "";' in body
    assert "border-right: 2px solid rgba(236, 242, 251, 0.78);" in body
    assert "border-bottom: 2px solid rgba(236, 242, 251, 0.78);" in body
    assert "transform: translateY(-50%) rotate(-45deg);" in body
    assert '.select-chevron-field[data-expanded="true"]::after {' in body
    assert "transform: translateY(-50%) rotate(45deg);" in body
    assert 'content: "â€º";' not in body
    assert 'content: "âŒ„";' not in body
    assert ".template-parameter-panel select," in body
    assert "height: var(--canvas-control-height);" in body
    assert ".template-select-field {" in body
    assert "grid-template-rows: var(--canvas-control-height);" in body
    assert "gap: 0;" in body
    assert ".template-select-field::after {" in body
    assert "top: 50%;" in body
    assert ".template-select-field select {" in body
    assert "appearance: none;" in body
    assert "padding-right: 2.2rem;" in body
    assert "min-width: 9rem;" in body
    assert "select:hover," in body
    assert "select:focus-visible {" in body
    assert ".template-settings-button:hover," in body
    assert "background: var(--control-hover-bg);" in body
    assert "box-shadow: var(--control-hover-shadow);" in body
    assert "min-width: 10.5rem;" not in body


def test_css_asset_styles_grouped_export_and_code_generation_controls(
    editor_server: EditorServer,
) -> None:
    html = request_text(f"{editor_server.base_url}/")
    body = request_text(f"{editor_server.base_url}/app.css")

    assert 'id="engine-select-field"' in html
    assert 'id="collection-format-select-field"' in html
    assert 'class="code-format-picker select-chevron-field"' in html
    assert ".toolbar-menu {" in body
    assert ".toolbar-menu-panel {" in body
    assert "top: var(--toolbar-menu-top, 0px);" in body
    assert "left: var(--toolbar-menu-left, 0px);" in body
    assert "position: fixed;" in body
    assert ".toolbar-menubar {" in body
    assert ".toolbar-menubar-button {" in body
    assert ".toolbar-menu-item {" in body
    assert ".toolbar-menu-item-content {" in body
    assert ".toolbar-menu-item-header {" in body
    assert ".toolbar-menu-item-description {" in body
    assert "display: none;" in body
    assert ".toolbar-menu-item-shortcut {" in body
    assert "background: #0e639c;" not in body
    assert ".code-header-controls {" in body
    assert ".code-header-controls .code-format-picker {" in body
    assert ".code-format-picker.select-chevron-field::after {" in body
    assert ".code-format-picker select {" in body
    assert "appearance: none;" in body
    assert "padding-right: 2.1rem;" in body
    assert ".code-header-row {" in body
    assert ".code-preview {" in body
    assert "padding: 3.4rem 1rem 1rem;" not in body
    assert "padding: 1rem;" in body
    assert ".code-preview .token.keyword {" in body
    assert ".code-preview .token.function {" in body


def test_css_asset_exposes_editor_dark_theme_tokens_and_compact_surfaces(
    editor_server: EditorServer,
) -> None:
    body = request_text(f"{editor_server.base_url}/app.css")

    assert "--bg-app:" in body
    assert "--surface-canvas:" in body
    assert "--surface-panel:" in body
    assert "--surface-elevated:" in body
    assert "--border-subtle:" in body
    assert "--accent-hover:" in body
    assert "--selection-accent:" in body
    assert "--font-ui:" in body
    assert "--font-mono:" in body
    assert "font-family: var(--font-ui);" in body
    assert "font-family: Georgia" not in body
    assert "border-radius: 24px;" not in body
    assert ".canvas-panel," in body
    assert ".help-dialog {" in body


def test_graph_assets_import_shared_editor_theme_palette(
    editor_server: EditorServer,
) -> None:
    graph_body = request_text(f"{editor_server.base_url}/js/graph/graphRender.js")
    export_body = request_text(f"{editor_server.base_url}/js/graph/exportMinimap.js")
    theme_body = request_text(f"{editor_server.base_url}/js/core/theme.js")

    assert 'from "../core/theme.js"' in graph_body
    assert 'from "../core/theme.js"' in export_body
    assert "export const GRAPH_THEME = Object.freeze(" in theme_body
    assert "export const UI_THEME = Object.freeze(" in theme_body
    assert "GRAPH_THEME.selection" in graph_body
    assert "GRAPH_THEME.pendingTensor" in graph_body
    assert "GRAPH_THEME.pendingIndex" in graph_body
    assert "GRAPH_THEME.canvasBackground" in export_body
    assert "GRAPH_THEME.selectionFill" in export_body


def test_css_asset_uses_two_row_shortcut_tooltips(
    editor_server: EditorServer,
) -> None:
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert ".shortcut-tooltip {" in css_body
    assert "width: max-content;" in css_body
    assert "max-width: min(18rem, calc(100vw - 1rem));" in css_body
    assert "border-radius: 6px;" in css_body
    assert "display: grid;" in css_body
    assert "gap: 0.22rem;" in css_body
    assert ".shortcut-tooltip-header {" in css_body
    assert "justify-content: flex-start;" in css_body
    assert "flex-wrap: wrap;" in css_body
    assert ".shortcut-tooltip-shortcut {" in css_body
    assert "white-space: nowrap;" in css_body
    assert ".shortcut-tooltip-description {" in css_body
    assert "line-height: 1.35;" in css_body


def test_css_asset_standardizes_hover_across_controls(
    editor_server: EditorServer,
) -> None:
    body = request_text(f"{editor_server.base_url}/app.css")

    assert "button:not(:disabled):hover," in body
    assert "button:not(:disabled):focus-visible," in body
    assert (
        'input:not([type="checkbox"]):not([type="radio"]):not([type="color"]):not([type="file"]):not([type="hidden"]):not([disabled]):hover,'
        in body
    )
    assert "textarea:not([disabled]):hover," in body
    assert ".properties-disclosure-summary:hover," in body
    assert ".properties-disclosure-summary:focus-visible {" in body
    assert ".button-accent-cool:hover {" not in body
    assert ".button-quiet:hover {" not in body
    assert ".button-accent-positive:hover," not in body
    assert ".button-accent-insert:hover {" not in body
    assert "button.danger:hover {" not in body


def test_sidebar_assets_expose_resize_handle(editor_server: EditorServer) -> None:
    html = request_text(f"{editor_server.base_url}/")
    css_body = request_text(f"{editor_server.base_url}/app.css")
    dom_body = request_text(f"{editor_server.base_url}/js/core/dom.js")
    sidebar_body = request_text(f"{editor_server.base_url}/js/core/sidebarTabs.js")

    assert 'id="sidebar-resize-handle"' in html
    assert 'class="sidebar-toggle-icon"' in html
    assert "&gt;&gt;" not in html
    assert 'role="separator"' in html
    assert "--sidebar-width: 360px;" in css_body
    assert ".sidebar-resize-handle {" in css_body
    assert ".sidebar-toggle-button {" in css_body
    assert "height: 2.5rem;" in css_body
    assert "align-self: center;" in css_body
    assert ".sidebar-toggle-button svg {" in css_body
    assert (
        "grid-template-columns: minmax(0, 1fr) minmax(280px, var(--sidebar-width));"
        in css_body
    )
    assert (
        'sidebarResizeHandle: document.getElementById("sidebar-resize-handle")'
        in dom_body
    )
    assert "function setSidebarWidth(" in sidebar_body
    assert "function buildSidebarToggleIconMarkup(" in sidebar_body
    assert (
        'windowRef.addEventListener("mousemove", handleSidebarResizeMove);'
        in sidebar_body
    )


def test_properties_asset_exposes_total_element_summaries_and_icon_delete_controls(
    editor_server: EditorServer,
) -> None:
    overview_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersOverview.js"
    )
    overview_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesMarkup.js"
    )
    tensor_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersTensor.js"
    )
    entities_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersEntities.js"
    )
    entity_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/entityPropertiesMarkup.js"
    )
    support_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesSupport.js"
    )
    metadata_body = request_text(
        f"{editor_server.base_url}/js/properties/metadataEditors.js"
    )
    summaries_body = request_text(
        f"{editor_server.base_url}/js/properties/propertySummaries.js"
    )

    combined_body = (
        overview_body
        + overview_markup_body
        + tensor_body
        + entities_body
        + entity_markup_body
        + metadata_body
    )
    assert "Total elements" in combined_body
    assert "Delete Selected" not in combined_body
    assert "Delete Connection" not in combined_body
    assert "Delete Note" not in combined_body
    assert 'aria-label="Delete selection"' in overview_body + overview_markup_body
    assert 'aria-label="Delete connection"' in entities_body + entity_markup_body
    assert 'aria-label="Delete note"' in entities_body + entity_markup_body
    assert 'from "./propertySummaries.js"' in support_body
    assert "function getSelectionTotalElementCount(" in summaries_body
    assert "function getTensorTotalElementCount(" in summaries_body


def test_properties_assets_use_compact_metadata_disclosures_and_tag_autocomplete(
    editor_server: EditorServer,
) -> None:
    overview_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersOverview.js"
    )
    overview_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesMarkup.js"
    )
    tensor_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersTensor.js"
    )
    entities_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersEntities.js"
    )
    entity_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/entityPropertiesMarkup.js"
    )
    support_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesSupport.js"
    )
    metadata_body = request_text(
        f"{editor_server.base_url}/js/properties/metadataEditors.js"
    )

    combined_body = (
        overview_body
        + overview_markup_body
        + tensor_body
        + entities_body
        + entity_markup_body
        + metadata_body
    )
    assert "Tags" in combined_body
    assert 'from "./metadataEditors.js"' in support_body
    assert "Custom metadata (JSON)" in metadata_body
    assert "metadata-editor-disclosure" in metadata_body
    assert 'summaryLabel = "Metadata"' in metadata_body
    assert 'rows="1"' in metadata_body
    assert "properties-disclosure-chevron" in metadata_body
    assert "field-label-with-help" in metadata_body
    assert "field-help-icon" in metadata_body
    assert (
        "Short reusable labels for filtering, search, and organization."
        in metadata_body
    )
    assert (
        "Store extra JSON fields that are not covered by the guided inputs."
        in metadata_body
    )
    assert "function buildTagAutocompleteSuggestions(" in metadata_body
    assert "function replaceActiveTagToken(" in metadata_body
    assert "{ scheduleOnInput: false }" in metadata_body
    assert (
        'export const RESERVED_METADATA_KEYS = new Set(["color", "collapsed", "tags"]);'
        not in metadata_body
    )
    assert (
        'const RESERVED_METADATA_KEYS = new Set(["color", "collapsed", "tags"]);'
        in metadata_body
    )
    assert "function bindMetadataEditors(" in metadata_body
    assert "function propertyInvalidation(overrides = {})" in support_body
    assert "Suggested annotations" not in metadata_body
    assert "function buildSuggestedAnnotationsMarkup(" not in metadata_body
    assert "function bindSuggestedAnnotationEditors(" not in metadata_body


def test_properties_assets_remove_guided_annotation_inputs_and_center_controls(
    editor_server: EditorServer,
) -> None:
    tensor_standard_body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesStandard.js"
    )
    tensor_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesStandardMarkup.js"
    )
    tensor_boundary_body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesBoundary.js"
    )
    overview_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesMarkup.js"
    )

    combined_body = tensor_standard_body + tensor_boundary_body + overview_markup_body

    assert "tensorAnnotationInputId" not in combined_body
    assert "indexAnnotationInputId" not in combined_body
    assert "bindSuggestedAnnotationEditors" not in combined_body
    assert 'id="center-tensor-button"' not in combined_body
    assert 'id="align-selection-center-button"' not in combined_body
    assert ">Center<" not in combined_body
    assert "createStandardTensorPropertiesRenderer" in tensor_standard_body
    assert "createPaleIndexColor" in tensor_markup_body
    assert "index-disclosure-state" in tensor_markup_body
    assert "Remove this tensor from the network." in tensor_markup_body


def test_canvas_tool_assets_expose_floating_filter_search_and_highlight_hooks(
    editor_server: EditorServer,
) -> None:
    html_body = request_text(f"{editor_server.base_url}/")
    dom_body = request_text(f"{editor_server.base_url}/js/core/dom.js")
    main_body = request_text(f"{editor_server.base_url}/js/main.js")
    filter_body = request_text(f"{editor_server.base_url}/js/graph/metadataFilters.js")
    filter_bindings_body = request_text(
        f"{editor_server.base_url}/js/graph/metadataFiltersBindings.js"
    )
    filter_renderers_body = request_text(
        f"{editor_server.base_url}/js/graph/metadataFiltersRenderers.js"
    )
    filter_state_body = request_text(
        f"{editor_server.base_url}/js/graph/metadataFiltersState.js"
    )
    graph_body = request_text(f"{editor_server.base_url}/js/graph/graphRender.js")
    minimap_body = request_text(f"{editor_server.base_url}/js/graph/exportMinimap.js")
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert 'id="canvas-tools"' in html_body
    assert 'id="canvas-context-menu-root"' in html_body
    assert 'id="metadata-filters-panel"' not in html_body
    assert 'canvasTools: document.getElementById("canvas-tools")' in dom_body
    assert (
        'canvasContextMenuRoot: document.getElementById("canvas-context-menu-root")'
        in dom_body
    )
    assert 'from "./graph/metadataFilters.js"' in main_body
    assert "registerMetadataFilters(context);" in main_body
    assert 'from "./metadataFiltersBindings.js"' in filter_body
    assert 'from "./metadataFiltersRenderers.js"' in filter_body
    assert 'from "./metadataFiltersState.js"' in filter_body
    assert "function openCanvasMetadataFilter(" in filter_body
    assert "function openCanvasNameSearch(" in filter_body
    assert "canvas-metadata-filter-button" in filter_renderers_body
    assert "canvas-name-search-button" in filter_renderers_body
    assert "canvas-metadata-filter-clear-button" in filter_renderers_body
    assert "canvas-metadata-filter-select-all-button" in filter_renderers_body
    assert "canvas-metadata-filter-select-none-button" in filter_renderers_body
    assert "Not specified" in filter_renderers_body + filter_state_body
    assert "canvas-name-search-input" in filter_renderers_body
    assert 'data-tooltip-enabled="true"' in filter_renderers_body
    assert (
        "Highlight tensors, indices, or bonds by metadata tags without hiding anything."
        in filter_renderers_body
    )
    assert (
        "Highlight tensors, indices, or bonds by exact name without changing the selection."
        in filter_renderers_body
    )
    assert (
        'class="canvas-tool-scope-field select-chevron-field"' in filter_renderers_body
    )
    assert '"bond"' in filter_renderers_body
    assert "function getMetadataFilterHighlight(" in filter_state_body
    assert "function bindMetadataFilterControls(" in filter_bindings_body
    assert "metadata-filter-dim" in graph_body
    assert "getMetadataFilterEntityState" in graph_body
    assert "getMetadataFilterEntityState" in minimap_body
    assert ".canvas-tool-popover" in css_body
    assert ".canvas-tool-scope-field {" in css_body
    assert ".canvas-tool-scope-field select {" in css_body
    assert "bottom: calc(100% +" in css_body
    assert "flex-wrap: wrap;" in css_body
    assert "transform: rotate(90deg)" in css_body
    assert ".metadata-editor-disclosure" in css_body
    assert "overflow: visible;" in css_body
    assert ".planner-chip-info {" in css_body
    assert ".planner-disclosure-state-show {" in css_body
    assert ".planner-disclosure-state-hide {" in css_body
    assert ".index-disclosure-state {" in css_body


def test_canvas_context_menu_assets_expose_minimal_selection_actions(
    editor_server: EditorServer,
) -> None:
    main_body = request_text(f"{editor_server.base_url}/js/main.js")
    context_menu_body = request_text(
        f"{editor_server.base_url}/js/graph/canvasContextMenu.js"
    )
    context_menu_bindings_body = request_text(
        f"{editor_server.base_url}/js/graph/canvasContextMenuBindings.js"
    )
    context_menu_markup_body = request_text(
        f"{editor_server.base_url}/js/graph/canvasContextMenuMarkup.js"
    )
    context_menu_targets_body = request_text(
        f"{editor_server.base_url}/js/graph/canvasContextMenuTargets.js"
    )
    graph_body = request_text(f"{editor_server.base_url}/js/graph/graphRender.js")
    overlays_body = request_text(
        f"{editor_server.base_url}/js/graph/overlaysLayoutTemplates.js"
    )
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert 'from "./graph/canvasContextMenu.js"' in main_body
    assert "registerCanvasContextMenu(context);" in main_body
    assert "function openCanvasContextMenu(" in context_menu_body
    assert 'from "./canvasContextMenuBindings.js"' in context_menu_body
    assert 'from "./canvasContextMenuMarkup.js"' in context_menu_body
    assert 'from "./canvasContextMenuTargets.js"' in context_menu_body
    assert 'id="context-menu-name-input"' in context_menu_markup_body
    assert 'id="context-menu-add-index-button"' in context_menu_markup_body
    assert 'id="context-menu-tensor-color-input"' in context_menu_markup_body
    assert 'id="context-menu-delete-tensor-button"' in context_menu_markup_body
    assert 'inputPrefix: "context-menu-tensor"' in context_menu_markup_body
    assert 'id="context-menu-dimension-input"' in context_menu_markup_body
    assert 'id="context-menu-index-color-input"' in context_menu_markup_body
    assert 'id="context-menu-move-up-button"' in context_menu_markup_body
    assert 'id="context-menu-move-down-button"' in context_menu_markup_body
    assert 'id="context-menu-delete-index-button"' in context_menu_markup_body
    assert 'inputPrefix: "context-menu-index"' in context_menu_markup_body
    assert 'id="context-menu-edge-color-input"' in context_menu_markup_body
    assert 'id="context-menu-delete-edge-button"' in context_menu_markup_body
    assert 'inputPrefix: "context-menu-edge"' in context_menu_markup_body
    assert 'id="context-menu-hyperedge-color-input"' in context_menu_markup_body
    assert 'id="context-menu-delete-hyperedge-button"' in context_menu_markup_body
    assert 'inputPrefix: "context-menu-hyperedge"' in context_menu_markup_body
    assert 'id="context-menu-add-index-to-selection-button"' in context_menu_markup_body
    assert 'id="context-menu-extract-selection-button"' in context_menu_markup_body
    assert (
        'id="context-menu-promote-selection-template-button"'
        in context_menu_markup_body
    )
    assert 'id="context-menu-create-hyperedge-button"' in context_menu_markup_body
    assert 'id="context-menu-selection-color-input"' in context_menu_markup_body
    assert 'id="context-menu-group-selection-button"' in context_menu_markup_body
    assert 'id="context-menu-delete-selection-button"' in context_menu_markup_body
    assert 'id="context-menu-toggle-group-button"' in context_menu_markup_body
    assert 'id="context-menu-add-index-to-group-button"' in context_menu_markup_body
    assert 'id="context-menu-extract-group-button"' in context_menu_markup_body
    assert 'id="context-menu-group-color-input"' in context_menu_markup_body
    assert 'id="context-menu-promote-group-template-button"' in context_menu_markup_body
    assert 'id="context-menu-delete-group-button"' in context_menu_markup_body
    assert "function buildTooltipAttributes(" in context_menu_markup_body
    assert '"Choose color"' in context_menu_markup_body
    assert '"Move index up"' in context_menu_markup_body
    assert '"Move index down"' in context_menu_markup_body
    assert '"Delete index"' in context_menu_markup_body
    assert '"Index dimension"' in context_menu_markup_body
    assert "Remove this tensor from the network." in context_menu_markup_body
    assert "Add one new open index to each selected tensor." in context_menu_markup_body
    assert (
        "Extract the selected tensors as a reusable subnetwork."
        in context_menu_markup_body
    )
    assert (
        "Save the selected tensors to the subnetwork library."
        in context_menu_markup_body
    )
    assert (
        "Promote the selected tensors to a reusable template."
        in context_menu_markup_body
    )
    assert (
        "Create a visual group from the selected tensors." in context_menu_markup_body
    )
    assert (
        "Create a hyperedge from the selected open indices." in context_menu_markup_body
    )
    assert (
        "Extract the tensors inside this group as a reusable subnetwork."
        in context_menu_markup_body
    )
    assert (
        "Save the tensors inside this group to the subnetwork library."
        in context_menu_markup_body
    )
    assert (
        "Promote the tensors inside this group to a reusable template."
        in context_menu_markup_body
    )
    assert "Add index to tensors" not in context_menu_markup_body
    assert "Extract selection" not in context_menu_markup_body
    assert "Promote to template" not in context_menu_markup_body
    assert "Add index" in context_menu_markup_body
    assert "Extract" in context_menu_markup_body
    assert "To Template" in context_menu_markup_body
    assert 'inputPrefix: "context-menu-group"' in context_menu_markup_body
    assert "Member tensors" in context_menu_markup_body
    assert "Total elements" in context_menu_markup_body
    assert "buildMetadataEditorMarkup" in context_menu_markup_body
    assert "bindMetadataEditors" in context_menu_bindings_body
    assert "renameHyperedge" in context_menu_bindings_body
    assert "deleteHyperedge" in context_menu_bindings_body
    assert "function resolveContextTarget(" in context_menu_targets_body
    assert "getHyperedgeContextTarget" in context_menu_targets_body
    assert "findHyperedgeById" in context_menu_body + context_menu_targets_body
    assert "canvas-context-menu-title" not in context_menu_markup_body
    assert "max-height: ${maxHeight}px;" not in context_menu_markup_body
    assert 'state.cy.on("cxttap"' in graph_body
    assert 'kind !== "tensor"' in graph_body
    assert 'kind !== "index"' in graph_body
    assert 'kind !== "edge"' in graph_body
    assert 'kind !== "hyperedge-hub"' in graph_body
    assert 'kind !== "hyperedge-spoke"' in graph_body
    assert 'addEventListener("contextmenu"' in overlays_body
    assert ".canvas-context-menu {" in css_body
    assert "max-height: calc(100% - 1rem);" not in css_body
    assert "overscroll-behavior: contain;" not in css_body


def test_properties_renderer_assets_are_split_by_selection_family(
    editor_server: EditorServer,
) -> None:
    facade_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderers.js"
    )
    overview_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersOverview.js"
    )
    tensor_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersTensor.js"
    )
    entities_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersEntities.js"
    )
    overview_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesMarkup.js"
    )
    overview_bindings_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesBindings.js"
    )
    entity_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/entityPropertiesMarkup.js"
    )
    entity_bindings_body = request_text(
        f"{editor_server.base_url}/js/properties/entityPropertiesBindings.js"
    )
    tensor_standard_body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesStandard.js"
    )
    tensor_boundary_body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesBoundary.js"
    )
    tensor_contraction_body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesContraction.js"
    )

    assert 'from "./propertiesRenderersOverview.js"' in facade_body
    assert 'from "./propertiesRenderersTensor.js"' in facade_body
    assert 'from "./propertiesRenderersEntities.js"' in facade_body
    assert 'from "./overviewPropertiesMarkup.js"' in overview_body
    assert 'from "./overviewPropertiesBindings.js"' in overview_body
    assert 'from "./entityPropertiesMarkup.js"' in entities_body
    assert 'from "./entityPropertiesBindings.js"' in entities_body
    assert 'from "./tensorPropertiesStandard.js"' in tensor_body
    assert 'from "./tensorPropertiesBoundary.js"' in tensor_body
    assert 'from "./tensorPropertiesContraction.js"' in tensor_body
    assert "renderNetworkProperties" in overview_body
    assert "renderTensorProperties" in tensor_body
    assert "renderGroupProperties" in entities_body
    assert "buildNetworkPropertiesMarkup" in overview_markup_body
    assert "createOverviewPropertiesBindings" in overview_bindings_body
    assert "buildGroupPropertiesMarkup" in entity_markup_body
    assert "createEntityPropertiesBindings" in entity_bindings_body
    assert "createStandardTensorPropertiesRenderer" in tensor_standard_body
    assert "createBoundaryTensorPropertiesRenderer" in tensor_boundary_body
    assert "createContractionTensorPropertiesRenderer" in tensor_contraction_body


def test_shell_and_properties_assets_delegate_bootstrap_and_panel_mutations_to_internal_modules(
    editor_server: EditorServer,
) -> None:
    bootstrap_body = request_text(f"{editor_server.base_url}/js/bootstrap.js")
    bootstrap_flow_body = request_text(
        f"{editor_server.base_url}/js/shell/editorBootstrapFlow.js"
    )
    shell_bindings_body = request_text(
        f"{editor_server.base_url}/js/shell/editorShellBindings.js"
    )
    tooltip_body = request_text(f"{editor_server.base_url}/js/shell/shortcutTooltip.js")
    properties_body = request_text(
        f"{editor_server.base_url}/js/properties/properties.js"
    )
    overview_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersOverview.js"
    )
    entities_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersEntities.js"
    )
    commands_body = request_text(
        f"{editor_server.base_url}/js/actions/propertyCommands.js"
    )

    assert 'from "./shell/editorBootstrapFlow.js"' in bootstrap_body
    assert 'from "./shell/editorShellBindings.js"' in bootstrap_body
    assert 'from "./shell/shortcutTooltip.js"' in bootstrap_body
    assert "ctx.services && ctx.services.session" not in bootstrap_body
    assert "typeof ctx.enforceLinearPeriodicEngineSupport" not in bootstrap_body
    assert "typeof ctx.refreshContractionAnalysis" not in bootstrap_body
    assert "typeof ctx.renderPlanner" not in bootstrap_body
    assert "function createEditorBootstrapFlow(" in bootstrap_flow_body
    assert "function createEditorShellBindings(" in shell_bindings_body
    assert "function createShortcutTooltip(" in tooltip_body
    assert "function applyTitleHint(" in tooltip_body
    assert "function showVirtualTooltip(" in tooltip_body
    assert "function hideActiveTooltip(" in tooltip_body
    assert "function escapeTooltipText(" in tooltip_body
    assert "function buildTooltipMarkup(" in tooltip_body
    assert 'class="shortcut-tooltip-header"' in tooltip_body
    assert 'class="shortcut-tooltip-description"' in tooltip_body
    assert "tooltip.innerHTML = buildTooltipMarkup(button);" in tooltip_body
    assert "ctx.shortcutTooltip = shortcutTooltip;" in bootstrap_body
    assert "shortcutTooltip.applyTitleHint(" in shell_bindings_body
    assert "ctx.applyDesignChange(" not in overview_body
    assert "ctx.applyDesignChange(" not in entities_body
    assert "ctx.removeEdge(" not in entities_body
    assert "ctx.removeNote(" not in entities_body
    assert 'from "../actions/propertyCommands.js"' in properties_body
    assert "function renameNetwork(" in commands_body
    assert "function applySelectionColor(" in commands_body
    assert "function addIndexToSelectedTensors(" in commands_body
    assert "function renameGroup(" in commands_body
    assert "function deleteGroup(" in commands_body
    assert "function renameEdge(" in commands_body
    assert "function deleteEdge(" in commands_body
    assert "function updateNoteText(" in commands_body
    assert "function deleteNote(" in commands_body


def test_graph_assets_expose_for_boundary_tensor_hovers(
    editor_server: EditorServer,
) -> None:
    graph_body = request_text(f"{editor_server.base_url}/js/graph/graphRender.js")
    graph_tooltips_body = request_text(
        f"{editor_server.base_url}/js/graph/graphRenderTooltips.js"
    )

    assert 'state.cy.on("mouseover", "node[kind = \'tensor\']"' in graph_body
    assert 'state.cy.on("mouseout", "node[kind = \'tensor\']"' in graph_body
    assert 'from "./graphRenderTooltips.js"' in graph_body
    assert "ctx.shortcutTooltip.showVirtualTooltip" in graph_tooltips_body
    assert (
        "Virtual boundary tensor for the next cell in For unidimensional mode."
        in graph_tooltips_body
    )
    assert (
        "Virtual boundary tensor for the cell on the right in For bidimensional mode."
        in graph_tooltips_body
    )


def test_tensor_property_assets_delegate_rendering_and_mutations_to_internal_modules(
    editor_server: EditorServer,
) -> None:
    css_body = request_text(f"{editor_server.base_url}/app.css")
    tensor_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersTensor.js"
    )
    tensor_standard_body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesStandard.js"
    )
    tensor_standard_data_body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesStandardData.js"
    )
    tensor_standard_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesStandardMarkup.js"
    )
    tensor_standard_bindings_body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesStandardBindings.js"
    )
    tensor_boundary_body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesBoundary.js"
    )
    tensor_contraction_body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesContraction.js"
    )

    assert "function renderTensorProperties(" not in tensor_body
    assert "function renderLinearPeriodicBoundaryTensorProperties(" not in tensor_body
    assert "function renderContractionTensorProperties(" not in tensor_body
    assert "function renderContractionIndexProperties(" not in tensor_body
    assert "ctx.applyDesignChange(" not in tensor_body
    assert "typeof ctx.syncConnectedIndexDimension" not in tensor_body
    assert "function renderTensorProperties(" in tensor_standard_body
    assert 'from "./tensorPropertiesStandardData.js"' in tensor_standard_body
    assert 'from "./tensorPropertiesStandardMarkup.js"' in tensor_standard_body
    assert 'from "./tensorPropertiesStandardBindings.js"' in tensor_standard_body
    assert (
        "function renderLinearPeriodicBoundaryTensorProperties(" in tensor_boundary_body
    )
    assert "function renderContractionTensorProperties(" in tensor_contraction_body
    assert "function renderContractionIndexProperties(" in tensor_contraction_body
    assert "function createStandardTensorDataSupport(" in tensor_standard_data_body
    assert (
        "function createStandardTensorPropertiesMarkupSupport("
        in tensor_standard_markup_body
    )
    assert (
        "function createStandardTensorPropertiesBindingSupport("
        in tensor_standard_bindings_body
    )
    assert 'id="tensor-values-disclosure"' in tensor_standard_markup_body
    assert '<div class="properties-disclosure-body">' in tensor_standard_markup_body
    assert "tensor-values-disclosure-body" not in tensor_standard_markup_body
    assert 'id="tensor-data-mode-select"' in tensor_standard_markup_body
    assert 'id="tensor-data-validation-message"' in tensor_standard_markup_body
    assert "Explicit values (JSON)" in tensor_standard_markup_body
    assert (
        "Choose how this tensor is initialized and edit explicit values when needed."
        in tensor_standard_markup_body
    )
    assert "Expected shape:" in tensor_standard_markup_body
    assert "Current initializer:" not in tensor_standard_markup_body
    assert (
        "Use JSON numbers that match the tensor shape exactly."
        not in tensor_standard_markup_body
    )
    assert (
        ".tensor-values-disclosure > .properties-disclosure-summary {" not in css_body
    )
    assert ".tensor-values-disclosure-body {" not in css_body
    assert "commands.updateTensorData" in tensor_standard_bindings_body


def test_index_disclosure_border_uses_the_port_color(
    editor_server: EditorServer,
) -> None:
    standard_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesStandardMarkup.js"
    )
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert "--index-border-color:" in standard_markup_body
    assert "ctx.getIndexColor(index, isConnected)" in standard_markup_body
    assert (
        "border-color: var(--index-border-color, rgba(76, 92, 120, 0.95));" in css_body
    )
    assert "border-color: var(--index-border-color, var(--accent));" in css_body


def test_contraction_result_properties_expose_a_delete_action(
    editor_server: EditorServer,
) -> None:
    body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesContraction.js"
    )

    assert 'id="delete-contraction-tensor-button"' in body
    assert 'aria-label="Delete result"' in body
    assert 'data-tooltip-enabled="true"' in body
    assert 'data-shortcut-label="Delete result"' in body
    assert 'title="Delete result"' not in body
    assert "commands.deleteCurrentSelection" in body


def test_note_assets_move_note_editing_into_canvas(
    editor_server: EditorServer,
) -> None:
    notes_body = request_text(f"{editor_server.base_url}/js/graph/notes.js")
    properties_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersEntities.js"
    )
    properties_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/entityPropertiesMarkup.js"
    )
    properties_bindings_body = request_text(
        f"{editor_server.base_url}/js/properties/entityPropertiesBindings.js"
    )
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert 'textarea.addEventListener("keydown", (event) => {' in notes_body
    assert "event.stopPropagation();" in notes_body
    assert 'className = "canvas-note-color-button"' in notes_body
    assert 'colorInput.type = "color";' in notes_body
    assert "ctx.bindDebouncedAutosave(" in notes_body
    assert "scheduleOnInput: false" in notes_body
    assert "scheduleOnInput: false" in properties_bindings_body
    assert 'button.dataset.tooltipEnabled = "true";' in notes_body
    assert 'button.removeAttribute("title");' in notes_body
    assert 'setAttribute("title"' not in notes_body
    assert (
        '<label for="note-text-input">Note text</label>'
        in properties_body + properties_markup_body
    )
    assert 'id="note-color-input"' in properties_body + properties_markup_body
    assert "Edit this note directly on the canvas." not in properties_body
    assert ".canvas-note-color-button {" in css_body


def test_dynamic_frontend_actions_use_shared_tooltips_and_consistent_labels(
    editor_server: EditorServer,
) -> None:
    context_menu_markup_body = request_text(
        f"{editor_server.base_url}/js/graph/canvasContextMenuMarkup.js"
    )
    overview_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesMarkup.js"
    )
    entity_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/entityPropertiesMarkup.js"
    )
    planner_body = request_runtime_bundle(
        editor_server,
        "js/planner/plannerRenderers.js",
        "js/planner/plannerRenderersPanel.js",
    )
    html = request_text(f"{editor_server.base_url}/")
    utilities_body = request_text(f"{editor_server.base_url}/js/utils/utilitiesUi.js")

    assert 'data-shortcut-label="Delete selection"' in overview_markup_body
    assert 'title="Delete selection"' not in overview_markup_body
    assert 'data-shortcut-label="Delete group"' in entity_markup_body
    assert 'data-shortcut-label="Delete connection"' in entity_markup_body
    assert 'data-shortcut-label="Delete note"' in entity_markup_body
    assert 'title="Delete group"' not in entity_markup_body
    assert 'title="Delete connection"' not in entity_markup_body
    assert 'title="Delete note"' not in entity_markup_body
    assert '"Delete selection",' in context_menu_markup_body
    assert '"Delete connection",' in context_menu_markup_body
    assert '"Delete group",' in context_menu_markup_body
    assert "Delete bond" not in context_menu_markup_body
    assert 'title="Delete selection"' not in context_menu_markup_body
    assert 'title="Delete connection"' not in context_menu_markup_body
    assert 'title="Delete group"' not in context_menu_markup_body
    assert (
        'data-shortcut-description="Remove all manual steps from the current contraction path."'
        in planner_body
    )
    assert 'title="Reset path"' not in planner_body
    assert 'data-shortcut-label="Code generation warning"' in html
    assert 'data-shortcut-label="Template warnings"' in html
    assert "codeGenerationWarning.title =" not in utilities_body
    assert "templateCatalogWarning.title =" not in utilities_body


def test_ui_utility_assets_route_panels_generated_code_toolbar_and_status_through_helpers(
    editor_server: EditorServer,
) -> None:
    utilities_ui_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUi.js"
    )
    utilities_ui_dom_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiDom.js"
    )
    utilities_ui_panels_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiPanels.js"
    )
    utilities_ui_generated_code_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiGeneratedCode.js"
    )
    utilities_ui_toolbar_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiToolbar.js"
    )
    utilities_ui_status_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiStatus.js"
    )

    assert 'from "./utilitiesUiDom.js"' in utilities_ui_body
    assert 'from "./utilitiesUiPanels.js"' in utilities_ui_body
    assert 'from "./utilitiesUiGeneratedCode.js"' in utilities_ui_body
    assert 'from "./utilitiesUiToolbar.js"' in utilities_ui_body
    assert 'from "./utilitiesUiStatus.js"' in utilities_ui_body
    assert "function positionFloatingPanel(" in utilities_ui_dom_body
    assert "function toggleToolbarMenu(" in utilities_ui_panels_body
    assert "function toggleGeneratedCodeModal(" in utilities_ui_generated_code_body
    assert "function updateToolbarState(" in utilities_ui_toolbar_body
    assert "function formatIssues(" in utilities_ui_status_body
    assert "function positionFloatingPanel(" not in utilities_ui_body
    assert "function updateToolbarState(" not in utilities_ui_body
    assert "function formatIssues(" not in utilities_ui_body


def test_note_assets_tint_the_full_note_frame_and_avoid_rerendering_text_edits(
    editor_server: EditorServer,
) -> None:
    notes_body = request_text(f"{editor_server.base_url}/js/graph/notes.js")
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert "invalidate: noteInvalidation({ overlays: false })" in notes_body
    assert 'frame.style.setProperty("--note-accent-color"' in notes_body
    assert "--note-surface-color" in notes_body
    assert "var(--note-accent-color" in css_body
    assert "var(--note-surface-color" in css_body


def test_collapsed_note_assets_leave_a_small_grab_margin_around_the_toggle(
    editor_server: EditorServer,
) -> None:
    notes_body = request_text(f"{editor_server.base_url}/js/graph/notes.js")
    css_body = request_text(f"{editor_server.base_url}/app.css")
    constants_body = request_text(f"{editor_server.base_url}/js/core/constants.js")
    collapsed_frame_start = css_body.index(".canvas-note-frame.is-collapsed {")
    collapsed_toggle_start = css_body.index(".canvas-note-collapsed-toggle {")
    collapsed_toggle_end = css_body.index(
        ".canvas-note-color-button svg,",
        collapsed_toggle_start,
    )
    collapsed_frame_block = css_body[collapsed_frame_start:collapsed_toggle_start]
    collapsed_toggle_block = css_body[collapsed_toggle_start:collapsed_toggle_end]

    assert 'collapsedToggle.classList.add("canvas-note-collapsed-toggle")' in notes_body
    assert "NOTE_COLLAPSED_SIZE: 36," in constants_body
    assert "width: 36px;" in collapsed_frame_block
    assert "min-width: 36px;" in collapsed_frame_block
    assert "min-height: 36px;" in collapsed_frame_block
    assert "height: 36px;" in collapsed_frame_block
    assert ".canvas-note-collapsed-toggle {" in collapsed_toggle_block
    assert "width: 100%;" not in collapsed_toggle_block
    assert "min-width: 100%;" not in collapsed_toggle_block
    assert "height: 100%;" not in collapsed_toggle_block


def test_interaction_assets_support_latest_contraction_scene_editing(
    editor_server: EditorServer,
) -> None:
    interactions_body = request_interactions_runtime_bundle(editor_server)
    planner_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerSupport.js"
    )
    planner_commands_body = request_text(
        f"{editor_server.base_url}/js/actions/plannerCommands.js"
    )
    graph_body = request_text(f"{editor_server.base_url}/js/graph/graphRender.js")
    graph_lifecycle_body = request_text(
        f"{editor_server.base_url}/js/graph/graphRenderLifecycle.js"
    )
    utilities_body = request_utilities_runtime_bundle(editor_server)

    assert (
        "Connect mode is only available in the base tensor view."
        not in interactions_body
    )
    assert "Selection cleared." in planner_body + planner_commands_body
    assert (
        "Choose a different tensor or intermediate; both selections refer to the same contracted operand."
        not in planner_body
    )
    assert 'from "./graphRenderLifecycle.js"' in graph_body
    assert "const indexNodesInteractive = !readOnlyScene;" in graph_lifecycle_body
    assert "selectable: !readOnlyScene," in graph_lifecycle_body
    assert "ctx.ensureContractionViewSnapshots();" in utilities_body


def test_toolbar_assets_route_file_and_template_actions_through_cursor_style_menus(
    editor_server: EditorServer,
) -> None:
    bootstrap_body = request_text(f"{editor_server.base_url}/js/bootstrap.js")
    shell_bindings_body = request_text(
        f"{editor_server.base_url}/js/shell/editorShellBindings.js"
    )
    dom_body = request_text(f"{editor_server.base_url}/js/core/dom.js")
    interactions_body = request_interactions_runtime_bundle(editor_server)
    utilities_body = request_utilities_runtime_bundle(editor_server)

    assert (
        'exportFormatSelect: document.getElementById("export-format-select")'
        in dom_body
    )
    assert (
        'codeGenerationWarning: document.getElementById("code-generation-warning")'
        in dom_body
    )
    assert 'fileMenuButton: document.getElementById("file-menu-button")' in dom_body
    assert 'fileMenuPanel: document.getElementById("file-menu-panel")' in dom_body
    assert (
        'templatesMenuButton: document.getElementById("templates-menu-button")'
        in dom_body
    )
    assert (
        'templatesMenuPanel: document.getElementById("templates-menu-panel")'
        in dom_body
    )
    assert (
        'exportPythonMenuItem: document.getElementById("export-python-menu-item")'
        in dom_body
    )
    assert (
        'exportPngMenuItem: document.getElementById("export-png-menu-item")' in dom_body
    )
    assert (
        'exportSvgMenuItem: document.getElementById("export-svg-menu-item")' in dom_body
    )
    assert 'from "./shell/editorShellBindings.js"' in bootstrap_body
    assert "bindMenubarMenu(menu.name, menu.button, menu.panel);" in shell_bindings_body
    assert 'bindListener(exportPythonMenuItem, "click", () => {' in shell_bindings_body
    assert 'bindListener(exportPngMenuItem, "click", () => {' in shell_bindings_body
    assert 'bindListener(exportSvgMenuItem, "click", () => {' in shell_bindings_body
    assert (
        'bindListener(saveSessionTemplateMenuItem, "click", () => {'
        in shell_bindings_body
    )
    assert (
        'bindListener(loadSessionTemplateMenuItem, "click", () => {'
        in shell_bindings_body
    )
    assert (
        'bindListener(editSessionTemplateMenuItem, "click", () => {'
        in shell_bindings_body
    )
    assert "async function downloadSelectedExport()" in interactions_body
    assert "async function downloadExportAs(format)" in interactions_body
    assert "const previousFormat = exportFormatSelect.value;" in interactions_body
    assert "await downloadSelectedExport();" in interactions_body
    assert "exportPythonMenuItem.disabled =" in utilities_body
    assert "exportPngMenuItem.disabled =" in utilities_body
    assert "exportSvgMenuItem.disabled =" in utilities_body


def test_template_insertion_assets_refresh_lookups_and_anchor_new_contract_operands(
    editor_server: EditorServer,
) -> None:
    interactions_body = request_interactions_runtime_bundle(editor_server)
    contraction_body = request_text(
        f"{editor_server.base_url}/js/graph/contractionScene.js"
    )
    contraction_cache_body = request_text(
        f"{editor_server.base_url}/js/graph/contractionSceneCache.js"
    )

    assert "invalidate: { lookups: true }" in interactions_body
    assert 'from "./contractionSceneCache.js"' in contraction_body
    assert "ensureSpecLookups()" in contraction_cache_body
    assert "state.tensorById[anchorTensorId] || null" in contraction_cache_body


def test_contraction_scene_assets_route_progression_and_snapshots_through_state_modules(
    editor_server: EditorServer,
) -> None:
    contraction_body = request_text(
        f"{editor_server.base_url}/js/graph/contractionScene.js"
    )
    contraction_cache_body = request_text(
        f"{editor_server.base_url}/js/graph/contractionSceneCache.js"
    )
    contraction_operands_body = request_text(
        f"{editor_server.base_url}/js/graph/contractionSceneOperands.js"
    )

    assert 'from "./contractionSceneCache.js"' in contraction_body
    assert 'from "./contractionSceneEditing.js"' in contraction_body
    assert 'from "./contractionSceneOperands.js"' in contraction_body
    assert 'from "../state/contractionSceneProgression.js"' in contraction_operands_body
    assert 'from "../state/contractionSceneProgression.js"' in contraction_cache_body
    assert 'from "../state/contractionSceneSnapshots.js"' in contraction_cache_body
    assert "export function cloneOperand(" not in contraction_body
    assert "export function analyzeOperandPair(" not in contraction_body
    assert (
        "function buildContractionOperandProgressionUncached("
        not in contraction_operands_body
    )
    assert "function buildSnapshotLayoutMap(" not in contraction_cache_body


def test_subnetwork_assets_expose_import_export_controls_and_routes(
    editor_server: EditorServer,
) -> None:
    html = request_text(f"{editor_server.base_url}/")
    dom_body = request_text(f"{editor_server.base_url}/js/core/dom.js")
    shell_bindings_body = request_text(
        f"{editor_server.base_url}/js/shell/editorShellBindings.js"
    )
    interactions_body = request_interactions_runtime_bundle(editor_server)
    overview_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersOverview.js"
    )
    overview_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesMarkup.js"
    )
    entities_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersEntities.js"
    )
    entity_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/entityPropertiesMarkup.js"
    )
    context_menu_markup_body = request_text(
        f"{editor_server.base_url}/js/graph/canvasContextMenuMarkup.js"
    )
    context_menu_bindings_body = request_text(
        f"{editor_server.base_url}/js/graph/canvasContextMenuBindings.js"
    )

    assert 'id="reflow-imported-button"' in html
    assert 'id="save-subnetwork-library-menu-item"' in html
    assert 'id="open-subnetwork-library-menu-item"' in html
    assert 'id="subnetwork-load-input"' in html
    assert 'id="subnetwork-library-modal"' in html
    assert 'id="subnetwork-library-search-input"' in html
    assert 'id="subnetwork-library-tag-filter"' in html
    assert 'id="subnetwork-library-select-all-input"' in html
    assert 'id="subnetwork-library-selection-summary"' in html
    assert 'id="subnetwork-library-add-selected-button"' in html
    assert 'id="subnetwork-library-list"' in html
    assert 'id="subnetwork-catalog-warning"' in html
    assert (
        'reflowImportedButton: document.getElementById("reflow-imported-button")'
        in dom_body
    )
    assert "saveSubnetworkLibraryMenuItem: document.getElementById(" in dom_body
    assert "openSubnetworkLibraryMenuItem: document.getElementById(" in dom_body
    assert (
        'subnetworkLoadInput: document.getElementById("subnetwork-load-input")'
        in dom_body
    )
    assert (
        'subnetworkLibraryModal: document.getElementById("subnetwork-library-modal")'
        in dom_body
    )
    assert (
        "subnetworkLibrarySelectAllInput: document.getElementById(" in dom_body
        and '"subnetwork-library-select-all-input"' in dom_body
    )
    assert (
        "subnetworkLibrarySelectionSummary: document.getElementById(" in dom_body
        and '"subnetwork-library-selection-summary"' in dom_body
    )
    assert (
        "subnetworkLibraryAddSelectedButton: document.getElementById(" in dom_body
        and '"subnetwork-library-add-selected-button"' in dom_body
    )
    assert (
        'subnetworkCatalogWarning: document.getElementById("subnetwork-catalog-warning")'
        in dom_body
    )
    assert (
        'bindListener(subnetworkLoadInput, "change", actions.loadSubnetworkFromFile);'
        in shell_bindings_body
    )
    assert (
        'bindListener(saveSubnetworkLibraryMenuItem, "click", () => {'
        in shell_bindings_body
    )
    assert (
        'bindListener(openSubnetworkLibraryMenuItem, "click", () => {'
        in shell_bindings_body
    )
    assert 'id="insert-subnetwork-button"' not in html
    assert '"/api/subnetwork/extract"' in interactions_body
    assert '"/api/subnetwork/prepare-insert"' in interactions_body
    assert '"/api/subnetwork-library/save"' in interactions_body
    assert '"/api/subnetwork-library/rename"' in interactions_body
    assert '"/api/subnetwork-library/delete"' in interactions_body
    assert '"/api/subnetwork-library/prepare-insert"' in interactions_body
    assert "prepareLibrarySubnetworkForInsert" in interactions_body
    assert "saveSelectionAsSessionTemplate" in interactions_body
    assert "saveSelectionToSubnetworkLibrary" in interactions_body
    assert "openSubnetworkLibrary" in interactions_body
    assert 'id="extract-selection-button"' in overview_body + overview_markup_body
    assert (
        'id="save-selection-subnetwork-library-button"'
        in overview_body + overview_markup_body
    )
    assert (
        'id="promote-selection-template-button"' in overview_body + overview_markup_body
    )
    assert 'id="extract-group-button"' in entities_body + entity_markup_body
    assert (
        'id="save-group-subnetwork-library-button"'
        in entities_body + entity_markup_body
    )
    assert 'id="promote-group-template-button"' in entities_body + entity_markup_body
    assert (
        'id="context-menu-save-selection-subnetwork-library-button"'
        in context_menu_markup_body
    )
    assert (
        'id="context-menu-save-group-subnetwork-library-button"'
        in context_menu_markup_body
    )
    assert (
        "context-menu-save-selection-subnetwork-library-button"
        in context_menu_bindings_body
    )
    assert (
        "context-menu-save-group-subnetwork-library-button"
        in context_menu_bindings_body
    )
    assert "Extract Group" not in entity_markup_body
    assert "Promote Group to Template" not in entity_markup_body
    assert "Extract" in entity_markup_body
    assert "To Template" in entity_markup_body
    assert "To Library" in overview_markup_body
    assert "To Library" in entity_markup_body
    assert (
        "Extract the selected tensors as a reusable subnetwork." in overview_markup_body
    )
    assert (
        "Save the selected tensors to the subnetwork library." in overview_markup_body
    )
    assert (
        "Promote the selected tensors to a reusable template." in overview_markup_body
    )
    assert "Create a visual group from the selected tensors." in overview_markup_body
    assert (
        "Extract the tensors inside this group as a reusable subnetwork."
        in entity_markup_body
    )
    assert (
        "Save the tensors inside this group to the subnetwork library."
        in entity_markup_body
    )
    assert (
        "Promote the tensors inside this group to a reusable template."
        in entity_markup_body
    )


def test_template_management_assets_expose_toolbar_controls_and_routes(
    editor_server: EditorServer,
) -> None:
    html = request_text(f"{editor_server.base_url}/")
    body = request_text(f"{editor_server.base_url}/app.css")
    dom_body = request_text(f"{editor_server.base_url}/js/core/dom.js")
    bootstrap_flow_body = request_text(
        f"{editor_server.base_url}/js/shell/editorBootstrapFlow.js"
    )
    shell_bindings_body = request_text(
        f"{editor_server.base_url}/js/shell/editorShellBindings.js"
    )
    interactions_body = request_interactions_runtime_bundle(editor_server)
    utilities_ui_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUi.js"
    )
    utilities_ui_panels_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiPanels.js"
    )
    utilities_ui_toolbar_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiToolbar.js"
    )
    utilities_ui_toolbar_warnings_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiToolbarWarnings.js"
    )
    utilities_ui_toolbar_mode_controls_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiToolbarModeControls.js"
    )
    session_template_body = request_text(
        f"{editor_server.base_url}/js/session/sessionTemplateFlows.js"
    )
    session_template_library_body = request_text(
        f"{editor_server.base_url}/js/session/sessionTemplateFlowSubnetworkLibrary.js"
    )
    session_template_dialogs_body = request_text(
        f"{editor_server.base_url}/js/session/sessionTemplateDialogs.js"
    )
    session_template_manager_body = request_text(
        f"{editor_server.base_url}/js/session/sessionTemplateManager.js"
    )

    assert re.search(
        r'<button id="insert-template-button"[^>]*>\s*\+\s*</button>',
        html,
    )
    assert 'id="template-settings-button"' in html
    assert 'class="template-settings-icon"' in html
    assert ">...<" not in html
    assert 'id="reflow-imported-button"' in html
    assert 'aria-haspopup="dialog"' in html
    assert 'id="reflow-layout-popover"' in html
    assert 'id="reflow-auto-layout-button"' in html
    assert 'id="reflow-align-left-button"' not in html
    assert 'id="reflow-indices-left-button"' in html
    assert 'id="reflow-indices-reset-button"' in html
    assert 'id="reflow-arrange-chain-button"' in html
    assert 'id="reflow-snap-grid-button"' in html
    assert 'id="reflow-distribute-horizontal-button"' not in html
    assert 'id="reflow-distribute-vertical-button"' not in html
    assert 'id="save-session-template-menu-item"' in html
    assert 'id="load-session-template-menu-item"' in html
    assert 'id="export-session-template-menu-item"' in html
    assert 'id="edit-session-template-menu-item"' in html
    assert 'id="template-load-input"' in html
    assert 'id="template-manager-modal"' in html
    assert 'id="template-manager-list"' in html
    assert 'id="template-manager-close-button"' in html
    assert 'id="template-manager-save-button"' in html
    assert 'id="template-manager-discard-button"' in html
    assert 'class="icon-button toolbar-icon-button danger button-close-static"' in html
    assert "Review the session templates you created" in html
    assert 'id="template-settings-popover"' in html
    assert 'id="template-catalog-warning"' in html
    assert 'id="about-schema-version"' in html
    assert 'id="insert-subnetwork-button"' not in html
    assert 'id="save-subnetwork-library-menu-item"' in html
    assert 'id="open-subnetwork-library-menu-item"' in html
    assert 'id="subnetwork-library-modal"' in html
    assert 'id="help-shared-header"' in html
    assert 'class="help-close-icon"' in html
    assert '<span class="template-parameter-title">Template</span>' not in html
    assert re.search(
        r'<div class="button-row reflow-action-row">[\s\S]*id="reflow-auto-layout-button"[\s\S]*>\s*Auto layout\s*<[\s\S]*id="reflow-arrange-chain-button"[\s\S]*>\s*Chain\s*<[\s\S]*id="reflow-arrange-tree-button"[\s\S]*>\s*Tree\s*<[\s\S]*id="reflow-arrange-grid-button"[\s\S]*>\s*Grid\s*<[\s\S]*id="reflow-snap-grid-button"[\s\S]*>\s*Snap to Grid\s*<',
        html,
    )
    assert re.search(
        r'Indices[\s\S]*<div class="button-row reflow-align-row reflow-indices-row">[\s\S]*id="reflow-indices-left-button"[\s\S]*id="reflow-indices-right-button"[\s\S]*id="reflow-indices-top-button"[\s\S]*id="reflow-indices-bottom-button"[\s\S]*id="reflow-indices-reset-button"',
        html,
    )
    assert 'aria-label="Align left"' not in html
    assert 'aria-label="Align middle"' not in html
    assert 'aria-label="Move indices left"' in html
    assert (
        'title="Align left: place selected tensors on the same left edge while keeping them separated."'
        not in html
    )
    assert (
        'title="Chain: place selected tensors in one ordered row, following bonds when present."'
        in html
    )
    assert (
        'title="Tree: place selected tensors in levels under a root, using bonds when possible."'
        in html
    )
    assert (
        'title="Grid: place selected tensors on an even grid, keeping nearby order stable."'
        in html
    )
    assert (
        'title="Snap to Grid: move each selected tensor to the nearest canvas grid point."'
        in html
    )
    assert (
        'title="Auto layout: detect the best arrangement for the current selection, or for the whole graph when nothing is selected."'
        in html
    )
    assert (
        'title="Indices reset: redistribute selected tensor indices evenly around each tensor."'
        in html
    )
    assert not re.search(r'<button id="help-close-button"[^>]*title=', html)
    assert re.search(r">\s*Reset\s*<", html)
    assert "&larr;" in html
    assert "&#8857;" not in html
    assert "Arrange Chain" not in html
    assert "Distribute Horizontally" not in html
    assert "Distribute Vertically" not in html
    assert not re.search(
        r'<button id="help-close-button"[^>]*>\s*Close\s*</button>',
        html,
    )
    assert 'dataset.shortcutLabel = "Sidebar";' in request_text(
        f"{editor_server.base_url}/js/core/sidebarTabs.js"
    )
    assert 'dataset.tooltipEnabled = "true";' in request_text(
        f"{editor_server.base_url}/js/core/sidebarTabs.js"
    )
    assert "Output type" in shell_bindings_body
    assert "Choose how generated code returns the tensors" in shell_bindings_body
    assert 'from "./utilitiesUiPanels.js"' in utilities_ui_body
    assert 'from "./utilitiesUiToolbar.js"' in utilities_ui_body
    assert 'from "./utilitiesUiToolbarModeControls.js"' in utilities_ui_toolbar_body
    assert (
        "Create a new benchmark scheme after the current one."
        in utilities_ui_toolbar_mode_controls_body
    )
    assert (
        "Cell navigation is available in For unidimensional, For bidimensional, and Benchmark modes."
        in utilities_ui_toolbar_mode_controls_body
    )
    shortcuts_section = re.search(
        r'<section id="help-shortcuts-section"[^>]*>(?P<body>.*?)</section>',
        html,
        re.DOTALL,
    )
    assert shortcuts_section is not None
    assert "<h3>" not in shortcuts_section.group("body")
    assert (
        "<strong>H</strong><span>Create hyperedge from selected indices</span>"
        in shortcuts_section.group("body")
    )
    about_section = re.search(
        r'<section id="help-about-section"[^>]*>(?P<body>.*?)</section>',
        html,
        re.DOTALL,
    )
    assert about_section is not None
    assert "<h3>" not in about_section.group("body")
    assert "Schema version" in about_section.group("body")
    assert "Support on YouTube" in about_section.group("body")
    assert 'href="https://www.youtube.com/@whenphysics"' in about_section.group("body")
    assert (
        'templateSettingsButton: document.getElementById("template-settings-button")'
        in dom_body
    )
    assert (
        'templateSettingsPopover: document.getElementById("template-settings-popover")'
        in dom_body
    )
    assert (
        'templateLoadInput: document.getElementById("template-load-input")' in dom_body
    )
    assert (
        'reflowLayoutPopover: document.getElementById("reflow-layout-popover")'
        in dom_body
    )
    assert (
        'reflowAlignLeftButton: document.getElementById("reflow-align-left-button")'
        in dom_body
    )
    assert (
        'reflowIndicesLeftButton: document.getElementById("reflow-indices-left-button")'
        in dom_body
    )
    assert (
        'reflowArrangeChainButton: document.getElementById("reflow-arrange-chain-button")'
        in dom_body
    )
    assert (
        'reflowSnapGridButton: document.getElementById("reflow-snap-grid-button")'
        in dom_body
    )
    assert (
        'aboutSchemaVersion: document.getElementById("about-schema-version")'
        in dom_body
    )
    assert (
        'templateManagerModal: document.getElementById("template-manager-modal")'
        in dom_body
    )
    assert "templateManagerSaveButton: document.getElementById(" in dom_body
    assert "templateManagerCloseButton: document.getElementById(" in dom_body
    assert "templateManagerDiscardButton: document.getElementById(" in dom_body
    assert (
        'templateCatalogWarning: document.getElementById("template-catalog-warning")'
        in dom_body
    )
    assert 'helpSharedHeader: document.getElementById("help-shared-header")' in dom_body
    assert ".help-about-grid {" in body
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in body
    assert ".help-dialog-close {" in body
    assert "width: 2.2rem;" in body
    assert "min-width: 2.2rem;" in body
    assert "height: 2.2rem;" in body
    assert "border-radius: 8px;" in body
    assert "color: #ffffff;" in body
    assert ".button-close-static {" in body
    assert "button.button-close-static:not(:disabled):hover {" in body
    assert ".help-dialog-close:hover," not in body
    assert ".help-dialog-close:focus-visible {" in body
    assert "rgba(116, 34, 44, 0.56)" not in body
    assert "rgba(234, 114, 126, 0.76)" not in body
    assert ".help-close-icon {" in body
    assert "fill: currentColor;" in body
    assert ".help-dialog-header[hidden] {" in body
    assert ".help-sections[hidden] {" in body
    assert ".help-shortcuts[hidden] {" in body
    assert 'sourceBadge.textContent = "Session"' in session_template_manager_body
    assert "buildSerializedNetworkPreviewMarkup" in session_template_manager_body
    assert "buildSerializedNetworkPreviewMarkup" in session_template_library_body
    assert (
        "deleteButton.innerHTML = renderTrashIcon();" in session_template_manager_body
    )
    assert "<span>Delete</span>" not in session_template_manager_body
    assert "templateManagerCloseButton" in utilities_ui_panels_body
    assert (
        'bindListener(templateManagerCloseButton, "click", () =>' in shell_bindings_body
    )
    assert 'from "./sessionTemplateDialogs.js"' in session_template_body
    assert 'from "./sessionTemplateManager.js"' in session_template_body
    assert 'from "./sessionTemplateFlowSubnetworkLibrary.js"' in session_template_body
    assert "deleteButton.innerHTML" in session_template_manager_body
    assert "title = `Delete ${entry.displayName}`" in session_template_manager_body
    assert "function buildTemplateManagerRow(" in session_template_manager_body
    assert "function saveTemplateManagerChanges(" in session_template_manager_body
    assert "function discardTemplateManagerChanges(" in session_template_manager_body
    assert "function promptForTemplateDisplayName(" in session_template_dialogs_body
    assert "function promptForSubnetworkName(" in session_template_dialogs_body
    assert "function promptForSubnetworkTags(" in session_template_dialogs_body
    assert "function createSubnetworkLibrarySupport(" in session_template_library_body
    assert "function renderSubnetworkLibrary(" in session_template_library_body
    assert (
        "function syncSubnetworkLibraryBatchControls(" in session_template_library_body
    )
    assert "getFilteredSubnetworkEntries," in session_template_library_body
    assert (
        'bindListener(saveSessionTemplateMenuItem, "click", () => {'
        in shell_bindings_body
    )
    assert (
        'bindListener(saveSubnetworkLibraryMenuItem, "click", () => {'
        in shell_bindings_body
    )
    assert (
        'bindListener(templateSettingsButton, "click", () => {' in shell_bindings_body
    )
    assert "toggleReflowLayoutPopover" in shell_bindings_body
    assert "reflowAlignLeftButton" in shell_bindings_body
    assert "reflowIndicesLeftButton" in shell_bindings_body
    assert "reflowArrangeGridButton" in shell_bindings_body
    assert 'bindReflowAction(reflowAutoLayoutButton, "auto");' in shell_bindings_body
    assert "applyReflowIndicesAction" in shell_bindings_body
    assert (
        'bindListener(loadSessionTemplateMenuItem, "click", () => {'
        in shell_bindings_body
    )
    assert (
        'bindListener(editSessionTemplateMenuItem, "click", () => {'
        in shell_bindings_body
    )
    assert (
        'bindListener(openSubnetworkLibraryMenuItem, "click", () => {'
        in shell_bindings_body
    )
    assert (
        'bindListener(subnetworkLibrarySelectAllInput, "change", (event) => {'
        in shell_bindings_body
    )
    assert (
        'bindListener(subnetworkLibraryAddSelectedButton, "click", () => {'
        in shell_bindings_body
    )
    assert (
        "templateCatalogWarnings: payload.template_catalog_warnings"
        in bootstrap_flow_body
    )
    assert (
        "subnetworkCatalogWarnings: payload.subnetwork_catalog_warnings"
        in bootstrap_flow_body
    )
    assert (
        'actions.setStatus(state.templateCatalogWarnings[0], "error");'
        in bootstrap_flow_body
    )
    assert "loadSessionTemplatesFromFile" in interactions_body
    assert "exportSelectedTemplateSpec" in interactions_body
    assert "toggleTemplateManager" in interactions_body
    assert 'from "./utilitiesUiToolbarWarnings.js"' in utilities_ui_toolbar_body
    assert "function syncTemplateCatalogWarning()" in utilities_ui_toolbar_warnings_body
    assert (
        "function syncSubnetworkCatalogWarning()" in utilities_ui_toolbar_warnings_body
    )
    assert "function toggleReflowLayoutPopover()" in utilities_ui_panels_body
    assert "function syncSubnetworkLibraryModalState()" in utilities_ui_panels_body
    assert "sessionUi.promptText(" in session_template_dialogs_body
    assert "Choose a name for this template." in session_template_dialogs_body
    assert "Choose a name for this subnetwork." in session_template_dialogs_body


def test_layout_assets_expose_reflow_helpers_and_selection_tensor_actions(
    editor_server: EditorServer,
) -> None:
    utilities_body = request_utilities_runtime_bundle(editor_server)
    utilities_module_body = request_text(
        f"{editor_server.base_url}/js/utils/utilities.js"
    )
    layout_body = request_text(f"{editor_server.base_url}/js/utils/utilitiesLayout.js")
    layout_algorithms_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesLayoutAlgorithms.js"
    )
    layout_algorithms_graph_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesLayoutAlgorithmsGraph.js"
    )
    layout_algorithms_positions_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesLayoutAlgorithmsPositions.js"
    )
    layout_indices_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesLayoutIndices.js"
    )
    layout_selection_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesLayoutSelection.js"
    )
    overview_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersOverview.js"
    )
    overview_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesMarkup.js"
    )
    overview_bindings_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesBindings.js"
    )
    properties_body = request_text(
        f"{editor_server.base_url}/js/properties/properties.js"
    )
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert 'from "./utilitiesLayout.js"' in utilities_module_body
    assert "createUtilityLayoutBindings" in layout_body
    assert 'from "./utilitiesLayoutAlgorithms.js"' in layout_body
    assert 'from "./utilitiesLayoutIndices.js"' in layout_body
    assert 'from "./utilitiesLayoutSelection.js"' in layout_body
    assert 'from "./utilitiesLayoutAlgorithmsGraph.js"' in layout_algorithms_body
    assert 'from "./utilitiesLayoutAlgorithmsPositions.js"' in layout_algorithms_body
    assert "function alignSelectedTensors(" in layout_body
    assert "function arrangeSelectedTensors(" in layout_body
    assert "function distributeSelectedTensors(" in layout_body
    assert "function snapSelectedTensorsToGrid(" in layout_body
    assert "function applyReflowLayoutAction(" in layout_body
    assert "function applyReflowIndicesAction(" in layout_body
    assert "function applyAutoLayout(" in layout_body
    assert "function reflowLastImportedTensors(" in layout_body
    assert "function createUtilityLayoutAlgorithmSupport(" in layout_algorithms_body
    assert "function buildArrangedSelectionPositions(" in layout_algorithms_graph_body
    assert "function buildAutoLayoutPositions(" in layout_algorithms_graph_body
    assert "function buildImportedReflowPositions(" in layout_algorithms_graph_body
    assert "function createLayoutAlgorithmGraphSupport(" in layout_algorithms_graph_body
    assert (
        "function createLayoutAlgorithmPositionSupport("
        in layout_algorithms_positions_body
    )
    assert "function buildReflowIndexOffsets(" in layout_indices_body
    assert "function getSelectedLayoutTensorIds(" in layout_selection_body
    assert "GRID_SNAP_SIZE" in utilities_body
    assert 'id="add-index-to-selection-button"' in overview_body + overview_markup_body
    assert 'id="extract-selection-button"' in overview_body + overview_markup_body
    assert (
        'id="promote-selection-template-button"' in overview_body + overview_markup_body
    )
    assert 'id="create-hyperedge-button"' in overview_body + overview_markup_body
    assert "Add Index to Tensors" not in overview_markup_body
    assert "Extract Selection" not in overview_markup_body
    assert "Promote to Template" not in overview_markup_body
    assert "Add index" in overview_markup_body
    assert "Extract" in overview_markup_body
    assert "To Template" in overview_markup_body
    assert "<h3>Hyperedge</h3>" not in overview_markup_body
    assert 'id="group-selection-button"' in overview_body + overview_markup_body
    assert (
        'id="align-selection-left-button"' not in overview_body + overview_markup_body
    )
    assert (
        'id="arrange-selection-chain-button"'
        not in overview_body + overview_markup_body
    )
    assert 'id="snap-selection-button"' not in overview_body + overview_markup_body
    assert (
        'id="distribute-selection-horizontal-button"'
        not in overview_body + overview_markup_body
    )
    assert "Arrange Chain" not in overview_markup_body
    assert "Distribute Horizontally" not in overview_markup_body
    assert 'class="button-row layout-align-row"' not in overview_markup_body
    assert "createGroupFromSelection" in properties_body
    assert 'bindClick("group-selection-button"' in overview_bindings_body
    assert ".reflow-action-row {" in css_body
    assert "grid-template-columns: repeat(5, var(--canvas-control-height));" in css_body
    assert "aspect-ratio: 1 / 1;" in css_body


def test_periodic_mode_assets_delegate_to_internal_helpers(
    editor_server: EditorServer,
) -> None:
    tree_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesTreePeriodic.js"
    )
    tree_state_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesTreePeriodicState.js"
    )
    tree_boundaries_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesTreePeriodicBoundaries.js"
    )
    tree_flow_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesTreePeriodicFlow.js"
    )
    grid_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesGridPeriodic.js"
    )
    grid_state_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesGridPeriodicState.js"
    )
    grid_boundaries_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesGridPeriodicBoundaries.js"
    )
    grid_flow_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesGridPeriodicFlow.js"
    )
    linear_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesLinearPeriodic.js"
    )
    linear_state_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesLinearPeriodicState.js"
    )
    linear_boundaries_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesLinearPeriodicBoundaries.js"
    )
    linear_flow_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesLinearPeriodicFlow.js"
    )
    toolbar_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiToolbar.js"
    )
    toolbar_warnings_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiToolbarWarnings.js"
    )
    toolbar_derived_state_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiToolbarDerivedState.js"
    )
    toolbar_mode_controls_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiToolbarModeControls.js"
    )
    toolbar_action_state_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiToolbarActionState.js"
    )

    assert 'from "./utilitiesTreePeriodicState.js"' in tree_body
    assert 'from "./utilitiesTreePeriodicBoundaries.js"' in tree_body
    assert 'from "./utilitiesTreePeriodicFlow.js"' in tree_body
    assert "function createTreePeriodicStateSupport(" in tree_state_body
    assert "function createTreePeriodicBoundarySupport(" in tree_boundaries_body
    assert "function createTreePeriodicFlowSupport(" in tree_flow_body
    assert 'from "./utilitiesGridPeriodicState.js"' in grid_body
    assert 'from "./utilitiesGridPeriodicBoundaries.js"' in grid_body
    assert 'from "./utilitiesGridPeriodicFlow.js"' in grid_body
    assert "function createGridPeriodicStateSupport(" in grid_state_body
    assert "function createGridPeriodicBoundarySupport(" in grid_boundaries_body
    assert "function createGridPeriodicFlowSupport(" in grid_flow_body
    assert 'from "./utilitiesLinearPeriodicState.js"' in linear_body
    assert 'from "./utilitiesLinearPeriodicBoundaries.js"' in linear_body
    assert 'from "./utilitiesLinearPeriodicFlow.js"' in linear_body
    assert "function createLinearPeriodicStateSupport(" in linear_state_body
    assert "function createLinearPeriodicBoundarySupport(" in linear_boundaries_body
    assert "function createLinearPeriodicFlowSupport(" in linear_flow_body
    assert 'from "./utilitiesUiToolbarWarnings.js"' in toolbar_body
    assert 'from "./utilitiesUiToolbarDerivedState.js"' in toolbar_body
    assert 'from "./utilitiesUiToolbarModeControls.js"' in toolbar_body
    assert 'from "./utilitiesUiToolbarActionState.js"' in toolbar_body
    assert "function createUiToolbarWarningSupport(" in toolbar_warnings_body
    assert "function createUiToolbarDerivedStateSupport(" in toolbar_derived_state_body
    assert "function createUiToolbarModeControlSupport(" in toolbar_mode_controls_body
    assert "function createUiToolbarActionStateSupport(" in toolbar_action_state_body


def test_periodic_frontend_assets_centralize_shared_constants(
    editor_server: EditorServer,
) -> None:
    planner_support_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerSupport.js"
    )
    contraction_operands_body = request_text(
        f"{editor_server.base_url}/js/graph/contractionSceneOperands.js"
    )
    toolbar_mode_controls_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesUiToolbarModeControls.js"
    )
    planner_formatting_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerAnalysisFormatting.js"
    )
    grid_state_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesGridPeriodicState.js"
    )
    linear_state_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesLinearPeriodicState.js"
    )
    tree_state_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesTreePeriodicState.js"
    )

    assert 'from "../utils/utilitiesLinearPeriodicState.js"' in planner_support_body
    assert "LINEAR_PERIODIC_PREVIOUS_OPERAND_ID" in planner_support_body
    assert "LINEAR_PERIODIC_NEXT_OPERAND_ID" in planner_support_body
    assert "__linear_previous__" not in planner_support_body
    assert "__linear_next__" not in planner_support_body
    assert (
        'from "../utils/utilitiesLinearPeriodicState.js"' in contraction_operands_body
    )
    assert "LINEAR_PERIODIC_PREVIOUS_OPERAND_ID" in contraction_operands_body
    assert "LINEAR_PERIODIC_NEXT_OPERAND_ID" in contraction_operands_body
    assert "__linear_previous__" not in contraction_operands_body
    assert "__linear_next__" not in contraction_operands_body
    assert 'from "./utilitiesLinearPeriodicState.js"' in toolbar_mode_controls_body
    assert 'from "./utilitiesGridPeriodicState.js"' in toolbar_mode_controls_body
    assert 'from "./utilitiesTreePeriodicState.js"' in toolbar_mode_controls_body
    assert "const LINEAR_PERIODIC_CELL_LABELS =" not in toolbar_mode_controls_body
    assert "const GRID_PERIODIC_CELL_LABELS =" not in toolbar_mode_controls_body
    assert "const TREE_PERIODIC_CELL_LABELS =" not in toolbar_mode_controls_body
    assert "export function formatShapeElementCount(" not in planner_formatting_body
    assert "function formatShapeElementCount(" in planner_formatting_body
    assert "export const GRID_PERIODIC_NAVIGATION =" not in grid_state_body
    assert "const GRID_PERIODIC_NAVIGATION =" in grid_state_body
    assert "export const GRID_PERIODIC_CELL_KEYS =" not in grid_state_body
    assert "const GRID_PERIODIC_CELL_KEYS =" in grid_state_body
    assert "export const GRID_PERIODIC_EXPECTED_ROLES =" not in grid_state_body
    assert "const GRID_PERIODIC_EXPECTED_ROLES =" in grid_state_body
    assert (
        "export const LINEAR_PERIODIC_RESERVED_OPERAND_ID_BY_ROLE ="
        not in linear_state_body
    )
    assert "const LINEAR_PERIODIC_RESERVED_OPERAND_ID_BY_ROLE =" in linear_state_body
    assert "export const TREE_PERIODIC_NAVIGATION =" not in tree_state_body
    assert "const TREE_PERIODIC_NAVIGATION =" in tree_state_body


def test_performance_sensitive_assets_use_lightweight_analysis_paths(
    editor_server: EditorServer,
) -> None:
    planner_body = request_text(f"{editor_server.base_url}/js/planner/planner.js")
    planner_support_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerSupport.js"
    )
    planner_service_body = request_text(
        f"{editor_server.base_url}/js/services/plannerAnalysisService.js"
    )
    interactions_body = request_interactions_runtime_bundle(editor_server)
    utilities_body = request_utilities_runtime_bundle(editor_server)
    minimap_body = request_text(f"{editor_server.base_url}/js/graph/exportMinimap.js")
    overlays_body = request_text(
        f"{editor_server.base_url}/js/graph/overlaysLayoutTemplates.js"
    )

    assert "function serializeCurrentSpec(options = {})" in utilities_body
    assert "persistViewSnapshots = false" in utilities_body
    assert (
        "ctx.serializeCurrentSpec({ persistViewSnapshots: false })"
        not in planner_support_body
    )
    assert (
        "serializeCurrentSpec({ persistViewSnapshots: false })" in planner_service_body
    )
    assert "serializeCurrentSpec({ persistViewSnapshots: false })" in interactions_body
    assert "serializeCurrentSpec({ persistViewSnapshots: true })" in interactions_body
    assert "ANALYSIS_REFRESH_DELAY_MS = 200" in planner_body
    assert "requestAnimationFrame" in minimap_body
    assert "requestAnimationFrame" in overlays_body


def test_editor_assets_use_lookup_caches_and_lighter_history_paths(
    editor_server: EditorServer,
) -> None:
    history_body = request_text(
        f"{editor_server.base_url}/js/graph/historySelection.js"
    )
    history_snapshots_body = request_text(
        f"{editor_server.base_url}/js/state/historySnapshots.js"
    )
    history_selection_body = request_text(
        f"{editor_server.base_url}/js/state/selectionEntries.js"
    )
    history_pipeline_body = request_text(
        f"{editor_server.base_url}/js/actions/designMutationPipeline.js"
    )
    utilities_body = request_utilities_runtime_bundle(editor_server)
    utilities_spec_body = request_text(
        f"{editor_server.base_url}/js/utils/utilitiesSpec.js"
    )
    spec_normalization_body = request_text(
        f"{editor_server.base_url}/js/spec/specNormalization.js"
    )
    spec_lookups_body = request_text(f"{editor_server.base_url}/js/spec/specLookups.js")
    spec_mutations_body = request_text(
        f"{editor_server.base_url}/js/spec/specMutations.js"
    )
    state_body = request_text(f"{editor_server.base_url}/js/state/state.js")
    notes_body = request_text(f"{editor_server.base_url}/js/graph/notes.js")
    notes_support_body = request_text(
        f"{editor_server.base_url}/js/graph/notesSupport.js"
    )
    properties_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesSupport.js"
    )
    properties_renderers_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersOverview.js"
    )
    properties_bindings_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesBindings.js"
    )
    interactions_body = request_interactions_runtime_bundle(editor_server)
    graph_body = request_text(f"{editor_server.base_url}/js/graph/graphRender.js")
    graph_model_body = request_text(
        f"{editor_server.base_url}/js/views/graphElementModel.js"
    )
    graph_adapter_body = request_text(
        f"{editor_server.base_url}/js/views/cytoscapeGraphAdapter.js"
    )

    assert (
        "JSON.stringify(leftSnapshot) === JSON.stringify(rightSnapshot)"
        not in history_body
    )
    assert 'from "../state/historySnapshots.js"' in history_body
    assert 'from "../state/selectionEntries.js"' in history_body
    assert 'from "../actions/designMutationPipeline.js"' in history_body
    assert "function normalizeInvalidations(" not in history_body
    assert "function createHistorySnapshot(" not in history_body
    assert "function getSelectedEntries(" not in history_body
    assert "function createHistorySnapshotSupport(" in history_snapshots_body
    assert "function createSelectionEntrySupport(" in history_selection_body
    assert "function createDesignMutationPipeline(" in history_pipeline_body
    assert "structuredClone" in utilities_body
    assert "function toggleLinearPeriodicMode()" in utilities_body
    assert "function switchLinearPeriodicCell(direction)" in utilities_body
    assert "linear_periodic_chain" in utilities_body
    assert 'from "../spec/specNormalization.js"' in utilities_spec_body
    assert 'from "../spec/specLookups.js"' in utilities_spec_body
    assert 'from "../spec/specMutations.js"' in utilities_spec_body
    assert "function normalizeGraphSectionInPlace(" not in utilities_spec_body
    assert "function ensureSpecLookups(" not in utilities_spec_body
    assert "function createSpecNormalizationBindings(" in spec_normalization_body
    assert "function createSpecLookupBindings(" in spec_lookups_body
    assert "function ensureSpecLookups()" in spec_lookups_body
    assert "function createSpecMutationBindings(" in spec_mutations_body
    assert "tensorById: {}" in state_body
    assert "edgeById: {}" in state_body
    assert "indexOwnerById: {}" in state_body
    assert "noteById: {}" in state_body
    assert 'from "./notesSupport.js"' in notes_body
    assert "return state.noteById[noteId] || null;" in notes_support_body
    assert "function propertyInvalidation(overrides = {})" in properties_body
    assert "function selectionColorInvalidation(selectedEntries)" in properties_body
    assert (
        "invalidate: selectionColorInvalidation(selectedEntries)"
        in properties_renderers_body + properties_bindings_body
    )
    assert 'if (typeof ctx.bumpSpecRevision === "function")' not in interactions_body
    assert "startOffset:" in graph_body
    assert "createGraphElementModelBuilder" in graph_model_body
    assert "createCytoscapeGraphAdapter" in graph_adapter_body


def test_properties_assets_lock_virtual_boundary_tensor_structure(
    editor_server: EditorServer,
) -> None:
    body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesRenderersTensor.js"
    )

    assert "ctx.isLinearPeriodicBoundaryTensor(tensor)" in body
    assert "renderLinearPeriodicBoundaryTensorProperties" in body
    assert "managed by For mode" not in body
    assert "Move this port directly on the canvas to adjust its position." not in body


def test_linear_periodic_assets_propagate_interface_dimensions_across_cells(
    editor_server: EditorServer,
) -> None:
    history_body = request_text(
        f"{editor_server.base_url}/js/graph/historySelection.js"
    )
    utilities_body = request_utilities_runtime_bundle(editor_server)

    assert "syncCurrentGraphIntoLinearPeriodicChain:" in history_body
    assert "function syncLinearPeriodicChainInterfaceDimensions(" in utilities_body
    assert "function getCanonicalLinearPeriodicInterfaceDimensions(" in utilities_body
    assert "boundaryTensor.indices = resolvedInterfaceDimensions.map(" in utilities_body
    assert (
        "syncLinearPeriodicBoundaryTensors(runtimeSpec, interfaceDimensions);"
        in utilities_body
    )
    assert "chain[`${cellName}_cell`] = seedLinearPeriodicCell(" in utilities_body
    assert "interfaceDimensions" in utilities_body


def test_linear_periodic_assets_do_not_force_two_engine_support_message(
    editor_server: EditorServer,
) -> None:
    utilities_body = request_utilities_runtime_bundle(editor_server)
    bootstrap_body = request_text(f"{editor_server.base_url}/js/bootstrap.js")
    interactions_body = request_interactions_runtime_bundle(editor_server)

    assert "LINEAR_PERIODIC_SUPPORTED_ENGINES" not in utilities_body
    assert (
        "For mode currently supports TensorNetwork and TensorKrowch."
        not in utilities_body
    )
    assert (
        "For mode currently supports TensorNetwork and TensorKrowch."
        not in bootstrap_body
    )
    assert (
        "For mode currently supports TensorNetwork and TensorKrowch."
        not in interactions_body
    )
    assert (
        "TensorNetwork is selected because this mode currently supports "
        "TensorNetwork and TensorKrowch."
    ) not in utilities_body


def test_properties_assets_sync_dimensions_across_connected_ports(
    editor_server: EditorServer,
) -> None:
    body = request_text(f"{editor_server.base_url}/js/actions/propertyCommands.js")
    support_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesSupport.js"
    )
    spec_mutations_body = request_text(
        f"{editor_server.base_url}/js/spec/specMutations.js"
    )

    assert 'from "./propertyInvalidation.js"' in support_body
    assert "createPropertyInvalidationSupport" in support_body
    assert "const currentOwner = findIndexOwner(indexId);" in body
    assert "const currentIndex = currentOwner ? currentOwner.index : null;" in body
    assert "if (!currentIndex) {" in body
    assert "syncConnectedIndexDimension(indexId, parsed);" in body
    assert (
        "function syncConnectedIndexDimension(indexId, nextDimension) {"
        in spec_mutations_body
    )
    assert "const connectedEdge = findEdgeByIndexId(indexId);" in spec_mutations_body
    assert "connectedOwner.index.dimension = nextDimension;" in spec_mutations_body


def test_planner_assets_expose_total_elements_and_step_spacing(
    editor_server: EditorServer,
) -> None:
    planner_body = request_runtime_bundle(
        editor_server,
        "js/planner/plannerRenderers.js",
        "js/planner/plannerRenderersManual.js",
    )
    planner_formatting_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerAnalysisFormatting.js"
    )
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert "Total elements" in planner_body + planner_formatting_body
    assert 'from "./plannerAnalysisFormatting.js"' in planner_body
    assert "function getShapeElementCount(" in planner_formatting_body
    assert "planner-manual-step-list" in planner_body
    assert ".planner-manual-step-list {" in css_body
    assert "border-top:" in css_body


def test_editor_shell_assets_split_session_ui_bindings_and_property_helpers(
    editor_server: EditorServer,
) -> None:
    interactions_body = request_text(
        f"{editor_server.base_url}/js/interactions/interactions.js"
    )
    session_body = request_text(
        f"{editor_server.base_url}/js/interactions/interactionsSession.js"
    )
    session_editor_body = request_text(
        f"{editor_server.base_url}/js/session/sessionEditorFlows.js"
    )
    session_template_body = request_text(
        f"{editor_server.base_url}/js/session/sessionTemplateFlows.js"
    )
    session_ui_body = request_text(
        f"{editor_server.base_url}/js/session/sessionUiAdapters.js"
    )
    planner_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerRenderers.js"
    )
    planner_bindings_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerPanelBindings.js"
    )
    properties_body = request_text(
        f"{editor_server.base_url}/js/properties/propertiesSupport.js"
    )
    property_autosave_body = request_text(
        f"{editor_server.base_url}/js/properties/propertyAutosave.js"
    )
    property_invalidation_body = request_text(
        f"{editor_server.base_url}/js/properties/propertyInvalidation.js"
    )

    assert "store: ctx.store" in interactions_body
    assert "selectors: ctx.selectors" in interactions_body
    assert "services: ctx.services" in interactions_body
    assert 'from "../session/sessionUiAdapters.js"' in interactions_body
    assert 'from "../session/sessionEditorFlows.js"' in session_body
    assert 'from "../session/sessionTemplateFlows.js"' in session_body
    assert "ctx.store ||" not in session_body
    assert "ctx.selectors ||" not in session_body
    assert "ctx.services && ctx.services.session" not in session_body
    assert "createEditorSessionService" not in session_body
    assert "createTemplateCatalogService" not in session_body
    assert "createSubnetworkService" not in session_body
    assert "createEditorSelectors" not in session_body
    assert "createEditorStore" not in session_body
    assert "function createSessionEditorFlows(" in session_editor_body
    assert "function createSessionTemplateFlows(" in session_template_body
    assert "function createSessionUiAdapters(" in session_ui_body
    assert 'from "./plannerPanelBindings.js"' in planner_body
    assert "ctx.render()" not in planner_body
    assert "typeof ctx.togglePastInspection" not in planner_body
    assert "function createPlannerPanelBindings(" in planner_bindings_body
    assert 'from "./propertyAutosave.js"' in properties_body
    assert 'from "./propertyInvalidation.js"' in properties_body
    assert "function bindDebouncedAutosave(" not in properties_body
    assert "function createPropertyAutosaveBindings(" in property_autosave_body
    assert "function createPropertyInvalidationSupport(" in property_invalidation_body


def test_graph_assets_expose_fixed_tensor_edge_port_layers_and_selection_border(
    editor_server: EditorServer,
) -> None:
    graph_body = request_text(f"{editor_server.base_url}/js/graph/graphRender.js")
    graph_drag_body = request_text(
        f"{editor_server.base_url}/js/graph/graphRenderDrag.js"
    )
    graph_lifecycle_body = request_text(
        f"{editor_server.base_url}/js/graph/graphRenderLifecycle.js"
    )
    graph_model_body = request_text(
        f"{editor_server.base_url}/js/views/graphElementModel.js"
    )
    graph_diff_body = request_text(
        f"{editor_server.base_url}/js/views/graphModelDiff.js"
    )
    graph_adapter_body = request_text(
        f"{editor_server.base_url}/js/views/cytoscapeGraphAdapter.js"
    )
    utilities_body = request_utilities_runtime_bundle(editor_server)

    assert "const TENSOR_BASE_Z_INDEX = 10;" in graph_body
    assert "const EDGE_Z_INDEX = 100;" in graph_body
    assert "const PORT_BASE_Z_INDEX = 200;" in graph_body
    assert 'from "../core/theme.js"' in graph_body
    assert 'from "../views/graphElementModel.js"' in graph_body
    assert 'from "../views/cytoscapeGraphAdapter.js"' in graph_body
    assert 'from "./graphRenderLifecycle.js"' in graph_body
    assert 'from "./graphRenderDrag.js"' in graph_body
    assert "selector: \"node[kind = 'tensor']:selected\"" in graph_body
    assert '"border-width": 4' in graph_body
    assert '"border-color": GRAPH_THEME.selection' in graph_body
    assert "function renderGraph(" in graph_lifecycle_body
    assert "function syncPendingInteractionClasses(" in graph_lifecycle_body
    assert "function createTensorDragState(" in graph_drag_body
    assert "function moveCompanionTensorsDuringDrag(" in graph_drag_body
    assert "createGraphElementModelBuilder" in graph_model_body
    assert "buildGraphElementUpdatePlan" in graph_diff_body
    assert "createCytoscapeGraphAdapter" in graph_adapter_body
    assert '"overlay-opacity": 0' in graph_body
    assert "function cloneGraphElementDescriptor(" not in graph_body
    assert "function graphElementDataEqual(" not in graph_body
    assert "function graphElementDescriptorsEqual(" not in graph_body
    assert "function updateGraphRenderCache(" not in graph_body
    assert "function buildGraphElementModel(" not in graph_body
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

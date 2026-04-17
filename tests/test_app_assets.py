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
        "js/utilities.js",
        "js/utilitiesBase.js",
        "js/utilitiesGeometry.js",
        "js/utilitiesLayout.js",
        "js/utilitiesLinearPeriodic.js",
        "js/utilitiesSpec.js",
        "js/utilitiesUi.js",
    )


def request_interactions_runtime_bundle(editor_server: EditorServer) -> str:
    return request_runtime_bundle(
        editor_server,
        "js/actions/sessionCommands.js",
        "js/interactions.js",
        "js/interactionsCanvas.js",
        "js/interactionsEditor.js",
        "js/interactionsSession.js",
        "js/interactionsShortcuts.js",
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
    assert ">+<" in html


def test_root_groups_export_actions_and_code_generation_controls_as_requested(
    editor_server: EditorServer,
) -> None:
    html = request_text(f"{editor_server.base_url}/")

    assert 'id="export-format-select"' in html
    assert 'id="export-button"' in html
    assert 'id="export-py-button"' not in html
    assert 'id="export-png-button"' not in html
    assert 'id="export-svg-button"' not in html
    assert html.index('id="export-format-select"') < html.index('id="export-button"')

    code_pane_index = html.index('id="sidebar-pane-code"')
    engine_index = html.index('id="engine-select"')
    collection_index = html.index('id="collection-format-select"')
    generate_index = html.index('id="generate-button"')
    warning_index = html.index('id="code-generation-warning"')

    assert (
        code_pane_index
        < engine_index
        < collection_index
        < generate_index
        < warning_index
    )
    assert 'id="generated-code-view"' in html
    assert 'id="generated-code"' in html
    assert "/vendor/prism-core.min.js?v=" in html
    assert "/vendor/prism-python.min.js?v=" in html


def test_root_renders_done_and_cancel_as_icon_toolbar_actions(
    editor_server: EditorServer,
) -> None:
    html = request_text(f"{editor_server.base_url}/")

    assert 'id="done-button"' in html
    assert 'id="cancel-button"' in html
    assert 'aria-label="Done"' in html
    assert 'aria-label="Cancel"' in html
    assert ">Done<" not in html
    assert ">Cancel<" not in html


def test_root_exposes_linear_periodic_toolbar_controls(
    editor_server: EditorServer,
) -> None:
    html = request_text(f"{editor_server.base_url}/")

    assert 'id="toggle-linear-periodic-button"' in html
    assert 'id="linear-periodic-previous-cell-button"' in html
    assert 'id="linear-periodic-cell-label"' in html
    assert 'id="linear-periodic-next-cell-button"' in html
    assert ">For<" in html


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
    body = request_text(f"{editor_server.base_url}/js/plannerRenderers.js")

    assert '"FLOPs"' not in body
    assert '"MACs"' not in body
    assert '"FLOP"' in body
    assert '"MAC"' in body


def test_notes_and_planner_feature_modules_are_served(
    editor_server: EditorServer,
) -> None:
    notes_body = request_text(f"{editor_server.base_url}/js/notes.js")
    planner_body = request_text(f"{editor_server.base_url}/js/planner.js")
    planner_support_body = request_text(
        f"{editor_server.base_url}/js/plannerSupport.js"
    )
    planner_renderers_body = request_text(
        f"{editor_server.base_url}/js/plannerRenderers.js"
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
    registrar_body = request_text(f"{editor_server.base_url}/js/notesPlanner.js")
    utilities_body = request_text(f"{editor_server.base_url}/js/utilities.js")
    utilities_templates_body = request_text(
        f"{editor_server.base_url}/js/utilitiesTemplates.js"
    )

    assert "registerNotesFeature" in notes_body
    assert "registerPlannerFeature" in planner_body
    assert 'from "./plannerSupport.js"' in planner_body
    assert 'from "./plannerRenderers.js"' in planner_body
    assert 'from "./state/plannerSelectors.js"' in planner_support_body
    assert 'from "./actions/plannerCommands.js"' in planner_support_body
    assert 'from "./services/plannerAnalysisService.js"' in planner_support_body
    assert "createPlannerSupport" in planner_support_body
    assert "createPlannerRenderers" in planner_renderers_body
    assert "buildPlannerOperandState" in planner_selectors_body
    assert "createPlannerCommands" in planner_commands_body
    assert "createPlannerAnalysisService" in planner_service_body
    assert 'from "./notes.js"' in registrar_body
    assert 'from "./planner.js"' in registrar_body
    assert 'from "./utilitiesTemplates.js"' in utilities_body
    assert "createTemplateOptionHelpers" in utilities_templates_body


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


def test_interactions_asset_exposes_updated_keyboard_shortcuts(
    editor_server: EditorServer,
) -> None:
    body = request_interactions_runtime_bundle(editor_server)

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
    assert 'if (lowerKey === "f") {' in body
    assert "toggleLinearPeriodicMode();" in body


def test_overlays_asset_reuses_shared_tensor_size_helpers(
    editor_server: EditorServer,
) -> None:
    body = request_text(f"{editor_server.base_url}/js/overlaysLayoutTemplates.js")

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


def test_css_asset_styles_grouped_export_and_code_generation_controls(
    editor_server: EditorServer,
) -> None:
    body = request_text(f"{editor_server.base_url}/app.css")

    assert ".toolbar-export-controls {" in body
    assert ".toolbar-export-controls select {" in body
    assert ".toolbar-export-controls button {" in body
    assert ".code-header-controls {" in body
    assert ".code-header-controls .code-format-picker {" in body
    assert ".code-header-row {" in body
    assert ".code-preview {" in body
    assert ".code-preview .token.keyword {" in body
    assert ".code-preview .token.function {" in body


def test_sidebar_assets_expose_resize_handle(editor_server: EditorServer) -> None:
    html = request_text(f"{editor_server.base_url}/")
    css_body = request_text(f"{editor_server.base_url}/app.css")
    dom_body = request_text(f"{editor_server.base_url}/js/dom.js")
    sidebar_body = request_text(f"{editor_server.base_url}/js/sidebarTabs.js")

    assert 'id="sidebar-resize-handle"' in html
    assert 'role="separator"' in html
    assert "--sidebar-width: 360px;" in css_body
    assert ".sidebar-resize-handle {" in css_body
    assert (
        "grid-template-columns: minmax(0, 1fr) minmax(280px, var(--sidebar-width));"
        in css_body
    )
    assert (
        'sidebarResizeHandle: document.getElementById("sidebar-resize-handle")'
        in dom_body
    )
    assert "function setSidebarWidth(" in sidebar_body
    assert (
        'windowRef.addEventListener("mousemove", handleSidebarResizeMove);'
        in sidebar_body
    )


def test_properties_asset_exposes_total_element_summaries_and_icon_delete_controls(
    editor_server: EditorServer,
) -> None:
    overview_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersOverview.js"
    )
    overview_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesMarkup.js"
    )
    tensor_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersTensor.js"
    )
    entities_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersEntities.js"
    )
    entity_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/entityPropertiesMarkup.js"
    )
    support_body = request_text(f"{editor_server.base_url}/js/propertiesSupport.js")
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
    assert 'from "./properties/propertySummaries.js"' in support_body
    assert "function getSelectionTotalElementCount(" in summaries_body
    assert "function getTensorTotalElementCount(" in summaries_body


def test_properties_assets_use_compact_metadata_disclosures_and_tag_autocomplete(
    editor_server: EditorServer,
) -> None:
    overview_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersOverview.js"
    )
    overview_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesMarkup.js"
    )
    tensor_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersTensor.js"
    )
    entities_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersEntities.js"
    )
    entity_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/entityPropertiesMarkup.js"
    )
    support_body = request_text(f"{editor_server.base_url}/js/propertiesSupport.js")
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
    assert 'from "./properties/metadataEditors.js"' in support_body
    assert "Custom metadata (JSON)" in metadata_body
    assert "metadata-editor-disclosure" in metadata_body
    assert 'summaryLabel = "Metadata"' in metadata_body
    assert 'rows="1"' in metadata_body
    assert "properties-disclosure-chevron" in metadata_body
    assert "function buildTagAutocompleteSuggestions(" in metadata_body
    assert "function replaceActiveTagToken(" in metadata_body
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


def test_canvas_tool_assets_expose_floating_filter_search_and_highlight_hooks(
    editor_server: EditorServer,
) -> None:
    html_body = request_text(f"{editor_server.base_url}/")
    dom_body = request_text(f"{editor_server.base_url}/js/dom.js")
    main_body = request_text(f"{editor_server.base_url}/js/main.js")
    filter_body = request_text(f"{editor_server.base_url}/js/metadataFilters.js")
    graph_body = request_text(f"{editor_server.base_url}/js/graphRender.js")
    minimap_body = request_text(f"{editor_server.base_url}/js/exportMinimap.js")
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert 'id="canvas-tools"' in html_body
    assert 'id="canvas-context-menu-root"' in html_body
    assert 'id="metadata-filters-panel"' not in html_body
    assert 'canvasTools: document.getElementById("canvas-tools")' in dom_body
    assert (
        'canvasContextMenuRoot: document.getElementById("canvas-context-menu-root")'
        in dom_body
    )
    assert 'from "./metadataFilters.js"' in main_body
    assert "registerMetadataFilters(context);" in main_body
    assert "canvas-metadata-filter-button" in filter_body
    assert "canvas-name-search-button" in filter_body
    assert "canvas-metadata-filter-clear-button" in filter_body
    assert "canvas-metadata-filter-select-all-button" in filter_body
    assert "canvas-metadata-filter-select-none-button" in filter_body
    assert "Not specified" in filter_body
    assert "canvas-name-search-input" in filter_body
    assert '"bond"' in filter_body
    assert "function getMetadataFilterHighlight(" in filter_body
    assert "metadata-filter-dim" in graph_body
    assert "getMetadataFilterEntityState" in graph_body
    assert "getMetadataFilterEntityState" in minimap_body
    assert ".canvas-tool-popover" in css_body
    assert "bottom: calc(100% +" in css_body
    assert "transform: rotate(90deg)" in css_body
    assert ".metadata-editor-disclosure" in css_body
    assert "overflow: visible;" in css_body


def test_canvas_context_menu_assets_expose_minimal_selection_actions(
    editor_server: EditorServer,
) -> None:
    main_body = request_text(f"{editor_server.base_url}/js/main.js")
    context_menu_body = request_text(
        f"{editor_server.base_url}/js/canvasContextMenu.js"
    )
    graph_body = request_text(f"{editor_server.base_url}/js/graphRender.js")
    overlays_body = request_text(
        f"{editor_server.base_url}/js/overlaysLayoutTemplates.js"
    )

    assert 'from "./canvasContextMenu.js"' in main_body
    assert "registerCanvasContextMenu(context);" in main_body
    assert "function openCanvasContextMenu(" in context_menu_body
    assert 'id="context-menu-name-input"' in context_menu_body
    assert 'id="context-menu-add-index-button"' in context_menu_body
    assert 'id="context-menu-dimension-input"' in context_menu_body
    assert 'id="context-menu-move-up-button"' in context_menu_body
    assert 'id="context-menu-move-down-button"' in context_menu_body
    assert 'id="context-menu-toggle-group-button"' in context_menu_body
    assert 'state.cy.on("cxttap"' in graph_body
    assert 'addEventListener("contextmenu"' in overlays_body


def test_properties_renderer_assets_are_split_by_selection_family(
    editor_server: EditorServer,
) -> None:
    facade_body = request_text(f"{editor_server.base_url}/js/propertiesRenderers.js")
    overview_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersOverview.js"
    )
    tensor_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersTensor.js"
    )
    entities_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersEntities.js"
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
    assert 'from "./properties/overviewPropertiesMarkup.js"' in overview_body
    assert 'from "./properties/overviewPropertiesBindings.js"' in overview_body
    assert 'from "./properties/entityPropertiesMarkup.js"' in entities_body
    assert 'from "./properties/entityPropertiesBindings.js"' in entities_body
    assert 'from "./properties/tensorPropertiesStandard.js"' in tensor_body
    assert 'from "./properties/tensorPropertiesBoundary.js"' in tensor_body
    assert 'from "./properties/tensorPropertiesContraction.js"' in tensor_body
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
    properties_body = request_text(f"{editor_server.base_url}/js/properties.js")
    overview_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersOverview.js"
    )
    entities_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersEntities.js"
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
    assert "ctx.applyDesignChange(" not in overview_body
    assert "ctx.applyDesignChange(" not in entities_body
    assert "ctx.removeEdge(" not in entities_body
    assert "ctx.removeNote(" not in entities_body
    assert 'from "./actions/propertyCommands.js"' in properties_body
    assert "function renameNetwork(" in commands_body
    assert "function applySelectionColor(" in commands_body
    assert "function addIndexToSelectedTensors(" in commands_body
    assert "function renameGroup(" in commands_body
    assert "function deleteGroup(" in commands_body
    assert "function renameEdge(" in commands_body
    assert "function deleteEdge(" in commands_body
    assert "function updateNoteText(" in commands_body
    assert "function deleteNote(" in commands_body


def test_tensor_property_assets_delegate_rendering_and_mutations_to_internal_modules(
    editor_server: EditorServer,
) -> None:
    tensor_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersTensor.js"
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

    assert "function renderTensorProperties(" not in tensor_body
    assert "function renderLinearPeriodicBoundaryTensorProperties(" not in tensor_body
    assert "function renderContractionTensorProperties(" not in tensor_body
    assert "function renderContractionIndexProperties(" not in tensor_body
    assert "ctx.applyDesignChange(" not in tensor_body
    assert "typeof ctx.syncConnectedIndexDimension" not in tensor_body
    assert "function renderTensorProperties(" in tensor_standard_body
    assert (
        "function renderLinearPeriodicBoundaryTensorProperties(" in tensor_boundary_body
    )
    assert "function renderContractionTensorProperties(" in tensor_contraction_body
    assert "function renderContractionIndexProperties(" in tensor_contraction_body


def test_index_disclosure_border_uses_the_port_color(
    editor_server: EditorServer,
) -> None:
    standard_body = request_text(
        f"{editor_server.base_url}/js/properties/tensorPropertiesStandard.js"
    )
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert "--index-border-color:" in standard_body
    assert "ctx.getIndexColor(index, isConnected)" in standard_body
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
    assert "commands.deleteCurrentSelection" in body


def test_note_assets_move_note_editing_into_canvas(
    editor_server: EditorServer,
) -> None:
    notes_body = request_text(f"{editor_server.base_url}/js/notes.js")
    properties_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersEntities.js"
    )
    properties_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/entityPropertiesMarkup.js"
    )
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert 'textarea.addEventListener("keydown", (event) => {' in notes_body
    assert "event.stopPropagation();" in notes_body
    assert 'className = "canvas-note-color-button"' in notes_body
    assert 'colorInput.type = "color";' in notes_body
    assert "ctx.bindDebouncedAutosave(" in notes_body
    assert (
        '<label for="note-text-input">Note text</label>'
        in properties_body + properties_markup_body
    )
    assert 'id="note-color-input"' in properties_body + properties_markup_body
    assert "Edit this note directly on the canvas." not in properties_body
    assert ".canvas-note-color-button {" in css_body


def test_note_assets_tint_the_full_note_frame_and_avoid_rerendering_text_edits(
    editor_server: EditorServer,
) -> None:
    notes_body = request_text(f"{editor_server.base_url}/js/notes.js")
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert "invalidate: noteInvalidation({ overlays: false })" in notes_body
    assert 'frame.style.setProperty("--note-accent-color"' in notes_body
    assert "--note-surface-color" in notes_body
    assert "var(--note-accent-color" in css_body
    assert "var(--note-surface-color" in css_body


def test_collapsed_note_assets_leave_a_small_grab_margin_around_the_toggle(
    editor_server: EditorServer,
) -> None:
    notes_body = request_text(f"{editor_server.base_url}/js/notes.js")
    css_body = request_text(f"{editor_server.base_url}/app.css")
    constants_body = request_text(f"{editor_server.base_url}/js/constants.js")
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
    planner_body = request_text(f"{editor_server.base_url}/js/plannerSupport.js")
    planner_commands_body = request_text(
        f"{editor_server.base_url}/js/actions/plannerCommands.js"
    )
    graph_body = request_text(f"{editor_server.base_url}/js/graphRender.js")
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
    assert "const indexNodesInteractive = !readOnlyScene;" in graph_body
    assert "selectable: !readOnlyScene," in graph_body
    assert "ctx.ensureContractionViewSnapshots();" in utilities_body


def test_toolbar_assets_route_export_actions_through_a_single_picker_and_button(
    editor_server: EditorServer,
) -> None:
    bootstrap_body = request_text(f"{editor_server.base_url}/js/bootstrap.js")
    shell_bindings_body = request_text(
        f"{editor_server.base_url}/js/shell/editorShellBindings.js"
    )
    dom_body = request_text(f"{editor_server.base_url}/js/dom.js")
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
    assert 'exportButton: document.getElementById("export-button")' in dom_body
    assert 'from "./shell/editorShellBindings.js"' in bootstrap_body
    assert (
        'bindListener(exportButton, "click", actions.downloadSelectedExport);'
        in shell_bindings_body
    )
    assert "async function downloadSelectedExport()" in interactions_body
    assert "switch (exportFormatSelect.value)" in interactions_body
    assert 'case "py":' in interactions_body
    assert 'case "png":' in interactions_body
    assert 'case "svg":' in interactions_body
    assert "await downloadPythonExport();" in interactions_body
    assert "actions.downloadPngExport();" in interactions_body
    assert "actions.downloadSvgExport();" in interactions_body
    assert "exportButton.disabled =" in utilities_body


def test_template_insertion_assets_refresh_lookups_and_anchor_new_contract_operands(
    editor_server: EditorServer,
) -> None:
    interactions_body = request_interactions_runtime_bundle(editor_server)
    contraction_body = request_text(f"{editor_server.base_url}/js/contractionScene.js")

    assert "invalidate: { lookups: true }" in interactions_body
    assert "ctx.ensureSpecLookups()" in contraction_body
    assert "state.tensorById[anchorTensorId] || null" in contraction_body


def test_contraction_scene_assets_route_progression_and_snapshots_through_state_modules(
    editor_server: EditorServer,
) -> None:
    contraction_body = request_text(f"{editor_server.base_url}/js/contractionScene.js")

    assert 'from "./state/contractionSceneProgression.js"' in contraction_body
    assert 'from "./state/contractionSceneSnapshots.js"' in contraction_body
    assert "function cloneOperand(" not in contraction_body
    assert "function analyzeOperandPair(" not in contraction_body
    assert (
        "function buildContractionOperandProgressionUncached(" not in contraction_body
    )
    assert "function buildSnapshotLayoutMap(" not in contraction_body


def test_subnetwork_assets_expose_import_export_controls_and_routes(
    editor_server: EditorServer,
) -> None:
    html = request_text(f"{editor_server.base_url}/")
    dom_body = request_text(f"{editor_server.base_url}/js/dom.js")
    shell_bindings_body = request_text(
        f"{editor_server.base_url}/js/shell/editorShellBindings.js"
    )
    interactions_body = request_interactions_runtime_bundle(editor_server)
    overview_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersOverview.js"
    )
    overview_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesMarkup.js"
    )
    entities_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersEntities.js"
    )
    entity_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/entityPropertiesMarkup.js"
    )

    assert 'id="insert-subnetwork-button"' in html
    assert 'id="reflow-imported-button"' in html
    assert 'id="subnetwork-load-input"' in html
    assert (
        'insertSubnetworkButton: document.getElementById("insert-subnetwork-button")'
        in dom_body
    )
    assert (
        'reflowImportedButton: document.getElementById("reflow-imported-button")'
        in dom_body
    )
    assert (
        'subnetworkLoadInput: document.getElementById("subnetwork-load-input")'
        in dom_body
    )
    assert (
        'bindListener(insertSubnetworkButton, "click", actions.openSubnetworkPicker);'
        in shell_bindings_body
    )
    assert (
        'bindListener(subnetworkLoadInput, "change", actions.loadSubnetworkFromFile);'
        in shell_bindings_body
    )
    assert '"/api/subnetwork/extract"' in interactions_body
    assert '"/api/subnetwork/prepare-insert"' in interactions_body
    assert '"/api/template/promote"' in interactions_body
    assert 'id="extract-selection-button"' in overview_body + overview_markup_body
    assert (
        'id="promote-selection-template-button"' in overview_body + overview_markup_body
    )
    assert 'id="extract-group-button"' in entities_body + entity_markup_body
    assert 'id="promote-group-template-button"' in entities_body + entity_markup_body


def test_template_management_assets_expose_toolbar_controls_and_routes(
    editor_server: EditorServer,
) -> None:
    html = request_text(f"{editor_server.base_url}/")
    dom_body = request_text(f"{editor_server.base_url}/js/dom.js")
    bootstrap_flow_body = request_text(
        f"{editor_server.base_url}/js/shell/editorBootstrapFlow.js"
    )
    shell_bindings_body = request_text(
        f"{editor_server.base_url}/js/shell/editorShellBindings.js"
    )
    interactions_body = request_interactions_runtime_bundle(editor_server)
    utilities_ui_body = request_text(f"{editor_server.base_url}/js/utilitiesUi.js")

    assert re.search(
        r'<button id="insert-template-button"[^>]*>\s*\+\s*</button>',
        html,
    )
    assert re.search(
        r'<button id="insert-subnetwork-button"[^>]*>\s*\+ subnetwork\s*</button>',
        html,
    )
    assert re.search(
        r'<button id="rename-template-button"[^>]*>\s*Rename\s*</button>',
        html,
    )
    assert re.search(
        r'<button id="reflow-imported-button"[^>]*>\s*Reflow\s*</button>',
        html,
    )
    assert 'id="rename-template-button"' in html
    assert 'id="delete-template-button"' in html
    assert 'id="template-catalog-warning"' in html
    assert 'aria-label="Delete template"' in html
    assert (
        'renameTemplateButton: document.getElementById("rename-template-button")'
        in dom_body
    )
    assert (
        'deleteTemplateButton: document.getElementById("delete-template-button")'
        in dom_body
    )
    assert (
        'templateCatalogWarning: document.getElementById("template-catalog-warning")'
        in dom_body
    )
    assert (
        'bindListener(renameTemplateButton, "click", actions.renameSelectedTemplate);'
        in shell_bindings_body
    )
    assert (
        'bindListener(deleteTemplateButton, "click", actions.deleteSelectedTemplate);'
        in shell_bindings_body
    )
    assert (
        "templateCatalogWarnings: payload.template_catalog_warnings"
        in bootstrap_flow_body
    )
    assert (
        'actions.setStatus(state.templateCatalogWarnings[0], "error");'
        in bootstrap_flow_body
    )
    assert '"/api/template/rename"' in interactions_body
    assert '"/api/template/delete"' in interactions_body
    assert "function syncTemplateCatalogWarning()" in utilities_ui_body


def test_layout_assets_expose_selection_alignment_distribution_and_snap_helpers(
    editor_server: EditorServer,
) -> None:
    utilities_body = request_utilities_runtime_bundle(editor_server)
    utilities_module_body = request_text(f"{editor_server.base_url}/js/utilities.js")
    layout_body = request_text(f"{editor_server.base_url}/js/utilitiesLayout.js")
    overview_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersOverview.js"
    )
    overview_markup_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesMarkup.js"
    )

    assert 'from "./utilitiesLayout.js"' in utilities_module_body
    assert "createUtilityLayoutBindings" in layout_body
    assert "function alignSelectedTensors(" in layout_body
    assert "function arrangeSelectedTensors(" in layout_body
    assert "function distributeSelectedTensors(" in layout_body
    assert "function snapSelectedTensorsToGrid(" in layout_body
    assert "function reflowLastImportedTensors(" in layout_body
    assert "GRID_SNAP_SIZE" in utilities_body
    assert 'id="align-selection-left-button"' in overview_body + overview_markup_body
    assert 'id="arrange-selection-chain-button"' in overview_body + overview_markup_body
    assert 'id="arrange-selection-tree-button"' in overview_body + overview_markup_body
    assert 'id="arrange-selection-grid-button"' in overview_body + overview_markup_body
    assert (
        'id="distribute-selection-horizontal-button"'
        in overview_body + overview_markup_body
    )
    assert 'id="snap-selection-button"' in overview_body + overview_markup_body


def test_performance_sensitive_assets_use_lightweight_analysis_paths(
    editor_server: EditorServer,
) -> None:
    planner_body = request_text(f"{editor_server.base_url}/js/planner.js")
    planner_support_body = request_text(
        f"{editor_server.base_url}/js/plannerSupport.js"
    )
    planner_service_body = request_text(
        f"{editor_server.base_url}/js/services/plannerAnalysisService.js"
    )
    interactions_body = request_interactions_runtime_bundle(editor_server)
    utilities_body = request_utilities_runtime_bundle(editor_server)
    minimap_body = request_text(f"{editor_server.base_url}/js/exportMinimap.js")
    overlays_body = request_text(
        f"{editor_server.base_url}/js/overlaysLayoutTemplates.js"
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
    history_body = request_text(f"{editor_server.base_url}/js/historySelection.js")
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
    utilities_spec_body = request_text(f"{editor_server.base_url}/js/utilitiesSpec.js")
    spec_normalization_body = request_text(
        f"{editor_server.base_url}/js/spec/specNormalization.js"
    )
    spec_lookups_body = request_text(f"{editor_server.base_url}/js/spec/specLookups.js")
    spec_mutations_body = request_text(
        f"{editor_server.base_url}/js/spec/specMutations.js"
    )
    state_body = request_text(f"{editor_server.base_url}/js/state.js")
    notes_body = request_text(f"{editor_server.base_url}/js/notes.js")
    properties_body = request_text(f"{editor_server.base_url}/js/propertiesSupport.js")
    properties_renderers_body = request_text(
        f"{editor_server.base_url}/js/propertiesRenderersOverview.js"
    )
    properties_bindings_body = request_text(
        f"{editor_server.base_url}/js/properties/overviewPropertiesBindings.js"
    )
    interactions_body = request_interactions_runtime_bundle(editor_server)
    graph_body = request_text(f"{editor_server.base_url}/js/graphRender.js")
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
    assert 'from "./state/historySnapshots.js"' in history_body
    assert 'from "./state/selectionEntries.js"' in history_body
    assert 'from "./actions/designMutationPipeline.js"' in history_body
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
    assert 'from "./spec/specNormalization.js"' in utilities_spec_body
    assert 'from "./spec/specLookups.js"' in utilities_spec_body
    assert 'from "./spec/specMutations.js"' in utilities_spec_body
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
    assert "return state.noteById[noteId] || null;" in notes_body
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
    body = request_text(f"{editor_server.base_url}/js/propertiesRenderersTensor.js")

    assert "ctx.isLinearPeriodicBoundaryTensor(tensor)" in body
    assert "renderLinearPeriodicBoundaryTensorProperties" in body
    assert "managed by For mode" not in body
    assert "Move this port directly on the canvas to adjust its position." not in body


def test_linear_periodic_assets_propagate_interface_dimensions_across_cells(
    editor_server: EditorServer,
) -> None:
    history_body = request_text(f"{editor_server.base_url}/js/historySelection.js")
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
    support_body = request_text(f"{editor_server.base_url}/js/propertiesSupport.js")
    spec_mutations_body = request_text(
        f"{editor_server.base_url}/js/spec/specMutations.js"
    )

    assert 'from "./properties/propertyInvalidation.js"' in support_body
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
    planner_body = request_text(f"{editor_server.base_url}/js/plannerRenderers.js")
    planner_formatting_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerAnalysisFormatting.js"
    )
    css_body = request_text(f"{editor_server.base_url}/app.css")

    assert "Total elements" in planner_body + planner_formatting_body
    assert 'from "./planner/plannerAnalysisFormatting.js"' in planner_body
    assert "function getShapeElementCount(" in planner_formatting_body
    assert "planner-manual-step-list" in planner_body
    assert ".planner-manual-step-list {" in css_body
    assert "border-top:" in css_body


def test_editor_shell_assets_split_session_ui_bindings_and_property_helpers(
    editor_server: EditorServer,
) -> None:
    interactions_body = request_text(f"{editor_server.base_url}/js/interactions.js")
    session_body = request_text(f"{editor_server.base_url}/js/interactionsSession.js")
    session_editor_body = request_text(
        f"{editor_server.base_url}/js/session/sessionEditorFlows.js"
    )
    session_template_body = request_text(
        f"{editor_server.base_url}/js/session/sessionTemplateFlows.js"
    )
    session_ui_body = request_text(
        f"{editor_server.base_url}/js/session/sessionUiAdapters.js"
    )
    planner_body = request_text(f"{editor_server.base_url}/js/plannerRenderers.js")
    planner_bindings_body = request_text(
        f"{editor_server.base_url}/js/planner/plannerPanelBindings.js"
    )
    properties_body = request_text(f"{editor_server.base_url}/js/propertiesSupport.js")
    property_autosave_body = request_text(
        f"{editor_server.base_url}/js/properties/propertyAutosave.js"
    )
    property_invalidation_body = request_text(
        f"{editor_server.base_url}/js/properties/propertyInvalidation.js"
    )

    assert "store: ctx.store" in interactions_body
    assert "selectors: ctx.selectors" in interactions_body
    assert "services: ctx.services" in interactions_body
    assert 'from "./session/sessionUiAdapters.js"' in interactions_body
    assert 'from "./session/sessionEditorFlows.js"' in session_body
    assert 'from "./session/sessionTemplateFlows.js"' in session_body
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
    assert 'from "./planner/plannerPanelBindings.js"' in planner_body
    assert "ctx.render()" not in planner_body
    assert "typeof ctx.togglePastInspection" not in planner_body
    assert "function createPlannerPanelBindings(" in planner_bindings_body
    assert 'from "./properties/propertyAutosave.js"' in properties_body
    assert 'from "./properties/propertyInvalidation.js"' in properties_body
    assert "function bindDebouncedAutosave(" not in properties_body
    assert "function createPropertyAutosaveBindings(" in property_autosave_body
    assert "function createPropertyInvalidationSupport(" in property_invalidation_body


def test_graph_assets_expose_fixed_tensor_edge_port_layers_and_selection_border(
    editor_server: EditorServer,
) -> None:
    graph_body = request_text(f"{editor_server.base_url}/js/graphRender.js")
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
    assert 'from "./views/graphElementModel.js"' in graph_body
    assert 'from "./views/cytoscapeGraphAdapter.js"' in graph_body
    assert "selector: \"node[kind = 'tensor']:selected\"" in graph_body
    assert '"border-width": 4' in graph_body
    assert '"border-color": "#8bc2ff"' in graph_body
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

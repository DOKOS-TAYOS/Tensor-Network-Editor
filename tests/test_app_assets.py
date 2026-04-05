from __future__ import annotations

import unittest

from tensor_network_editor.app.server import EditorServer
from tensor_network_editor.app.session import EditorSession
from tensor_network_editor.models import EngineName
from tests.app_support import request_text, request_with_headers
from tests.test_api import build_sample_spec


class AppAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM,
        )
        self.server = EditorServer(self.session)
        self.server.start()
        self.addCleanup(self.server.stop)

    def test_root_serves_editor_shell_with_module_entry(self) -> None:
        html = request_text(f"{self.server.base_url}/")

        self.assertIn("Tensor Network Editor", html)
        self.assertIn('type="module"', html)
        self.assertIn("/js/main.js", html)
        self.assertIn("?v=", html)
        self.assertIn('id="generate-button"', html)
        self.assertIn('id="template-select"', html)
        self.assertIn('id="create-group-button"', html)
        self.assertIn('id="add-note-button"', html)
        self.assertIn('id="notes-layer"', html)
        self.assertIn('id="sidebar-panel"', html)
        self.assertIn('id="sidebar-tabs"', html)
        self.assertIn('id="sidebar-tab-selection"', html)
        self.assertIn('id="sidebar-tab-planner"', html)
        self.assertIn('id="sidebar-tab-code"', html)
        self.assertIn('id="sidebar-pane-selection"', html)
        self.assertIn('id="sidebar-pane-planner"', html)
        self.assertIn('id="sidebar-pane-code"', html)
        self.assertIn('id="planner-panel"', html)
        self.assertIn('id="help-modal"', html)
        self.assertIn('id="minimap"', html)
        self.assertNotIn('id="status-message"', html)
        self.assertNotIn("/app.js?v=", html)

    def test_frontend_module_entry_is_served_from_js_subdirectory(self) -> None:
        script_body = request_text(f"{self.server.base_url}/js/main.js")

        self.assertIn("createEditorContext", script_body)
        self.assertIn("registerUtilities", script_body)
        self.assertIn("registerGraphRender", script_body)
        self.assertIn("registerNotesPlanner", script_body)
        self.assertIn("startEditor", script_body)

    def test_frontend_modules_are_split_by_responsibility(self) -> None:
        history_body = request_text(f"{self.server.base_url}/js/historySelection.js")
        graph_body = request_text(f"{self.server.base_url}/js/graphRender.js")
        properties_body = request_text(f"{self.server.base_url}/js/properties.js")
        interactions_body = request_text(f"{self.server.base_url}/js/interactions.js")
        export_body = request_text(f"{self.server.base_url}/js/exportMinimap.js")
        overlays_body = request_text(
            f"{self.server.base_url}/js/overlaysLayoutTemplates.js"
        )
        notes_planner_body = request_text(f"{self.server.base_url}/js/notesPlanner.js")
        sidebar_tabs_body = request_text(f"{self.server.base_url}/js/sidebarTabs.js")
        utilities_body = request_text(f"{self.server.base_url}/js/utilities.js")

        self.assertIn("registerHistorySelection", history_body)
        self.assertIn("function setSelection", history_body)
        self.assertNotIn('setActiveSidebarTab("selection")', history_body)
        self.assertIn("registerGraphRender", graph_body)
        self.assertIn("function initGraph", graph_body)
        self.assertIn("registerProperties", properties_body)
        self.assertIn("function renderProperties", properties_body)
        self.assertNotIn("Current size:", properties_body)
        self.assertNotIn(
            "Resize from the corner handles on the canvas.", properties_body
        )
        self.assertIn("registerInteractions", interactions_body)
        self.assertIn("function handleKeydown", interactions_body)
        self.assertIn("registerExportMinimap", export_body)
        self.assertIn("function renderMinimap", export_body)
        self.assertIn("registerOverlaysLayoutTemplates", overlays_body)
        self.assertIn("function createGroupFromSelection", overlays_body)
        self.assertIn("registerNotesPlanner", notes_planner_body)
        self.assertIn("function renderNotes", notes_planner_body)
        self.assertIn("function renderPlanner", notes_planner_body)
        self.assertIn("function resolvePlannerOperandId", notes_planner_body)
        self.assertIn("Automatic global", notes_planner_body)
        self.assertIn("Automatic local", notes_planner_body)
        self.assertIn("Network output shape", notes_planner_body)
        self.assertIn("Preview", notes_planner_body)
        self.assertIn("Accept", notes_planner_body)
        self.assertIn("MACs", notes_planner_body)
        self.assertIn("estimated_macs", notes_planner_body)
        self.assertIn("Contract", notes_planner_body)
        self.assertIn("stepOrdersByTensorId", notes_planner_body)
        self.assertIn("step.step_id", notes_planner_body)
        self.assertNotIn("firstStepByTensorId", notes_planner_body)
        self.assertIn("toggle-note-collapse", notes_planner_body)
        self.assertIn("canvas-note-resize-handle", notes_planner_body)
        self.assertNotIn("Refresh", notes_planner_body)
        self.assertNotIn("Remove Last", notes_planner_body)
        self.assertNotIn(
            "Manual mode uses clicks on tensors and previously created steps.",
            notes_planner_body,
        )
        self.assertNotIn("Global automatic preview active.", notes_planner_body)
        self.assertNotIn("<h3>Intermediates</h3>", notes_planner_body)
        self.assertIn("same contracted operand", notes_planner_body)
        self.assertIn("registerSidebarTabs", sidebar_tabs_body)
        self.assertIn("function setActiveSidebarTab", sidebar_tabs_body)
        self.assertIn("registerUtilities", utilities_body)
        self.assertIn("function normalizeSpec", utilities_body)
        self.assertIn("planner-order-badge", overlays_body)
        self.assertIn("planner-preview-badge", overlays_body)
        self.assertIn("planner-order-badge-stack", overlays_body)
        self.assertIn("is-preview", overlays_body)
        self.assertIn("showingPreview", overlays_body)
        self.assertIn("planner-pending-tensor", graph_body)
        self.assertIn("planner-pending-index", graph_body)
        self.assertIn("activeNoteResize", interactions_body)

    def test_vendor_asset_is_served_locally(self) -> None:
        asset_body = request_text(f"{self.server.base_url}/vendor/cytoscape.min.js")

        self.assertIn("cytoscape", asset_body)

    def test_frontend_uses_dark_color_scheme(self) -> None:
        css_body = request_text(f"{self.server.base_url}/app.css")

        self.assertIn("color-scheme: dark", css_body)
        self.assertIn("height: 100dvh", css_body)
        self.assertIn("overflow: hidden", css_body)
        self.assertIn(".minimap-shell", css_body)
        self.assertIn(".code-output-shell", css_body)
        self.assertIn(".help-modal", css_body)
        self.assertIn(".canvas-note", css_body)
        self.assertIn(".planner-card", css_body)
        self.assertIn(".planner-step", css_body)
        self.assertIn(".sidebar-tabs", css_body)
        self.assertIn(".sidebar-tab", css_body)
        self.assertIn(".sidebar-pane", css_body)
        self.assertIn(".sidebar-pane[hidden]", css_body)
        self.assertIn("padding: 0.48rem 0.72rem", css_body)
        self.assertRegex(
            css_body,
            r"\.planner-summary-grid\s*\{[^}]*grid-template-columns: 1fr;",
        )
        self.assertRegex(
            css_body,
            r"\.planner-chip-grid\s*\{[^}]*grid-template-columns: 1fr;",
        )
        self.assertRegex(
            css_body,
            r"\.planner-intermediate-list\s*\{[^}]*grid-template-columns: 1fr;",
        )
        self.assertIn(".planner-disclosure", css_body)
        self.assertIn(".planner-network-output", css_body)
        self.assertIn(".planner-order-badge", css_body)
        self.assertIn(".planner-preview-badge", css_body)
        self.assertIn(".planner-order-badge-stack", css_body)
        self.assertIn(".planner-order-badge-stack.is-preview", css_body)
        self.assertIn(".planner-pending-tensor", css_body)
        self.assertIn(".planner-pending-index", css_body)
        self.assertIn(".canvas-note.is-collapsed", css_body)
        self.assertIn(".canvas-note-resize-handle", css_body)

    def test_help_modal_mentions_current_editor_capabilities(self) -> None:
        html = request_text(f"{self.server.base_url}/")

        self.assertIn("copy and paste", html)
        self.assertIn("Create Group", html)
        self.assertIn("Contract planner", html)
        self.assertIn("Ctrl/Cmd+C", html)
        self.assertIn("Ctrl/Cmd+V", html)
        self.assertIn("Resize tensors from the corner handles", html)

    def test_static_assets_disable_browser_cache(self) -> None:
        _, headers = request_with_headers(f"{self.server.base_url}/js/main.js")

        self.assertIn("no-store", headers["Cache-Control"])


if __name__ == "__main__":
    unittest.main()

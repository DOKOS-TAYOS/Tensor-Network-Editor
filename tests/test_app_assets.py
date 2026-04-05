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
        self.assertIn('id="help-modal"', html)
        self.assertIn('id="minimap"', html)
        self.assertNotIn('id="status-message"', html)
        self.assertNotIn('/app.js?v=', html)

    def test_frontend_module_entry_is_served_from_js_subdirectory(self) -> None:
        script_body = request_text(f"{self.server.base_url}/js/main.js")

        self.assertIn("createEditorContext", script_body)
        self.assertIn("registerUtilities", script_body)
        self.assertIn("registerGraphRender", script_body)
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
        utilities_body = request_text(f"{self.server.base_url}/js/utilities.js")

        self.assertIn("registerHistorySelection", history_body)
        self.assertIn("function setSelection", history_body)
        self.assertIn("registerGraphRender", graph_body)
        self.assertIn("function initGraph", graph_body)
        self.assertIn("registerProperties", properties_body)
        self.assertIn("function renderProperties", properties_body)
        self.assertIn("registerInteractions", interactions_body)
        self.assertIn("function handleKeydown", interactions_body)
        self.assertIn("registerExportMinimap", export_body)
        self.assertIn("function renderMinimap", export_body)
        self.assertIn("registerOverlaysLayoutTemplates", overlays_body)
        self.assertIn("function createGroupFromSelection", overlays_body)
        self.assertIn("registerUtilities", utilities_body)
        self.assertIn("function normalizeSpec", utilities_body)

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

    def test_static_assets_disable_browser_cache(self) -> None:
        _, headers = request_with_headers(f"{self.server.base_url}/js/main.js")

        self.assertIn("no-store", headers["Cache-Control"])


if __name__ == "__main__":
    unittest.main()

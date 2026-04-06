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
    body = request_text(f"{editor_server.base_url}/js/notesPlanner.js")

    assert '"FLOPs"' not in body
    assert '"MACs"' not in body
    assert '"FLOP"' in body
    assert '"MAC"' in body


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

    assert "#canvas {" in body
    assert "z-index: 0;" in body
    assert "#group-layer {" in body
    assert "z-index: 10;" in body
    assert "#notes-layer {" in body
    assert "z-index: 30;" in body


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

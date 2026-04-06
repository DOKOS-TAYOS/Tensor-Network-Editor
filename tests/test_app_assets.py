from __future__ import annotations

import pytest

from tensor_network_editor.app.server import EditorServer
from tests.app_support import request_with_headers


def test_root_serves_editor_shell_with_versioned_module_entry(
    editor_server: EditorServer,
) -> None:
    html, headers = request_with_headers(f"{editor_server.base_url}/")

    assert "Tensor Network Editor" in html
    assert 'type="module"' in html
    assert "/js/main.js?v=" in html
    assert 'id="collection-format-select"' in html
    assert 'id="sidebar-toggle-button"' in html
    assert headers["Content-Type"].startswith("text/html")


def test_main_module_is_served_from_static_directory(
    editor_server: EditorServer,
) -> None:
    body, headers = request_with_headers(f"{editor_server.base_url}/js/main.js")

    assert body.strip()
    assert "startEditor" in body
    assert headers["Content-Type"].startswith("application/javascript")


def test_vendor_asset_is_served_locally(editor_server: EditorServer) -> None:
    body, headers = request_with_headers(
        f"{editor_server.base_url}/vendor/cytoscape.min.js"
    )

    assert "cytoscape" in body
    assert headers["Content-Type"].startswith("application/javascript")


@pytest.mark.parametrize("path", ["/", "/js/main.js", "/vendor/cytoscape.min.js"])
def test_static_assets_disable_browser_cache(
    editor_server: EditorServer,
    path: str,
) -> None:
    _, headers = request_with_headers(f"{editor_server.base_url}{path}")

    assert "no-store" in headers["Cache-Control"]
    assert headers["Pragma"] == "no-cache"
    assert headers["Expires"] == "0"

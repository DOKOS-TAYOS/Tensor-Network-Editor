from __future__ import annotations

import importlib
import os
from typing import Any

import pytest

from tensor_network_editor.app.server import EditorServer

pytestmark = pytest.mark.browser


def test_editor_shell_loads_in_real_browser(editor_server: EditorServer) -> None:
    if os.environ.get("TNE_RUN_BROWSER_SMOKE") != "1":
        pytest.skip("Set TNE_RUN_BROWSER_SMOKE=1 to run the browser smoke test.")
    try:
        sync_api = importlib.import_module("playwright.sync_api")
    except ModuleNotFoundError:
        pytest.fail(
            "The browser smoke test requires the playwright package and a "
            "Chromium browser installation.",
            pytrace=False,
        )

    with sync_api.sync_playwright() as playwright:
        browser: Any = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(editor_server.base_url, wait_until="domcontentloaded")
            page.locator("#add-tensor-button").wait_for(state="visible", timeout=5000)
            page.locator("#canvas").wait_for(state="visible", timeout=5000)

            assert page.title() == "Tensor Network Editor"
            assert page.locator("#engine-select").input_value() == "einsum_numpy"
            assert page.locator("#generated-code").count() == 1
        finally:
            browser.close()

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pytest

from tensor_network_editor.app.server import EditorServer
from tensor_network_editor.app.session import EditorSession

pytestmark = pytest.mark.browser


def _browser_e2e_enabled() -> bool:
    """Return whether the real-browser E2E suite is explicitly enabled."""
    return (
        os.environ.get("TNE_RUN_BROWSER_E2E") == "1"
        or os.environ.get("TNE_RUN_BROWSER_SMOKE") == "1"
    )


def _require_browser_e2e_enabled() -> None:
    """Skip when the explicit real-browser opt-in is not enabled."""
    if not _browser_e2e_enabled():
        pytest.skip(
            "Set TNE_RUN_BROWSER_E2E=1 to run the real-browser editor E2E tests."
        )


def _import_playwright_sync_api() -> Any:
    """Import the Playwright sync API or fail with an actionable message."""
    try:
        return importlib.import_module("playwright.sync_api")
    except ModuleNotFoundError:
        pytest.fail(
            "The browser E2E tests require the playwright package and a "
            "Chromium browser installation.",
            pytrace=False,
        )


def _request_json(url: str) -> dict[str, Any]:
    """Read one JSON response from a local editor server URL."""
    with urlopen(url, timeout=1) as response:
        return json.load(response)


def _wait_for_recoverable_draft_name(
    draft_url: str,
    expected_name: str,
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Wait until the autosave endpoint exposes the edited draft name."""
    deadline = time.monotonic() + timeout_seconds
    latest_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest_payload = _request_json(draft_url)
        draft = latest_payload.get("draft")
        if isinstance(draft, dict):
            spec = draft.get("spec")
            if isinstance(spec, dict):
                network = spec.get("network")
                if isinstance(network, dict) and network.get("name") == expected_name:
                    return latest_payload
        time.sleep(0.05)
    pytest.fail(
        "The browser editor did not autosave the expected draft name. "
        f"Latest /api/draft payload: {latest_payload!r}",
        pytrace=False,
    )


def test_editor_shell_loads_in_real_browser(editor_server: EditorServer) -> None:
    _require_browser_e2e_enabled()
    sync_api = _import_playwright_sync_api()

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


def test_editor_autosaves_recoverable_draft_after_mutation(tmp_path: Path) -> None:
    _require_browser_e2e_enabled()
    sync_api = _import_playwright_sync_api()
    draft_path = tmp_path / "editor-draft.json"
    session = EditorSession(draft_path=draft_path)
    server = EditorServer(session)
    server.start()
    try:
        with sync_api.sync_playwright() as playwright:
            browser: Any = playwright.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.goto(server.base_url, wait_until="domcontentloaded")
                page.locator("#network-name-input").wait_for(
                    state="visible", timeout=5000
                )
                expected_name = "Recoverable Browser Draft"
                network_name_input = page.locator("#network-name-input")
                network_name_input.fill(expected_name)
                network_name_input.press("Enter")
                draft_payload = _wait_for_recoverable_draft_name(
                    f"{server.base_url}/api/draft",
                    expected_name,
                )

                assert draft_payload["ok"] is True
                assert draft_payload["draft"]["spec"]["network"]["name"] == (
                    "Recoverable Browser Draft"
                )
                assert draft_path.is_file()
            finally:
                browser.close()
    finally:
        server.stop()


def test_editor_can_generate_code_and_complete_session(
    editor_server: EditorServer,
) -> None:
    _require_browser_e2e_enabled()
    sync_api = _import_playwright_sync_api()
    initial_tensor_count = len(editor_server.session.initial_spec.tensors)

    with sync_api.sync_playwright() as playwright:
        browser: Any = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(editor_server.base_url, wait_until="domcontentloaded")
            page.locator("#add-tensor-button").click()
            page.locator("#tensor-name-input").wait_for(state="visible", timeout=5000)
            page.locator("#sidebar-tab-code").click()
            page.locator("#generate-button").click()
            page.locator("#generated-code-view").wait_for(state="visible", timeout=5000)
            page.wait_for_function(
                """
                () => {
                  const node = document.getElementById("generated-code-view");
                  return Boolean(node && node.textContent && node.textContent.trim());
                }
                """,
                timeout=5000,
            )
            generated_code = page.locator("#generated-code-view").text_content()
            assert generated_code is not None
            assert generated_code.strip()

            page.locator("#file-menu-button").click()
            page.locator("#close-with-info-menu-item").click()

            result = editor_server.session.wait_for_result(timeout=5)
            assert result is not None
            assert result.codegen is not None
            assert len(result.spec.tensors) == initial_tensor_count + 1
        finally:
            browser.close()

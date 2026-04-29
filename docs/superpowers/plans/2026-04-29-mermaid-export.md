# Mermaid Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Mermaid export format that users can generate from Python, the CLI, and the browser editor for documentation-friendly tensor-network diagrams.

**Architecture:** Extend the existing static rendering family with a new text renderer in `rendering.py`, then thread the new format through the `/api/render` backend route, the CLI `render` subcommand, and the browser export UI. Reuse `DotRenderOptions` for label toggles, keep the Mermaid output structure-oriented rather than geometry-oriented, and degrade gracefully for notes and groups.

**Tech Stack:** Python 3.12, typed package exports, argparse CLI, browser editor JavaScript, pytest, pyright, ruff

---

### Task 1: Add the core Mermaid renderer with TDD

**Files:**
- Modify: `C:\Users\aleja\Documents\draw_to_tensor_network\src\tensor_network_editor\rendering.py`
- Modify: `C:\Users\aleja\Documents\draw_to_tensor_network\src\tensor_network_editor\__init__.py`
- Test: `C:\Users\aleja\Documents\draw_to_tensor_network\tests\test_rendering.py`
- Test: `C:\Users\aleja\Documents\draw_to_tensor_network\tests\test_api.py`

- [ ] **Step 1: Write the failing renderer tests**

Add focused tests to `tests/test_rendering.py` for:

```python
def test_render_spec_mermaid_returns_flowchart_for_normal_network() -> None:
    mermaid = render_spec_mermaid(build_sample_spec())

    assert mermaid.startswith("flowchart LR\n")
    assert 'tensor_tensor_a["A"]' in mermaid
    assert 'tensor_tensor_b["B"]' in mermaid
    assert 'tensor_tensor_a <-->|"bond_x / x=3"| tensor_tensor_b' in mermaid


def test_render_spec_mermaid_can_hide_tensor_index_and_bond_labels() -> None:
    mermaid = render_spec_mermaid(
        build_sample_spec(),
        options=DotRenderOptions(
            show_tensor_labels=False,
            show_index_labels=False,
            show_edge_labels=False,
        ),
    )

    assert 'tensor_tensor_a["tensor_a"]' in mermaid
    assert 'tensor_tensor_b["tensor_b"]' in mermaid
    assert 'bond_x' not in mermaid
    assert 'x=3' not in mermaid


def test_render_spec_mermaid_includes_hyperedges_groups_and_notes() -> None:
    spec = build_three_tensor_hyperedge_spec()
    mermaid = render_spec_mermaid(spec)

    assert "subgraph group_demo [Demo Group]" in mermaid
    assert 'hyperedge_h["shared_h"]' in mermaid
    assert '%% Note: Check the contraction order' in mermaid


def test_render_spec_mermaid_writes_output_path(tmp_path: Path) -> None:
    output_path = tmp_path / "network.mmd"

    mermaid = render_spec_mermaid(build_sample_spec(), output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == mermaid
```

Add a public API coverage check in `tests/test_api.py` similar to the existing static render exports:

```python
assert tensor_network_editor.render_spec_mermaid is render_spec_mermaid
```

- [ ] **Step 2: Run the renderer tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests\test_rendering.py -k mermaid`

Expected: FAIL because `render_spec_mermaid` does not exist yet.

- [ ] **Step 3: Write the minimal Mermaid renderer**

Implement in `src/tensor_network_editor/rendering.py`:

```python
def render_spec_mermaid(
    spec: NetworkSpec,
    *,
    options: DotRenderOptions | None = None,
    output_path: StrPath | None = None,
) -> str:
    ...
```

Add a small renderer class and helpers that:
- emit `flowchart LR`
- create safe Mermaid ids from existing stable ids
- render tensors, pairwise edges, open indices, hyperedge hubs, groups, and note comments
- reuse existing DOT label logic where possible
- write UTF-8 text when `output_path` is provided

Then export it from `__all__` and the lazy exports in `src/tensor_network_editor/__init__.py`.

- [ ] **Step 4: Run the renderer tests to verify they pass**

Run:
- `.\.venv\Scripts\python -m pytest tests\test_rendering.py -k mermaid`
- `.\.venv\Scripts\python -m pytest tests\test_api.py -k render_spec_mermaid`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tensor_network_editor/rendering.py src/tensor_network_editor/__init__.py tests/test_rendering.py tests/test_api.py
git commit -m "Add Mermaid renderer"
```

### Task 2: Integrate Mermaid into the backend route and CLI with TDD

**Files:**
- Modify: `C:\Users\aleja\Documents\draw_to_tensor_network\src\tensor_network_editor\cli.py`
- Modify: `C:\Users\aleja\Documents\draw_to_tensor_network\src\tensor_network_editor\app\routes.py`
- Modify: `C:\Users\aleja\Documents\draw_to_tensor_network\src\tensor_network_editor\internal\cli\_cli_handlers.py`
- Modify: `C:\Users\aleja\Documents\draw_to_tensor_network\src\tensor_network_editor\internal\cli\_cli_parser.py`
- Test: `C:\Users\aleja\Documents\draw_to_tensor_network\tests\test_cli.py`
- Test: `C:\Users\aleja\Documents\draw_to_tensor_network\tests\test_app_routes.py`

- [ ] **Step 1: Write the failing integration tests**

Add route coverage in `tests/test_app_routes.py`:

```python
def test_render_route_returns_mermaid_export(editor_server: EditorServer) -> None:
    spec = build_sample_spec()
    serialized_spec = {"schema_version": SCHEMA_VERSION, "network": spec.to_dict()}

    payload = request_json(
        f"{editor_server.base_url}/api/render",
        method="POST",
        payload={"format": "mermaid", "spec": serialized_spec},
    )

    assert payload["format"] == "mermaid"
    assert payload["content_type"] == "text/plain;charset=utf-8"
    assert payload["text"].startswith("flowchart LR\n")
```

Add CLI coverage in `tests/test_cli.py`:

```python
def test_render_subcommand_writes_mermaid_output(sample_spec: NetworkSpec) -> None:
    with (
        patch("tensor_network_editor.cli.load_spec", return_value=sample_spec),
        patch("tensor_network_editor.cli.render_spec_mermaid", return_value="flowchart LR\n"),
    ):
        exit_code = main(
            ["render", "saved-network.json", "--format", "mermaid", "--output", "graph.mmd"]
        )

    assert exit_code == 0


def test_render_subcommand_prints_mermaid_when_no_output(
    sample_spec: NetworkSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch("tensor_network_editor.cli.load_spec", return_value=sample_spec),
        patch("tensor_network_editor.cli.render_spec_mermaid", return_value="flowchart LR\n"),
    ):
        exit_code = main(["render", "saved-network.json", "--format", "mermaid"])

    assert exit_code == 0
    assert capsys.readouterr().out == "flowchart LR\n\n"
```

- [ ] **Step 2: Run the integration tests to verify they fail**

Run:
- `.\.venv\Scripts\python -m pytest tests\test_app_routes.py -k mermaid`
- `.\.venv\Scripts\python -m pytest tests\test_cli.py -k mermaid`

Expected: FAIL because the route and CLI do not accept `mermaid` yet.

- [ ] **Step 3: Wire Mermaid into the route and CLI**

Add `render_spec_mermaid` imports and format branches in:
- `src/tensor_network_editor/cli.py`
- `src/tensor_network_editor/internal/cli/_cli_handlers.py`
- `src/tensor_network_editor/internal/cli/_cli_parser.py`
- `src/tensor_network_editor/app/routes.py`

Use:

```python
content_type = "text/plain;charset=utf-8"
```

Use `.mmd` as the expected file extension in user-facing messages and downloads.

- [ ] **Step 4: Run the integration tests to verify they pass**

Run:
- `.\.venv\Scripts\python -m pytest tests\test_app_routes.py -k mermaid`
- `.\.venv\Scripts\python -m pytest tests\test_cli.py -k mermaid`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tensor_network_editor/cli.py src/tensor_network_editor/app/routes.py src/tensor_network_editor/internal/cli/_cli_handlers.py src/tensor_network_editor/internal/cli/_cli_parser.py tests/test_cli.py tests/test_app_routes.py
git commit -m "Integrate Mermaid export into CLI and API"
```

### Task 3: Add Mermaid to the browser editor and docs with TDD

**Files:**
- Modify: `C:\Users\aleja\Documents\draw_to_tensor_network\src\tensor_network_editor\app\static\index.html`
- Modify: `C:\Users\aleja\Documents\draw_to_tensor_network\src\tensor_network_editor\app\static\js\core\dom.js`
- Modify: `C:\Users\aleja\Documents\draw_to_tensor_network\src\tensor_network_editor\app\static\js\shell\editorShellBindings.js`
- Modify: `C:\Users\aleja\Documents\draw_to_tensor_network\src\tensor_network_editor\app\static\js\session\sessionEditorFlows.js`
- Modify: `C:\Users\aleja\Documents\draw_to_tensor_network\README.md`
- Modify: `C:\Users\aleja\Documents\draw_to_tensor_network\CHANGELOG.md`
- Test: `C:\Users\aleja\Documents\draw_to_tensor_network\tests\test_app_assets.py`
- Test: `C:\Users\aleja\Documents\draw_to_tensor_network\tests\test_frontend_architecture.py`
- Test: `C:\Users\aleja\Documents\draw_to_tensor_network\tests\test_frontend_runtime.py`

- [ ] **Step 1: Write the failing editor and docs tests**

Add asset checks in `tests/test_app_assets.py` for:
- `id="export-mermaid-menu-item"`
- `<option value="mermaid">Mermaid</option>`
- Mermaid listed in help text when export formats are enumerated
- DOM wiring for `exportMermaidMenuItem`

Add runtime coverage in `tests/test_frontend_runtime.py` similar to the existing academic export flow:

```javascript
await flows.downloadExportAs("mermaid");
```

Then assert:
- one `renderSpec` call with `payload.format === "mermaid"`
- one text download for `draft_demo.mmd`
- `contentType === "text/plain;charset=utf-8"`

- [ ] **Step 2: Run the editor tests to verify they fail**

Run:
- `.\.venv\Scripts\python -m pytest tests\test_app_assets.py -k mermaid`
- `.\.venv\Scripts\python -m pytest tests\test_frontend_runtime.py -k mermaid`

Expected: FAIL because the editor does not expose Mermaid yet.

- [ ] **Step 3: Implement the editor export wiring**

Update:
- `index.html` to add a Mermaid export menu item and selector option
- `dom.js` to expose `exportMermaidMenuItem`
- `editorShellBindings.js` to bind `downloadExportAs("mermaid")`
- `sessionEditorFlows.js` to add Mermaid to `exportDetails`, use `.mmd`, and route it through text download

Also update `README.md` and `CHANGELOG.md` to mention Mermaid export support.

- [ ] **Step 4: Run the editor tests to verify they pass**

Run:
- `.\.venv\Scripts\python -m pytest tests\test_app_assets.py -k mermaid`
- `.\.venv\Scripts\python -m pytest tests\test_frontend_runtime.py -k mermaid`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tensor_network_editor/app/static/index.html src/tensor_network_editor/app/static/js/core/dom.js src/tensor_network_editor/app/static/js/shell/editorShellBindings.js src/tensor_network_editor/app/static/js/session/sessionEditorFlows.js README.md CHANGELOG.md tests/test_app_assets.py tests/test_frontend_architecture.py tests/test_frontend_runtime.py
git commit -m "Add Mermaid export to browser editor"
```

### Task 4: Final verification and cleanup

**Files:**
- Verify only

- [ ] **Step 1: Run focused pytest coverage**

Run:
- `.\.venv\Scripts\python -m pytest tests\test_rendering.py`
- `.\.venv\Scripts\python -m pytest tests\test_api.py -k render_spec_mermaid`
- `.\.venv\Scripts\python -m pytest tests\test_cli.py -k mermaid`
- `.\.venv\Scripts\python -m pytest tests\test_app_routes.py -k mermaid`
- `.\.venv\Scripts\python -m pytest tests\test_app_assets.py -k mermaid`
- `.\.venv\Scripts\python -m pytest tests\test_frontend_runtime.py -k mermaid`

Expected: PASS

- [ ] **Step 2: Run Python quality checks**

Run:
- `.\.venv\Scripts\python -m ruff check . --fix`
- `.\.venv\Scripts\python -m ruff format .`

Expected: PASS

- [ ] **Step 3: Run type checking and note pre-existing failures separately**

Run:
- `.\.venv\Scripts\python -m pyright`

Expected: either PASS or only the already-known unrelated failures in:
- `tests/test_app_routes.py`
- `tests/test_app_server.py`
- `tests/test_session.py`

- [ ] **Step 4: Commit any final cleanup**

```bash
git add README.md CHANGELOG.md src tests
git commit -m "Polish Mermaid export integration"
```

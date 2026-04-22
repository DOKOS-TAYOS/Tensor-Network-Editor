# Hyperedge Context Menu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a right-click mini menu for hyperedges that opens from the hub or any spoke and exposes name, color, metadata, and delete actions.

**Architecture:** Extend the existing canvas context-menu resolver so `hyperedge-hub` and `hyperedge-spoke` normalize to one canonical hyperedge target. Reuse the current hyperedge command surface and metadata editor helpers so the new menu behaves like the sidebar properties, not like a separate editing system.

**Tech Stack:** Python test harness, JavaScript canvas context-menu modules, Cytoscape graph events, pytest, pyright, ruff.

---

### Task 1: Add Failing Coverage for Hyperedge Context Menus

**Files:**
- Modify: `tests/test_frontend_architecture.py`
- Modify: `tests/test_app_assets.py`
- Test: `tests/test_frontend_architecture.py`
- Test: `tests/test_app_assets.py`

- [ ] **Step 1: Write the failing architecture test**

Add a hyperedge fixture to the existing context-menu architecture test and assert that opening a menu for a hyperedge exposes the new controls and dispatches hyperedge-specific commands.

- [ ] **Step 2: Run the focused architecture test and verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_frontend_architecture.py -k context_menu -v`
Expected: FAIL because the current context-menu system does not resolve `hyperedge` targets or render hyperedge controls.

- [ ] **Step 3: Write the failing asset assertions**

Add asset checks for the new hyperedge menu ids, target resolver wiring, and graph right-click acceptance of hyperedge hub/spoke elements.

- [ ] **Step 4: Run the focused asset test and verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_app_assets.py -k context_menu -v`
Expected: FAIL because the static assets do not yet include the hyperedge context-menu controls and bindings.

### Task 2: Implement Hyperedge Context-Menu Resolution and UI

**Files:**
- Modify: `src/tensor_network_editor/app/static/js/graph/graphRender.js`
- Modify: `src/tensor_network_editor/app/static/js/graph/canvasContextMenu.js`
- Modify: `src/tensor_network_editor/app/static/js/graph/canvasContextMenuTargets.js`
- Modify: `src/tensor_network_editor/app/static/js/graph/canvasContextMenuMarkup.js`
- Modify: `src/tensor_network_editor/app/static/js/graph/canvasContextMenuBindings.js`

- [ ] **Step 1: Accept hyperedge right-click entry points**

Extend the graph `cxttap` handler so `hyperedge-hub` and `hyperedge-spoke` open the context menu and normalize spokes to the hub selection id.

- [ ] **Step 2: Add a hyperedge target resolver**

Pass `findHyperedgeById` into the canvas context-menu module and add `kind: "hyperedge"` resolution with canonical id, target object, and derived color.

- [ ] **Step 3: Render the hyperedge menu**

Add a hyperedge-specific renderer that shows:

```text
Name
Color
Tags
Custom metadata
Delete
```

- [ ] **Step 4: Bind the hyperedge actions**

Wire the new menu to:

```text
renameHyperedge(...)
updateTargetColor(...)
bindInlineMetadataEditor(...)
deleteHyperedge(...)
```

- [ ] **Step 5: Re-run the focused tests**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_frontend_architecture.py -k context_menu -v
.\.venv\Scripts\python -m pytest tests/test_app_assets.py -k context_menu -v
```

Expected: PASS.

### Task 3: Final Verification and Cleanup

**Files:**
- Modify: `CHANGELOG.md` only if implementation changes user-visible behavior beyond the already documented design intent

- [ ] **Step 1: Re-check whether `CHANGELOG.md` needs an update**

If the existing hyperedge entry already covers this right-click-menu addition clearly, do not add noise. If it does not, add one short line.

- [ ] **Step 2: Run repository-required formatting**

Run:

```powershell
.\.venv\Scripts\python -m ruff check . --fix
.\.venv\Scripts\python -m ruff format .
```

Expected: exit code 0.

- [ ] **Step 3: Run verification**

Run:

```powershell
.\.venv\Scripts\python -m pytest
pyright
```

Expected:
- pytest: PASS
- pyright: 0 errors

- [ ] **Step 4: Review git diff**

Run: `git diff --stat`
Expected: only the planned context-menu and test files changed.

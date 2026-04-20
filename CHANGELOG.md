# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed

- Importing `tensor_network_editor` now resolves the public headless/API exports lazily, so the package root loads faster and avoids importing analysis, template, diffing, linting, and editor helpers until they are first accessed.
- Editor sessions now reuse a shared in-process cache of static asset bytes and the rendered `index.html`, defer contraction analysis until the planner actually needs it, cache serialized specs and analysis results by spec revision, and load Prism syntax-highlighting assets on demand instead of during the initial page load.

### Fixed

- Checkout-based test runs now skip installed-distribution metadata assertions when `importlib.metadata` resolves a different package installation than the active `src/` checkout.
- Frontend runtime contract/regression tests now copy the full `utilities.js` dependency set, including the tree-periodic utility module, and the bootstrap architecture fixture no longer hardcodes a stale app version.
- Frontend runtime test scaffolding now reuses named local JS dependency presets for shortcut, interaction, utility, and layout/subnetwork runtime scripts, reducing repeated copy boilerplate without changing test behavior.
- Tree-periodic codegen renderers now include the missing type annotations needed for the targeted CLI `mypy` regression check, and the linear-periodic headless analysis regression test now matches the current single-analysis flow.

## [0.3.0] - 2026-04-20

### Changed

- Major structural refactor across the project (layout, modules, and responsibilities reorganized for clarity and maintainability).
- Several rounds of performance work: repeated profiling, hot-path optimizations, and follow-up tuning after each pass.
- Contraction analysis and code generation now avoid repeated validation on already-checked specs, prepared networks cache tensor lookups by id, and the editor defers planner analysis refreshes plus visible-scene fallback scans until they are actually needed.
- The editor now uses a tighter dark IDE visual language with neutral charcoal surfaces, restrained purple focus/selection accents, shared graph/minimap theme tokens, sans-serif panel titles, and more compact radii across panels, notes, planner cards, and dialogs.
- Close and discard icon buttons now keep a stable appearance on mouse hover while still showing a visible keyboard focus state.
- The toolbar `Reflow` control now opens a layout popover and applies the chosen layout action to the current multi-tensor selection instead of only targeting the last imported template or subnetwork.
- Multi-selection and group workflows now expose a more consistent action set across the sidebar and right-click menus, including extraction, template promotion, index insertion, color changes, grouping, and deletion.
- The template manager uses a more compact icon-based delete action, the About panel now includes the project support link on YouTube, and the close icon styling in the help/about dialogs is now white for better contrast.
- The README header now displays the transparent project logo using the repository-hosted image, while documentation images remain excluded from the published sdist and wheel artifacts.
- Added a first complete `For bidimensional` mode across the typed model, validation, serialization, frontend editor flow, and code-generation registry, with dedicated `grid_periodic_grid` / `grid_periodic_role` payloads alongside the existing 1D `For` mode.
- The toolbar now exposes four-way cell navigation for bidimensional `For` mode, keeps the active-cell label to the left of the navigation controls, and disables contraction planning and benchmark actions while a 2D `For` grid is active.

### Fixed

- The wheel smoke test now clears checkout imports before loading `tensor_network_editor`, so packaging validation checks the installed wheel instead of the local `src/` tree.
- Right-clicking a tensor that already belongs to a multi-selection now preserves that selection and opens the selection action menu instead of collapsing to a single-tensor context menu.
- Grid-periodic frontend/runtime flows now preserve the active 2D cell when returning to `Single`, seed the center cell from the current graph when entering 2D mode, and keep payload synchronization stable across save/load/history/codegen paths without executing expensive generated contractions in tests.
- Editor sessions now keep the confirmed result if a late cancellation request arrives after completion.

## [0.2.2] - 2026-04-13

### Changed

- Performance improvements and project restructuring.

### Fixed

- Many minor bug fixes across the editor and tooling.

## [0.2.1] - 2026-04-12

### Changed

- The editor and CLI now default the code-generation engine to `tensorkrowch`, and the Generate Code engine picker now shows engines in the order `TensorKrowch`, `PyTorch einsum`, `NumPy einsum`, `Quimb`, `TensorNetwork`.
- The toolbar keeps export actions grouped behind a single export-format picker plus `Export` button, while the Generate Code panel keeps the engine picker inside the code workflow.

### Fixed

- For mode now keeps repeated index-dimension edits stable across updates, propagates interface dimensions more reliably between initial, periodic, and final cells, and synchronizes connected-port dimensions automatically when one side changes.
- Linear periodic `For` mode now supports manual `Previous cell` / `Next cell` carry steps even when each cell is only partially contracted; generated code threads the selected carry operand to the next cell and preserves the remaining tensors in the output network.
- Notes now tint the whole note frame correctly without stealing focus while you type, and template insertion in contraction mode no longer collapses new tensors into one point.
- Many additional bug fixes and minor improvements across the editor and tooling.

## [0.2.0] - 2026-04-07

### Added

- Many new editor/CLI commands.
- Generated export code now includes contraction steps for the current scheme (not only tensor wiring).

### Changed

- Improved controls, forms, and menus; various UI elements relocated for clearer layout and behavior.

### Fixed

- Assorted bug fixes across the editor and tooling.

## [0.1.2] - 2026-04-05

### Added

- `analyze_contraction()`, `ContractionAnalysisResult`, and related summaries in `_contraction_analysis.py`: validates manual `contraction_plan` steps, reports pairwise costs and completeness, and computes automatic global/local greedy contraction paths; wired to `/api/analyze-contraction` for the in-app contract planner. Export/code generation does not yet consume these analysis results.
- Tests for contraction analysis and the `/api/analyze-contraction` route.

### Changed

- Editor UI: grouping controls, layout refinements, and the contraction planner surfaced in the static client.
- Replaced the single `einsum` target with explicit `einsum_numpy` and `einsum_torch` engines across the API, CLI, editor bootstrap payload, and generated code.
- HTTP layer refactored around shared JSON/spec helpers (`_protocol`); validation responses return structured issues and a normalized spec snapshot.

### Fixed

- Built-in templates can now be inserted with configurable graph size, bond dimension, and physical dimension while keeping the existing template catalog.
- Built-in templates now expose the expected tensor valences and open legs instead of starting from a generic four-port tensor shape.
- Built-in templates: expected tensor valences, open legs, and wiring for each catalog layout instead of incorrect or generic default shapes.

## [0.1.1] - 2026-04-05

### Added

- Template parameter metadata in the bootstrap payload so the editor can show graph size, bond dimension, and physical dimension controls per template.
- Editor layout templates (MPS, MPO, 2×2 PEPS, MERA, binary tree) with `/api/template` and template names exposed in the bootstrap payload.
- Tensor groups in the save model (`GroupSpec` / `groups` on `NetworkSpec`) for organizing nodes on the canvas.
- Per-tensor canvas size (`TensorSize`) and per-index label offsets on `IndexSpec`.
- `THIRD_PARTY_LICENSES` for bundled frontend assets (Cytoscape.js), included in the sdist/wheel and asserted in CI smoke tests.
- `scripts/clean.sh` for removing build and cache artifacts on Unix-like systems (documented in the README).
- `analyze_network()` and `NetworkAnalysis` for deriving connected versus open indices and edge endpoint lookups from a `NetworkSpec`.
- Tests covering analysis, templates, validation/model edges, HTTP API behavior, packaging metadata, and cleanup scripts.

### Changed

- Substantial editor UI refresh: layout, styling, and graph interactions in the static web client.
- Stricter deserialization via dedicated payload coercion helpers; packaging manifest lists third-party license text.

### Fixed

- New notes open taller, keep their text area inside the card, participate in right-drag box selection, and move together with selected tensors or groups.
- README, example code, and in-app help now reflect the split einsum engines, corrected templates, and current editor interactions.

## [0.1.0] - 2026-04-03

### Added

- Initial public release metadata and package layout for PyPI.
- Development tooling with `pytest`, `ruff`, `mypy`, `build`, and `twine`.
- Cross-platform GitHub Actions checks for Windows and Linux on Python 3.11 and 3.12.

### Changed

- Kept a single installable package for the library, CLI, and local editor.
- Added structured logging and clearer package-specific exceptions for I/O and serialization failures.
- Simplified the public API by keeping `launch_tensor_network_editor` as the canonical editor entry point.

### Fixed

- Request handling now returns `400` for malformed JSON payloads instead of generic server failures.
- Unexpected server-side errors are logged internally and exposed as generic `500` responses.

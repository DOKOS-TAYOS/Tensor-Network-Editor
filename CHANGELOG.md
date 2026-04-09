# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed

- The editor and CLI now default the code-generation engine to `tensorkrowch`, and the Generate Code engine picker now shows engines in the order `TensorKrowch`, `PyTorch einsum`, `NumPy einsum`, `Quimb`, `TensorNetwork`.
- The toolbar keeps export actions grouped behind a single export-format picker plus `Export` button, while the Generate Code panel keeps the engine picker inside the code workflow.

### Fixed

- For mode now keeps repeated index-dimension edits stable across updates, propagates interface dimensions more reliably between initial, periodic, and final cells, and synchronizes connected-port dimensions automatically when one side changes.
- Notes now tint the whole note frame correctly without stealing focus while you type, and template insertion in contraction mode no longer collapses new tensors into one point.

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

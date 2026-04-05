# Changelog

All notable changes to this project will be documented in this file.

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
- Replaced the single `einsum` target with explicit `einsum_numpy` and `einsum_torch` engines across the API, CLI, editor bootstrap payload, and generated code.
- HTTP layer refactored around shared JSON/spec helpers (`_protocol`); validation responses return structured issues and a normalized spec snapshot.
- Stricter deserialization via dedicated payload coercion helpers; packaging manifest lists third-party license text.

### Fixed

- Built-in templates can now be inserted with configurable graph size, bond dimension, and physical dimension while keeping the existing template catalog.
- Built-in templates now expose the expected tensor valences and open legs instead of starting from a generic four-port tensor shape.
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

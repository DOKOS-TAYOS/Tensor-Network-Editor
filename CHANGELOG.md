# Changelog

All notable changes to this project will be documented in this file.

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

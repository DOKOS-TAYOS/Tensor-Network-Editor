# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Contributor and citation metadata now live in `CONTRIBUTING.md` and
  `CITATION.cff`.
- Real-browser editor E2E coverage now goes beyond shell startup: CI runs the
  browser marker on Linux and Windows, and the suite now checks recoverable
  draft autosave plus full session completion with generated code.
- Academic figure exports now support PDF alongside SVG, PNG, TikZ/LaTeX, and
  Graphviz/DOT through the browser editor, render API, CLI, and public Python
  helpers.
- Pillow is now a base runtime dependency instead of an optional extra because
  academic PNG/PDF export is part of the core rendering surface.
- Practical example scripts now cover MPS template code generation, PEPS
  TikZ/DOT rendering, first-class hyperedges, contraction benchmarking, and
  tensor initializers without requiring optional backend packages.
- Public `NetworkBuilder`, `TensorHandle`, and `IndexHandle` helpers now make
  it easier to build normal-mode `NetworkSpec` objects from Python without
  hand-writing ids and endpoint references.
- Built-in template parameters now come from catalog-defined field metadata in
  the backend and browser editor, so templates can expose integer, numeric,
  boolean, and choice controls without hardcoded UI wiring.
- The `mpo` template now absorbs the old `transverse_ising_mpo` role, adds
  `boundary_condition`, `j`, and `h` parameters, and supports structural
  periodic wiring plus matching template metadata.
- The `ttn` template now absorbs the old `binary_tree` role, uses `depth` as
  its public size parameter, and adds `leaf_physical_legs`, `root_open_leg`,
  and `isometric` configuration options.
- Built-in templates now include `tebd_gate_layer`, while `mps` still covers
  common 1D physics starting points through configurable boundary, symmetry,
  and initialization presets.
- Saved designs can now be rendered to academic text formats with TikZ/LaTeX
  and Graphviz/DOT through the public rendering API and `render` CLI command.
- The browser editor's File menu can now export the current canvas directly as
  TikZ/LaTeX (`.tex`) or Graphviz/DOT (`.dot`) using the same academic
  renderers.
- TikZ/LaTeX and Graphviz/DOT renders now draw tensors as circles and let users
  independently show or hide tensor, index, and bond names.
- Academic SVG, PNG, PDF, and TikZ exports now share the same tensor-network
  diagram style: tensors render as circles, dangling ports render as plain legs
  instead of port circles, and the tensor/index/bond label toggles now apply to
  every academic export mode.
- TikZ/LaTeX figure exports now emit tensor and port geometry before using it in
  edges, so generated diagrams compile reliably instead of referencing missing
  node shapes.
- TikZ/LaTeX figure exports now define a single `\tneGlobalWidth` control
  (defaulting to `\linewidth`) and express coordinates, node sizes, note
  widths, and line widths relative to that scale so exported figures resize
  consistently from one place.
- The editor now closes through explicit File-menu actions (`Close with info`
  and `Close without info`) instead of top-right session buttons, and browser
  refreshes no longer auto-cancel the session and tear down the local server.
- Editor sessions now wait with short polling intervals so `Ctrl+C` remains
  responsive on Windows terminals while the browser editor is open.
- `tensor-network-editor doctor` now adds more actionable suggestions for
  suspicious model structure, available backend choices, incomplete manual
  plans, and manual paths that are clearly more expensive than `auto_full`.
- Browser editor sessions now keep a project-local recoverable draft under
  `.tensor-network-editor/drafts/`, prompt to restore it on startup, and clear
  it only after explicit save, Done, Cancel, or Start fresh actions.

### Changed

- Browser editor startup now makes the template catalog available before
  recoverable draft loading finishes, and preloads the largest startup scripts
  so the initial toolbar becomes interactive sooner.
- Non-dark browser editor themes now use theme-aware semantic UI tokens for
  planner highlights, accent buttons, sidebar tabs, code controls, benchmark
  emphasis, and reusable-subnetwork previews so light, contrast, colorblind,
  and shiny modes stay visually consistent instead of inheriting dark-only
  surfaces.
- Light and colorblind-friendly themes now keep pale hover states readable,
  repaint tensor colors immediately after a theme switch, draw default tensor
  cards with dark borders, and show bond labels with light backgrounds instead
  of stale dark-theme styling.
- `opt_einsum` is now a required runtime dependency instead of the optional
  `planner` extra, so automatic planner suggestions and automatic benchmark
  rows are available from the base installation.
- Documentation now states that TenPy support is out of scope, symbolic tensor
  support stays limited to the current portable model, and the known
  `tensorkrowch` export restriction is a stable project boundary rather than a
  planned future expansion.
- Documentation now treats a future `pywebview` integration as a desktop layer
  on top of the existing local browser-served editor, not a replacement for the
  current core interface.
- Browser editor sessions can now start with `dark`, `light`, `contrast`,
  `colorblind`, or `shiny` color themes through `EditorLaunchOptions.theme`,
  `open_editor(theme=...)`, or `tensor-network-editor edit --theme`.
- The browser editor now exposes a dedicated `Theme` menu so users can switch
  themes live in the UI and keep their preferred palette for later sessions.
- Tensor initializers now support explicit zeros, identity/delta, generalized
  copy tensors, seeded normal/uniform random values, portable dtype choices,
  and JSON-friendly complex scalars across saved specs and generated code.
- Tensor initializers now support external `.npy` and `.npz` data references,
  including editor fields, validation, generated NumPy loading helpers, CLI
  path anchoring, and generated-source round-trips.
- External tensor initializers now also support safe `.pt` tensor references,
  including optional mapping keys, backend-specific generated loaders, and
  generated-source round-trips.
- Saved designs can now be rendered headlessly to SVG from the public
  `tensor_network_editor.rendering` API or the `tensor-network-editor render`
  CLI command, without requiring a browser or Node runtime.
- Saved designs can now be rendered headlessly to PNG through the public
  rendering API and CLI when the optional `png` extra is installed.
- Generated Python for linear, grid, and tree periodic modes now embeds
  compact round-trip metadata so supported imports recover the editable
  periodic-mode payload instead of treating those exports as one-way artifacts.
- The CLI now includes `tensor-network-editor doctor`, a friendly diagnostic
  command that combines validation, lint, analysis, benchmark summaries, and
  backend/extra availability checks in text or JSON.
- Built-in templates now include `ttn` and `pepo` for common tree and
  operator-grid starting points.
- Packaging now declares `numpy` and `torch` optional extras for generated
  `einsum_numpy` and `einsum_torch` workflows while keeping base dependencies
  empty.

### Changed

- Saved design payloads now use schema version `2`, while schema version `1`
  remains loadable for existing files.
- `For bidimensional` and `For Tree` now accept manual contraction plans with
  clickable virtual boundary operands; grid cells expose upper/right/lower/left
  neighbors, tree cells expose parent/child boundaries, and generated periodic
  exports keep partial networks in `remaining_operands` instead of forcing a
  single final tensor.
- Frontend benchmark state and comparison export helpers now use shared modules
  instead of keeping duplicate definitions in multiple runtime files.
- Contraction analysis now reuses equivalent automatic greedy path lookups within
  one analysis run, and generated hyperedge copy tensors no longer materialize
  large nested literal payloads during backend preparation.
- Contraction analysis and benchmark mode now analyze normal-mode hyperedges by
  lowering them to internal generated copy tensors, while preserving the saved
  visual model and surfacing a clear warning.
- Einsum code generation now caches repeated automatic random-route lookups per
  generator instance, and validation skips full JSON serialization for plainly
  serializable metadata payloads.
- Frontend runtime tests now share the state module dependency preset used by
  scripts that copy `state.js`, avoiding repeated benchmark-state wiring.
- Documentation screenshots now use real browser-captured editor states with a
  consistent viewport and corrected canvas framing for metadata, tensor
  initializers, periodic modes, templates, shortcuts, and planner workflows.
- Removed obsolete private compatibility shims for split model and periodic
  code-generation internals.
- The `mps` template now owns the configurable 1D-state presets that were
  previously split across separate Heisenberg and Ising MPS entries: users can
  choose open or periodic boundaries, `none`/`u1`/`z2` symmetry tags, and
  explicit `zeros`, `random`, `all_up`, `all_down`, or `neel` initial states
  from the browser editor, Python helpers, and `template build` CLI command.

### Fixed

- Browser draft autosave now saves edited design names within the real-browser
  E2E timing budget.
- Split frontend helper modules now keep tensor border-color fallbacks and theme
  menu checked states in sync when exercised through their runtime facades.
- Third-party bundled-asset notices now describe Cytoscape.js and PrismJS notice
  handling more accurately.
- The browser editor File menu now groups export formats behind an `Export`
  submenu, the two-index mini menu keeps its shared dimension controls while
  disabling hyperedge creation, and open ports no longer draw over front tensor
  boxes unless they are connected or actively selected.
- The browser editor now keeps the `Export` submenu open from hover/focus with
  label toggles inside it, keeps selected tensor outlines and ports visible
  during clicks and drags, adds breathing room to the subnetwork library list,
  and keeps tensor resize handles below the minimap.
- Tensor ports now stay visible when their tensor is pressed or dragged, and
  stay attached throughout live dragging.
- Academic text exports now default missing label-visibility state to visible,
  so isolated session flows can still export TikZ/LaTeX and Graphviz/DOT.
- Documentation now describes grid/tree manual contraction plans as supported
  with clickable virtual boundary operands, including the row-major grid pass,
  bottom-up tree pass, and partial `remaining_operands` exports.
- Grid/tree periodic cell normalization, hydration, serialization, and cell
  switching now preserve each cell's saved `contraction_plan` instead of
  silently clearing it.
- Subnetwork catalog tests now isolate their working directory while preserving
  the default project-local catalog lookup used by normal editor sessions.
- Supported generated Python round-trips now reconstruct first-class
  `HyperedgeSpec` objects from structured generated copy-tensor comments, and
  live Python imports preserve finite complex runtime tensor values.

## [0.4.0] - 2026-04-25

### Changed

- Public documentation has been refreshed against the current package surface,
  including the local editor workflows, current CLI commands, schema wrapper,
  public facade modules, reusable subnetworks, hyperedges, tensor
  initializers, metadata filters, benchmark mode, and periodic modes.
- Public documentation now includes GitHub- and PyPI-friendly screenshots
  stored under `docs/images`, with absolute raw image URLs in the package
  README and relative image links inside the documentation folder, covering
  editor overview, templates, subnetworks, tensor initializers, hyperedges,
  planner/benchmark flows, Python imports, periodic modes, and CLI diagnostics.
- Benchmark mode now preserves user-typed spaces in scheme names while you are editing them in the toolbar instead of trimming the field on every keystroke.
- The editor now adds shortcuts for frequent building flow actions: `I` adds indices to selected tensors, `R` opens Reflow, and `Ctrl/Cmd+Enter` finishes the editor session.
- Benchmark comparison tables now show the partial FLOP, MAC, peak-size, and memory results of incomplete schemes when that analysis summary is available, while still reserving best/worst highlighting for fully complete schemes.
- All four cell-navigation arrow buttons in the top toolbar now use the shared shortcut hover system, so their tooltips show the matching `Alt+Arrow` shortcut instead of only a plain browser title.
- Multi-index selections now expose one shared `Dimension` input in both the `Selection` sidebar and the index mini menu, with the mini menu keeping that field on the same compact row as the `Indices` summary, so you can resize several selected indices at once even when their owner tensors are selected alongside them.
- Hover tooltips now include the real keyboard shortcut whenever that action has one, including the shared `Create hyperedge` buttons plus matching `Group`, `Delete`, `Search`, and `Filter` actions across the dynamic frontend panels.
- Hyperedge copy tensors in generated Python now use a compact zeros-plus-diagonal-fill pattern instead of giant nested literals, generated-source round-trips still recover those tensors correctly, and `Ctrl/Cmd+C` now gives priority to tensor-subgraph copy only for text selections inside the drawing area while preserving native text copy in the side panels and other UI text outside the canvas.
- Hyperedge creation now accepts selections that include the owning tensors alongside the selected open indices, so `Selection`, the `H` shortcut, and the index context menu still work when several chosen indices belong to the same tensor.
- `Shift+E` and the `Templates` dropdown now save the selected subnetwork JSON instead of exporting a session template, `Extract` tooltips surface that shortcut in the sidebar and mini menus, and `.py` loads now fall back from live import to the static parser when a generated file cannot import its backend modules.
- The canvas title toolbar now places the template controls before the cell-navigation controls, keeps the vertical separator between those two zones, and anchors the template block so it does not slide around when switching modes.
- The `Selection` sidebar now keeps compact row heights when there is spare vertical space, the tensor `Initialization` dropdown fills the usual field width, and the top toolbar adds a dedicated separator before the template controls so the base actions, cell controls, and templates read as three clearer zones.
- The `Info` help panel now works as a short practical guide to the current editor workflows and limits, and tensor `Initialization` now uses the same chevron disclosure behavior as the template selector.
- The planner reset button now opts into the shared shortcut hover, planner comparisons color automatic improvements in green and regressions in red both in the panel and in auto-past hover summaries, `Accept` actions use the positive green styling, the sidebar collapse toggle uses a solid black background for stronger contrast, and the top toolbar dropdown entries are back to the neutral menu styling instead of semantic accent colors.
- The planner now keeps `Auto past` visible without the redundant unlock helper text before the first manual contraction, auto-past comparison tooltips render as a compact four-line `FLOP` / `MAC` / `Peak` / `Peak Memory` summary, tensor value controls are now shown inline under `Initialization`, and `For`-mode boundary tensors expose only informational, color, and metadata controls instead of structural index editing actions.
- Editor action buttons now follow a clearer semantic color palette: add/create actions reuse the tensor-insert cyan, delete/trash actions stay red, template/library actions use a light yellow, contraction/planner actions use orange, and save/generate/export actions use green where they fit best.
- The guided public API is now centered on `open_editor`, `load_python_spec`, `tensor_network_editor.editor`, and `tensor_network_editor.io`; compatibility-only root exports and wrapper modules such as `api`, `serialization`, `diffing`, and legacy `codegen.*` re-export shims have been removed.
- Saved designs now keep the current payload shape but reset the public file wrapper to `schema_version = 1`, and loaders now reject the old compatibility-only schema numbers `4`, `5`, and `6`.
- Importing `tensor_network_editor` now resolves the public headless/API exports lazily, so the package root loads faster and avoids importing analysis, template, diffing, linting, and editor helpers until they are first accessed.
- Frontend planner and periodic-mode assets now reuse canonical shared constants for linear boundary operand ids and periodic cell labels, while helper-only formatting/navigation maps stay private to their defining modules to reduce drift in the browser codebase.
- Editor sessions now reuse a shared in-process cache of static asset bytes and the rendered `index.html`, defer contraction analysis until the planner actually needs it, cache serialized specs and analysis results by spec revision, and load Prism syntax-highlighting assets on demand instead of during the initial page load.
- The library is now organized around a new `tensor_network_editor.internal` implementation tree, stable public facade modules at the package root, a domain-based `codegen/` layout (`shared`, `backends`, `modes`), and a modularized frontend `app/static/js/` tree with only the browser entrypoints left at the top level.
- A second conservative modularization pass now moves canonicalization, diffing, linting, serialization, and subnetwork extraction internals behind thin public facades, while internal packages import those implementations directly instead of routing through compatibility wrappers.
- A third conservative modularization pass now splits linear-periodic carry semantics into `tensor_network_editor.internal.modes`, breaks linear-periodic codegen into smaller `common` / `standard` / `carry` modules, and removes internal dependencies on the old `_linear_periodic_shared` implementation path.
- The editor server now rebuilds its shared static-asset cache when source files change and uses nanosecond asset versions, avoiding stale in-process assets after fast local edits.
- The CLI now exposes a first-class `benchmark` subcommand with stable `text`, `json`, `csv`, and `latex` outputs for comparing the manual, auto-full, auto-future, and auto-past contraction variants of one saved design.
- Public docs now surface benchmark workflows and the advanced periodic modes more clearly, including dedicated guidance for `For bidimensional` and `For Tree` instead of focusing almost entirely on the linear workflow.
- Editor sessions now support a reusable subnetwork library with dedicated project/shared catalogs, CRUD routes, bootstrap payloads, preview/tag metadata, and matching `subnetwork list/save/export` CLI commands built on top of the existing extract/prepare-insert primitives.
- The `Reflow` popover now exposes an explicit `Auto layout` action that can arrange the active tensor selection or the whole graph when nothing is selected, while keeping benchmark-scheme and `For`-mode restrictions aligned with the rest of the toolbar.
- Tensors can now store editor-managed value initializers (`ones`, `fill`, and explicit numeric literals), generated code emits backend-native data initializers, and supported generated Python round-trips recover those tensor values.
- Networks can now store first-class `hyperedges` in normal mode, editor rendering shows hub-and-spoke hyperedge geometry, and code generation lowers hyperedges to autogenerated copy tensors for backend exports.
- Python loading now accepts explicit or autodetected source profiles (`generated`, `quimb`, `tensornetwork`, and `einsum`), conservative external AST imports can recover simple static ecosystem sources without executing user code, and linting now uses guided metadata keys like `role`, `symmetry`, `leg_kind`, and `observable` for higher-signal modeling warnings.
- Hyperedge hubs now store a persistent relative `hub_offset`, can be dragged directly on the canvas, and share one creation workflow across the Selection panel, a dedicated multi-index context menu, and the global `H` shortcut.
- Public docs now explain `HyperedgeSpec.hub_offset`, the draggable hyperedge hub, and the shared `H` / Selection / context-menu creation workflow.
- Hyperedges now expose their own right-click mini menu from either the hub or any spoke, including quick edits for name, color, metadata, and deletion.

### Fixed

- `For bidimensional` mode now recomputes center-cell boundary ports after every tensor addition, so the virtual neighbor-cell tensors no longer keep the previous slot count after the second edit.
- `Ctrl/Cmd+Enter` now finishes the editor session even when focus is inside an editable text or number field, including after the session completion handler is registered by the interaction runtime.
- Benchmark scheme name editing now preserves typed spaces in the toolbar input instead of trimming them away on each keystroke.
- SVG exports now escape quoted font-family attributes correctly so the generated XML opens reliably, and minimap index captions no longer leak the mojibake `Â·` separator.
- Unexpected editor-server failures now return safer but more actionable browser-visible messages, including retry guidance and the local session reference instead of a flat generic `500`.
- When automatic browser opening fails, the editor now explains that the local server is still running before printing the manual URL to open.
- Checkout-based test runs now skip installed-distribution metadata assertions when `importlib.metadata` resolves a different package installation than the active `src/` checkout.
- Frontend runtime contract/regression tests now copy the full `utilities.js` dependency set, including the tree-periodic utility module, and the bootstrap architecture fixture no longer hardcodes a stale app version.
- Frontend runtime test scaffolding now reuses named local JS dependency presets for shortcut, interaction, utility, and layout/subnetwork runtime scripts, reducing repeated copy boilerplate without changing test behavior.
- Tree-periodic codegen renderers now include the missing type annotations needed for the targeted CLI `mypy` regression check, and the linear-periodic headless analysis regression test now matches the current single-analysis flow.
- Editor HTTP request handling now rejects truncated request bodies deterministically, repeated session completion requests keep the first confirmed result instead of overwriting it, and benchmark-base planner guards clear all transient disclosure/inspection state consistently.
- The editor toolbar no longer presents `For Tree` as “Not available yet”; its copy now matches the real support level shipped in 0.4.0.

- Irregular imported graphs no longer fall back to a coarse grid-only reflow path; auto layout now uses layered component placement with overlap-safe spacing, and the toolbar keeps whole-graph layout available without requiring a temporary tensor selection.
- Hyperedges now participate correctly in canonicalization, subnetworks, clipboard flows, metadata filters, graph rendering, and connection lookups, while planner/manual contraction editing and benchmark mode fail fast with clear messages whenever a design contains hyperedges.
- Live Python imports now only fall back to the static parser for generated
  sources that fail because a backend import is missing, so ambiguous runtime
  globals still raise a `python_object_name` error instead of being silently
  parsed statically.

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

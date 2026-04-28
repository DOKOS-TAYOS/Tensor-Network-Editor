from __future__ import annotations

import json
import posixpath
import re
import shutil
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


JS_RELOCATION_MAP: dict[str, str] = {
    "api.js": "services/api.js",
    "benchmarkState.js": "state/benchmarkState.js",
    "codeHighlighting.js": "core/codeHighlighting.js",
    "constants.js": "core/constants.js",
    "dom.js": "core/dom.js",
    "editorContext.js": "core/editorContext.js",
    "exportMinimap.js": "graph/exportMinimap.js",
    "graphRender.js": "graph/graphRender.js",
    "graphRenderDrag.js": "graph/graphRenderDrag.js",
    "graphRenderLifecycle.js": "graph/graphRenderLifecycle.js",
    "graphRenderTooltips.js": "graph/graphRenderTooltips.js",
    "historySelection.js": "graph/historySelection.js",
    "interactions.js": "interactions/interactions.js",
    "interactionsCanvas.js": "interactions/interactionsCanvas.js",
    "interactionsEditor.js": "interactions/interactionsEditor.js",
    "interactionsSession.js": "interactions/interactionsSession.js",
    "interactionsShortcuts.js": "interactions/interactionsShortcuts.js",
    "metadataFilters.js": "graph/metadataFilters.js",
    "metadataFiltersBindings.js": "graph/metadataFiltersBindings.js",
    "metadataFiltersRenderers.js": "graph/metadataFiltersRenderers.js",
    "metadataFiltersState.js": "graph/metadataFiltersState.js",
    "notes.js": "graph/notes.js",
    "notesClipboard.js": "graph/notesClipboard.js",
    "notesSupport.js": "graph/notesSupport.js",
    "notesPlanner.js": "planner/notesPlanner.js",
    "overlaysLayoutTemplates.js": "graph/overlaysLayoutTemplates.js",
    "planner.js": "planner/planner.js",
    "plannerAutomaticSupport.js": "planner/plannerAutomaticSupport.js",
    "plannerRenderersAutomatic.js": "planner/plannerRenderersAutomatic.js",
    "plannerRenderersCommon.js": "planner/plannerRenderersCommon.js",
    "plannerRenderersManual.js": "planner/plannerRenderersManual.js",
    "plannerRenderersPanel.js": "planner/plannerRenderersPanel.js",
    "plannerRenderers.js": "planner/plannerRenderers.js",
    "plannerSupportActions.js": "planner/plannerSupportActions.js",
    "plannerSupportAnalysis.js": "planner/plannerSupportAnalysis.js",
    "plannerSupportGuards.js": "planner/plannerSupportGuards.js",
    "plannerSupportOperands.js": "planner/plannerSupportOperands.js",
    "plannerSupport.js": "planner/plannerSupport.js",
    "properties.js": "properties/properties.js",
    "propertiesRenderers.js": "properties/propertiesRenderers.js",
    "propertiesRenderersEntities.js": "properties/propertiesRenderersEntities.js",
    "propertiesRenderersOverview.js": "properties/propertiesRenderersOverview.js",
    "propertiesRenderersTensor.js": "properties/propertiesRenderersTensor.js",
    "propertiesSupport.js": "properties/propertiesSupport.js",
    "sidebarTabs.js": "core/sidebarTabs.js",
    "state.js": "state/state.js",
    "theme.js": "core/theme.js",
    "utilities.js": "utils/utilities.js",
    "utilitiesBase.js": "utils/utilitiesBase.js",
    "utilitiesBenchmark.js": "utils/utilitiesBenchmark.js",
    "utilitiesBenchmarkExports.js": "utils/utilitiesBenchmarkExports.js",
    "utilitiesBenchmarkSession.js": "utils/utilitiesBenchmarkSession.js",
    "utilitiesBenchmarkSessionState.js": "utils/utilitiesBenchmarkSessionState.js",
    "utilitiesBenchmarkTable.js": "utils/utilitiesBenchmarkTable.js",
    "utilitiesGeometry.js": "utils/utilitiesGeometry.js",
    "utilitiesGridPeriodic.js": "utils/utilitiesGridPeriodic.js",
    "utilitiesGridPeriodicBoundaries.js": "utils/utilitiesGridPeriodicBoundaries.js",
    "utilitiesGridPeriodicFlow.js": "utils/utilitiesGridPeriodicFlow.js",
    "utilitiesGridPeriodicState.js": "utils/utilitiesGridPeriodicState.js",
    "utilitiesLayout.js": "utils/utilitiesLayout.js",
    "utilitiesLayoutAlgorithms.js": "utils/utilitiesLayoutAlgorithms.js",
    "utilitiesLayoutAlgorithmsGraph.js": "utils/utilitiesLayoutAlgorithmsGraph.js",
    "utilitiesLayoutAlgorithmsPositions.js": "utils/utilitiesLayoutAlgorithmsPositions.js",
    "utilitiesLayoutIndices.js": "utils/utilitiesLayoutIndices.js",
    "utilitiesLayoutSelection.js": "utils/utilitiesLayoutSelection.js",
    "utilitiesLinearPeriodic.js": "utils/utilitiesLinearPeriodic.js",
    "utilitiesLinearPeriodicBoundaries.js": "utils/utilitiesLinearPeriodicBoundaries.js",
    "utilitiesLinearPeriodicFlow.js": "utils/utilitiesLinearPeriodicFlow.js",
    "utilitiesLinearPeriodicState.js": "utils/utilitiesLinearPeriodicState.js",
    "utilitiesSpec.js": "utils/utilitiesSpec.js",
    "utilitiesTemplates.js": "utils/utilitiesTemplates.js",
    "utilitiesTreePeriodic.js": "utils/utilitiesTreePeriodic.js",
    "utilitiesTreePeriodicBoundaries.js": "utils/utilitiesTreePeriodicBoundaries.js",
    "utilitiesTreePeriodicFlow.js": "utils/utilitiesTreePeriodicFlow.js",
    "utilitiesTreePeriodicState.js": "utils/utilitiesTreePeriodicState.js",
    "utilitiesUi.js": "utils/utilitiesUi.js",
    "utilitiesUiDom.js": "utils/utilitiesUiDom.js",
    "utilitiesUiGeneratedCode.js": "utils/utilitiesUiGeneratedCode.js",
    "utilitiesUiPanels.js": "utils/utilitiesUiPanels.js",
    "utilitiesUiStatus.js": "utils/utilitiesUiStatus.js",
    "utilitiesUiToolbar.js": "utils/utilitiesUiToolbar.js",
    "utilitiesUiToolbarActionState.js": "utils/utilitiesUiToolbarActionState.js",
    "utilitiesUiToolbarDerivedState.js": "utils/utilitiesUiToolbarDerivedState.js",
    "utilitiesUiToolbarModeControls.js": "utils/utilitiesUiToolbarModeControls.js",
    "utilitiesUiToolbarWarnings.js": "utils/utilitiesUiToolbarWarnings.js",
    "sessionTemplateDialogs.js": "session/sessionTemplateDialogs.js",
    "sessionTemplateFlowSubnetworkLibrary.js": "session/sessionTemplateFlowSubnetworkLibrary.js",
    "sessionTemplateImports.js": "session/sessionTemplateImports.js",
    "sessionTemplateManager.js": "session/sessionTemplateManager.js",
}


def _js_source_name(module_name: str) -> str:
    return JS_RELOCATION_MAP.get(module_name, module_name)


def _mapped_js_modules(module_names: tuple[str, ...]) -> dict[str, str]:
    return {module_name: _js_source_name(module_name) for module_name in module_names}


def _write_text_file(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


_RELATIVE_JS_IMPORT_PATTERN = re.compile(
    r'(?P<prefix>\bfrom\s+["\']|\bimport\s+["\'])(?P<spec>\.{1,2}/[^"\']+)(?P<suffix>["\'])'
)

_STATE_RUNTIME_DEPENDENCY_MODULES: dict[str, str] = {
    "benchmarkState.js": _js_source_name("benchmarkState.js"),
    "state.runtime.mjs": _js_source_name("state.js"),
    "theme.js": _js_source_name("theme.js"),
}

_UTILITY_RUNTIME_DEPENDENCY_MODULES: dict[str, str] = {
    **_STATE_RUNTIME_DEPENDENCY_MODULES,
    "utilities.runtime.mjs": _js_source_name("utilities.js"),
    "codeHighlighting.js": _js_source_name("codeHighlighting.js"),
    "utilitiesTemplates.js": _js_source_name("utilitiesTemplates.js"),
    "spec/specLookups.js": "spec/specLookups.js",
    "spec/specMutations.js": "spec/specMutations.js",
    "spec/specNormalization.js": "spec/specNormalization.js",
    "utilitiesBase.js": _js_source_name("utilitiesBase.js"),
    "theme.js": _js_source_name("theme.js"),
    "utilitiesBenchmark.js": _js_source_name("utilitiesBenchmark.js"),
    "utilitiesBenchmarkExports.js": _js_source_name("utilitiesBenchmarkExports.js"),
    "utilitiesBenchmarkSession.js": _js_source_name("utilitiesBenchmarkSession.js"),
    "utilitiesBenchmarkSessionState.js": _js_source_name(
        "utilitiesBenchmarkSessionState.js"
    ),
    "utilitiesBenchmarkTable.js": _js_source_name("utilitiesBenchmarkTable.js"),
    "utilitiesGeometry.js": _js_source_name("utilitiesGeometry.js"),
    "utilitiesGridPeriodic.js": _js_source_name("utilitiesGridPeriodic.js"),
    "utilitiesGridPeriodicBoundaries.js": _js_source_name(
        "utilitiesGridPeriodicBoundaries.js"
    ),
    "utilitiesGridPeriodicFlow.js": _js_source_name("utilitiesGridPeriodicFlow.js"),
    "utilitiesGridPeriodicState.js": _js_source_name("utilitiesGridPeriodicState.js"),
    "utilitiesLayout.js": _js_source_name("utilitiesLayout.js"),
    "utilitiesLayoutAlgorithms.js": _js_source_name("utilitiesLayoutAlgorithms.js"),
    "utilitiesLayoutAlgorithmsGraph.js": _js_source_name(
        "utilitiesLayoutAlgorithmsGraph.js"
    ),
    "utilitiesLayoutAlgorithmsPositions.js": _js_source_name(
        "utilitiesLayoutAlgorithmsPositions.js"
    ),
    "utilitiesLayoutIndices.js": _js_source_name("utilitiesLayoutIndices.js"),
    "utilitiesLayoutSelection.js": _js_source_name("utilitiesLayoutSelection.js"),
    "utilitiesLinearPeriodic.js": _js_source_name("utilitiesLinearPeriodic.js"),
    "utilitiesLinearPeriodicBoundaries.js": _js_source_name(
        "utilitiesLinearPeriodicBoundaries.js"
    ),
    "utilitiesLinearPeriodicFlow.js": _js_source_name("utilitiesLinearPeriodicFlow.js"),
    "utilitiesLinearPeriodicState.js": _js_source_name(
        "utilitiesLinearPeriodicState.js"
    ),
    "utilitiesTreePeriodic.js": _js_source_name("utilitiesTreePeriodic.js"),
    "utilitiesTreePeriodicBoundaries.js": _js_source_name(
        "utilitiesTreePeriodicBoundaries.js"
    ),
    "utilitiesTreePeriodicFlow.js": _js_source_name("utilitiesTreePeriodicFlow.js"),
    "utilitiesTreePeriodicState.js": _js_source_name("utilitiesTreePeriodicState.js"),
    "utilitiesSpec.js": _js_source_name("utilitiesSpec.js"),
    "utilitiesUi.js": _js_source_name("utilitiesUi.js"),
    "utilitiesUiDom.js": _js_source_name("utilitiesUiDom.js"),
    "utilitiesUiGeneratedCode.js": _js_source_name("utilitiesUiGeneratedCode.js"),
    "utilitiesUiPanels.js": _js_source_name("utilitiesUiPanels.js"),
    "utilitiesUiStatus.js": _js_source_name("utilitiesUiStatus.js"),
    "utilitiesUiToolbar.js": _js_source_name("utilitiesUiToolbar.js"),
    "utilitiesUiToolbarActionState.js": _js_source_name(
        "utilitiesUiToolbarActionState.js"
    ),
    "utilitiesUiToolbarDerivedState.js": _js_source_name(
        "utilitiesUiToolbarDerivedState.js"
    ),
    "utilitiesUiToolbarModeControls.js": _js_source_name(
        "utilitiesUiToolbarModeControls.js"
    ),
    "utilitiesUiToolbarWarnings.js": _js_source_name("utilitiesUiToolbarWarnings.js"),
    "metadataFiltersBindings.js": _js_source_name("metadataFiltersBindings.js"),
    "metadataFiltersRenderers.js": _js_source_name("metadataFiltersRenderers.js"),
    "metadataFiltersState.js": _js_source_name("metadataFiltersState.js"),
}

_RUNTIME_EDITOR_SUPPORT_MODULES: dict[str, str] = _mapped_js_modules(
    (
        "actions/designMutationPipeline.js",
        "actions/plannerCommands.js",
        "actions/propertyCommands.js",
        "actions/sessionCommands.js",
        "benchmarkState.js",
        "codeHighlighting.js",
        "planner/plannerAnalysisFormatting.js",
        "planner/plannerRenderersAutomatic.js",
        "planner/plannerRenderersCommon.js",
        "planner/plannerRenderersManual.js",
        "planner/plannerRenderersPanel.js",
        "planner/plannerPanelBindings.js",
        "planner/plannerSupportActions.js",
        "planner/plannerSupportAnalysis.js",
        "planner/plannerSupportGuards.js",
        "planner/plannerSupportOperands.js",
        "properties/entityPropertiesBindings.js",
        "properties/entityPropertiesMarkup.js",
        "properties/metadataEditors.js",
        "properties/overviewPropertiesBindings.js",
        "properties/overviewPropertiesMarkup.js",
        "properties/propertyAutosave.js",
        "properties/propertyInvalidation.js",
        "properties/propertySummaries.js",
        "properties/tensorPropertiesBoundary.js",
        "properties/tensorPropertiesContraction.js",
        "properties/tensorPropertiesStandardBindings.js",
        "properties/tensorPropertiesStandardData.js",
        "properties/tensorPropertiesStandardMarkup.js",
        "properties/tensorPropertiesStandard.js",
        "session/sessionEditorFlows.js",
        "session/sessionTemplateFlows.js",
        "session/sessionTemplateDialogs.js",
        "session/sessionTemplateFlowSubnetworkLibrary.js",
        "session/sessionTemplateImports.js",
        "session/sessionTemplateManager.js",
        "session/sessionUiAdapters.js",
        "utilitiesBase.js",
        "utilitiesBenchmark.js",
        "utilitiesBenchmarkExports.js",
        "utilitiesBenchmarkSession.js",
        "utilitiesBenchmarkSessionState.js",
        "utilitiesBenchmarkTable.js",
        "utilitiesGeometry.js",
        "utilitiesGridPeriodic.js",
        "utilitiesGridPeriodicBoundaries.js",
        "utilitiesGridPeriodicFlow.js",
        "utilitiesGridPeriodicState.js",
        "utilitiesLayout.js",
        "utilitiesLayoutAlgorithms.js",
        "utilitiesLayoutAlgorithmsGraph.js",
        "utilitiesLayoutAlgorithmsPositions.js",
        "utilitiesLayoutIndices.js",
        "utilitiesLayoutSelection.js",
        "utilitiesLinearPeriodic.js",
        "utilitiesLinearPeriodicBoundaries.js",
        "utilitiesLinearPeriodicFlow.js",
        "utilitiesLinearPeriodicState.js",
        "utilitiesTreePeriodic.js",
        "utilitiesTreePeriodicBoundaries.js",
        "utilitiesTreePeriodicFlow.js",
        "utilitiesTreePeriodicState.js",
        "utilitiesSpec.js",
        "utilitiesTemplates.js",
        "utilitiesUi.js",
        "utilitiesUiDom.js",
        "utilitiesUiGeneratedCode.js",
        "utilitiesUiPanels.js",
        "utilitiesUiStatus.js",
        "utilitiesUiToolbar.js",
        "utilitiesUiToolbarActionState.js",
        "utilitiesUiToolbarDerivedState.js",
        "utilitiesUiToolbarModeControls.js",
        "utilitiesUiToolbarWarnings.js",
        "interactionsCanvas.js",
        "interactionsEditor.js",
        "interactionsSession.js",
        "interactionsShortcuts.js",
        "graph/canvasContextMenuBindings.js",
        "graph/canvasContextMenuMarkup.js",
        "graph/canvasContextMenuTargets.js",
        "graph/contractionSceneCache.js",
        "graph/contractionSceneEditing.js",
        "graph/contractionSceneOperands.js",
        "graphRenderDrag.js",
        "graphRenderLifecycle.js",
        "graphRenderTooltips.js",
        "metadataFiltersBindings.js",
        "metadataFiltersRenderers.js",
        "metadataFiltersState.js",
        "notesClipboard.js",
        "notesSupport.js",
        "plannerAutomaticSupport.js",
        "propertiesRenderersOverview.js",
        "propertiesRenderersTensor.js",
        "propertiesRenderersEntities.js",
        "services/editorSessionService.js",
        "services/plannerAnalysisService.js",
        "services/subnetworkService.js",
        "services/templateCatalogService.js",
        "spec/specLookups.js",
        "spec/specMutations.js",
        "spec/specNormalization.js",
        "state/historySnapshots.js",
        "state/selectionEntries.js",
        "state/contractionSceneProgression.js",
        "state/contractionSceneSnapshots.js",
        "state/editorSelectors.js",
        "state/editorStore.js",
        "state/plannerSelectors.js",
        "theme.js",
    )
)

_SHORTCUT_RUNTIME_DEPENDENCY_MODULES: dict[str, str] = {
    **_STATE_RUNTIME_DEPENDENCY_MODULES,
    "interactionsShortcuts.js": _js_source_name("interactionsShortcuts.js"),
}

_MINIMAP_SHORTCUT_RUNTIME_DEPENDENCY_MODULES: dict[str, str] = {
    **_SHORTCUT_RUNTIME_DEPENDENCY_MODULES,
    "exportMinimap.js": _js_source_name("exportMinimap.js"),
    "theme.js": _js_source_name("theme.js"),
}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_theme_module_applies_runtime_graph_palette(tmp_path: Path) -> None:
    script_path = tmp_path / "theme_runtime_regression.mjs"
    theme_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "core"
        / "theme.js"
    )
    script_path.write_text(
        textwrap.dedent(
            f"""
            import {{ pathToFileURL }} from "node:url";

            const themeUrl = pathToFileURL({str(theme_module_path)!r}).href;
            const theme = await import(themeUrl);
            const root = {{
              dataset: {{}},
              style: {{}},
            }};
            const documentRef = {{ documentElement: root }};
            const storageRef = {{
              values: {{}},
              getItem(name) {{
                return Object.prototype.hasOwnProperty.call(this.values, name)
                  ? this.values[name]
                  : null;
              }},
              setItem(name, value) {{
                this.values[name] = String(value);
              }},
            }};
            const initialSelection = theme.GRAPH_THEME.selection;
            const appliedName = theme.applyEditorTheme("light", {{
              documentRef,
              storageRef,
              persist: true,
            }});

            if (appliedName !== "light") {{
              throw new Error(`Expected light theme, received ${{appliedName}}.`);
            }}
            if (root.dataset.theme !== "light") {{
              throw new Error(`Expected root data-theme=light, received ${{root.dataset.theme}}.`);
            }}
            if (theme.GRAPH_THEME.selection === initialSelection) {{
              throw new Error("Applying a theme should update the shared graph palette object.");
            }}
            if (!theme.EDITOR_THEME_NAMES.includes("colorblind")) {{
              throw new Error("Supported theme names should include colorblind.");
            }}
            if (storageRef.values[theme.EDITOR_THEME_STORAGE_KEY] !== "light") {{
              throw new Error(
                `Expected the persisted theme to be stored, received ${{storageRef.values[theme.EDITOR_THEME_STORAGE_KEY]}}.`
              );
            }}
            if (theme.readStoredEditorThemeName({{ storageRef }}) !== "light") {{
              throw new Error("Expected readStoredEditorThemeName to return the saved theme.");
            }}
            storageRef.values[theme.EDITOR_THEME_STORAGE_KEY] = "colorblind";
            if (
              theme.resolvePreferredEditorThemeName({{
                bootstrapThemeName: "dark",
                storageRef,
              }}) !== "colorblind"
            ) {{
              throw new Error("Stored preferences should override the bootstrap theme.");
            }}
            if (theme.formatEditorThemeLabel("contrast") !== "High contrast") {{
              throw new Error("Expected the theme formatter to return a human-friendly label.");
            }}
            """
        ),
        encoding="utf-8",
    )
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The theme runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


def _write_editor_bootstrap_theme_runtime_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "editor_bootstrap_theme_runtime.mjs"
    _copy_js_modules(
        tmp_path,
        {
            "editorBootstrapFlow.js": "shell/editorBootstrapFlow.js",
            "theme.js": "core/theme.js",
        },
    )
    script_path.write_text(
        textwrap.dedent(
            """
            const baseUrl = new URL("./", import.meta.url);
            const { createEditorBootstrapFlow } = await import(
              new URL("./editorBootstrapFlow.js", baseUrl).href
            );

            const state = {
              templateCatalogWarnings: [],
              subnetworkCatalogWarnings: [],
              availableCollectionFormats: [],
              selectedTheme: "dark",
            };
            const calls = [];
            const store = {
              setSelectedTheme(value) {
                state.selectedTheme = value;
                calls.push({ type: "setSelectedTheme", value });
              },
              setSpec(value) {
                state.spec = value;
              },
              setSchemaVersion(value) {
                state.schemaVersion = value;
              },
              setAppMetadata(value) {
                state.appMetadata = value;
              },
              setAvailableCollectionFormats(value) {
                state.availableCollectionFormats = value;
              },
              setAnnotationDefinitions(value) {
                state.annotationDefinitions = value;
              },
              setSelectedEngine(value) {
                state.selectedEngine = value;
              },
              setSelectedCollectionFormat(value) {
                state.selectedCollectionFormat = value;
              },
              setSubnetworkCatalogData(value) {
                state.subnetworkCatalogData = value;
              },
            };
            const root = {
              dataset: {},
              style: {},
            };
            const flow = createEditorBootstrapFlow({
              state,
              store,
              sessionService: {
                async loadBootstrap() {
                  return {
                    theme: "dark",
                    spec: {
                      schema_version: 4,
                      network: { id: "network_bootstrap", tensors: [], edges: [], groups: [], notes: [], metadata: {} },
                    },
                    schema_version: 4,
                    app_metadata: {},
                    collection_formats: ["list"],
                    templates: [],
                    template_definitions: {},
                    template_catalog_warnings: [],
                    subnetworks: [],
                    subnetwork_definitions: {},
                    subnetwork_catalog_warnings: [],
                    selected_subnetwork: "",
                    annotation_definitions: {},
                    default_engine: "tensornetwork",
                    default_collection_format: "list",
                    engines: ["tensornetwork"],
                  };
                },
                async loadDraft() {
                  return { draft: null };
                },
              },
              actions: {
                normalizeSpec(spec) {
                  return spec;
                },
                applyTemplateCatalogPayload(payload) {
                  state.templatePayload = payload;
                },
                reconcileTensorOrder() {},
                populateEngineOptions(value) {
                  state.engineOptions = value;
                },
                enforceLinearPeriodicEngineSupport() {},
                populateCollectionFormatOptions(value) {
                  state.collectionOptions = value;
                },
                initGraph() {},
                clearHistory() {},
                render() {},
                setStatus(message, level) {
                  calls.push({ type: "setStatus", message, level });
                },
              },
              documentRef: { documentElement: root },
              windowRef: {
                localStorage: {
                  getItem(name) {
                    return name === "tensor-network-editor.theme" ? "colorblind" : null;
                  },
                },
              },
              confirmAction() {
                return false;
              },
            });

            await flow.bootstrap();

            if (state.selectedTheme !== "colorblind") {
              throw new Error(`Expected bootstrap to prefer the saved theme, received ${state.selectedTheme}.`);
            }
            if (root.dataset.theme !== "colorblind") {
              throw new Error(`Expected bootstrap to apply the saved theme to the document root, received ${root.dataset.theme}.`);
            }
            if (root.style.colorScheme !== "light") {
              throw new Error(`Expected the colorblind theme to set a light color scheme, received ${root.style.colorScheme}.`);
            }
            if (!calls.some((entry) => entry.type === "setSelectedTheme" && entry.value === "colorblind")) {
              throw new Error(`Expected bootstrap to persist the selected theme in state, received ${JSON.stringify(calls)}.`);
            }
          """
        ),
        encoding="utf-8",
    )
    return script_path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_editor_bootstrap_prefers_saved_theme_over_backend_default(
    tmp_path: Path,
) -> None:
    script_path = _write_editor_bootstrap_theme_runtime_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The editor bootstrap theme runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


_INTERACTION_SESSION_BINDING_DEPENDENCY_MODULES: dict[str, str] = _mapped_js_modules(
    (
        "actions/sessionCommands.js",
        "codeHighlighting.js",
        "interactionsSession.js",
        "session/sessionEditorFlows.js",
        "session/sessionTemplateDialogs.js",
        "session/sessionTemplateFlowSubnetworkLibrary.js",
        "session/sessionTemplateFlows.js",
        "session/sessionTemplateImports.js",
        "session/sessionTemplateManager.js",
        "session/sessionUiAdapters.js",
        "state/editorSelectors.js",
        "state/editorStore.js",
        "utilitiesTemplates.js",
    )
)

_SESSION_EDITOR_FLOWS_DEPENDENCY_MODULES: dict[str, str] = _mapped_js_modules(
    ("session/sessionEditorFlows.js",)
)

_INTERACTION_RUNTIME_CONTRACT_DEPENDENCY_MODULES: dict[str, str] = {
    **_STATE_RUNTIME_DEPENDENCY_MODULES,
    "interactions.runtime.mjs": _js_source_name("interactions.js"),
    **_INTERACTION_SESSION_BINDING_DEPENDENCY_MODULES,
    **_mapped_js_modules(
        (
            "interactionsCanvas.js",
            "interactionsEditor.js",
            "interactionsShortcuts.js",
            "interactions/editorActionGroups.js",
            "services/editorSessionService.js",
            "services/subnetworkService.js",
            "services/templateCatalogService.js",
        )
    ),
}

_LAYOUT_SUBNETWORK_RUNTIME_DEPENDENCY_MODULES: dict[str, str] = {
    **_UTILITY_RUNTIME_DEPENDENCY_MODULES,
    "historySelection.runtime.mjs": _js_source_name("historySelection.js"),
    **_mapped_js_modules(
        (
            "actions/designMutationPipeline.js",
            "actions/sessionCommands.js",
            "interactionsSession.js",
            "session/sessionEditorFlows.js",
            "session/sessionTemplateDialogs.js",
            "session/sessionTemplateFlowSubnetworkLibrary.js",
            "session/sessionTemplateFlows.js",
            "session/sessionTemplateImports.js",
            "session/sessionTemplateManager.js",
            "session/sessionUiAdapters.js",
            "services/editorSessionService.js",
            "services/subnetworkService.js",
            "services/templateCatalogService.js",
            "state/editorSelectors.js",
            "state/editorStore.js",
            "state/historySnapshots.js",
            "state/selectionEntries.js",
        )
    ),
}


def _copy_js_modules(tmp_path: Path, copied_modules: dict[str, str]) -> None:
    js_root = REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js"
    source_to_target = {
        source_name.replace("\\", "/"): target_name.replace("\\", "/")
        for target_name, source_name in copied_modules.items()
    }

    def rewrite_imports(content: str, *, source_name: str, target_name: str) -> str:
        source_dir = posixpath.dirname(source_name) or "."
        target_dir = posixpath.dirname(target_name) or "."

        def replace_import(match: re.Match[str]) -> str:
            spec = match.group("spec")
            source_target = posixpath.normpath(posixpath.join(source_dir, spec))
            mapped_target = source_to_target.get(source_target)
            if mapped_target is None:
                return match.group(0)
            rewritten_spec = posixpath.relpath(mapped_target, start=target_dir)
            if not rewritten_spec.startswith("."):
                rewritten_spec = f"./{rewritten_spec}"
            return f"{match.group('prefix')}{rewritten_spec}{match.group('suffix')}"

        return _RELATIVE_JS_IMPORT_PATTERN.sub(replace_import, content)

    for target_name, source_name in copied_modules.items():
        target_path = tmp_path / target_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_key = source_name.replace("\\", "/")
        target_key = target_name.replace("\\", "/")
        source_content = (js_root / source_name).read_text(encoding="utf-8")
        rewritten_content = rewrite_imports(
            source_content,
            source_name=source_key,
            target_name=target_key,
        )
        target_path.write_text(rewritten_content, encoding="utf-8")


def _copy_runtime_editor_support_modules(tmp_path: Path) -> None:
    _copy_js_modules(tmp_path, _RUNTIME_EDITOR_SUPPORT_MODULES)


def _copy_runtime_bundle(
    tmp_path: Path,
    entry_modules: dict[str, str],
    support_modules: dict[str, str] | None = None,
) -> None:
    copied_modules: dict[str, str] = {}
    if support_modules:
        copied_modules.update(support_modules)
    copied_modules.update(entry_modules)
    _copy_js_modules(tmp_path, copied_modules)


def test_copy_runtime_editor_support_modules_includes_planner_automatic_support(
    tmp_path: Path,
) -> None:
    _copy_runtime_editor_support_modules(tmp_path)

    assert (tmp_path / "plannerAutomaticSupport.js").exists()
    assert (tmp_path / "planner" / "plannerRenderersAutomatic.js").exists()
    assert (tmp_path / "planner" / "plannerSupportActions.js").exists()
    assert (tmp_path / "utilitiesBenchmark.js").exists()


def test_state_runtime_dependency_modules_include_benchmark_state(
    tmp_path: Path,
) -> None:
    assert "_STATE_RUNTIME_DEPENDENCY_MODULES" in globals()

    _copy_js_modules(tmp_path, _STATE_RUNTIME_DEPENDENCY_MODULES)

    state_runtime_path = tmp_path / "state.runtime.mjs"
    assert state_runtime_path.exists()
    assert (tmp_path / "benchmarkState.js").exists()
    assert 'from "./benchmarkState.js"' in state_runtime_path.read_text(
        encoding="utf-8"
    )


def _write_for_mode_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "for_mode_runtime_regression.mjs"
    state_runtime_path = tmp_path / "state.runtime.mjs"
    utilities_runtime_path = tmp_path / "utilities.runtime.mjs"
    history_runtime_path = tmp_path / "historySelection.runtime.mjs"
    _copy_runtime_bundle(
        tmp_path,
        {
            "state.runtime.mjs": "state/state.js",
            "utilities.runtime.mjs": "utils/utilities.js",
            "historySelection.runtime.mjs": "graph/historySelection.js",
        },
        _RUNTIME_EDITOR_SUPPORT_MODULES,
    )
    script_body = textwrap.dedent(
        f"""
        import {{ pathToFileURL }} from "node:url";

        function createClassList() {{
          return {{
            add() {{}},
            remove() {{}},
            toggle() {{}},
          }};
        }}

        function createButton() {{
          return {{
            disabled: false,
            classList: createClassList(),
          }};
        }}

        function createLinearPeriodicSpec() {{
          return {{
            id: "network_linear_periodic",
            name: "linear-periodic-chain",
            tensors: [],
            groups: [],
            edges: [],
            notes: [],
            contraction_plan: null,
            metadata: {{}},
            linear_periodic_chain: {{
              active_cell: "initial",
              metadata: {{}},
              initial_cell: {{
                tensors: [
                  {{
                    id: "initial_tensor",
                    name: "Initial",
                    position: {{ x: 100, y: 140 }},
                    size: {{ width: 140, height: 84 }},
                    metadata: {{}},
                    indices: [
                      {{
                        id: "initial_phys",
                        name: "phys",
                        dimension: 2,
                        offset: {{ x: -38, y: 0 }},
                        metadata: {{}},
                      }},
                      {{
                        id: "initial_bond",
                        name: "bond",
                        dimension: 3,
                        offset: {{ x: 38, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                  {{
                    id: "initial_next_boundary",
                    name: "Next cell",
                    position: {{ x: 320, y: 140 }},
                    size: {{ width: 140, height: 84 }},
                    linear_periodic_role: "next",
                    metadata: {{}},
                    indices: [
                      {{
                        id: "initial_next_slot_1",
                        name: "slot_1",
                        dimension: 2,
                        offset: {{ x: -38, y: 0 }},
                        metadata: {{}},
                      }},
                      {{
                        id: "initial_next_slot_2",
                        name: "slot_2",
                        dimension: 3,
                        offset: {{ x: 38, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                ],
                groups: [],
                edges: [],
                notes: [],
                contraction_plan: null,
                metadata: {{}},
              }},
              periodic_cell: {{
                tensors: [],
                groups: [],
                edges: [],
                notes: [],
                contraction_plan: null,
                metadata: {{}},
              }},
              final_cell: {{
                tensors: [],
                groups: [],
                edges: [],
                notes: [],
                contraction_plan: null,
                metadata: {{}},
              }},
            }},
          }};
        }}

        function getActualIndexDimension(spec) {{
          const tensor = spec.tensors.find((candidate) => candidate.id === "initial_tensor");
          if (!tensor) {{
            throw new Error("Missing initial tensor in active spec.");
          }}
          const index = tensor.indices.find((candidate) => candidate.id === "initial_phys");
          if (!index) {{
            throw new Error("Missing initial phys index in active spec.");
          }}
          return index.dimension;
        }}

        function applyIndexDimensionChange(ctx, nextDimension) {{
          ctx.applyDesignChange(
            () => {{
              const located = ctx.findIndexOwner("initial_phys");
              if (!located) {{
                throw new Error("The active index owner could not be resolved.");
              }}
              located.index.dimension = nextDimension;
            }},
            {{
              invalidate: {{
                graph: true,
                lookups: false,
                analysis: true,
                properties: true,
                toolbar: false,
                overlays: false,
                planner: false,
                sidebarTabs: false,
                minimap: false,
                code: false,
              }},
            }}
          );
        }}

        const [stateModule, utilitiesModule, historyModule] = await Promise.all([
          import(pathToFileURL({json.dumps(str(state_runtime_path))}).href),
          import(pathToFileURL({json.dumps(str(utilities_runtime_path))}).href),
          import(pathToFileURL({json.dumps(str(history_runtime_path))}).href),
        ]);
        const {{ createInitialState }} = stateModule;
        const {{ registerUtilities }} = utilitiesModule;
        const {{ registerHistorySelection }} = historyModule;

        const ctx = {{
          state: createInitialState(),
          constants: {{
            TENSOR_WIDTH: 140,
            TENSOR_HEIGHT: 84,
            MIN_TENSOR_WIDTH: 96,
            MIN_TENSOR_HEIGHT: 60,
            INDEX_RADIUS: 10,
            INDEX_PADDING: 6,
            NOTE_WIDTH: 220,
            NOTE_HEIGHT: 120,
            NOTE_MIN_WIDTH: 120,
            NOTE_MIN_HEIGHT: 90,
            HISTORY_LIMIT: 100,
            REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
            DEFAULT_INDEX_SLOTS: [
              {{ x: -38, y: 0 }},
              {{ x: 38, y: 0 }},
              {{ x: 0, y: -24 }},
              {{ x: 0, y: 24 }},
            ],
          }},
          dom: {{
            workspace: {{}},
            statusMessage: {{
              textContent: "",
              classList: createClassList(),
            }},
            propertiesPanel: {{ innerHTML: "" }},
            generatedCode: {{ value: "" }},
            engineSelect: {{ options: [], value: "tensornetwork" }},
            collectionFormatSelect: {{ options: [], value: "list" }},
            exportFormatSelect: {{ value: "py" }},
            addNoteButton: createButton(),
            connectButton: {{ classList: createClassList() }},
            loadInput: {{}},
            undoButton: createButton(),
            redoButton: createButton(),
            exportButton: createButton(),
            toggleLinearPeriodicButton: {{ classList: createClassList() }},
            linearPeriodicPreviousCellButton: createButton(),
            linearPeriodicCellLabel: {{ textContent: "" }},
            linearPeriodicNextCellButton: createButton(),
            templateSelect: {{ value: "" }},
            templateParameterPanel: {{ hidden: true }},
            templateGraphSizeLabel: {{ textContent: "" }},
            templateGraphSizeInput: {{ value: "2", min: "1" }},
            templateBondDimensionInput: {{ value: "3", min: "1" }},
            templatePhysicalDimensionInput: {{ value: "2", min: "1" }},
            insertTemplateButton: createButton(),
            createGroupButton: createButton(),
            helpButton: createButton(),
            helpModal: {{ classList: createClassList() }},
            helpBackdrop: createButton(),
            helpCloseButton: createButton(),
            canvasShell: {{
              getBoundingClientRect() {{
                return {{ left: 0, top: 0, width: 1000, height: 800 }};
              }},
            }},
            groupLayer: {{}},
            resizeLayer: {{}},
            notesLayer: {{}},
            selectionBox: {{}},
            minimapCanvas: {{}},
            sidebar: {{}},
            plannerPanel: {{}},
            generateButton: createButton(),
          }},
          apiGet: async () => {{
            throw new Error("apiGet should not be used in this runtime regression test.");
          }},
          apiPost: async () => {{
            throw new Error("apiPost should not be used in this runtime regression test.");
          }},
          window: {{
            structuredClone: globalThis.structuredClone,
            crypto: globalThis.crypto,
            setTimeout,
            clearTimeout,
            confirm: () => true,
          }},
          document: {{
            activeElement: null,
            createElement() {{
              return {{
                value: "",
                textContent: "",
                selected: false,
                appendChild() {{}},
                click() {{}},
              }};
            }},
            querySelectorAll() {{
              return [];
            }},
          }},
          cytoscape: null,
          tensorWidth: (tensor) => tensor?.size?.width ?? 140,
          tensorHeight: (tensor) => tensor?.size?.height ?? 84,
          render: () => {{}},
          renderOverlayDecorations: () => {{}},
          renderMinimap: () => {{}},
          renderPlanner: () => {{}},
          renderSidebarTabs: () => {{}},
          refreshContractionAnalysis: () => {{}},
          repairContractionPlan: () => {{}},
        }};

        registerUtilities(ctx);
        registerHistorySelection(ctx);

        ctx.captureEditableFocus = () => null;
        ctx.restoreEditableFocus = () => {{}};
        ctx.render = () => {{}};
        ctx.updateToolbarState = () => {{}};
        ctx.renderOverlayDecorations = () => {{}};
        ctx.renderMinimap = () => {{}};
        ctx.renderPlanner = () => {{}};
        ctx.renderSidebarTabs = () => {{}};
        ctx.refreshContractionAnalysis = () => {{}};
        ctx.repairContractionPlan = () => {{}};
        ctx.state.selectedEngine = "tensornetwork";
        ctx.state.selectedCollectionFormat = "list";
        ctx.state.spec = ctx.normalizeSpec(createLinearPeriodicSpec());

        if (getActualIndexDimension(ctx.state.spec) !== 2) {{
          throw new Error("The initial test spec was not normalised as expected.");
        }}

        applyIndexDimensionChange(ctx, 5);
        if (getActualIndexDimension(ctx.state.spec) !== 5) {{
          throw new Error(
            `Expected the first dimension update to persist as 5, received ${{getActualIndexDimension(ctx.state.spec)}}.`
          );
        }}

        applyIndexDimensionChange(ctx, 7);
        if (getActualIndexDimension(ctx.state.spec) !== 7) {{
          throw new Error(
            `Expected the second dimension update to persist as 7, received ${{getActualIndexDimension(ctx.state.spec)}}.`
          );
        }}
        """
    )
    script_path.write_text(script_body, encoding="utf-8")
    return script_path


def _write_benchmark_mode_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "benchmark_mode_runtime_regression.mjs"
    script_body = textwrap.dedent(
        f"""
        import {{ pathToFileURL }} from "node:url";

        const stateUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "state/state.js")!r}).href;
        const baseUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "utils/utilitiesBase.js")!r}).href;
        const specNormalizationUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "spec" / "specNormalization.js")!r}).href;
        const uiUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "utils/utilitiesUi.js")!r}).href;
        const benchmarkUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "utils/utilitiesBenchmark.js")!r}).href;
        const plannerUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "planner/planner.js")!r}).href;

        const [
          stateModule,
          baseModule,
          specNormalizationModule,
          uiModule,
          benchmarkModule,
          plannerModule,
        ] = await Promise.all([
          import(stateUrl),
          import(baseUrl),
          import(specNormalizationUrl),
          import(uiUrl),
          import(benchmarkUrl),
          import(plannerUrl),
        ]);

        function createClassList() {{
          return {{
            toggle() {{}},
            add() {{}},
            remove() {{}},
          }};
        }}

        function createStyle() {{
          return {{
            values: {{}},
            setProperty(name, value) {{
              this.values[name] = value;
            }},
            getPropertyValue(name) {{
              return this.values[name] || "";
            }},
          }};
        }}

        function createButton(initialText = "") {{
          return {{
            disabled: false,
            hidden: false,
            textContent: initialText,
            innerHTML: initialText,
            value: "",
            dataset: {{}},
            classList: createClassList(),
            style: createStyle(),
            setAttribute(name, value) {{
              this[name] = value;
            }},
            removeAttribute(name) {{
              delete this[name];
            }},
          }};
        }}

        function createInput() {{
          return {{
            value: "",
            disabled: false,
            hidden: false,
            dataset: {{}},
            classList: createClassList(),
            style: createStyle(),
            setAttribute(name, value) {{
              this[name] = value;
            }},
            removeAttribute(name) {{
              delete this[name];
            }},
          }};
        }}

        function createPanel() {{
          return {{
            innerHTML: "",
            querySelectorAll() {{
              return [];
            }},
          }};
        }}

        const state = stateModule.createInitialState();
        const statusEvents = [];
        const plannerPanel = createPanel();
        let analysisRequestCount = 0;
        const ctx = {{
          state,
          constants: {{
            TENSOR_WIDTH: 140,
            TENSOR_HEIGHT: 84,
            MIN_TENSOR_WIDTH: 120,
            MIN_TENSOR_HEIGHT: 72,
            NOTE_WIDTH: 220,
            NOTE_HEIGHT: 152,
            NOTE_MIN_WIDTH: 176,
            NOTE_MIN_HEIGHT: 152,
          }},
          dom: {{
            statusMessage: {{ textContent: "", classList: createClassList() }},
            singleModeMenuItem: createButton(),
            linearPeriodicModeMenuItem: createButton(),
            gridPeriodicModeMenuItem: createButton(),
            treeModeMenuItem: createButton(),
            benchmarkModeMenuItem: createButton(),
            toolbarModeControls: {{ hidden: true }},
            linearPeriodicPreviousCellButton: createButton("<"),
            linearPeriodicCellLabel: {{ textContent: "" }},
            linearPeriodicNextCellButton: createButton(">"),
            benchmarkCompareButton: createButton("Compare"),
            benchmarkSchemeNameInput: createInput(),
            benchmarkCompareModal: {{ classList: createClassList(), hidden: true }},
            benchmarkCompareCloseButton: createButton("Close"),
            benchmarkCompareTableBody: {{ innerHTML: "" }},
            plannerPanel,
          }},
          document: {{
            getElementById() {{
              return null;
            }},
          }},
          window: {{
            confirm: () => true,
            setTimeout,
            clearTimeout,
          }},
          getSelectedIdsByKind: () => [],
          getSelectedEntries: () => [],
          escapeHtml: (value) => String(value),
          formatIssues: (issues) => issues.map((issue) => issue.message).join(" "),
          setStatus(message, kind = "info") {{
            statusEvents.push({{ message, kind }});
          }},
          render() {{}},
          renderOverlayDecorations() {{}},
          renderSidebarTabs() {{}},
          clearGeneratedCodePreview() {{
            return false;
          }},
          bumpSpecRevision() {{
            state.specRevision += 1;
          }},
          findTensorById(tensorId) {{
            return (
              Array.isArray(state.spec?.tensors)
                ? state.spec.tensors.find((tensor) => tensor.id === tensorId)
                : null
            ) || null;
          }},
          serializeCurrentSpec: () => ({{
            schema_version: "1.0",
            network: runtime.buildSerializedSpec
              ? runtime.buildSerializedSpec()
              : state.spec,
          }}),
          apiPost: async () => {{
            analysisRequestCount += 1;
            return {{
              ok: true,
              network_output_shape: [2],
              automatic_full: {{
                status: "complete",
                summary: {{}},
              }},
              automatic_future: {{
                status: "complete",
                summary: {{}},
              }},
              automatic_past: {{
                status: "complete",
                summary: {{}},
              }},
              manual: {{
                status: "complete",
                summary: {{}},
                steps: [],
              }},
              comparisons: {{}},
            }};
          }},
        }};

        const runtime = {{}};
        const env = {{
          ctx,
          state,
          constants: ctx.constants,
          dom: ctx.dom,
          runtime,
        }};

        Object.assign(runtime, baseModule.createUtilityBaseBindings(env));
        Object.assign(runtime, specNormalizationModule.createSpecNormalizationBindings(env));
        runtime.isLinearPeriodicMode = () => false;
        runtime.isGridPeriodicMode = () => false;
        runtime.getActiveLinearPeriodicCellName = () => null;
        runtime.enforceLinearPeriodicEngineSupport = () => false;
        Object.assign(runtime, uiModule.createUtilityUiBindings(env));
        Object.assign(runtime, benchmarkModule.createUtilityBenchmarkBindings(env));
        Object.assign(ctx, runtime);
        plannerModule.registerPlannerFeature(ctx);

        state.spec = runtime.normalizeSpec({{
          id: "network_benchmark",
          name: "Benchmark demo",
          tensors: [
            {{
              id: "tensor_a",
              name: "A",
              position: {{ x: 100, y: 100 }},
              size: {{ width: 140, height: 84 }},
              indices: [
                {{
                  id: "index_a",
                  name: "i",
                  dimension: 2,
                  offset: {{ x: 0, y: 0 }},
                  metadata: {{}},
                }},
              ],
              metadata: {{}},
            }},
            {{
              id: "tensor_b",
              name: "B",
              position: {{ x: 320, y: 100 }},
              size: {{ width: 140, height: 84 }},
              indices: [
                {{
                  id: "index_b",
                  name: "i",
                  dimension: 2,
                  offset: {{ x: 0, y: 0 }},
                  metadata: {{}},
                }},
              ],
              metadata: {{}},
            }},
          ],
          edges: [],
          groups: [],
          notes: [],
          contraction_plan: {{
            id: "original_plan",
            name: "Original path",
            steps: [],
            metadata: {{}},
          }},
          metadata: {{}},
        }});

        runtime.toggleBenchmarkMode();
        if (!state.benchmarkSession.enabled || state.benchmarkSession.activePosition !== 0) {{
          throw new Error(`Expected benchmark mode to start at the base tensor network view, received ${{JSON.stringify(state.benchmarkSession)}}.`);
        }}
        if (state.benchmarkSession.schemes.length !== 1 || state.benchmarkSession.schemes[0].name !== "Original path") {{
          throw new Error(`Expected the existing manual path to seed the first benchmark scheme, received ${{JSON.stringify(state.benchmarkSession.schemes)}}.`);
        }}
        if (state.spec.contraction_plan !== null) {{
          throw new Error("Expected the base tensor network position to clear the active contraction plan.");
        }}
        ctx.renderPlanner();
        if (!plannerPanel.innerHTML.includes("Move right to open or create a contraction scheme.")) {{
          throw new Error(`Expected the benchmark planner panel to explain the base position, received ${{plannerPanel.innerHTML}}.`);
        }}

        runtime.switchBenchmarkPosition(1);
        if (!state.spec.contraction_plan || state.spec.contraction_plan.name !== "Original path") {{
          throw new Error(`Expected moving right to project scheme 1 into the live plan, received ${{JSON.stringify(state.spec.contraction_plan)}}.`);
        }}
        runtime.renameActiveBenchmarkScheme("Alpha ");
        runtime.updateToolbarState();
        if (ctx.dom.benchmarkSchemeNameInput.value !== "Alpha ") {{
          throw new Error(`Expected benchmark scheme names to preserve a typed trailing space while editing, received '${{ctx.dom.benchmarkSchemeNameInput.value}}'.`);
        }}
        runtime.renameActiveBenchmarkScheme(`${{ctx.dom.benchmarkSchemeNameInput.value}}Beta`);
        runtime.updateToolbarState();
        if (!state.spec.contraction_plan || state.spec.contraction_plan.name !== "Alpha Beta") {{
          throw new Error(`Expected benchmark scheme names to allow internal spaces, received ${{JSON.stringify(state.spec.contraction_plan)}}.`);
        }}
        await new Promise((resolve) => setTimeout(resolve, 0));
        ctx.renderPlanner();
        if (analysisRequestCount !== 1) {{
          throw new Error(`Expected entering a benchmark scheme to start contraction analysis immediately, received ${{analysisRequestCount}} requests.`);
        }}
        if (!state.contractionAnalysis || state.contractionAnalysis.status !== "ready") {{
          throw new Error(`Expected benchmark scheme analysis to be ready after the immediate refresh, received ${{JSON.stringify(state.contractionAnalysis)}}.`);
        }}
        if (!plannerPanel.innerHTML.includes("Auto future")) {{
          throw new Error(`Expected benchmark schemes to expose Auto future once analysis is ready, received ${{plannerPanel.innerHTML}}.`);
        }}
        if (!plannerPanel.innerHTML.includes("Auto past")) {{
          throw new Error(`Expected benchmark schemes to expose Auto past once analysis is ready, received ${{plannerPanel.innerHTML}}.`);
        }}
        ctx.renderPlanner();
        if (plannerPanel.innerHTML.includes("Move right to open or create a contraction scheme.")) {{
          throw new Error(`Expected the planner to leave the benchmark-base state after moving right, received ${{plannerPanel.innerHTML}}.`);
        }}
        if (
          !plannerPanel.innerHTML.includes('id="toggle-planner-mode-button"') ||
          !plannerPanel.innerHTML.includes("Contract")
        ) {{
          throw new Error(`Expected benchmark schemes to expose the Contract action, received ${{plannerPanel.innerHTML}}.`);
        }}
        const contractButtonMarkupMatch = plannerPanel.innerHTML.match(
          /<button[^>]*id="toggle-planner-mode-button"[^>]*>/
        );
        if (!contractButtonMarkupMatch) {{
          throw new Error(`Expected the planner to render a Contract button, received ${{plannerPanel.innerHTML}}.`);
        }}
        if (contractButtonMarkupMatch[0].includes("disabled")) {{
          throw new Error(`Expected the Contract action to be enabled for a benchmark scheme, received ${{plannerPanel.innerHTML}}.`);
        }}
        state.spec.contraction_plan.name = "Alpha";
        runtime.updateToolbarState();
        if (ctx.dom.linearPeriodicNextCellButton.textContent !== "+") {{
          throw new Error(`Expected the last benchmark position to expose '+', received '${{ctx.dom.linearPeriodicNextCellButton.textContent}}'.`);
        }}

        runtime.switchBenchmarkPosition(1);
        if (state.benchmarkSession.schemes.length !== 2) {{
          throw new Error(`Expected the last-position '+' action to create a new scheme, received ${{state.benchmarkSession.schemes.length}} schemes.`);
        }}
        if (!state.spec.contraction_plan || state.spec.contraction_plan.name !== "Scheme 2") {{
          throw new Error(`Expected the new scheme to become active with the default name, received ${{JSON.stringify(state.spec.contraction_plan)}}.`);
        }}

        runtime.toggleBenchmarkMode();
        if (state.benchmarkSession.enabled) {{
          throw new Error("Expected benchmark mode to discard the temporary session on exit.");
        }}
        if (!state.spec.contraction_plan || state.spec.contraction_plan.name !== "Scheme 2") {{
          throw new Error(`Expected the active benchmark scheme to become the normal manual path on exit, received ${{JSON.stringify(state.spec.contraction_plan)}}.`);
        }}
      """
    )
    script_path.write_text(script_body, encoding="utf-8")
    return script_path


def _write_planner_auto_paths_immediate_refresh_runtime_regression_script(
    tmp_path: Path,
) -> Path:
    script_path = (
        tmp_path / "planner_auto_paths_immediate_refresh_runtime_regression.mjs"
    )
    script_body = textwrap.dedent(
        f"""
        import {{ pathToFileURL }} from "node:url";

        const stateUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "state/state.js")!r}).href;
        const baseUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "utils/utilitiesBase.js")!r}).href;
        const specNormalizationUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "spec" / "specNormalization.js")!r}).href;
        const historySelectionUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "graph/historySelection.js")!r}).href;
        const plannerUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "planner/planner.js")!r}).href;

        const [
          stateModule,
          baseModule,
          specNormalizationModule,
          historySelectionModule,
          plannerModule,
        ] = await Promise.all([
          import(stateUrl),
          import(baseUrl),
          import(specNormalizationUrl),
          import(historySelectionUrl),
          import(plannerUrl),
        ]);

        function createClassList() {{
          return {{
            toggle() {{}},
            add() {{}},
            remove() {{}},
          }};
        }}

        function createStyle() {{
          return {{
            values: {{}},
            setProperty(name, value) {{
              this.values[name] = value;
            }},
            getPropertyValue(name) {{
              return this.values[name] || "";
            }},
          }};
        }}

        function createButton(initialText = "") {{
          return {{
            disabled: false,
            hidden: false,
            textContent: initialText,
            innerHTML: initialText,
            value: "",
            dataset: {{}},
            classList: createClassList(),
            style: createStyle(),
            addEventListener() {{}},
            setAttribute(name, value) {{
              this[name] = value;
            }},
            removeAttribute(name) {{
              delete this[name];
            }},
          }};
        }}

        function createPanel() {{
          return {{
            innerHTML: "",
            querySelectorAll() {{
              return [];
            }},
          }};
        }}

        const state = stateModule.createInitialState();
        const plannerPanel = createPanel();
        let analysisRequestCount = 0;
        const ctx = {{
          state,
          constants: {{
            HISTORY_LIMIT: 100,
            TENSOR_WIDTH: 140,
            TENSOR_HEIGHT: 84,
            MIN_TENSOR_WIDTH: 120,
            MIN_TENSOR_HEIGHT: 72,
            NOTE_WIDTH: 220,
            NOTE_HEIGHT: 152,
            NOTE_MIN_WIDTH: 176,
            NOTE_MIN_HEIGHT: 152,
          }},
          dom: {{
            generatedCode: {{ value: "" }},
            plannerPanel,
            statusMessage: {{ textContent: "", classList: createClassList() }},
          }},
          document: {{
            getElementById() {{
              return null;
            }},
          }},
          window: {{
            structuredClone: globalThis.structuredClone,
            crypto: globalThis.crypto,
            setTimeout,
            clearTimeout,
            confirm: () => true,
          }},
          escapeHtml: (value) => String(value),
          formatIssues: (issues) => issues.map((issue) => issue.message).join(" "),
          setStatus() {{}},
          render() {{}},
          renderOverlayDecorations() {{}},
          renderSidebarTabs() {{}},
          updateToolbarState() {{}},
          reconcileTensorOrder() {{}},
          renderGeneratedCodePreview() {{}},
          syncPendingInteractionClasses() {{}},
          clearGeneratedCodePreview() {{
            return false;
          }},
          bumpSpecRevision() {{
            state.specRevision += 1;
          }},
          getSelectedIdsByKind: () => [],
          getSelectedEntries: () => [],
          getVisibleTensors: () => state.spec?.tensors || [],
          getContractibleTensors: () => state.spec?.tensors || [],
          findTensorById(tensorId) {{
            return (
              Array.isArray(state.spec?.tensors)
                ? state.spec.tensors.find((tensor) => tensor.id === tensorId)
                : null
            ) || null;
          }},
          findGroupById: () => null,
          findIndexOwner: () => null,
          findEdgeById: () => null,
          serializeCurrentSpec: () => ({{
            schema_version: "1.0",
            network: state.spec,
          }}),
          apiPost: async () => {{
            analysisRequestCount += 1;
            return {{
              ok: true,
              network_output_shape: [2],
              automatic_full: {{
                status: "complete",
                summary: {{}},
                steps: [
                  {{
                    step_id: "auto_full_step_1",
                    left_operand_id: "tensor_a",
                    right_operand_id: "tensor_b",
                    result_operand_id: "auto_full_step_1",
                  }},
                ],
              }},
              automatic_future: {{
                status: "complete",
                summary: {{}},
                steps: [
                  {{
                    step_id: "auto_future_step_1",
                    left_operand_id: "tensor_a",
                    right_operand_id: "tensor_b",
                    result_operand_id: "auto_future_step_1",
                  }},
                ],
              }},
              automatic_past: {{
                status: "complete",
                summary: {{}},
                steps: [
                  {{
                    step_id: "step_ab",
                    left_operand_id: "tensor_a",
                    right_operand_id: "tensor_b",
                    result_operand_id: "step_ab",
                  }},
                ],
              }},
              manual: {{
                status: "complete",
                summary: {{}},
                steps: [],
              }},
              comparisons: {{}},
            }};
          }},
        }};

        const runtime = {{}};
        const env = {{
          ctx,
          state,
          constants: ctx.constants,
          dom: ctx.dom,
          runtime,
        }};

        Object.assign(runtime, baseModule.createUtilityBaseBindings(env));
        Object.assign(runtime, specNormalizationModule.createSpecNormalizationBindings(env));
        Object.assign(runtime, {{
          isLinearPeriodicMode: () => false,
          isGridPeriodicMode: () => false,
          isForMode: () => false,
          isContractionSceneVisible: () => false,
          isInspectingPastStage: () => false,
        }});
        Object.assign(ctx, runtime);
        historySelectionModule.registerHistorySelection(ctx);
        plannerModule.registerPlannerFeature(ctx);

        state.activeSidebarTab = "planner";
        state.spec = runtime.normalizeSpec({{
          id: "network_planner_refresh",
          name: "Planner refresh demo",
          tensors: [
            {{
              id: "tensor_a",
              name: "A",
              position: {{ x: 100, y: 100 }},
              size: {{ width: 140, height: 84 }},
              indices: [
                {{
                  id: "index_a",
                  name: "i",
                  dimension: 2,
                  offset: {{ x: 0, y: 0 }},
                  metadata: {{}},
                }},
              ],
              metadata: {{}},
            }},
            {{
              id: "tensor_b",
              name: "B",
              position: {{ x: 320, y: 100 }},
              size: {{ width: 140, height: 84 }},
              indices: [
                {{
                  id: "index_b",
                  name: "i",
                  dimension: 2,
                  offset: {{ x: 0, y: 0 }},
                  metadata: {{}},
                }},
              ],
              metadata: {{}},
            }},
          ],
          edges: [],
          groups: [],
          notes: [],
          contraction_plan: {{
            id: "manual_plan",
            name: "Manual path",
            steps: [],
            metadata: {{}},
          }},
          metadata: {{}},
        }});
        state.contractionAnalysis = {{
          status: "ready",
          payload: {{
            network_output_shape: [2],
            automatic_full: {{ status: "complete", summary: {{}}, steps: [] }},
            automatic_future: {{ status: "complete", summary: {{}}, steps: [] }},
            automatic_past: {{ status: "complete", summary: {{}}, steps: [] }},
            manual: {{ status: "complete", summary: {{}}, steps: [] }},
            comparisons: {{}},
          }},
        }};

        ctx.applyDesignChange(
          () => {{
            state.spec.metadata.last_change = "refresh-auto-paths";
          }},
          {{
            statusMessage: "Updated planner state.",
          }}
        );

        if (analysisRequestCount !== 1) {{
          throw new Error(
            `Expected planner-visible design changes to refresh contraction analysis immediately, received ${{analysisRequestCount}} requests.`
          );
        }}
        if (!state.contractionAnalysis || state.contractionAnalysis.status !== "loading") {{
          throw new Error(
            `Expected planner-visible design changes to enter loading immediately, received ${{JSON.stringify(state.contractionAnalysis)}}.`
          );
        }}

        await new Promise((resolve) => setTimeout(resolve, 0));
        ctx.renderPlanner();

        if (!state.contractionAnalysis || state.contractionAnalysis.status !== "ready") {{
          throw new Error(
            `Expected the refreshed planner analysis to become ready, received ${{JSON.stringify(state.contractionAnalysis)}}.`
          );
        }}
        if (!plannerPanel.innerHTML.includes("Auto future")) {{
          throw new Error(
            `Expected the refreshed planner analysis to expose Auto future, received ${{plannerPanel.innerHTML}}.`
          );
        }}
        if (!plannerPanel.innerHTML.includes("Auto past")) {{
          throw new Error(
            `Expected the refreshed planner analysis to expose Auto past, received ${{plannerPanel.innerHTML}}.`
          );
        }}
        """
    )
    script_path.write_text(script_body, encoding="utf-8")
    return script_path


def _write_benchmark_compare_export_runtime_regression_script(
    tmp_path: Path,
) -> Path:
    script_path = tmp_path / "benchmark_compare_export_runtime_regression.mjs"
    script_body = textwrap.dedent(
        f"""
        import {{ pathToFileURL }} from "node:url";

        const stateUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "state/state.js")!r}).href;
        const baseUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "utils/utilitiesBase.js")!r}).href;
        const specNormalizationUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "spec" / "specNormalization.js")!r}).href;
        const benchmarkUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "utils/utilitiesBenchmark.js")!r}).href;
        const uiUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "utils/utilitiesUi.js")!r}).href;

        const [
          stateModule,
          baseModule,
          specNormalizationModule,
          benchmarkModule,
          uiModule,
        ] = await Promise.all([
          import(stateUrl),
          import(baseUrl),
          import(specNormalizationUrl),
          import(benchmarkUrl),
          import(uiUrl),
        ]);

        function createClassList() {{
          return {{
            toggle() {{}},
            add() {{}},
            remove() {{}},
          }};
        }}

        function createStyle() {{
          return {{
            values: {{}},
            setProperty(name, value) {{
              this.values[name] = value;
            }},
            getPropertyValue(name) {{
              return this.values[name] || "";
            }},
          }};
        }}

        function createButton(initialText = "") {{
          return {{
            disabled: false,
            hidden: false,
            textContent: initialText,
            innerHTML: initialText,
            value: "",
            dataset: {{}},
            classList: createClassList(),
            style: createStyle(),
            addEventListener() {{}},
            setAttribute(name, value) {{
              this[name] = value;
            }},
            removeAttribute(name) {{
              delete this[name];
            }},
          }};
        }}

        const state = stateModule.createInitialState();
        const exportEvents = [];
        const ctx = {{
          state,
          constants: {{
            TENSOR_WIDTH: 140,
            TENSOR_HEIGHT: 84,
            MIN_TENSOR_WIDTH: 120,
            MIN_TENSOR_HEIGHT: 72,
            NOTE_WIDTH: 220,
            NOTE_HEIGHT: 152,
            NOTE_MIN_WIDTH: 176,
            NOTE_MIN_HEIGHT: 152,
          }},
          dom: {{
            statusMessage: {{ textContent: "", classList: createClassList() }},
            singleModeMenuItem: createButton(),
            linearPeriodicModeMenuItem: createButton(),
            gridPeriodicModeMenuItem: createButton(),
            treeModeMenuItem: createButton(),
            benchmarkModeMenuItem: createButton(),
            toolbarModeControls: {{ hidden: true }},
            linearPeriodicPreviousCellButton: createButton("<"),
            linearPeriodicCellLabel: {{ textContent: "" }},
            linearPeriodicNextCellButton: createButton(">"),
            benchmarkSchemeNameInput: createButton(),
            benchmarkCompareButton: createButton("Compare"),
            benchmarkCompareModal: {{ classList: createClassList(), hidden: true }},
            benchmarkCompareBackdrop: createButton(),
            benchmarkCompareCloseButton: createButton("Close"),
            benchmarkCompareTableBody: {{ innerHTML: "" }},
            benchmarkCompareExportCsvButton: createButton("CSV"),
            benchmarkCompareExportTextButton: createButton("TXT"),
            benchmarkCompareCopyLatexButton: createButton("Copy LaTeX"),
          }},
          document: {{
            getElementById() {{
              return null;
            }},
          }},
          window: {{
            confirm: () => true,
            setTimeout,
            clearTimeout,
            Blob,
          }},
          getSelectedIdsByKind: () => [],
          getSelectedEntries: () => [],
          escapeHtml: (value) => String(value),
          formatIssues: (issues) => issues.map((issue) => issue.message).join(" "),
          setStatus() {{}},
          render() {{}},
          renderOverlayDecorations() {{}},
          renderSidebarTabs() {{}},
          clearGeneratedCodePreview() {{
            return false;
          }},
          bumpSpecRevision() {{
            state.specRevision += 1;
          }},
          findTensorById(tensorId) {{
            return (
              Array.isArray(state.spec?.tensors)
                ? state.spec.tensors.find((tensor) => tensor.id === tensorId)
                : null
            ) || null;
          }},
          downloadText(filename, text, contentType) {{
            exportEvents.push({{ kind: "download", filename, text, contentType }});
          }},
          async copyText(text) {{
            exportEvents.push({{ kind: "copy", text }});
          }},
          apiPost: async (path, payload) => {{
            if (path !== "/api/analyze-contraction") {{
              throw new Error(`Unexpected API path: ${{path}}.`);
            }}
            const planName = payload?.spec?.network?.contraction_plan?.name || "";
            if (planName === "Alpha") {{
              return {{
                ok: true,
                manual: {{
                  status: "complete",
                  summary: {{
                    total_estimated_flops: 10,
                    total_estimated_macs: 20,
                    peak_intermediate_size: 30,
                    peak_intermediate_bytes: 40,
                  }},
                }},
              }};
            }}
            if (planName === "Beta & Co") {{
              return {{
                ok: true,
                manual: {{
                  status: "complete",
                  summary: {{
                    total_estimated_flops: 12,
                    total_estimated_macs: 18,
                    peak_intermediate_size: 24,
                    peak_intermediate_bytes: 36,
                  }},
                }},
              }};
            }}
            if (planName === "Gamma") {{
              return {{
                ok: true,
                manual: {{
                  status: "incomplete",
                  summary: {{
                    total_estimated_flops: 7,
                    total_estimated_macs: 9,
                    peak_intermediate_size: 11,
                    peak_intermediate_bytes: 44,
                  }},
                }},
              }};
            }}
            throw new Error(`Unexpected scheme name: ${{planName}}.`);
          }},
        }};

        const runtime = {{}};
        const env = {{
          ctx,
          state,
          constants: ctx.constants,
          dom: ctx.dom,
          runtime,
        }};

        Object.assign(runtime, baseModule.createUtilityBaseBindings(env));
        Object.assign(runtime, specNormalizationModule.createSpecNormalizationBindings(env));
        runtime.isLinearPeriodicMode = () => false;
        runtime.isGridPeriodicMode = () => false;
        runtime.getActiveLinearPeriodicCellName = () => null;
        runtime.enforceLinearPeriodicEngineSupport = () => false;
        Object.assign(runtime, uiModule.createUtilityUiBindings(env));
        Object.assign(runtime, benchmarkModule.createUtilityBenchmarkBindings(env));
        Object.assign(ctx, runtime);

        state.schemaVersion = "1.0";
        state.spec = runtime.normalizeSpec({{
          id: "network_benchmark_exports",
          name: "Benchmark Demo",
          tensors: [
            {{
              id: "tensor_a",
              name: "A",
              position: {{ x: 100, y: 100 }},
              size: {{ width: 140, height: 84 }},
              indices: [
                {{
                  id: "index_a",
                  name: "i",
                  dimension: 2,
                  offset: {{ x: 0, y: 0 }},
                  metadata: {{}},
                }},
              ],
              metadata: {{}},
            }},
            {{
              id: "tensor_b",
              name: "B",
              position: {{ x: 320, y: 100 }},
              size: {{ width: 140, height: 84 }},
              indices: [
                {{
                  id: "index_b",
                  name: "i",
                  dimension: 2,
                  offset: {{ x: 0, y: 0 }},
                  metadata: {{}},
                }},
              ],
              metadata: {{}},
            }},
          ],
          edges: [],
          groups: [],
          notes: [],
          contraction_plan: {{
            id: "alpha_plan",
            name: "Alpha",
            steps: [],
            metadata: {{}},
          }},
          metadata: {{}},
        }});

        runtime.toggleBenchmarkMode();
        runtime.switchBenchmarkPosition(1);
        runtime.switchBenchmarkPosition(1);
        runtime.renameActiveBenchmarkScheme("Beta & Co");
        runtime.switchBenchmarkPosition(1);
        runtime.renameActiveBenchmarkScheme("Gamma");
        await runtime.openBenchmarkCompareModal();

        runtime.exportBenchmarkCompareAsCsv();
        runtime.exportBenchmarkCompareAsText();
        await runtime.copyBenchmarkCompareAsLatex();

        const csvExport = exportEvents.find((entry) => entry.kind === "download" && entry.filename.endsWith(".csv"));
        if (!csvExport) {{
          throw new Error(`Expected a CSV download event, received ${{JSON.stringify(exportEvents)}}.`);
        }}
        if (csvExport.filename !== "benchmark-demo-benchmark-compare.csv") {{
          throw new Error(`Unexpected CSV filename: ${{csvExport.filename}}.`);
        }}
        if (!csvExport.text.includes("Alpha,10,20,30,40 bytes")) {{
          throw new Error(`Expected the CSV export to include the Alpha metrics, received ${{csvExport.text}}.`);
        }}
        if (!csvExport.text.includes("Gamma,7,9,11,44 bytes")) {{
          throw new Error(`Expected the CSV export to include the partial Gamma metrics, received ${{csvExport.text}}.`);
        }}

        const textExport = exportEvents.find((entry) => entry.kind === "download" && entry.filename.endsWith(".txt"));
        if (!textExport) {{
          throw new Error(`Expected a TXT download event, received ${{JSON.stringify(exportEvents)}}.`);
        }}
        if (!textExport.text.includes("Beta & Co") || !textExport.text.includes("Peak Memory")) {{
          throw new Error(`Expected the TXT export to include the table headers and scheme names, received ${{textExport.text}}.`);
        }}
        if (!textExport.text.includes("Gamma") || !textExport.text.includes("44 bytes")) {{
          throw new Error(`Expected the TXT export to include the partial Gamma metrics, received ${{textExport.text}}.`);
        }}

        const latexExport = exportEvents.find((entry) => entry.kind === "copy");
        if (!latexExport) {{
          throw new Error(`Expected a LaTeX copy event, received ${{JSON.stringify(exportEvents)}}.`);
        }}
        if (!latexExport.text.includes("\\\\begin{{tabular}}{{lrrrr}}")) {{
          throw new Error(`Expected the copied LaTeX to include a tabular block, received ${{latexExport.text}}.`);
        }}
        if (!latexExport.text.includes("Beta \\\\& Co & 12 & 18 & 24 & 36 bytes \\\\\\\\")) {{
          throw new Error(`Expected the copied LaTeX to include the escaped Beta row, received ${{latexExport.text}}.`);
        }}
        if (!latexExport.text.includes("Gamma & 7 & 9 & 11 & 44 bytes \\\\\\\\")) {{
          throw new Error(`Expected the copied LaTeX to include the partial Gamma row, received ${{latexExport.text}}.`);
        }}
        """
    )
    script_path.write_text(script_body, encoding="utf-8")
    return script_path


def _write_for_mode_reserved_operand_runtime_regression_script(
    tmp_path: Path,
) -> Path:
    script_path = tmp_path / "for_mode_reserved_operands_runtime_regression.mjs"
    state_runtime_path = tmp_path / "state.runtime.mjs"
    utilities_runtime_path = tmp_path / "utilities.runtime.mjs"
    planner_runtime_path = tmp_path / "planner.runtime.mjs"
    contraction_scene_runtime_path = tmp_path / "contractionScene.runtime.mjs"
    _copy_runtime_bundle(
        tmp_path,
        {
            "state.runtime.mjs": "state/state.js",
            "utilities.runtime.mjs": "utils/utilities.js",
            "planner.runtime.mjs": "planner/planner.js",
            "plannerSupport.js": "planner/plannerSupport.js",
            "plannerRenderers.js": "planner/plannerRenderers.js",
            "contractionScene.runtime.mjs": "graph/contractionScene.js",
        },
        _RUNTIME_EDITOR_SUPPORT_MODULES,
    )
    script_body = textwrap.dedent(
        f"""
        import {{ pathToFileURL }} from "node:url";

        function createClassList() {{
          return {{
            add() {{}},
            remove() {{}},
            toggle() {{}},
          }};
        }}

        function createButton() {{
          return {{
            disabled: false,
            classList: createClassList(),
            addEventListener() {{}},
          }};
        }}

        function createPlannerPanel() {{
          return {{
            innerHTML: "",
            querySelectorAll() {{
              return [];
            }},
          }};
        }}

        function emptyGraphSection() {{
          return {{
            tensors: [],
            groups: [],
            edges: [],
            notes: [],
            contraction_plan: null,
            metadata: {{}},
          }};
        }}

        function createLinearPeriodicSpec() {{
          return {{
            id: "network_linear_periodic_reserved",
            name: "linear-periodic-reserved",
            tensors: [],
            groups: [],
            edges: [],
            notes: [],
            contraction_plan: null,
            metadata: {{}},
            linear_periodic_chain: {{
              active_cell: "initial",
              metadata: {{}},
              initial_cell: {{
                tensors: [
                  {{
                    id: "initial_tensor",
                    name: "Initial tensor",
                    position: {{ x: 100, y: 140 }},
                    size: {{ width: 140, height: 84 }},
                    metadata: {{}},
                    indices: [
                      {{
                        id: "initial_phys",
                        name: "phys",
                        dimension: 2,
                        offset: {{ x: -38, y: 0 }},
                        metadata: {{}},
                      }},
                      {{
                        id: "initial_to_next",
                        name: "bond",
                        dimension: 3,
                        offset: {{ x: 38, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                  {{
                    id: "initial_next_boundary",
                    name: "Next cell",
                    position: {{ x: 320, y: 140 }},
                    size: {{ width: 140, height: 84 }},
                    linear_periodic_role: "next",
                    metadata: {{}},
                    indices: [
                      {{
                        id: "initial_next_slot_1",
                        name: "slot_1",
                        dimension: 3,
                        offset: {{ x: -38, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                ],
                groups: [],
                edges: [
                  {{
                    id: "initial_next_edge",
                    name: "bond_1",
                    left: {{
                      tensor_id: "initial_tensor",
                      index_id: "initial_to_next",
                    }},
                    right: {{
                      tensor_id: "initial_next_boundary",
                      index_id: "initial_next_slot_1",
                    }},
                    metadata: {{}},
                  }},
                ],
                notes: [],
                contraction_plan: null,
                metadata: {{}},
              }},
              periodic_cell: {{
                tensors: [
                  {{
                    id: "periodic_tensor",
                    name: "Periodic tensor",
                    position: {{ x: 160, y: 140 }},
                    size: {{ width: 140, height: 84 }},
                    metadata: {{}},
                    indices: [
                      {{
                        id: "periodic_from_previous",
                        name: "prev",
                        dimension: 3,
                        offset: {{ x: -38, y: 0 }},
                        metadata: {{}},
                      }},
                      {{
                        id: "periodic_to_next",
                        name: "next",
                        dimension: 3,
                        offset: {{ x: 38, y: 0 }},
                        metadata: {{}},
                      }},
                      {{
                        id: "periodic_phys",
                        name: "phys",
                        dimension: 2,
                        offset: {{ x: 0, y: -24 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                  {{
                    id: "periodic_previous_boundary",
                    name: "Previous cell",
                    position: {{ x: -60, y: 140 }},
                    size: {{ width: 140, height: 84 }},
                    linear_periodic_role: "previous",
                    metadata: {{}},
                    indices: [
                      {{
                        id: "periodic_previous_slot_1",
                        name: "slot_1",
                        dimension: 3,
                        offset: {{ x: -38, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                  {{
                    id: "periodic_next_boundary",
                    name: "Next cell",
                    position: {{ x: 380, y: 140 }},
                    size: {{ width: 140, height: 84 }},
                    linear_periodic_role: "next",
                    metadata: {{}},
                    indices: [
                      {{
                        id: "periodic_next_slot_1",
                        name: "slot_1",
                        dimension: 3,
                        offset: {{ x: 38, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                ],
                groups: [],
                edges: [
                  {{
                    id: "periodic_previous_edge",
                    name: "bond_1",
                    left: {{
                      tensor_id: "periodic_previous_boundary",
                      index_id: "periodic_previous_slot_1",
                    }},
                    right: {{
                      tensor_id: "periodic_tensor",
                      index_id: "periodic_from_previous",
                    }},
                    metadata: {{}},
                  }},
                  {{
                    id: "periodic_next_edge",
                    name: "bond_2",
                    left: {{
                      tensor_id: "periodic_tensor",
                      index_id: "periodic_to_next",
                    }},
                    right: {{
                      tensor_id: "periodic_next_boundary",
                      index_id: "periodic_next_slot_1",
                    }},
                    metadata: {{}},
                  }},
                ],
                notes: [],
                contraction_plan: null,
                metadata: {{}},
              }},
              final_cell: {{
                tensors: [
                  {{
                    id: "final_tensor",
                    name: "Final tensor",
                    position: {{ x: 180, y: 140 }},
                    size: {{ width: 140, height: 84 }},
                    metadata: {{}},
                    indices: [
                      {{
                        id: "final_from_previous",
                        name: "prev",
                        dimension: 3,
                        offset: {{ x: -38, y: 0 }},
                        metadata: {{}},
                      }},
                      {{
                        id: "final_phys",
                        name: "phys",
                        dimension: 2,
                        offset: {{ x: 38, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                  {{
                    id: "final_previous_boundary",
                    name: "Previous cell",
                    position: {{ x: -40, y: 140 }},
                    size: {{ width: 140, height: 84 }},
                    linear_periodic_role: "previous",
                    metadata: {{}},
                    indices: [
                      {{
                        id: "final_previous_slot_1",
                        name: "slot_1",
                        dimension: 3,
                        offset: {{ x: -38, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                ],
                groups: [],
                edges: [
                  {{
                    id: "final_previous_edge",
                    name: "bond_1",
                    left: {{
                      tensor_id: "final_previous_boundary",
                      index_id: "final_previous_slot_1",
                    }},
                    right: {{
                      tensor_id: "final_tensor",
                      index_id: "final_from_previous",
                    }},
                    metadata: {{}},
                  }},
                ],
                notes: [],
                contraction_plan: null,
                metadata: {{}},
              }},
            }},
          }};
        }}

        function createGridPeriodicSpec() {{
          return {{
            id: "network_grid_periodic_reserved",
            name: "grid-periodic-reserved",
            tensors: [],
            groups: [],
            edges: [],
            notes: [],
            contraction_plan: null,
            metadata: {{}},
            grid_periodic_grid: {{
              active_cell: "center",
              metadata: {{}},
              top_left_cell: emptyGraphSection(),
              top_cell: emptyGraphSection(),
              top_right_cell: emptyGraphSection(),
              left_cell: emptyGraphSection(),
              center_cell: {{
                tensors: [
                  {{
                    id: "grid_tensor",
                    name: "Grid tensor",
                    position: {{ x: 180, y: 140 }},
                    size: {{ width: 140, height: 84 }},
                    metadata: {{}},
                    indices: [
                      {{
                        id: "grid_from_left",
                        name: "left",
                        dimension: 3,
                        offset: {{ x: -38, y: 0 }},
                        metadata: {{}},
                      }},
                      {{
                        id: "grid_to_right",
                        name: "right",
                        dimension: 5,
                        offset: {{ x: 38, y: 0 }},
                        metadata: {{}},
                      }},
                      {{
                        id: "grid_phys",
                        name: "phys",
                        dimension: 2,
                        offset: {{ x: 0, y: -24 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                  {{
                    id: "grid_left_boundary",
                    name: "Left cell",
                    position: {{ x: -40, y: 140 }},
                    size: {{ width: 140, height: 84 }},
                    grid_periodic_role: "left",
                    metadata: {{}},
                    indices: [
                      {{
                        id: "grid_left_slot",
                        name: "left_slot",
                        dimension: 3,
                        offset: {{ x: -38, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                  {{
                    id: "grid_right_boundary",
                    name: "Right cell",
                    position: {{ x: 400, y: 140 }},
                    size: {{ width: 140, height: 84 }},
                    grid_periodic_role: "right",
                    metadata: {{}},
                    indices: [
                      {{
                        id: "grid_right_slot",
                        name: "right_slot",
                        dimension: 5,
                        offset: {{ x: 38, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                ],
                groups: [],
                edges: [
                  {{
                    id: "grid_left_edge",
                    name: "grid_left",
                    left: {{
                      tensor_id: "grid_left_boundary",
                      index_id: "grid_left_slot",
                    }},
                    right: {{
                      tensor_id: "grid_tensor",
                      index_id: "grid_from_left",
                    }},
                    metadata: {{}},
                  }},
                  {{
                    id: "grid_right_edge",
                    name: "grid_right",
                    left: {{
                      tensor_id: "grid_tensor",
                      index_id: "grid_to_right",
                    }},
                    right: {{
                      tensor_id: "grid_right_boundary",
                      index_id: "grid_right_slot",
                    }},
                    metadata: {{}},
                  }},
                ],
                notes: [],
                contraction_plan: {{
                  id: "grid_plan",
                  name: "Manual path",
                  steps: [
                    {{
                      id: "grid_left_step",
                      left_operand_id: "__grid_left__",
                      right_operand_id: "grid_tensor",
                      metadata: {{}},
                    }},
                  ],
                  view_snapshots: [],
                  metadata: {{}},
                }},
                metadata: {{}},
              }},
              right_cell: emptyGraphSection(),
              bottom_left_cell: emptyGraphSection(),
              bottom_cell: emptyGraphSection(),
              bottom_right_cell: emptyGraphSection(),
            }},
          }};
        }}

        function createTreePeriodicSpec() {{
          return {{
            id: "network_tree_periodic_reserved",
            name: "tree-periodic-reserved",
            tensors: [],
            groups: [],
            edges: [],
            notes: [],
            contraction_plan: null,
            metadata: {{}},
            tree_periodic_tree: {{
              active_cell: "branch",
              branching_factor: 2,
              metadata: {{}},
              root_cell: emptyGraphSection(),
              branch_cell: {{
                tensors: [
                  {{
                    id: "tree_branch_tensor",
                    name: "Branch tensor",
                    position: {{ x: 180, y: 140 }},
                    size: {{ width: 140, height: 84 }},
                    metadata: {{}},
                    indices: [
                      {{
                        id: "tree_from_parent",
                        name: "parent",
                        dimension: 3,
                        offset: {{ x: 0, y: -24 }},
                        metadata: {{}},
                      }},
                      {{
                        id: "tree_to_child_0",
                        name: "child_0",
                        dimension: 5,
                        offset: {{ x: -38, y: 0 }},
                        metadata: {{}},
                      }},
                      {{
                        id: "tree_phys",
                        name: "phys",
                        dimension: 2,
                        offset: {{ x: 38, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                  {{
                    id: "tree_parent_boundary",
                    name: "Parent cell",
                    position: {{ x: 180, y: -60 }},
                    size: {{ width: 140, height: 84 }},
                    tree_periodic_role: "parent",
                    metadata: {{}},
                    indices: [
                      {{
                        id: "tree_parent_slot",
                        name: "parent_slot",
                        dimension: 3,
                        offset: {{ x: 0, y: -24 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                  {{
                    id: "tree_child_0_boundary",
                    name: "Child 0",
                    position: {{ x: -40, y: 260 }},
                    size: {{ width: 140, height: 84 }},
                    tree_periodic_role: "child",
                    tree_periodic_child_index: 0,
                    metadata: {{}},
                    indices: [
                      {{
                        id: "tree_child_0_slot",
                        name: "child_0_slot",
                        dimension: 5,
                        offset: {{ x: -38, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                  {{
                    id: "tree_child_1_boundary",
                    name: "Child 1",
                    position: {{ x: 400, y: 260 }},
                    size: {{ width: 140, height: 84 }},
                    tree_periodic_role: "child",
                    tree_periodic_child_index: 1,
                    metadata: {{}},
                    indices: [
                      {{
                        id: "tree_child_1_slot",
                        name: "child_1_slot",
                        dimension: 7,
                        offset: {{ x: 38, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                  }},
                ],
                groups: [],
                edges: [
                  {{
                    id: "tree_parent_edge",
                    name: "tree_parent",
                    left: {{
                      tensor_id: "tree_parent_boundary",
                      index_id: "tree_parent_slot",
                    }},
                    right: {{
                      tensor_id: "tree_branch_tensor",
                      index_id: "tree_from_parent",
                    }},
                    metadata: {{}},
                  }},
                  {{
                    id: "tree_child_0_edge",
                    name: "tree_child_0",
                    left: {{
                      tensor_id: "tree_branch_tensor",
                      index_id: "tree_to_child_0",
                    }},
                    right: {{
                      tensor_id: "tree_child_0_boundary",
                      index_id: "tree_child_0_slot",
                    }},
                    metadata: {{}},
                  }},
                ],
                notes: [],
                contraction_plan: {{
                  id: "tree_plan",
                  name: "Manual path",
                  steps: [
                    {{
                      id: "tree_parent_step",
                      left_operand_id: "__tree_parent__",
                      right_operand_id: "tree_branch_tensor",
                      metadata: {{}},
                    }},
                    {{
                      id: "tree_child_step",
                      left_operand_id: "tree_parent_step",
                      right_operand_id: "__tree_child_0__",
                      metadata: {{}},
                    }},
                  ],
                  view_snapshots: [],
                  metadata: {{}},
                }},
                metadata: {{}},
              }},
              leaf_cell: emptyGraphSection(),
            }},
          }};
        }}

        function assertSceneHasOperandIndex(scene, operandId, indexName, label) {{
          if (!scene) {{
            throw new Error(`${{label}} did not build a visible contraction scene.`);
          }}
          const operand = scene.operandMap[operandId];
          if (!operand) {{
            throw new Error(`${{label}} did not include operand ${{operandId}}.`);
          }}
          if (!operand.indices.some((index) => index.name === indexName)) {{
            throw new Error(`${{label}} did not keep the ${{indexName}} interface visible.`);
          }}
        }}

        const [stateModule, utilitiesModule, plannerModule, contractionSceneModule] =
          await Promise.all([
            import(pathToFileURL({json.dumps(str(state_runtime_path))}).href),
            import(pathToFileURL({json.dumps(str(utilities_runtime_path))}).href),
            import(pathToFileURL({json.dumps(str(planner_runtime_path))}).href),
            import(pathToFileURL({json.dumps(str(contraction_scene_runtime_path))}).href),
          ]);
        const {{ createInitialState }} = stateModule;
        const {{ registerUtilities }} = utilitiesModule;
        const {{ registerPlannerFeature }} = plannerModule;
        const {{ registerContractionScene }} = contractionSceneModule;

        const ctx = {{
          state: createInitialState(),
          constants: {{
            TENSOR_WIDTH: 140,
            TENSOR_HEIGHT: 84,
            MIN_TENSOR_WIDTH: 96,
            MIN_TENSOR_HEIGHT: 60,
            INDEX_RADIUS: 10,
            INDEX_PADDING: 6,
            NOTE_WIDTH: 220,
            NOTE_HEIGHT: 120,
            NOTE_MIN_WIDTH: 120,
            NOTE_MIN_HEIGHT: 90,
            HISTORY_LIMIT: 100,
            REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
            DEFAULT_INDEX_SLOTS: [
              {{ x: -38, y: 0 }},
              {{ x: 38, y: 0 }},
              {{ x: 0, y: -24 }},
              {{ x: 0, y: 24 }},
            ],
          }},
          dom: {{
            workspace: {{}},
            statusMessage: {{ textContent: "", classList: createClassList() }},
            propertiesPanel: {{ innerHTML: "" }},
            generatedCode: {{ value: "" }},
            engineSelect: {{ options: [], value: "tensornetwork" }},
            collectionFormatSelect: {{ options: [], value: "list" }},
            exportFormatSelect: {{ value: "py" }},
            addNoteButton: createButton(),
            connectButton: {{ classList: createClassList() }},
            loadInput: {{}},
            undoButton: createButton(),
            redoButton: createButton(),
            exportButton: createButton(),
            toggleLinearPeriodicButton: {{ classList: createClassList() }},
            linearPeriodicPreviousCellButton: createButton(),
            linearPeriodicCellLabel: {{ textContent: "" }},
            linearPeriodicNextCellButton: createButton(),
            templateSelect: {{ value: "" }},
            templateParameterPanel: {{ hidden: true }},
            templateGraphSizeLabel: {{ textContent: "" }},
            templateGraphSizeInput: {{ value: "2", min: "1" }},
            templateBondDimensionInput: {{ value: "3", min: "1" }},
            templatePhysicalDimensionInput: {{ value: "2", min: "1" }},
            insertTemplateButton: createButton(),
            createGroupButton: createButton(),
            helpButton: createButton(),
            helpModal: {{ classList: createClassList() }},
            helpBackdrop: createButton(),
            helpCloseButton: createButton(),
            canvasShell: {{
              getBoundingClientRect() {{
                return {{ left: 0, top: 0, width: 1000, height: 800 }};
              }},
            }},
            groupLayer: {{}},
            resizeLayer: {{}},
            notesLayer: {{}},
            selectionBox: {{}},
            minimapCanvas: {{}},
            sidebar: {{}},
            plannerPanel: createPlannerPanel(),
            generateButton: createButton(),
          }},
          apiGet: async () => null,
          apiPost: async () => null,
          window: {{
            structuredClone: globalThis.structuredClone,
            crypto: globalThis.crypto,
            setTimeout,
            clearTimeout,
            confirm: () => true,
          }},
          document: {{
            getElementById() {{
              return createButton();
            }},
            querySelectorAll() {{
              return [];
            }},
          }},
          cytoscape: null,
          tensorWidth: (tensor) => tensor?.size?.width ?? 140,
          tensorHeight: (tensor) => tensor?.size?.height ?? 84,
          render: () => {{}},
          renderOverlayDecorations: () => {{}},
          renderMinimap: () => {{}},
          renderPlanner: () => {{}},
          renderSidebarTabs: () => {{}},
          refreshContractionAnalysis: () => {{}},
          syncPendingInteractionClasses: () => {{}},
          setActiveSidebarTab: () => {{}},
        }};

        registerUtilities(ctx);
        registerContractionScene(ctx);
        registerPlannerFeature(ctx);

        ctx.state.selectedEngine = "tensornetwork";
        ctx.state.selectedCollectionFormat = "list";
        ctx.state.spec = ctx.normalizeSpec(createLinearPeriodicSpec());

        if (ctx.resolvePlannerOperandId("initial_next_boundary") !== "__linear_next__") {{
          throw new Error("The initial next boundary did not resolve to the reserved next operand id.");
        }}

        ctx.applyManualContractionStep("initial_tensor", "__linear_next__");
        ctx.syncCurrentGraphIntoLinearPeriodicChain();
        const initialSteps =
          ctx.state.spec.linear_periodic_chain.initial_cell.contraction_plan.steps;
        if (initialSteps.length !== 1 || initialSteps[0].right_operand_id !== "__linear_next__") {{
          throw new Error("The initial cell did not persist the reserved next operand id.");
        }}
        const initialScene = ctx.buildContractionScene();
        assertSceneHasOperandIndex(
          initialScene,
          initialSteps[0].id,
          "bond",
          "Initial carry scene"
        );

        ctx.switchLinearPeriodicCell(1);
        if (ctx.resolvePlannerOperandId("periodic_previous_boundary") !== "__linear_previous__") {{
          throw new Error("The periodic previous boundary did not resolve to the reserved previous operand id.");
        }}
        if (ctx.resolvePlannerOperandId("periodic_next_boundary") !== "__linear_next__") {{
          throw new Error("The periodic next boundary did not resolve to the reserved next operand id.");
        }}

        ctx.state.spec.contraction_plan = {{
          id: "periodic_plan",
          name: "Manual path",
          steps: [
            {{
              id: "periodic_prev_step",
              left_operand_id: "__linear_previous__",
              right_operand_id: "periodic_tensor",
              metadata: {{}},
            }},
            {{
              id: "periodic_carry_step",
              left_operand_id: "periodic_prev_step",
              right_operand_id: "__linear_next__",
              metadata: {{}},
            }},
          ],
          view_snapshots: [],
          metadata: {{}},
        }};
        ctx.repairContractionPlan();
        const periodicState = ctx.buildContractionOperandState();
        const periodicOperandIds = periodicState.activeOperands.map((operand) => operand.id);
        if (periodicState.validSteps.length !== 2) {{
          throw new Error(`Expected 2 valid periodic steps, received ${{periodicState.validSteps.length}}.`);
        }}
        if (!periodicOperandIds.includes("periodic_carry_step")) {{
          throw new Error("The periodic carry step should remain active after the next contraction.");
        }}
        if (periodicOperandIds.includes("__linear_previous__") || periodicOperandIds.includes("__linear_next__")) {{
          throw new Error("Reserved carry operands should not remain active after the periodic carry plan finishes.");
        }}

        const snapshots = ctx.ensureContractionViewSnapshots();
        if (!Array.isArray(snapshots) || snapshots.length !== 3) {{
          throw new Error(`Expected 3 snapshots for the periodic plan, received ${{snapshots && snapshots.length}}.`);
        }}
        const periodicScene = ctx.buildContractionScene();
        if (periodicScene.totalStepCount !== 2 || periodicScene.appliedStepCount !== 2) {{
          throw new Error("The periodic carry scene did not stay on the latest two-step scheme.");
        }}
        assertSceneHasOperandIndex(
          periodicScene,
          "periodic_carry_step",
          "next",
          "Periodic carry scene"
        );

        ctx.syncCurrentGraphIntoLinearPeriodicChain();
        const periodicSteps =
          ctx.state.spec.linear_periodic_chain.periodic_cell.contraction_plan.steps;
        if (
          periodicSteps.length !== 2 ||
          periodicSteps[0].left_operand_id !== "__linear_previous__" ||
          periodicSteps[1].right_operand_id !== "__linear_next__"
        ) {{
          throw new Error("The periodic cell did not preserve the reserved carry operands when syncing back into the chain.");
        }}

        ctx.switchLinearPeriodicCell(2);
        ctx.state.spec.contraction_plan = {{
          id: "final_plan",
          name: "Manual path",
          steps: [
            {{
              id: "final_prev_step",
              left_operand_id: "__linear_previous__",
              right_operand_id: "final_tensor",
              metadata: {{}},
            }},
          ],
          view_snapshots: [],
          metadata: {{}},
        }};
        ctx.repairContractionPlan();
        const finalScene = ctx.buildContractionScene();
        if (finalScene.totalStepCount !== 1 || finalScene.appliedStepCount !== 1) {{
          throw new Error("The final carry scene did not expose the one-step scheme.");
        }}
        assertSceneHasOperandIndex(
          finalScene,
          "final_prev_step",
          "phys",
          "Final carry scene"
        );
        ctx.syncCurrentGraphIntoLinearPeriodicChain();
        const finalSteps =
          ctx.state.spec.linear_periodic_chain.final_cell.contraction_plan.steps;
        if (
          finalSteps.length !== 1 ||
          finalSteps[0].left_operand_id !== "__linear_previous__"
        ) {{
          throw new Error("The final cell did not preserve the reserved previous operand when syncing back into the chain.");
        }}

        ctx.state.spec = ctx.normalizeSpec(createGridPeriodicSpec());
        let gridSteps =
          ctx.state.spec.grid_periodic_grid.center_cell.contraction_plan &&
          ctx.state.spec.grid_periodic_grid.center_cell.contraction_plan.steps;
        if (!Array.isArray(gridSteps) || gridSteps[0].left_operand_id !== "__grid_left__") {{
          throw new Error("Normalizing the grid should preserve the center cell's reserved-border plan.");
        }}
        ctx.hydrateActiveGridPeriodicCell();
        gridSteps =
          ctx.state.spec.contraction_plan && ctx.state.spec.contraction_plan.steps;
        if (!Array.isArray(gridSteps) || gridSteps[0].left_operand_id !== "__grid_left__") {{
          throw new Error("Hydrating the grid center cell should expose its reserved-border plan.");
        }}
        const gridLeftResolution = ctx.resolvePlannerOperandId("grid_left_boundary");
        if (gridLeftResolution !== "grid_left_step") {{
          throw new Error(`The grid left boundary should resolve to the contraction result after the plan consumes it. Received ${{gridLeftResolution}}.`);
        }}
        if (ctx.resolvePlannerOperandId("grid_right_boundary") !== "__grid_right__") {{
          throw new Error("The grid right boundary did not resolve to the reserved right operand id.");
        }}
        ctx.repairContractionPlan();
        const gridOperandState = ctx.buildContractionOperandState();
        const gridOperandIds = gridOperandState.activeOperands.map((operand) => operand.id);
        if (gridOperandState.validSteps.length !== 1) {{
          throw new Error(`Expected 1 valid grid step, received ${{gridOperandState.validSteps.length}}.`);
        }}
        if (!gridOperandIds.includes("grid_left_step") || !gridOperandIds.includes("__grid_right__")) {{
          throw new Error(`Expected the grid result and live right border to remain active, received ${{JSON.stringify(gridOperandIds)}}.`);
        }}
        ctx.syncCurrentGraphIntoGridPeriodicGrid();
        gridSteps =
          ctx.state.spec.grid_periodic_grid.center_cell.contraction_plan &&
          ctx.state.spec.grid_periodic_grid.center_cell.contraction_plan.steps;
        if (!Array.isArray(gridSteps) || gridSteps[0].left_operand_id !== "__grid_left__") {{
          throw new Error("The grid cell did not preserve the reserved border plan when syncing back into the grid.");
        }}
        ctx.switchGridPeriodicCell("right");
        ctx.switchGridPeriodicCell("left");
        gridSteps =
          ctx.state.spec.contraction_plan && ctx.state.spec.contraction_plan.steps;
        if (!Array.isArray(gridSteps) || gridSteps[0].left_operand_id !== "__grid_left__") {{
          throw new Error("Switching away from and back to the grid center should restore its plan.");
        }}

        ctx.state.spec = ctx.normalizeSpec(createTreePeriodicSpec());
        let treeSteps =
          ctx.state.spec.tree_periodic_tree.branch_cell.contraction_plan &&
          ctx.state.spec.tree_periodic_tree.branch_cell.contraction_plan.steps;
        if (!Array.isArray(treeSteps) || treeSteps[0].left_operand_id !== "__tree_parent__") {{
          throw new Error("Normalizing the tree should preserve the branch cell's reserved-border plan.");
        }}
        ctx.hydrateActiveTreePeriodicCell();
        treeSteps =
          ctx.state.spec.contraction_plan && ctx.state.spec.contraction_plan.steps;
        if (!Array.isArray(treeSteps) || treeSteps[1].right_operand_id !== "__tree_child_0__") {{
          throw new Error("Hydrating the tree branch cell should expose its reserved-border plan.");
        }}
        if (ctx.resolvePlannerOperandId("tree_parent_boundary") !== "tree_child_step") {{
          throw new Error("The tree parent boundary should resolve to the final result after the plan consumes it.");
        }}
        if (ctx.resolvePlannerOperandId("tree_child_0_boundary") !== "tree_child_step") {{
          throw new Error("The consumed tree child boundary should resolve to the final result.");
        }}
        if (ctx.resolvePlannerOperandId("tree_child_1_boundary") !== "__tree_child_1__") {{
          throw new Error("The live tree child boundary did not resolve to the reserved child operand id.");
        }}
        ctx.repairContractionPlan();
        const treeOperandState = ctx.buildContractionOperandState();
        const treeOperandIds = treeOperandState.activeOperands.map((operand) => operand.id);
        if (treeOperandState.validSteps.length !== 2) {{
          throw new Error(`Expected 2 valid tree steps, received ${{treeOperandState.validSteps.length}}.`);
        }}
        if (!treeOperandIds.includes("tree_child_step") || !treeOperandIds.includes("__tree_child_1__")) {{
          throw new Error(`Expected the tree result and live child-1 border to remain active, received ${{JSON.stringify(treeOperandIds)}}.`);
        }}
        ctx.syncCurrentGraphIntoTreePeriodicTree();
        treeSteps =
          ctx.state.spec.tree_periodic_tree.branch_cell.contraction_plan &&
          ctx.state.spec.tree_periodic_tree.branch_cell.contraction_plan.steps;
        if (!Array.isArray(treeSteps) || treeSteps[1].right_operand_id !== "__tree_child_0__") {{
          throw new Error("The tree cell did not preserve the reserved border plan when syncing back into the tree.");
        }}
        """
    )
    script_path.write_text(script_body, encoding="utf-8")
    return script_path


def _write_manual_contraction_anchor_runtime_regression_script(
    tmp_path: Path,
) -> Path:
    script_path = tmp_path / "manual_contraction_anchor_runtime_regression.mjs"
    state_runtime_path = tmp_path / "state.runtime.mjs"
    utilities_runtime_path = tmp_path / "utilities.runtime.mjs"
    history_runtime_path = tmp_path / "historySelection.runtime.mjs"
    planner_runtime_path = tmp_path / "planner.runtime.mjs"
    contraction_scene_runtime_path = tmp_path / "contractionScene.runtime.mjs"
    _copy_runtime_bundle(
        tmp_path,
        {
            "state.runtime.mjs": "state/state.js",
            "utilities.runtime.mjs": "utils/utilities.js",
            "historySelection.runtime.mjs": "graph/historySelection.js",
            "planner.runtime.mjs": "planner/planner.js",
            "plannerSupport.js": "planner/plannerSupport.js",
            "plannerRenderers.js": "planner/plannerRenderers.js",
            "contractionScene.runtime.mjs": "graph/contractionScene.js",
        },
        _RUNTIME_EDITOR_SUPPORT_MODULES,
    )
    script_body = textwrap.dedent(
        f"""
        import {{ pathToFileURL }} from "node:url";

        function createClassList() {{
          return {{
            add() {{}},
            remove() {{}},
            toggle() {{}},
          }};
        }}

        function createButton() {{
          return {{
            disabled: false,
            classList: createClassList(),
            addEventListener() {{}},
          }};
        }}

        function createPlannerPanel() {{
          return {{
            innerHTML: "",
            querySelectorAll() {{
              return [];
            }},
          }};
        }}

        function createSpec() {{
          return {{
            id: "network_manual_anchor",
            name: "manual-anchor",
            tensors: [
              {{
                id: "tensor_a",
                name: "A",
                position: {{ x: 120, y: 140 }},
                size: {{ width: 140, height: 84 }},
                metadata: {{}},
                indices: [
                  {{
                    id: "tensor_a_left",
                    name: "left",
                    dimension: 2,
                    offset: {{ x: -38, y: 0 }},
                    metadata: {{}},
                  }},
                  {{
                    id: "tensor_a_bond",
                    name: "bond",
                    dimension: 3,
                    offset: {{ x: 38, y: 0 }},
                    metadata: {{}},
                  }},
                ],
              }},
              {{
                id: "tensor_b",
                name: "B",
                position: {{ x: 360, y: 220 }},
                size: {{ width: 140, height: 84 }},
                metadata: {{}},
                indices: [
                  {{
                    id: "tensor_b_bond",
                    name: "bond",
                    dimension: 3,
                    offset: {{ x: -38, y: 0 }},
                    metadata: {{}},
                  }},
                  {{
                    id: "tensor_b_right",
                    name: "carry",
                    dimension: 5,
                    offset: {{ x: 38, y: 0 }},
                    metadata: {{}},
                  }},
                ],
              }},
              {{
                id: "tensor_c",
                name: "C",
                position: {{ x: 620, y: 300 }},
                size: {{ width: 140, height: 84 }},
                metadata: {{}},
                indices: [
                  {{
                    id: "tensor_c_left",
                    name: "carry",
                    dimension: 5,
                    offset: {{ x: -38, y: 0 }},
                    metadata: {{}},
                  }},
                  {{
                    id: "tensor_c_right",
                    name: "right",
                    dimension: 7,
                    offset: {{ x: 38, y: 0 }},
                    metadata: {{}},
                  }},
                ],
              }},
            ],
            groups: [],
            edges: [
              {{
                id: "edge_ab",
                name: "bond_ab",
                left: {{ tensor_id: "tensor_a", index_id: "tensor_a_bond" }},
                right: {{ tensor_id: "tensor_b", index_id: "tensor_b_bond" }},
                metadata: {{}},
              }},
              {{
                id: "edge_bc",
                name: "bond_bc",
                left: {{ tensor_id: "tensor_b", index_id: "tensor_b_right" }},
                right: {{ tensor_id: "tensor_c", index_id: "tensor_c_left" }},
                metadata: {{}},
              }},
            ],
            notes: [],
            contraction_plan: null,
            metadata: {{}},
          }};
        }}

        function assertLatestResultPosition(ctx, expectedPosition, label) {{
          const plan = ctx.state.spec && ctx.state.spec.contraction_plan;
          const steps = plan && Array.isArray(plan.steps) ? plan.steps : [];
          if (!steps.length) {{
            throw new Error(`${{label}} should produce at least one step.`);
          }}
          const scene = ctx.buildContractionScene();
          if (!scene) {{
            throw new Error(`${{label}} did not produce a contraction scene.`);
          }}
          const result = scene.operandMap[steps[steps.length - 1].id];
          if (!result) {{
            throw new Error(`${{label}} did not expose the result operand in the scene.`);
          }}
          if (
            result.position.x !== expectedPosition.x ||
            result.position.y !== expectedPosition.y
          ) {{
            throw new Error(
              `${{label}} anchored the result at (${{result.position.x}}, ${{result.position.y}}) instead of (${{expectedPosition.x}}, ${{expectedPosition.y}}).`
            );
          }}
        }}

        function createContext() {{
          const ctx = {{
            state: createInitialState(),
            constants: {{
              TENSOR_WIDTH: 140,
              TENSOR_HEIGHT: 84,
              MIN_TENSOR_WIDTH: 96,
              MIN_TENSOR_HEIGHT: 60,
              INDEX_RADIUS: 10,
              INDEX_PADDING: 6,
              NOTE_WIDTH: 220,
              NOTE_HEIGHT: 120,
              NOTE_MIN_WIDTH: 120,
              NOTE_MIN_HEIGHT: 90,
              HISTORY_LIMIT: 100,
              REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
              DEFAULT_INDEX_SLOTS: [
                {{ x: -38, y: 0 }},
                {{ x: 38, y: 0 }},
                {{ x: 0, y: -24 }},
                {{ x: 0, y: 24 }},
              ],
            }},
            dom: {{
              workspace: {{}},
              statusMessage: {{ textContent: "", classList: createClassList() }},
              propertiesPanel: {{ innerHTML: "" }},
              generatedCode: {{ value: "" }},
              engineSelect: {{ options: [], value: "tensornetwork" }},
              collectionFormatSelect: {{ options: [], value: "list" }},
              exportFormatSelect: {{ value: "py" }},
              addNoteButton: createButton(),
              connectButton: {{ classList: createClassList() }},
              loadInput: {{}},
              undoButton: createButton(),
              redoButton: createButton(),
              exportButton: createButton(),
              toggleLinearPeriodicButton: {{ classList: createClassList() }},
              linearPeriodicPreviousCellButton: createButton(),
              linearPeriodicCellLabel: {{ textContent: "" }},
              linearPeriodicNextCellButton: createButton(),
              templateSelect: {{ value: "" }},
              templateParameterPanel: {{ hidden: true }},
              templateGraphSizeLabel: {{ textContent: "" }},
              templateGraphSizeInput: {{ value: "2", min: "1" }},
              templateBondDimensionInput: {{ value: "3", min: "1" }},
              templatePhysicalDimensionInput: {{ value: "2", min: "1" }},
              insertTemplateButton: createButton(),
              createGroupButton: createButton(),
              helpButton: createButton(),
              helpModal: {{ classList: createClassList() }},
              helpBackdrop: createButton(),
              helpCloseButton: createButton(),
              canvasShell: {{
                getBoundingClientRect() {{
                  return {{ left: 0, top: 0, width: 1000, height: 800 }};
                }},
              }},
              groupLayer: {{}},
              resizeLayer: {{}},
              notesLayer: {{}},
              selectionBox: {{}},
              minimapCanvas: {{}},
              sidebar: {{}},
              plannerPanel: createPlannerPanel(),
              generateButton: createButton(),
            }},
            apiGet: async () => null,
            apiPost: async () => null,
            window: {{
              structuredClone: globalThis.structuredClone,
              crypto: globalThis.crypto,
              setTimeout,
              clearTimeout,
              confirm: () => true,
            }},
            document: {{
              activeElement: null,
              createElement() {{
                return {{
                  value: "",
                  textContent: "",
                  selected: false,
                  appendChild() {{}},
                  click() {{}},
                }};
              }},
              getElementById() {{
                return createButton();
              }},
              querySelectorAll() {{
                return [];
              }},
            }},
            cytoscape: null,
            tensorWidth: (tensor) => tensor?.size?.width ?? 140,
            tensorHeight: (tensor) => tensor?.size?.height ?? 84,
            render: () => {{}},
            renderOverlayDecorations: () => {{}},
            renderMinimap: () => {{}},
            renderPlanner: () => {{}},
            renderSidebarTabs: () => {{}},
            refreshContractionAnalysis: () => {{}},
            syncPendingInteractionClasses: () => {{}},
            setActiveSidebarTab: () => {{}},
            updateToolbarState: () => {{}},
            captureEditableFocus: () => null,
            restoreEditableFocus: () => {{}},
          }};

          registerUtilities(ctx);
          registerContractionScene(ctx);
          registerHistorySelection(ctx);
          registerPlannerFeature(ctx);

          ctx.state.selectedEngine = "tensornetwork";
          ctx.state.selectedCollectionFormat = "list";
          ctx.state.spec = ctx.normalizeSpec(createSpec());
          return ctx;
        }}

        const [
          stateModule,
          utilitiesModule,
          historyModule,
          plannerModule,
          contractionSceneModule,
        ] = await Promise.all([
          import(pathToFileURL({json.dumps(str(state_runtime_path))}).href),
          import(pathToFileURL({json.dumps(str(utilities_runtime_path))}).href),
          import(pathToFileURL({json.dumps(str(history_runtime_path))}).href),
          import(pathToFileURL({json.dumps(str(planner_runtime_path))}).href),
          import(pathToFileURL({json.dumps(str(contraction_scene_runtime_path))}).href),
        ]);
        const {{ createInitialState }} = stateModule;
        const {{ registerUtilities }} = utilitiesModule;
        const {{ registerHistorySelection }} = historyModule;
        const {{ registerPlannerFeature }} = plannerModule;
        const {{ registerContractionScene }} = contractionSceneModule;

        const directContext = createContext();
        directContext.applyManualContractionStep("tensor_a", "tensor_b");
        assertLatestResultPosition(
          directContext,
          {{ x: 360, y: 220 }},
          "Direct manual contraction"
        );
        const directFirstStepId = directContext.state.spec.contraction_plan.steps[0].id;
        directContext.buildContractionScene();
        const directSnapshot = directContext.getSnapshotForStepCount(1);
        const directFirstResultLayout = directSnapshot.operand_layouts.find(
          (layout) => layout.operand_id === directFirstStepId
        );
        directFirstResultLayout.position = {{ x: 210, y: 80 }};
        directContext.applyManualContractionStep(directFirstStepId, "tensor_c");
        assertLatestResultPosition(
          directContext,
          {{ x: 620, y: 300 }},
          "Direct follow-up contraction"
        );

        const plannerContext = createContext();
        plannerContext.state.plannerMode = true;
        plannerContext.handlePlannerOperandClick("tensor_a");
        plannerContext.handlePlannerOperandClick("tensor_b");
        const plannerSteps = plannerContext.state.spec.contraction_plan.steps;
        if (
          plannerSteps.length !== 1 ||
          plannerSteps[0].left_operand_id !== "tensor_a" ||
          plannerSteps[0].right_operand_id !== "tensor_b"
        ) {{
          throw new Error(
            `Planner click order changed the recorded step order: ${{JSON.stringify(plannerSteps)}}.`
          );
        }}
        assertLatestResultPosition(
          plannerContext,
          {{ x: 360, y: 220 }},
          "Planner click contraction"
        );
        const plannerFirstStepId = plannerSteps[0].id;
        plannerContext.buildContractionScene();
        const plannerSnapshot = plannerContext.getSnapshotForStepCount(1);
        const plannerFirstResultLayout = plannerSnapshot.operand_layouts.find(
          (layout) => layout.operand_id === plannerFirstStepId
        );
        plannerFirstResultLayout.position = {{ x: 260, y: 100 }};
        plannerContext.handlePlannerOperandClick(plannerFirstStepId);
        plannerContext.handlePlannerOperandClick("tensor_c");
        assertLatestResultPosition(
          plannerContext,
          {{ x: 620, y: 300 }},
          "Planner follow-up contraction"
        );
        """
    )
    script_path.write_text(script_body, encoding="utf-8")
    return script_path


def _write_engine_order_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "engine_order_runtime_regression.mjs"
    utilities_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utils/utilities.js"
    )
    utilities_templates_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utils/utilitiesTemplates.js"
    )
    utilities_runtime_path = tmp_path / "utilities.runtime.mjs"
    utilities_templates_runtime_path = tmp_path / "utilitiesTemplates.js"
    utilities_runtime_path.write_text(
        utilities_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    utilities_templates_runtime_path.write_text(
        utilities_templates_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _copy_runtime_editor_support_modules(tmp_path)
    script_body = textwrap.dedent(
        f"""
        import {{ pathToFileURL }} from "node:url";

        function createClassList() {{
          return {{
            add() {{}},
            remove() {{}},
            toggle() {{}},
          }};
        }}

        function createButton() {{
          return {{
            disabled: false,
            classList: createClassList(),
          }};
        }}

        function createSelectElement() {{
          const select = {{
            options: [],
            value: "",
            appendChild(option) {{
              this.options.push(option);
              if (option.selected) {{
                this.value = option.value;
              }}
            }},
          }};
          Object.defineProperty(select, "innerHTML", {{
            get() {{
              return "";
            }},
            set(_value) {{
              this.options = [];
              this.value = "";
            }},
          }});
          return select;
        }}

        const {{ registerUtilities }} = await import(
          pathToFileURL({json.dumps(str(utilities_runtime_path))}).href
        );

        const ctx = {{
          state: {{
            spec: null,
            selectedEngine: "einsum_torch",
            selectedCollectionFormat: "list",
          }},
          constants: {{
            TENSOR_WIDTH: 140,
            TENSOR_HEIGHT: 84,
            MIN_TENSOR_WIDTH: 96,
            MIN_TENSOR_HEIGHT: 60,
            INDEX_RADIUS: 10,
            INDEX_PADDING: 6,
            NOTE_WIDTH: 220,
            NOTE_HEIGHT: 120,
            NOTE_MIN_WIDTH: 120,
            NOTE_MIN_HEIGHT: 90,
            HISTORY_LIMIT: 100,
            REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
            DEFAULT_INDEX_SLOTS: [],
          }},
          dom: {{
            workspace: {{}},
            statusMessage: {{ textContent: "", classList: createClassList() }},
            propertiesPanel: {{ innerHTML: "" }},
            generatedCode: {{ value: "" }},
            engineSelect: createSelectElement(),
            collectionFormatSelect: createSelectElement(),
            exportFormatSelect: {{ value: "py" }},
            addNoteButton: createButton(),
            connectButton: {{ classList: createClassList() }},
            loadInput: {{}},
            undoButton: createButton(),
            redoButton: createButton(),
            exportButton: createButton(),
            toggleLinearPeriodicButton: {{ classList: createClassList() }},
            linearPeriodicPreviousCellButton: createButton(),
            linearPeriodicCellLabel: {{ textContent: "" }},
            linearPeriodicNextCellButton: createButton(),
            templateSelect: {{ value: "" }},
            templateParameterPanel: {{ hidden: true }},
            templateGraphSizeLabel: {{ textContent: "" }},
            templateGraphSizeInput: {{ value: "2", min: "1" }},
            templateBondDimensionInput: {{ value: "3", min: "1" }},
            templatePhysicalDimensionInput: {{ value: "2", min: "1" }},
            insertTemplateButton: createButton(),
            createGroupButton: createButton(),
            helpButton: createButton(),
            helpModal: {{ classList: createClassList() }},
            helpBackdrop: createButton(),
            helpCloseButton: createButton(),
            canvasShell: {{
              getBoundingClientRect() {{
                return {{ left: 0, top: 0, width: 1000, height: 800 }};
              }},
            }},
            groupLayer: {{}},
            resizeLayer: {{}},
            notesLayer: {{}},
            selectionBox: {{}},
            minimapCanvas: {{}},
            sidebar: {{}},
            plannerPanel: {{}},
            generateButton: createButton(),
          }},
          apiGet: async () => null,
          apiPost: async () => null,
          window: {{
            structuredClone: globalThis.structuredClone,
            crypto: globalThis.crypto,
            setTimeout,
            clearTimeout,
            confirm: () => true,
          }},
          document: {{
            createElement(tagName) {{
              return {{
                tagName,
                value: "",
                textContent: "",
                selected: false,
              }};
            }},
            querySelectorAll() {{
              return [];
            }},
          }},
          cytoscape: null,
          tensorWidth: (tensor) => tensor?.size?.width ?? 140,
          tensorHeight: (tensor) => tensor?.size?.height ?? 84,
        }};

        registerUtilities(ctx);

        ctx.populateEngineOptions([
          "tensornetwork",
          "quimb",
          "tensorkrowch",
          "einsum_numpy",
          "einsum_torch",
        ]);

        const optionOrder = ctx.dom.engineSelect.options.map((option) => option.value);
        const expectedOrder = [
          "tensorkrowch",
          "einsum_torch",
          "einsum_numpy",
          "quimb",
          "tensornetwork",
        ];

        if (JSON.stringify(optionOrder) !== JSON.stringify(expectedOrder)) {{
          throw new Error(
            `Expected engine option order ${{JSON.stringify(expectedOrder)}}, received ${{JSON.stringify(optionOrder)}}.`
          );
        }}

        if (ctx.dom.engineSelect.value !== "einsum_torch") {{
          throw new Error(
            `Expected the selected engine to remain einsum_torch, received ${{ctx.dom.engineSelect.value}}.`
          );
        }}
        """
    )
    script_path.write_text(script_body, encoding="utf-8")
    return script_path


def _write_tensor_index_move_properties_runtime_regression_script(
    tmp_path: Path,
) -> Path:
    script_path = tmp_path / "tensor_index_move_properties_regression.mjs"
    state_runtime_path = tmp_path / "state.runtime.mjs"
    utilities_runtime_path = tmp_path / "utilities.runtime.mjs"
    history_runtime_path = tmp_path / "historySelection.runtime.mjs"
    properties_runtime_path = tmp_path / "properties.runtime.mjs"
    _copy_runtime_bundle(
        tmp_path,
        {
            "state.runtime.mjs": "state/state.js",
            "utilities.runtime.mjs": "utils/utilities.js",
            "historySelection.runtime.mjs": "graph/historySelection.js",
            "properties.runtime.mjs": "properties/properties.js",
            "propertiesSupport.js": "properties/propertiesSupport.js",
            "propertiesRenderers.js": "properties/propertiesRenderers.js",
        },
        _RUNTIME_EDITOR_SUPPORT_MODULES,
    )
    script_body = textwrap.dedent(
        """
        import { pathToFileURL } from "node:url";

        function createClassList() {
          return {
            add() {},
            remove() {},
            toggle() {},
          };
        }

        function createFakeElement(id = null, tagName = "div") {
          return {
            id,
            tagName,
            value: "",
            textContent: "",
            selected: false,
            disabled: false,
            dataset: {},
            style: {},
            classList: createClassList(),
            listeners: {},
            addEventListener(eventName, listener) {
              if (!this.listeners[eventName]) {
                this.listeners[eventName] = [];
              }
              this.listeners[eventName].push(listener);
            },
            dispatchEvent(eventName, event = {}) {
              (this.listeners[eventName] || []).forEach((listener) => {
                listener({
                  preventDefault() {},
                  target: this,
                  ...event,
                });
              });
            },
            click() {
              this.dispatchEvent("click");
            },
            focus() {},
            setAttribute() {},
            removeAttribute() {},
            appendChild() {},
          };
        }

        function createFakeDocument() {
          const elements = new Map();
          const toggleElements = [];
          return {
            activeElement: null,
            toggleElements,
            registerHtml(html) {
              elements.clear();
              toggleElements.length = 0;

              const idPattern = /id="([^"]+)"/g;
              let idMatch = idPattern.exec(html);
              while (idMatch) {
                elements.set(idMatch[1], createFakeElement(idMatch[1]));
                idMatch = idPattern.exec(html);
              }

              const togglePattern = /<button[\\s\\S]*?data-index-toggle="([^"]+)"[\\s\\S]*?>/g;
              let toggleMatch = togglePattern.exec(html);
              while (toggleMatch) {
                const element = createFakeElement(null, "button");
                element.dataset.indexToggle = toggleMatch[1];
                toggleElements.push(element);
                toggleMatch = togglePattern.exec(html);
              }
            },
            getElementById(id) {
              return elements.get(id) || null;
            },
            createElement(tagName) {
              return createFakeElement(null, tagName);
            },
            querySelectorAll() {
              return [];
            },
          };
        }

        function createPropertiesPanel(document) {
          let html = "";
          return {
            get innerHTML() {
              return html;
            },
            set innerHTML(value) {
              html = value;
              document.registerHtml(value);
            },
            querySelectorAll(selector) {
              if (selector === "[data-index-toggle]") {
                return document.toggleElements;
              }
              return [];
            },
          };
        }

        function createButton() {
          return createFakeElement(null, "button");
        }

        function createSpec() {
          return {
            id: "network_index_move",
            name: "index move regression",
            tensors: [
              {
                id: "tensor_a",
                name: "Tensor",
                position: { x: 120, y: 120 },
                size: { width: 140, height: 84 },
                metadata: {},
                indices: [
                  {
                    id: "index_a",
                    name: "A",
                    dimension: 2,
                    offset: { x: -38, y: 0 },
                    metadata: {},
                  },
                  {
                    id: "index_b",
                    name: "B",
                    dimension: 3,
                    offset: { x: 38, y: 0 },
                    metadata: {},
                  },
                  {
                    id: "index_c",
                    name: "C",
                    dimension: 5,
                    offset: { x: 0, y: -24 },
                    metadata: {},
                  },
                ],
              },
            ],
            edges: [],
            groups: [],
            notes: [],
            contraction_plan: null,
            metadata: {},
          };
        }

        function getToggleChunk(html, indexId) {
          const token = `data-index-toggle="${indexId}"`;
          const position = html.indexOf(token);
          if (position < 0) {
            throw new Error(`Missing toggle for ${indexId}.`);
          }
          return html.slice(Math.max(0, position - 220), position + token.length + 120);
        }

        function assertToggleState(html, indexId, { open, focused }) {
          const chunk = getToggleChunk(html, indexId);
          const isOpen = chunk.includes("index-disclosure-toggle is-open");
          const isFocused = chunk.includes("is-focused");
          if (isOpen !== open) {
            throw new Error(`Expected ${indexId} open=${open}, received ${isOpen}.`);
          }
          if (isFocused !== focused) {
            throw new Error(`Expected ${indexId} focused=${focused}, received ${isFocused}.`);
          }
        }

        function assertDisclosureState(html, disclosureId, expectedOpen) {
          const match = html.match(
            new RegExp(`<details[^>]*id="${disclosureId}"[^>]*>`)
          );
          if (!match) {
            throw new Error(`Missing disclosure ${disclosureId}.`);
          }
          const isOpen = /\\sopen(?:\\s|>)/.test(match[0]);
          if (isOpen !== expectedOpen) {
            throw new Error(
              `Expected disclosure ${disclosureId} open=${expectedOpen}, received ${isOpen}.`
            );
          }
        }

        function assertHtmlOrder(html, firstLabel, secondLabel) {
          const firstPosition = html.indexOf(firstLabel);
          const secondPosition = html.indexOf(secondLabel);
          if (firstPosition < 0 || secondPosition < 0 || firstPosition >= secondPosition) {
            throw new Error(
              `Expected ${firstLabel} before ${secondLabel}.\\nHTML:\\n${html}`
            );
          }
        }

        const [stateModule, utilitiesModule, historyModule, propertiesModule] =
          await Promise.all([
            import(pathToFileURL(__STATE_PATH__).href),
            import(pathToFileURL(__UTILITIES_PATH__).href),
            import(pathToFileURL(__HISTORY_PATH__).href),
            import(pathToFileURL(__PROPERTIES_PATH__).href),
          ]);
        const { createInitialState } = stateModule;
        const { registerUtilities } = utilitiesModule;
        const { registerHistorySelection } = historyModule;
        const { registerProperties } = propertiesModule;

        const document = createFakeDocument();
        const propertiesPanel = createPropertiesPanel(document);
        const ctx = {
          state: createInitialState(),
          constants: {
            TENSOR_WIDTH: 140,
            TENSOR_HEIGHT: 84,
            MIN_TENSOR_WIDTH: 96,
            MIN_TENSOR_HEIGHT: 60,
            INDEX_RADIUS: 10,
            INDEX_PADDING: 6,
            NOTE_WIDTH: 220,
            NOTE_HEIGHT: 120,
            NOTE_MIN_WIDTH: 120,
            NOTE_MIN_HEIGHT: 90,
            HISTORY_LIMIT: 100,
            REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
            DEFAULT_INDEX_SLOTS: [
              { x: -38, y: 0 },
              { x: 38, y: 0 },
              { x: 0, y: -24 },
              { x: 0, y: 24 },
            ],
          },
          dom: {
            workspace: {},
            statusMessage: { textContent: "", classList: createClassList() },
            propertiesPanel,
            generatedCode: { value: "" },
            engineSelect: { options: [], value: "tensornetwork" },
            collectionFormatSelect: { options: [], value: "list" },
            exportFormatSelect: { value: "py" },
            addNoteButton: createButton(),
            connectButton: createButton(),
            loadInput: {},
            undoButton: createButton(),
            redoButton: createButton(),
            exportButton: createButton(),
            toggleLinearPeriodicButton: createButton(),
            linearPeriodicPreviousCellButton: createButton(),
            linearPeriodicCellLabel: { textContent: "" },
            linearPeriodicNextCellButton: createButton(),
            templateSelect: { value: "" },
            templateParameterPanel: { hidden: true },
            templateGraphSizeLabel: { textContent: "" },
            templateGraphSizeInput: { value: "2", min: "1" },
            templateBondDimensionInput: { value: "3", min: "1" },
            templatePhysicalDimensionInput: { value: "2", min: "1" },
            insertTemplateButton: createButton(),
            createGroupButton: createButton(),
            helpButton: createButton(),
            helpModal: { classList: createClassList() },
            helpBackdrop: createButton(),
            helpCloseButton: createButton(),
            canvasShell: {
              getBoundingClientRect() {
                return { left: 0, top: 0, width: 1000, height: 800 };
              },
            },
            groupLayer: {},
            resizeLayer: {},
            notesLayer: {},
            selectionBox: {},
            minimapCanvas: {},
            sidebar: {},
            plannerPanel: {},
            generateButton: createButton(),
          },
          apiGet: async () => null,
          apiPost: async () => null,
          window: {
            structuredClone: globalThis.structuredClone,
            crypto: globalThis.crypto,
            setTimeout,
            clearTimeout,
            confirm: () => true,
          },
          document,
          cytoscape: null,
          tensorWidth: (tensor) => tensor?.size?.width ?? 140,
          tensorHeight: (tensor) => tensor?.size?.height ?? 84,
          renderOverlayDecorations: () => {},
          renderMinimap: () => {},
          renderPlanner: () => {},
          renderSidebarTabs: () => {},
          refreshContractionAnalysis: () => {},
          repairContractionPlan: () => {},
          updateToolbarState: () => {},
        };

        registerUtilities(ctx);
        registerHistorySelection(ctx);
        registerProperties(ctx);
        ctx.findNoteById = (noteId) =>
          (Array.isArray(ctx.state.spec?.notes)
            ? ctx.state.spec.notes.find((note) => note.id === noteId)
            : null) || null;

        ctx.captureEditableFocus = () => null;
        ctx.restoreEditableFocus = () => {};
        ctx.render = (options = {}) => {
          const resolvedOptions = {
            graph: true,
            properties: true,
            code: true,
            toolbar: true,
            overlays: true,
            planner: true,
            sidebarTabs: true,
            minimap: true,
            syncSelection: false,
            ...options,
          };
          if (resolvedOptions.properties) {
            ctx.renderProperties();
          }
          if (resolvedOptions.toolbar) {
            ctx.updateToolbarState();
          }
        };

        ctx.state.selectedEngine = "tensornetwork";
        ctx.state.selectedCollectionFormat = "list";
        ctx.state.spec = ctx.normalizeSpec(createSpec());

        ctx.setSelection(["index_a"], { primaryId: "index_a" });
        const initialHtml = ctx.dom.propertiesPanel.innerHTML;
        assertHtmlOrder(initialHtml, "<strong>1. A</strong>", "<strong>2. B</strong>");
        assertToggleState(initialHtml, "index_a", { open: true, focused: true });
        assertToggleState(initialHtml, "index_b", { open: false, focused: false });
        assertDisclosureState(initialHtml, "tensor-tags-input-disclosure", false);

        ctx.state.metadataDisclosureState["tensor:tensor_a:metadata"] = true;
        ctx.renderProperties();
        const metadataOpenHtml = ctx.dom.propertiesPanel.innerHTML;
        assertDisclosureState(metadataOpenHtml, "tensor-tags-input-disclosure", true);

        const moveDownButton = document.getElementById("move-index-down-button-index_a");
        if (!moveDownButton) {
          throw new Error("The open index editor did not expose the move-down button.");
        }
        moveDownButton.click();

        const indexOrder = ctx.state.spec.tensors[0].indices.map((index) => index.id);
        const expectedOrder = ["index_b", "index_a", "index_c"];
        if (JSON.stringify(indexOrder) !== JSON.stringify(expectedOrder)) {
          throw new Error(
            `Expected model order ${JSON.stringify(expectedOrder)}, received ${JSON.stringify(indexOrder)}.`
          );
        }

        const updatedHtml = ctx.dom.propertiesPanel.innerHTML;
        if (updatedHtml === initialHtml) {
          throw new Error("The properties panel did not re-render after moving the index.");
        }
        assertHtmlOrder(updatedHtml, "<strong>1. B</strong>", "<strong>2. A</strong>");
        assertToggleState(updatedHtml, "index_b", { open: false, focused: false });
        assertToggleState(updatedHtml, "index_a", { open: true, focused: true });
        assertDisclosureState(updatedHtml, "tensor-tags-input-disclosure", true);
        """
    )
    script_body = script_body.replace(
        "__STATE_PATH__", json.dumps(str(state_runtime_path))
    )
    script_body = script_body.replace(
        "__UTILITIES_PATH__", json.dumps(str(utilities_runtime_path))
    )
    script_body = script_body.replace(
        "__HISTORY_PATH__", json.dumps(str(history_runtime_path))
    )
    script_body = script_body.replace(
        "__PROPERTIES_PATH__", json.dumps(str(properties_runtime_path))
    )
    script_path.write_text(script_body, encoding="utf-8")
    return script_path


def _write_sidebar_resize_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "sidebar_resize_runtime_regression.mjs"
    state_runtime_path = tmp_path / "state.runtime.mjs"
    sidebar_tabs_runtime_path = tmp_path / "sidebarTabs.runtime.mjs"
    _copy_runtime_bundle(
        tmp_path,
        {"sidebarTabs.runtime.mjs": _js_source_name("sidebarTabs.js")},
        _STATE_RUNTIME_DEPENDENCY_MODULES,
    )
    script_body = textwrap.dedent(
        """
        import { pathToFileURL } from "node:url";

        function createClassList() {
          const classes = new Set();
          return {
            add(className) {
              classes.add(className);
            },
            remove(className) {
              classes.delete(className);
            },
            toggle(className, force) {
              const shouldHaveClass = typeof force === "boolean" ? force : !classes.has(className);
              if (shouldHaveClass) {
                classes.add(className);
              } else {
                classes.delete(className);
              }
            },
            contains(className) {
              return classes.has(className);
            },
          };
        }

        function createStyle() {
          const properties = {};
          return {
            properties,
            setProperty(name, value) {
              properties[name] = value;
            },
          };
        }

        function createElement() {
          return {
            hidden: false,
            innerHTML: "",
            dataset: {},
            classList: createClassList(),
            style: createStyle(),
            listeners: {},
            addEventListener(eventName, listener) {
              this.listeners[eventName] = listener;
            },
            setAttribute(name, value) {
              this[name] = value;
            },
            removeAttribute(name) {
              delete this[name];
            },
          };
        }

        function fire(target, eventName, event = {}) {
          const listener = target.listeners[eventName];
          if (!listener) {
            throw new Error(`Missing ${eventName} listener.`);
          }
          listener({
            clientX: 0,
            key: "",
            preventDefault() {},
            ...event,
          });
        }

        const [stateModule, sidebarTabsModule] = await Promise.all([
          import(pathToFileURL(__STATE_PATH__).href),
          import(pathToFileURL(__SIDEBAR_TABS_PATH__).href),
        ]);
        const { createInitialState } = stateModule;
        const { registerSidebarTabs } = sidebarTabsModule;

        const windowListeners = {};
        let canvasResizeCount = 0;
        let overlayRenderCount = 0;
        let minimapRenderCount = 0;
        let plannerRefreshCount = 0;
        const ctx = {
          state: createInitialState(),
          dom: {
            workspace: createElement(),
            sidebar: createElement(),
            sidebarPanel: createElement(),
            sidebarResizeHandle: createElement(),
            sidebarToggleButton: createElement(),
            sidebarTabSelection: createElement(),
            sidebarTabPlanner: createElement(),
            sidebarTabCode: createElement(),
            sidebarPaneSelection: createElement(),
            sidebarPanePlanner: createElement(),
            sidebarPaneCode: createElement(),
          },
          window: {
            innerWidth: 1200,
            addEventListener(eventName, listener) {
              windowListeners[eventName] = listener;
            },
          },
          cy: {
            resize() {
              canvasResizeCount += 1;
            },
          },
          renderOverlayDecorations() {
            overlayRenderCount += 1;
          },
          renderMinimap() {
            minimapRenderCount += 1;
          },
          refreshContractionAnalysis() {
            plannerRefreshCount += 1;
            ctx.state.contractionAnalysisDirty = false;
          },
        };

        registerSidebarTabs(ctx);

        if (ctx.state.sidebarWidth !== 360) {
          throw new Error(`Expected default sidebar width 360, received ${ctx.state.sidebarWidth}.`);
        }
        if (ctx.dom.sidebarToggleButton.dataset.tooltipEnabled !== "true") {
          throw new Error("Expected the sidebar toggle to expose the shared tooltip behavior.");
        }
        if (ctx.dom.sidebarToggleButton.dataset.shortcutLabel !== "Sidebar") {
          throw new Error(`Expected the sidebar toggle tooltip label to stay compact, received ${ctx.dom.sidebarToggleButton.dataset.shortcutLabel}.`);
        }
        if ("shortcutDescription" in ctx.dom.sidebarToggleButton.dataset) {
          throw new Error("Expected the sidebar toggle tooltip to expose only the shortcut.");
        }
        if (ctx.dom.workspace.style.properties["--sidebar-width"] !== "360px") {
          throw new Error("The workspace did not receive the initial sidebar width CSS variable.");
        }

        fire(ctx.dom.sidebarResizeHandle, "mousedown", { clientX: 900 });
        if (!ctx.state.activeSidebarResize) {
          throw new Error("Dragging the resize handle did not start sidebar resizing.");
        }
        windowListeners.mousemove({ clientX: 800, preventDefault() {} });
        if (ctx.state.sidebarWidth !== 460) {
          throw new Error(`Expected dragging left to widen the sidebar to 460, received ${ctx.state.sidebarWidth}.`);
        }
        if (ctx.dom.workspace.style.properties["--sidebar-width"] !== "460px") {
          throw new Error("The workspace sidebar width CSS variable was not updated while dragging.");
        }
        if (!canvasResizeCount || !overlayRenderCount || !minimapRenderCount) {
          throw new Error("Resizing the sidebar should refresh the canvas, overlays, and minimap.");
        }

        windowListeners.mousemove({ clientX: 1200, preventDefault() {} });
        if (ctx.state.sidebarWidth !== 280) {
          throw new Error(`Expected the sidebar width to clamp to 280, received ${ctx.state.sidebarWidth}.`);
        }
        windowListeners.mousemove({ clientX: 100, preventDefault() {} });
        if (ctx.state.sidebarWidth !== 640) {
          throw new Error(`Expected the sidebar width to clamp to 640, received ${ctx.state.sidebarWidth}.`);
        }
        windowListeners.mouseup();
        if (ctx.state.activeSidebarResize) {
          throw new Error("Mouseup should stop sidebar resizing.");
        }

        ctx.toggleSidebarCollapsed(true);
        ctx.toggleSidebarCollapsed(false);
        if (ctx.state.sidebarWidth !== 640) {
          throw new Error("Collapsing and expanding the sidebar should preserve the custom width.");
        }

        fire(ctx.dom.sidebarResizeHandle, "keydown", { key: "ArrowRight" });
        if (ctx.state.sidebarWidth !== 616) {
          throw new Error(`Expected ArrowRight to narrow the sidebar by 24px, received ${ctx.state.sidebarWidth}.`);
        }
        fire(ctx.dom.sidebarResizeHandle, "keydown", { key: "ArrowLeft" });
        if (ctx.state.sidebarWidth !== 640) {
          throw new Error(`Expected ArrowLeft to widen the sidebar by 24px, received ${ctx.state.sidebarWidth}.`);
        }
        ctx.state.contractionAnalysisDirty = true;
        ctx.setActiveSidebarTab("planner");
        if (plannerRefreshCount !== 1) {
          throw new Error(`Expected opening the planner tab with dirty analysis to trigger one refresh, received ${plannerRefreshCount}.`);
        }
        ctx.setActiveSidebarTab("planner");
        if (plannerRefreshCount !== 1) {
          throw new Error(`Expected reopening a clean planner tab to avoid duplicate refreshes, received ${plannerRefreshCount}.`);
        }
        """
    )
    script_body = script_body.replace(
        "__STATE_PATH__", json.dumps(str(state_runtime_path))
    )
    script_body = script_body.replace(
        "__SIDEBAR_TABS_PATH__", json.dumps(str(sidebar_tabs_runtime_path))
    )
    script_path.write_text(script_body, encoding="utf-8")
    return script_path


def _write_minimap_shortcut_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "minimap_shortcut_runtime_regression.mjs"
    _copy_js_modules(tmp_path, _MINIMAP_SHORTCUT_RUNTIME_DEPENDENCY_MODULES)

    script_path.write_text(
        textwrap.dedent(
            """
            const baseUrl = new URL("./", import.meta.url);
            const [stateModule, shortcutsModule, exportMinimapModule] =
              await Promise.all([
                import(new URL("./state.runtime.mjs", baseUrl).href),
                import(new URL("./interactionsShortcuts.js", baseUrl).href),
                import(new URL("./exportMinimap.js", baseUrl).href),
              ]);

            const { createInitialState } = stateModule;
            const { createInteractionShortcutBindings } = shortcutsModule;
            const { registerExportMinimap } = exportMinimapModule;

            function createClassList() {
              const classes = new Set();
              return {
                add(name) {
                  classes.add(name);
                },
                remove(name) {
                  classes.delete(name);
                },
                toggle(name, force) {
                  const shouldHaveClass =
                    typeof force === "boolean" ? force : !classes.has(name);
                  if (shouldHaveClass) {
                    classes.add(name);
                  } else {
                    classes.delete(name);
                  }
                },
                contains(name) {
                  return classes.has(name);
                },
              };
            }

            const minimapShell = { classList: createClassList() };
            const minimapCanvas = {
              classList: createClassList(),
              getContext() {
                return null;
              },
            };
            const ctx = {
              state: createInitialState(),
              constants: {
                TENSOR_WIDTH: 140,
                TENSOR_HEIGHT: 84,
                MIN_TENSOR_WIDTH: 96,
                MIN_TENSOR_HEIGHT: 60,
                INDEX_RADIUS: 10,
                INDEX_PADDING: 6,
                HISTORY_LIMIT: 100,
                REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
                DEFAULT_INDEX_SLOTS: [],
              },
              dom: {
                statusMessage: { textContent: "", classList: createClassList() },
                propertiesPanel: {},
                generatedCode: {},
                engineSelect: { options: [], value: "tensornetwork" },
                connectButton: {},
                loadInput: { click() {} },
                undoButton: {},
                redoButton: {},
                helpCloseButton: { focus() {} },
                helpModal: { classList: createClassList() },
                minimapShell,
                minimapCanvas,
              },
              apiGet: async () => null,
              apiPost: async () => null,
              window: {
                structuredClone: globalThis.structuredClone,
                crypto: globalThis.crypto,
                setTimeout,
                clearTimeout,
                cancelAnimationFrame() {},
                requestAnimationFrame(callback) {
                  callback();
                  return 1;
                },
              },
              document: {
                activeElement: null,
              },
              cytoscape: null,
              isTextInput() {
                return false;
              },
              setStatus() {},
              clamp(value, min, max) {
                return Math.min(max, Math.max(min, value));
              },
              computeDesignBounds() {
                return { x1: 0, y1: 0, x2: 100, y2: 100 };
              },
              getVisibleTensors() {
                return [];
              },
              getVisibleEdges() {
                return [];
              },
            };

            registerExportMinimap(ctx);
            Object.assign(
              ctx,
              createInteractionShortcutBindings({
                ctx,
                state: ctx.state,
                dom: ctx.dom,
                runtime: {},
              })
            );

            ctx.handleKeydown({
              key: "M",
              shiftKey: true,
              ctrlKey: false,
              metaKey: false,
              preventDefault() {},
              target: null,
            });

            if (!ctx.state.minimapHidden) {
              throw new Error("Shift+M should hide the minimap.");
            }
            if (!minimapShell.classList.contains("is-hidden")) {
              throw new Error("Shift+M should hide the minimap shell.");
            }
            """
        ),
        encoding="utf-8",
    )
    return script_path


def _write_svg_export_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "svg_export_runtime_regression.mjs"
    script_path.write_text(
        textwrap.dedent(
            f"""
            import {{ pathToFileURL }} from "node:url";

            const exportMinimapUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "graph" / "exportMinimap.js")!r}).href;
            const {{ registerExportMinimap }} = await import(exportMinimapUrl);

            function escapeSvgText(value) {{
              return String(value)
                .replaceAll("&", "&amp;")
                .replaceAll("<", "&lt;")
                .replaceAll(">", "&gt;")
                .replaceAll('"', "&quot;")
                .replaceAll("'", "&apos;");
            }}

            function escapeSvgAttribute(value) {{
              return escapeSvgText(value);
            }}

            function buildQuadraticCurve(source, target) {{
              const midpoint = {{
                x: (source.x + target.x) / 2,
                y: (source.y + target.y) / 2,
              }};
              const deltaX = target.x - source.x;
              const deltaY = target.y - source.y;
              const distance = Math.max(1, Math.sqrt(deltaX * deltaX + deltaY * deltaY));
              const normal = {{ x: -deltaY / distance, y: deltaX / distance }};
              const bend = Math.min(60, Math.max(18, distance * 0.18));
              return {{
                control: {{
                  x: midpoint.x + normal.x * bend,
                  y: midpoint.y + normal.y * bend,
                }},
              }};
            }}

            function quadraticPointAt(source, control, target, t) {{
              const inverse = 1 - t;
              return {{
                x: inverse * inverse * source.x + 2 * inverse * t * control.x + t * t * target.x,
                y: inverse * inverse * source.y + 2 * inverse * t * control.y + t * t * target.y,
              }};
            }}

            const spec = {{
              name: "svg-regression",
              tensors: [
                {{
                  id: "tensor_a1",
                  name: "A1",
                  position: {{ x: 120, y: 100 }},
                  metadata: {{}},
                  indices: [
                    {{
                      id: "index_a1_right",
                      name: "right",
                      dimension: 3,
                      offset: {{ x: 58, y: 0 }},
                      metadata: {{}},
                    }},
                  ],
                }},
                {{
                  id: "tensor_a2",
                  name: "A2",
                  position: {{ x: 320, y: 100 }},
                  metadata: {{}},
                  indices: [
                    {{
                      id: "index_a2_left",
                      name: "left",
                      dimension: 3,
                      offset: {{ x: -58, y: 0 }},
                      metadata: {{}},
                    }},
                  ],
                }},
              ],
              edges: [
                {{
                  id: "edge_a1_a2",
                  name: "edge-r1-c1-right",
                  leftIndexId: "index_a1_right",
                  rightIndexId: "index_a2_left",
                  metadata: {{}},
                }},
              ],
            }};

            const ctx = {{
              state: {{
                spec,
                selectionIds: [],
              }},
              constants: {{
                INDEX_RADIUS: 15,
              }},
              dom: {{}},
              apiGet: async () => null,
              apiPost: async () => null,
              window: globalThis,
              document: {{}},
              cytoscape: null,
              computeDesignBounds() {{
                return {{ x1: 0, y1: 0, x2: 440, y2: 220 }};
              }},
              getVisibleTensors() {{
                return spec.tensors;
              }},
              getVisibleEdges() {{
                return spec.edges;
              }},
              findIndexOwner(indexId) {{
                for (const tensor of spec.tensors) {{
                  const index = tensor.indices.find((candidate) => candidate.id === indexId);
                  if (index) {{
                    return {{ tensor, index }};
                  }}
                }}
                return null;
              }},
              findEdgeByIndexId(indexId) {{
                return (
                  spec.edges.find(
                    (edge) =>
                      edge.leftIndexId === indexId || edge.rightIndexId === indexId
                  ) || null
                );
              }},
              indexAbsolutePosition(tensor, index) {{
                return {{
                  x: tensor.position.x + index.offset.x,
                  y: tensor.position.y + index.offset.y,
                }};
              }},
              buildQuadraticCurve,
              quadraticPointAt,
              getMetadataColor(metadata, fallback) {{
                return metadata && typeof metadata.color === "string" ? metadata.color : fallback;
              }},
              shiftColor(color) {{
                return color;
              }},
              tensorWidth() {{
                return 180;
              }},
              tensorHeight() {{
                return 108;
              }},
              readableTextColor() {{
                return "#f5f9ff";
              }},
              getIndexColor(index, isConnected) {{
                return isConnected ? "#7ed3cf" : "#d7ae68";
              }},
              escapeSvgText,
              escapeSvgAttribute,
            }};

            registerExportMinimap(ctx);
            process.stdout.write(ctx.buildSvgExport());
            """
        ),
        encoding="utf-8",
    )
    return script_path


def _write_hyperedge_shortcut_and_context_menu_runtime_regression_script(
    tmp_path: Path,
) -> Path:
    script_path = tmp_path / "hyperedge_shortcut_and_context_menu_runtime.mjs"
    _copy_runtime_bundle(
        tmp_path,
        {
            "state.runtime.mjs": "state/state.js",
            "utilities.runtime.mjs": "utils/utilities.js",
            "historySelection.runtime.mjs": "graph/historySelection.js",
            "properties.runtime.mjs": "properties/properties.js",
            "propertiesSupport.js": "properties/propertiesSupport.js",
            "propertiesRenderers.js": "properties/propertiesRenderers.js",
            "canvasContextMenu.runtime.mjs": "graph/canvasContextMenu.js",
            "interactionsShortcuts.js": "interactions/interactionsShortcuts.js",
        },
        _RUNTIME_EDITOR_SUPPORT_MODULES,
    )

    script_path.write_text(
        textwrap.dedent(
            """
            const baseUrl = new URL("./", import.meta.url);
            const [
              stateModule,
              utilitiesModule,
              historyModule,
              propertiesModule,
              canvasContextMenuModule,
              shortcutsModule,
            ] = await Promise.all([
              import(new URL("./state.runtime.mjs", baseUrl).href),
              import(new URL("./utilities.runtime.mjs", baseUrl).href),
              import(new URL("./historySelection.runtime.mjs", baseUrl).href),
              import(new URL("./properties.runtime.mjs", baseUrl).href),
              import(new URL("./canvasContextMenu.runtime.mjs", baseUrl).href),
              import(new URL("./interactionsShortcuts.js", baseUrl).href),
            ]);

            const { createInitialState } = stateModule;
            const { registerUtilities } = utilitiesModule;
            const { registerHistorySelection } = historyModule;
            const { registerProperties } = propertiesModule;
            const { registerCanvasContextMenu } = canvasContextMenuModule;
            const { createInteractionShortcutBindings } = shortcutsModule;

            function createClassList() {
              const names = new Set();
              return {
                add(name) {
                  names.add(name);
                },
                remove(name) {
                  names.delete(name);
                },
                toggle(name, force) {
                  if (force === true) {
                    names.add(name);
                    return true;
                  }
                  if (force === false) {
                    names.delete(name);
                    return false;
                  }
                  if (names.has(name)) {
                    names.delete(name);
                    return false;
                  }
                  names.add(name);
                  return true;
                },
                contains(name) {
                  return names.has(name);
                },
              };
            }

            function createFakeElement(id = null, tagName = "div") {
              return {
                id,
                tagName: String(tagName || "div").toUpperCase(),
                value: "",
                textContent: "",
                selected: false,
                checked: false,
                disabled: false,
                hidden: false,
                dataset: {},
                attributes: {},
                style: {},
                className: "",
                classList: createClassList(),
                listeners: {},
                ownerDocument: null,
                addEventListener(eventName, listener) {
                  if (!this.listeners[eventName]) {
                    this.listeners[eventName] = [];
                  }
                  this.listeners[eventName].push(listener);
                },
                dispatchEvent(eventName, event = {}) {
                  (this.listeners[eventName] || []).forEach((listener) => {
                    listener({
                      preventDefault() {},
                      stopPropagation() {},
                      target: this,
                      ...event,
                    });
                  });
                },
                click() {
                  this.dispatchEvent("click");
                },
                focus() {
                  if (this.ownerDocument) {
                    this.ownerDocument.activeElement = this;
                  }
                },
                setAttribute(name, value) {
                  this.attributes[name] = value;
                },
                removeAttribute(name) {
                  delete this.attributes[name];
                },
                appendChild() {},
                closest() {
                  return null;
                },
              };
            }

            function createFakeDocument() {
              const elements = new Map();
              return {
                activeElement: null,
                registerHtml(html) {
                  elements.clear();
                  const tagPattern = /<(input|textarea|button|select)[^>]*id="([^"]+)"[^>]*>/g;
                  let tagMatch = tagPattern.exec(html);
                  while (tagMatch) {
                    const element = createFakeElement(tagMatch[2], tagMatch[1]);
                    element.ownerDocument = this;
                    elements.set(tagMatch[2], element);
                    tagMatch = tagPattern.exec(html);
                  }
                },
                getElementById(id) {
                  return elements.get(id) || null;
                },
                createElement(tagName) {
                  const element = createFakeElement(null, tagName);
                  element.ownerDocument = this;
                  return element;
                },
                querySelectorAll() {
                  return [];
                },
                addEventListener() {},
                removeEventListener() {},
                body: {
                  appendChild() {},
                },
              };
            }

            function createPanel(document) {
              let html = "";
              return {
                get innerHTML() {
                  return html;
                },
                set innerHTML(value) {
                  html = value;
                  document.registerHtml(value);
                },
                querySelectorAll() {
                  return [];
                },
                getBoundingClientRect() {
                  return {
                    left: 0,
                    top: 0,
                    width: 1000,
                    height: 800,
                    right: 1000,
                    bottom: 800,
                  };
                },
              };
            }

            function createButton() {
              return createFakeElement(null, "button");
            }

            function createSelectElement(value = "") {
              const element = createFakeElement(null, "select");
              element.value = value;
              element.options = [];
              element.appendChild = function appendOption(option) {
                this.options.push(option);
                if (option.selected) {
                  this.value = option.value;
                }
              };
              return element;
            }

            function createEvent({
              key,
              altKey = false,
              ctrlKey = false,
              metaKey = false,
              shiftKey = false,
            }) {
              return {
                key,
                altKey,
                ctrlKey,
                metaKey,
                shiftKey,
                preventDefaultCalls: 0,
                preventDefault() {
                  this.preventDefaultCalls += 1;
                },
                target: null,
              };
            }

            function buildSpec() {
              return {
                id: "network_hyperedge_shortcuts",
                name: "hyperedge-shortcuts",
                tensors: [
                  {
                    id: "tensor_a",
                    name: "A",
                    position: { x: 120, y: 120 },
                    size: { width: 140, height: 84 },
                    indices: [
                      {
                        id: "tensor_a_left",
                        name: "left",
                        dimension: 3,
                        offset: { x: -38, y: 0 },
                        metadata: {},
                      },
                    ],
                    metadata: {},
                  },
                  {
                    id: "tensor_b",
                    name: "B",
                    position: { x: 320, y: 120 },
                    size: { width: 140, height: 84 },
                    indices: [
                      {
                        id: "tensor_b_left",
                        name: "left",
                        dimension: 3,
                        offset: { x: -38, y: 0 },
                        metadata: {},
                      },
                      {
                        id: "tensor_b_right",
                        name: "right",
                        dimension: 3,
                        offset: { x: 38, y: 0 },
                        metadata: {},
                      },
                    ],
                    metadata: {},
                  },
                  {
                    id: "tensor_c",
                    name: "C",
                    position: { x: 520, y: 120 },
                    size: { width: 140, height: 84 },
                    indices: [
                      {
                        id: "tensor_c_left",
                        name: "left",
                        dimension: 3,
                        offset: { x: -38, y: 0 },
                        metadata: {},
                      },
                    ],
                    metadata: {},
                  },
                ],
                edges: [],
                hyperedges: [],
                groups: [],
                notes: [],
                contraction_plan: null,
                metadata: {},
              };
            }

            function assertSingleHyperedge(ctx, sourceLabel) {
              if (!Array.isArray(ctx.state.spec.hyperedges) || ctx.state.spec.hyperedges.length !== 1) {
                throw new Error(`${sourceLabel} should create exactly one hyperedge, received ${JSON.stringify(ctx.state.spec.hyperedges)}.`);
              }
              const createdHyperedge = ctx.state.spec.hyperedges[0];
              if (
                !createdHyperedge.hub_offset
                || createdHyperedge.hub_offset.x !== 0
                || createdHyperedge.hub_offset.y !== 0
              ) {
                throw new Error(`${sourceLabel} should create a hyperedge with a zero hub offset, received ${JSON.stringify(createdHyperedge)}.`);
              }
              const expectedSelection = ctx.hyperedgeHubNodeId(createdHyperedge.id);
              if (JSON.stringify(ctx.state.selectionIds) !== JSON.stringify([expectedSelection])) {
                throw new Error(`${sourceLabel} should select the new hyperedge hub, received ${JSON.stringify(ctx.state.selectionIds)}.`);
              }
            }

            const document = createFakeDocument();
            const propertiesPanel = createPanel(document);
            const canvasContextMenuRoot = createPanel(document);
            const statusCalls = [];
            const ctx = {
              state: createInitialState(),
              constants: {
                TENSOR_WIDTH: 140,
                TENSOR_HEIGHT: 84,
                MIN_TENSOR_WIDTH: 96,
                MIN_TENSOR_HEIGHT: 60,
                INDEX_RADIUS: 10,
                INDEX_PADDING: 6,
                NOTE_WIDTH: 220,
                NOTE_HEIGHT: 120,
                NOTE_MIN_WIDTH: 120,
                NOTE_MIN_HEIGHT: 90,
                HISTORY_LIMIT: 100,
                REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
                DEFAULT_INDEX_SLOTS: [
                  { x: -38, y: 0 },
                  { x: 38, y: 0 },
                  { x: 0, y: -24 },
                  { x: 0, y: 24 },
                ],
              },
              dom: {
                workspace: {},
                statusMessage: { textContent: "", classList: createClassList() },
                propertiesPanel,
                canvasContextMenuRoot,
                generatedCode: { value: "" },
                engineSelect: createSelectElement("tensornetwork"),
                collectionFormatSelect: createSelectElement("list"),
                exportFormatSelect: { value: "py" },
                addNoteButton: createButton(),
                connectButton: createButton(),
                loadInput: { click() {} },
                undoButton: createButton(),
                redoButton: createButton(),
                exportButton: createButton(),
                toggleLinearPeriodicButton: createButton(),
                linearPeriodicPreviousCellButton: createButton(),
                linearPeriodicCellLabel: { textContent: "" },
                linearPeriodicNextCellButton: createButton(),
                templateSelect: createSelectElement(""),
                templateParameterPanel: { hidden: true },
                templateGraphSizeLabel: { textContent: "" },
                templateGraphSizeInput: { value: "2", min: "1", addEventListener() {} },
                templateBondDimensionInput: { value: "3", min: "1", addEventListener() {} },
                templatePhysicalDimensionInput: { value: "2", min: "1", addEventListener() {} },
                insertTemplateButton: createButton(),
                createGroupButton: createButton(),
                helpButton: createButton(),
                helpModal: { classList: createClassList() },
                helpBackdrop: createButton(),
                helpCloseButton: createButton(),
                canvasShell: {
                  getBoundingClientRect() {
                    return { left: 0, top: 0, width: 1000, height: 800 };
                  },
                  addEventListener() {},
                },
                groupLayer: {},
                resizeLayer: {},
                notesLayer: {},
                selectionBox: {},
                minimapCanvas: {},
                sidebar: {},
                plannerPanel: {},
                generateButton: createButton(),
              },
              apiGet: async () => null,
              apiPost: async () => null,
              window: {
                structuredClone: globalThis.structuredClone,
                crypto: globalThis.crypto,
                setTimeout,
                clearTimeout,
                confirm: () => true,
                addEventListener() {},
                removeEventListener() {},
              },
              document,
              cytoscape: null,
              tensorWidth: (tensor) => tensor?.size?.width ?? 140,
              tensorHeight: (tensor) => tensor?.size?.height ?? 84,
              renderGraph() {},
              renderOverlayDecorations() {},
              renderMinimap() {},
              renderPlanner() {},
              renderSidebarTabs() {},
              refreshContractionAnalysis() {},
              repairContractionPlan() {},
              updateToolbarState() {},
              captureEditableFocus() {
                return null;
              },
              restoreEditableFocus() {},
              findNoteById() {
                return null;
              },
              isTextInput() {
                return false;
              },
              setStatus(message, level) {
                statusCalls.push({ message, level });
                this.dom.statusMessage.textContent = message;
              },
              toggleSidebarCollapsed() {},
              setActiveSidebarTab() {},
              createGroupFromSelection() {},
              addNoteAtCenter() {},
              toggleTemplateManager() {},
              openCanvasMetadataFilter() {},
              openCanvasNameSearch() {},
              toggleLinearPeriodicMode() {},
              setLinearPeriodicMode() {},
              setGridPeriodicMode() {},
              setTreePeriodicMode() {},
              setBenchmarkMode() {},
              switchLinearPeriodicCell() {},
              switchGridPeriodicCell() {},
              switchTreePeriodicCell() {},
              switchBenchmarkPosition() {},
              nudgeSelectedElements() {
                return false;
              },
              openSessionTemplatePicker() {},
              exportSelectedTemplateSpec() {},
              closeBenchmarkCompareModal() {},
              addTensorAtCenter() {},
              toggleConnectMode() {},
              insertTemplate() {},
              saveDesign() {},
              performUndo() {},
              performRedo() {},
              toggleGeneratedCodeModal() {},
              finishBoxSelection() {},
              generateCode() {},
              deleteSelection() {},
            };

            registerUtilities(ctx);
            const runtimeSetStatus = ctx.setStatus.bind(ctx);
            ctx.setStatus = (message, level) => {
              statusCalls.push({ message, level });
              runtimeSetStatus(message, level);
            };
            registerHistorySelection(ctx);
            registerProperties(ctx);
            registerCanvasContextMenu(ctx);
            Object.assign(
              ctx,
              createInteractionShortcutBindings({
                ctx,
                state: ctx.state,
                dom: ctx.dom,
                runtime: {},
                shortcutActions: {},
              })
            );

            ctx.render = (options = {}) => {
              const resolvedOptions = {
                graph: true,
                properties: true,
                code: true,
                toolbar: true,
                overlays: true,
                planner: true,
                sidebarTabs: true,
                minimap: true,
                syncSelection: false,
                ...options,
              };
              if (resolvedOptions.properties) {
                ctx.renderProperties();
              }
            };

            ctx.state.selectedEngine = "tensornetwork";
            ctx.state.selectedCollectionFormat = "list";

            ctx.state.spec = ctx.normalizeSpec(buildSpec());
            ctx.bumpSpecRevision();
            ctx.setSelection(
              ["tensor_a_left", "tensor_b_left", "tensor_c_left"],
              { primaryId: "tensor_b_left" }
            );
            ctx.renderProperties();

            const selectionButton = document.getElementById("create-hyperedge-button");
            if (!selectionButton) {
              throw new Error("Expected the Selection panel to expose the hyperedge button.");
            }
            selectionButton.click();
            assertSingleHyperedge(ctx, "The Selection panel button");

            ctx.state.spec = ctx.normalizeSpec(buildSpec());
            ctx.bumpSpecRevision();
            ctx.setSelection(
              ["tensor_a", "tensor_a_left", "tensor_b", "tensor_b_left", "tensor_b_right"],
              { primaryId: "tensor_b_right" }
            );
            ctx.renderProperties();

            const ownerSelectionButton = document.getElementById("create-hyperedge-button");
            if (!ownerSelectionButton) {
              throw new Error("Expected the Selection panel to expose the hyperedge button when owner tensors are selected together with their indices.");
            }
            ownerSelectionButton.click();
            assertSingleHyperedge(ctx, "The Selection panel button with owner tensors");

            ctx.state.spec = ctx.normalizeSpec(buildSpec());
            ctx.bumpSpecRevision();
            ctx.setSelection(
              ["tensor_a_left", "tensor_b_left", "tensor_c_left"],
              { primaryId: "tensor_b_left" }
            );
            const validShortcutEvent = createEvent({ key: "h" });
            ctx.handleKeydown(validShortcutEvent);
            if (validShortcutEvent.preventDefaultCalls !== 1) {
              throw new Error("H should prevent the browser default for a valid hyperedge selection.");
            }
            assertSingleHyperedge(ctx, "The H shortcut");

            ctx.state.spec = ctx.normalizeSpec(buildSpec());
            ctx.bumpSpecRevision();
            ctx.setSelection(
              ["tensor_a", "tensor_a_left", "tensor_b", "tensor_b_left", "tensor_b_right"],
              { primaryId: "tensor_b_right" }
            );
            const ownerShortcutEvent = createEvent({ key: "h" });
            ctx.handleKeydown(ownerShortcutEvent);
            if (ownerShortcutEvent.preventDefaultCalls !== 1) {
              throw new Error("H should prevent the browser default when owner tensors are selected together with valid hyperedge indices.");
            }
            assertSingleHyperedge(ctx, "The H shortcut with owner tensors");

            statusCalls.length = 0;
            ctx.state.spec = ctx.normalizeSpec(buildSpec());
            ctx.bumpSpecRevision();
            ctx.setSelection(
              ["tensor_a_left", "tensor_b_left"],
              { primaryId: "tensor_b_left" }
            );
            const invalidShortcutEvent = createEvent({ key: "h" });
            ctx.handleKeydown(invalidShortcutEvent);
            if (invalidShortcutEvent.preventDefaultCalls !== 1) {
              throw new Error("H should still prevent the browser default when the selection is invalid.");
            }
            if (ctx.state.spec.hyperedges.length !== 0) {
              throw new Error("An invalid H shortcut should not create a hyperedge.");
            }
            if (
              !statusCalls.length
              || !statusCalls[statusCalls.length - 1].message.includes(
                "Select at least three open indices"
              )
            ) {
              throw new Error(`Expected the invalid H shortcut to reuse the hyperedge validation message, received ${JSON.stringify(statusCalls)}.`);
            }

            ctx.state.spec = ctx.normalizeSpec(buildSpec());
            ctx.bumpSpecRevision();
            ctx.setSelection(
              ["tensor_a", "tensor_a_left", "tensor_b", "tensor_b_left", "tensor_b_right"],
              { primaryId: "tensor_b_right" }
            );
            ctx.openCanvasContextMenu({
              kind: "index",
              id: "tensor_b_left",
              clientX: 120,
              clientY: 160,
            });
            if (
              !ctx.dom.canvasContextMenuRoot.innerHTML.includes(
                "context-menu-create-hyperedge-button"
              )
            ) {
              throw new Error(`Expected the mixed owner/index context menu to expose a hyperedge action, received ${ctx.dom.canvasContextMenuRoot.innerHTML}.`);
            }
            const ownerContextButton = document.getElementById(
              "context-menu-create-hyperedge-button"
            );
            if (!ownerContextButton) {
              throw new Error("Expected the mixed owner/index context-menu hyperedge button to be registered in the fake document.");
            }
            ownerContextButton.click();
            assertSingleHyperedge(ctx, "The multi-index context menu with owner tensors");

            ctx.state.spec = ctx.normalizeSpec(buildSpec());
            ctx.bumpSpecRevision();
            ctx.setSelection(
              ["tensor_a_left", "tensor_b_left", "tensor_c_left"],
              { primaryId: "tensor_b_left" }
            );
            ctx.openCanvasContextMenu({
              kind: "index",
              id: "tensor_b_left",
              clientX: 120,
              clientY: 160,
            });
            if (
              !ctx.dom.canvasContextMenuRoot.innerHTML.includes(
                "context-menu-create-hyperedge-button"
              )
            ) {
              throw new Error(`Expected the multi-index context menu to expose a hyperedge action, received ${ctx.dom.canvasContextMenuRoot.innerHTML}.`);
            }
            const contextButton = document.getElementById(
              "context-menu-create-hyperedge-button"
            );
            if (!contextButton) {
              throw new Error("Expected the context-menu hyperedge button to be registered in the fake document.");
            }
            contextButton.click();
            assertSingleHyperedge(ctx, "The multi-index context menu");
            """
        ),
        encoding="utf-8",
    )
    return script_path


def _write_multi_index_dimension_batch_runtime_regression_script(
    tmp_path: Path,
) -> Path:
    script_path = tmp_path / "multi_index_dimension_batch_runtime.mjs"
    _copy_runtime_bundle(
        tmp_path,
        {
            "state.runtime.mjs": "state/state.js",
            "utilities.runtime.mjs": "utils/utilities.js",
            "historySelection.runtime.mjs": "graph/historySelection.js",
            "properties.runtime.mjs": "properties/properties.js",
            "propertiesSupport.js": "properties/propertiesSupport.js",
            "propertiesRenderers.js": "properties/propertiesRenderers.js",
            "canvasContextMenu.runtime.mjs": "graph/canvasContextMenu.js",
        },
        _RUNTIME_EDITOR_SUPPORT_MODULES,
    )

    script_path.write_text(
        textwrap.dedent(
            """
            const baseUrl = new URL("./", import.meta.url);
            const [
              stateModule,
              utilitiesModule,
              historyModule,
              propertiesModule,
              canvasContextMenuModule,
            ] = await Promise.all([
              import(new URL("./state.runtime.mjs", baseUrl).href),
              import(new URL("./utilities.runtime.mjs", baseUrl).href),
              import(new URL("./historySelection.runtime.mjs", baseUrl).href),
              import(new URL("./properties.runtime.mjs", baseUrl).href),
              import(new URL("./canvasContextMenu.runtime.mjs", baseUrl).href),
            ]);

            const { createInitialState } = stateModule;
            const { registerUtilities } = utilitiesModule;
            const { registerHistorySelection } = historyModule;
            const { registerProperties } = propertiesModule;
            const { registerCanvasContextMenu } = canvasContextMenuModule;

            function createClassList() {
              return {
                add() {},
                remove() {},
                toggle() {},
              };
            }

            function createFakeElement(id = null, tagName = "div") {
              return {
                id,
                tagName: String(tagName || "div").toUpperCase(),
                value: "",
                textContent: "",
                selected: false,
                checked: false,
                disabled: false,
                hidden: false,
                dataset: {},
                attributes: {},
                style: {},
                className: "",
                classList: createClassList(),
                listeners: {},
                ownerDocument: null,
                addEventListener(eventName, listener) {
                  if (!this.listeners[eventName]) {
                    this.listeners[eventName] = [];
                  }
                  this.listeners[eventName].push(listener);
                },
                dispatchEvent(eventName, event = {}) {
                  (this.listeners[eventName] || []).forEach((listener) => {
                    listener({
                      preventDefault() {},
                      stopPropagation() {},
                      target: this,
                      ...event,
                    });
                  });
                },
                click() {
                  this.dispatchEvent("click");
                },
                focus() {
                  if (this.ownerDocument) {
                    this.ownerDocument.activeElement = this;
                  }
                },
                setAttribute(name, value) {
                  this.attributes[name] = value;
                },
                removeAttribute(name) {
                  delete this.attributes[name];
                },
                appendChild() {},
                closest() {
                  return null;
                },
              };
            }

            function createFakeDocument() {
              const elements = new Map();
              return {
                activeElement: null,
                registerHtml(html) {
                  elements.clear();
                  const tagPattern = /<(input|textarea|button|select)[^>]*id="([^"]+)"[^>]*>/g;
                  let tagMatch = tagPattern.exec(html);
                  while (tagMatch) {
                    const element = createFakeElement(tagMatch[2], tagMatch[1]);
                    element.ownerDocument = this;
                    elements.set(tagMatch[2], element);
                    tagMatch = tagPattern.exec(html);
                  }
                },
                getElementById(id) {
                  return elements.get(id) || null;
                },
                createElement(tagName) {
                  const element = createFakeElement(null, tagName);
                  element.ownerDocument = this;
                  return element;
                },
                querySelectorAll() {
                  return [];
                },
                addEventListener() {},
                removeEventListener() {},
                body: {
                  appendChild() {},
                },
              };
            }

            function createPanel(document) {
              let html = "";
              return {
                get innerHTML() {
                  return html;
                },
                set innerHTML(value) {
                  html = value;
                  document.registerHtml(value);
                },
                querySelectorAll() {
                  return [];
                },
                getBoundingClientRect() {
                  return {
                    left: 0,
                    top: 0,
                    width: 1000,
                    height: 800,
                    right: 1000,
                    bottom: 800,
                  };
                },
              };
            }

            function createButton() {
              return createFakeElement(null, "button");
            }

            function createSelectElement(value = "") {
              const element = createFakeElement(null, "select");
              element.value = value;
              element.options = [];
              element.appendChild = function appendOption(option) {
                this.options.push(option);
                if (option.selected) {
                  this.value = option.value;
                }
              };
              return element;
            }

            function buildSpec() {
              return {
                id: "network_multi_index_dimension_batch",
                name: "multi-index-dimension-batch",
                tensors: [
                  {
                    id: "tensor_a",
                    name: "A",
                    position: { x: 120, y: 120 },
                    size: { width: 140, height: 84 },
                    indices: [
                      {
                        id: "tensor_a_left",
                        name: "left",
                        dimension: 2,
                        offset: { x: -38, y: 0 },
                        metadata: {},
                      },
                    ],
                    metadata: {},
                  },
                  {
                    id: "tensor_b",
                    name: "B",
                    position: { x: 320, y: 120 },
                    size: { width: 140, height: 84 },
                    indices: [
                      {
                        id: "tensor_b_left",
                        name: "left",
                        dimension: 3,
                        offset: { x: -38, y: 0 },
                        metadata: {},
                      },
                      {
                        id: "tensor_b_right",
                        name: "right",
                        dimension: 7,
                        offset: { x: 38, y: 0 },
                        metadata: {},
                      },
                    ],
                    metadata: {},
                  },
                  {
                    id: "tensor_c",
                    name: "C",
                    position: { x: 520, y: 120 },
                    size: { width: 140, height: 84 },
                    indices: [
                      {
                        id: "tensor_c_left",
                        name: "left",
                        dimension: 5,
                        offset: { x: -38, y: 0 },
                        metadata: {},
                      },
                    ],
                    metadata: {},
                  },
                ],
                edges: [],
                hyperedges: [],
                groups: [],
                notes: [],
                contraction_plan: null,
                metadata: {},
              };
            }

            function assertDimensions(ctx, indexIds, expectedDimension, sourceLabel) {
              const dimensions = indexIds.map((indexId) => {
                const located = ctx.findIndexOwner(indexId);
                return located && located.index ? located.index.dimension : null;
              });
              if (!dimensions.every((dimension) => dimension === expectedDimension)) {
                throw new Error(`${sourceLabel} should set every selected index to ${expectedDimension}, received ${JSON.stringify(dimensions)}.`);
              }
            }

            const selectedIndexIds = ["tensor_a_left", "tensor_b_left", "tensor_c_left"];
            const document = createFakeDocument();
            const propertiesPanel = createPanel(document);
            const canvasContextMenuRoot = createPanel(document);
            const ctx = {
              state: createInitialState(),
              constants: {
                TENSOR_WIDTH: 140,
                TENSOR_HEIGHT: 84,
                MIN_TENSOR_WIDTH: 96,
                MIN_TENSOR_HEIGHT: 60,
                INDEX_RADIUS: 10,
                INDEX_PADDING: 6,
                NOTE_WIDTH: 220,
                NOTE_HEIGHT: 120,
                NOTE_MIN_WIDTH: 120,
                NOTE_MIN_HEIGHT: 90,
                HISTORY_LIMIT: 100,
                REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
                DEFAULT_INDEX_SLOTS: [
                  { x: -38, y: 0 },
                  { x: 38, y: 0 },
                  { x: 0, y: -24 },
                  { x: 0, y: 24 },
                ],
              },
              dom: {
                workspace: {},
                statusMessage: { textContent: "", classList: createClassList() },
                propertiesPanel,
                canvasContextMenuRoot,
                generatedCode: { value: "" },
                engineSelect: createSelectElement("tensornetwork"),
                collectionFormatSelect: createSelectElement("list"),
                exportFormatSelect: { value: "py" },
                addNoteButton: createButton(),
                connectButton: createButton(),
                loadInput: { click() {} },
                undoButton: createButton(),
                redoButton: createButton(),
                exportButton: createButton(),
                toggleLinearPeriodicButton: createButton(),
                linearPeriodicPreviousCellButton: createButton(),
                linearPeriodicCellLabel: { textContent: "" },
                linearPeriodicNextCellButton: createButton(),
                templateSelect: createSelectElement(""),
                templateParameterPanel: { hidden: true },
                templateGraphSizeLabel: { textContent: "" },
                templateGraphSizeInput: { value: "2", min: "1", addEventListener() {} },
                templateBondDimensionInput: { value: "3", min: "1", addEventListener() {} },
                templatePhysicalDimensionInput: { value: "2", min: "1", addEventListener() {} },
                insertTemplateButton: createButton(),
                createGroupButton: createButton(),
                helpButton: createButton(),
                helpModal: { classList: createClassList() },
                helpBackdrop: createButton(),
                helpCloseButton: createButton(),
                canvasShell: {
                  getBoundingClientRect() {
                    return { left: 0, top: 0, width: 1000, height: 800 };
                  },
                  addEventListener() {},
                },
                groupLayer: {},
                resizeLayer: {},
                notesLayer: {},
                selectionBox: {},
                minimapCanvas: {},
                sidebar: {},
                plannerPanel: {},
                generateButton: createButton(),
              },
              apiGet: async () => null,
              apiPost: async () => null,
              window: {
                structuredClone: globalThis.structuredClone,
                crypto: globalThis.crypto,
                setTimeout,
                clearTimeout,
                confirm: () => true,
                addEventListener() {},
                removeEventListener() {},
              },
              document,
              cytoscape: null,
              tensorWidth: (tensor) => tensor?.size?.width ?? 140,
              tensorHeight: (tensor) => tensor?.size?.height ?? 84,
              renderGraph() {},
              renderOverlayDecorations() {},
              renderMinimap() {},
              renderPlanner() {},
              renderSidebarTabs() {},
              refreshContractionAnalysis() {},
              repairContractionPlan() {},
              updateToolbarState() {},
              captureEditableFocus() {
                return null;
              },
              restoreEditableFocus() {},
              findNoteById() {
                return null;
              },
              isTextInput() {
                return false;
              },
              setStatus(message) {
                this.dom.statusMessage.textContent = message;
              },
              toggleSidebarCollapsed() {},
              setActiveSidebarTab() {},
              createGroupFromSelection() {},
              addNoteAtCenter() {},
              toggleTemplateManager() {},
              openCanvasMetadataFilter() {},
              openCanvasNameSearch() {},
              toggleLinearPeriodicMode() {},
              setLinearPeriodicMode() {},
              setGridPeriodicMode() {},
              setTreePeriodicMode() {},
              setBenchmarkMode() {},
              switchLinearPeriodicCell() {},
              switchGridPeriodicCell() {},
              switchTreePeriodicCell() {},
              switchBenchmarkPosition() {},
              nudgeSelectedElements() {
                return false;
              },
              openSessionTemplatePicker() {},
              exportSelectedTemplateSpec() {},
              closeBenchmarkCompareModal() {},
              addTensorAtCenter() {},
              toggleConnectMode() {},
              insertTemplate() {},
              saveDesign() {},
              performUndo() {},
              performRedo() {},
              toggleGeneratedCodeModal() {},
              finishBoxSelection() {},
              generateCode() {},
              deleteSelection() {},
            };

            registerUtilities(ctx);
            registerHistorySelection(ctx);
            registerProperties(ctx);
            registerCanvasContextMenu(ctx);

            ctx.render = (options = {}) => {
              const resolvedOptions = {
                graph: true,
                properties: true,
                code: true,
                toolbar: true,
                overlays: true,
                planner: true,
                sidebarTabs: true,
                minimap: true,
                syncSelection: false,
                ...options,
              };
              if (resolvedOptions.properties) {
                ctx.renderProperties();
              }
            };

            ctx.state.selectedEngine = "tensornetwork";
            ctx.state.selectedCollectionFormat = "list";

            ctx.state.spec = ctx.normalizeSpec(buildSpec());
            ctx.bumpSpecRevision();
            ctx.setSelection(selectedIndexIds, { primaryId: "tensor_b_left" });
            ctx.renderProperties();

            const selectionDimensionInput = document.getElementById(
              "multi-index-dimension-input"
            );
            if (!selectionDimensionInput) {
              throw new Error(
                "Expected the Selection panel to expose a shared dimension input for multiple selected indices."
              );
            }
            selectionDimensionInput.value = "11";
            selectionDimensionInput.dispatchEvent("blur");
            assertDimensions(ctx, selectedIndexIds, 11, "The Selection panel");

            ctx.state.spec = ctx.normalizeSpec(buildSpec());
            ctx.bumpSpecRevision();
            ctx.setSelection(
              ["tensor_a", "tensor_a_left", "tensor_b", "tensor_b_left", "tensor_c", "tensor_c_left"],
              { primaryId: "tensor_b_left" }
            );
            ctx.renderProperties();

            const ownerSelectionDimensionInput = document.getElementById(
              "multi-index-dimension-input"
            );
            if (!ownerSelectionDimensionInput) {
              throw new Error(
                "Expected the Selection panel to keep the shared dimension input when owner tensors are selected together with their indices."
              );
            }
            ownerSelectionDimensionInput.value = "12";
            ownerSelectionDimensionInput.dispatchEvent("blur");
            assertDimensions(
              ctx,
              selectedIndexIds,
              12,
              "The Selection panel with owner tensors"
            );

            ctx.state.spec = ctx.normalizeSpec(buildSpec());
            ctx.bumpSpecRevision();
            ctx.setSelection(
              ["tensor_a", "tensor_a_left", "tensor_b", "tensor_b_left", "tensor_c", "tensor_c_left"],
              { primaryId: "tensor_b_left" }
            );
            ctx.openCanvasContextMenu({
              kind: "index",
              id: "tensor_b_left",
              clientX: 120,
              clientY: 160,
            });

            if (
              !ctx.dom.canvasContextMenuRoot.innerHTML.includes(
                "context-menu-selection-dimension-input"
              )
            ) {
              throw new Error(
                `Expected the multi-index context menu to expose a shared dimension input, received ${ctx.dom.canvasContextMenuRoot.innerHTML}.`
              );
            }
            const contextDimensionInput = document.getElementById(
              "context-menu-selection-dimension-input"
            );
            if (!contextDimensionInput) {
              throw new Error(
                "Expected the context menu shared dimension input to be registered in the fake document."
              );
            }
            contextDimensionInput.value = "13";
            contextDimensionInput.dispatchEvent("blur");
            assertDimensions(
              ctx,
              selectedIndexIds,
              13,
              "The multi-index context menu"
            );

            ctx.state.spec = ctx.normalizeSpec(buildSpec());
            ctx.bumpSpecRevision();
            const twoSelectedIndexIds = ["tensor_a_left", "tensor_b_left"];
            ctx.setSelection(twoSelectedIndexIds, { primaryId: "tensor_b_left" });
            ctx.openCanvasContextMenu({
              kind: "index",
              id: "tensor_b_left",
              clientX: 120,
              clientY: 160,
            });

            const twoIndexMenuHtml = ctx.dom.canvasContextMenuRoot.innerHTML;
            if (!twoIndexMenuHtml.includes("context-menu-selection-dimension-input")) {
              throw new Error(
                `Expected exactly two selected indices to keep the multi-index context menu, received ${twoIndexMenuHtml}.`
              );
            }
            if (
              !/id="context-menu-create-hyperedge-button"[^>]*disabled/.test(
                twoIndexMenuHtml
              )
            ) {
              throw new Error(
                `Expected the two-index context menu to expose a disabled hyperedge button, received ${twoIndexMenuHtml}.`
              );
            }
            const twoIndexDimensionInput = document.getElementById(
              "context-menu-selection-dimension-input"
            );
            if (!twoIndexDimensionInput) {
              throw new Error(
                "Expected the two-index context menu shared dimension input to be registered."
              );
            }
            twoIndexDimensionInput.value = "17";
            twoIndexDimensionInput.dispatchEvent("blur");
            assertDimensions(
              ctx,
              twoSelectedIndexIds,
              17,
              "The two-index context menu"
            );
            """
        ),
        encoding="utf-8",
    )
    return script_path


def _write_port_layering_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "port_layering_runtime_regression.mjs"
    script_path.write_text(
        textwrap.dedent(
            f"""
            import {{ pathToFileURL }} from "node:url";

            const graphModelUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "views" / "graphElementModel.js")!r}).href;
            const graphModelModule = await import(graphModelUrl);
            const {{ createGraphElementModelBuilder }} = graphModelModule;

            const state = {{
              pendingIndexId: null,
              pendingPlannerSelectionId: null,
              selectionIds: [],
              spec: {{
                tensors: [
                  {{
                    id: "tensor_back",
                    name: "Back",
                    position: {{ x: 100, y: 100 }},
                    size: {{ width: 140, height: 84 }},
                    indices: [
                      {{
                        id: "back_open",
                        name: "open",
                        dimension: 2,
                        offset: {{ x: 40, y: 0 }},
                        metadata: {{}},
                      }},
                      {{
                        id: "back_connected",
                        name: "connected",
                        dimension: 2,
                        offset: {{ x: -40, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                    metadata: {{}},
                  }},
                  {{
                    id: "tensor_front",
                    name: "Front",
                    position: {{ x: 130, y: 100 }},
                    size: {{ width: 140, height: 84 }},
                    indices: [
                      {{
                        id: "front_connected",
                        name: "connected",
                        dimension: 2,
                        offset: {{ x: 40, y: 0 }},
                        metadata: {{}},
                      }},
                    ],
                    metadata: {{}},
                  }},
                ],
                edges: [
                  {{
                    id: "edge_shared",
                    left: {{ index_id: "back_connected" }},
                    right: {{ index_id: "front_connected" }},
                    metadata: {{}},
                  }},
                ],
                hyperedges: [],
              }},
            }};

            const builder = createGraphElementModelBuilder({{
              state,
              buildContractionScene: () => null,
              ensureTensorIndexOffsets: () => {{}},
              findIndexOwner: (indexId) => {{
                for (const tensor of state.spec.tensors) {{
                  const index = tensor.indices.find((candidate) => candidate.id === indexId);
                  if (index) {{
                    return {{ tensor, index }};
                  }}
                }}
                return null;
              }},
              findTensorById: (tensorId) =>
                state.spec.tensors.find((tensor) => tensor.id === tensorId) || null,
              getIndexColor: () => "#456cbf",
              getMetadataColor: (metadata, fallbackColor) => fallbackColor,
              getMetadataFilterEntityState: () => "",
              getMetadataFilterHighlight: () => null,
              getHyperedgeHubPosition: () => null,
              hyperedgeHubNodeId: (hyperedgeId) => `hyperedge-hub:${{hyperedgeId}}`,
              hyperedgeSpokeEdgeId: (hyperedgeId, endpointPosition) =>
                `hyperedge-spoke:${{hyperedgeId}}:${{endpointPosition}}`,
              indexAbsolutePosition: (tensor, index) => ({{
                x: tensor.position.x + index.offset.x,
                y: tensor.position.y + index.offset.y,
              }}),
              indexLabelNodeId: (indexId) => `${{indexId}}__label`,
              indexLabelPosition: (position) => position,
              isInspectingPastStage: () => false,
              readableTextColor: () => "#111111",
              shiftColor: (color) => color,
              tensorHeight: (tensor) => tensor.size.height,
              tensorLayerRank: (tensorId) => (tensorId === "tensor_front" ? 1 : 0),
              tensorWidth: (tensor) => tensor.size.width,
              zIndexes: {{
                edge: 100,
                indexLabel: 230,
                port: 200,
                tensor: 10,
              }},
            }});

            const model = builder();
            const zIndexFor = (elementId) => model.descriptorsById[elementId].data.zIndex;
            const frontTensorZIndex = zIndexFor("tensor_front");

            if (!(zIndexFor("tensor_back") < zIndexFor("back_open"))) {{
              throw new Error("An open port should still sit above its owning tensor.");
            }}
            if (!(zIndexFor("back_open") < frontTensorZIndex)) {{
              throw new Error(
                `An open port from a rear tensor should not cover a front tensor: open=${{zIndexFor("back_open")}}, front=${{frontTensorZIndex}}.`
              );
            }}
            if (!(zIndexFor("back_connected") > frontTensorZIndex)) {{
              throw new Error(
                `A connected port should stay above tensors so connections remain visible: connected=${{zIndexFor("back_connected")}}, front=${{frontTensorZIndex}}.`
              );
            }}

            state.selectionIds = ["tensor_back"];
            const selectedModel = builder();
            const selectedZIndexFor = (elementId) =>
              selectedModel.descriptorsById[elementId].data.zIndex;
            if (!(selectedZIndexFor("back_open") > selectedZIndexFor("tensor_front"))) {{
              throw new Error(
                `A selected tensor should keep its open ports visible above front tensors: open=${{selectedZIndexFor("back_open")}}, front=${{selectedZIndexFor("tensor_front")}}.`
              );
            }}
            """
        ),
        encoding="utf-8",
    )
    return script_path


def _write_planner_auto_shortcut_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "planner_auto_shortcut_runtime_regression.mjs"
    _copy_js_modules(tmp_path, _SHORTCUT_RUNTIME_DEPENDENCY_MODULES)

    script_path.write_text(
        textwrap.dedent(
            """
            const baseUrl = new URL("./", import.meta.url);
            const [stateModule, shortcutsModule] =
              await Promise.all([
                import(new URL("./state.runtime.mjs", baseUrl).href),
                import(new URL("./interactionsShortcuts.js", baseUrl).href),
              ]);

            const { createInitialState } = stateModule;
            const { createInteractionShortcutBindings } = shortcutsModule;

            function createClassList() {
              return {
                add() {},
                remove() {},
                toggle() {},
              };
            }

            function createEvent({
              key,
              altKey = false,
              ctrlKey = false,
              metaKey = false,
              shiftKey = false,
            }) {
              return {
                key,
                altKey,
                ctrlKey,
                metaKey,
                shiftKey,
                preventDefaultCalls: 0,
                preventDefault() {
                  this.preventDefaultCalls += 1;
                },
                target: null,
              };
            }

            const shortcutCalls = [];
            const ctx = {
              state: createInitialState(),
              constants: {
                TENSOR_WIDTH: 140,
                TENSOR_HEIGHT: 84,
                MIN_TENSOR_WIDTH: 96,
                MIN_TENSOR_HEIGHT: 60,
                INDEX_RADIUS: 10,
                INDEX_PADDING: 6,
                HISTORY_LIMIT: 100,
                REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
                DEFAULT_INDEX_SLOTS: [],
              },
              dom: {
                statusMessage: { textContent: "", classList: createClassList() },
                propertiesPanel: {},
                generatedCode: {},
                engineSelect: { options: [], value: "tensornetwork" },
                connectButton: {},
                loadInput: { click() {} },
                undoButton: {},
                redoButton: {},
                helpCloseButton: { focus() {} },
                helpModal: { classList: createClassList() },
              },
              document: {
                activeElement: null,
              },
              isTextInput() {
                return false;
              },
              setStatus() {},
            };

            Object.assign(
              ctx,
              createInteractionShortcutBindings({
                ctx,
                state: ctx.state,
                dom: ctx.dom,
                runtime: {},
                shortcutActions: {
                  toggleSidebarCollapsed(force) {
                    shortcutCalls.push({ kind: "sidebar", force });
                  },
                  setActiveSidebarTab(tabId) {
                    shortcutCalls.push({ kind: "tab", tabId });
                  },
                  startAutomaticPreview(mode) {
                    shortcutCalls.push({ kind: "preview", mode });
                  },
                  acceptAutomaticPlan(mode) {
                    shortcutCalls.push({ kind: "accept", mode });
                  },
                  selectAllTensors() {
                    shortcutCalls.push({ kind: "select-all-tensors" });
                  },
                },
              })
            );

            const altAEvent = createEvent({ key: "a", altKey: true });
            ctx.handleKeydown(altAEvent);
            if (altAEvent.preventDefaultCalls !== 1) {
              throw new Error("Alt+A should prevent the browser default.");
            }
            if (
              !shortcutCalls.some(
                (entry) =>
                  entry.kind === "preview" && entry.mode === "automaticFuture"
              )
            ) {
              throw new Error(
                `Alt+A should preview the auto future path, received ${JSON.stringify(shortcutCalls)}.`
              );
            }

            const ctrlAltAEvent = createEvent({ key: "a", ctrlKey: true, altKey: true });
            ctx.handleKeydown(ctrlAltAEvent);
            if (ctrlAltAEvent.preventDefaultCalls !== 1) {
              throw new Error("Ctrl+Alt+A should prevent the browser default.");
            }
            if (
              !shortcutCalls.some(
                (entry) =>
                  entry.kind === "accept" && entry.mode === "automaticFuture"
              )
            ) {
              throw new Error(
                `Ctrl+Alt+A should accept the auto future path, received ${JSON.stringify(shortcutCalls)}.`
              );
            }

            const callsBeforeCtrlA = shortcutCalls.length;
            const ctrlAEvent = createEvent({ key: "a", ctrlKey: true });
            ctx.handleKeydown(ctrlAEvent);
            if (ctrlAEvent.preventDefaultCalls !== 1) {
              throw new Error("Ctrl+A should prevent the browser default and select the visible tensors.");
            }
            if (
              shortcutCalls.length !== callsBeforeCtrlA + 1 ||
              shortcutCalls[shortcutCalls.length - 1].kind !== "select-all-tensors"
            ) {
              throw new Error(
                `Ctrl+A should select the visible tensors, received ${JSON.stringify(shortcutCalls.slice(callsBeforeCtrlA))}.`
              );
            }
            """
        ),
        encoding="utf-8",
    )
    return script_path


def _write_mode_and_template_shortcut_runtime_regression_script(
    tmp_path: Path,
) -> Path:
    script_path = tmp_path / "mode_and_template_shortcut_runtime_regression.mjs"
    _copy_js_modules(tmp_path, _SHORTCUT_RUNTIME_DEPENDENCY_MODULES)

    script_path.write_text(
        textwrap.dedent(
            """
            const baseUrl = new URL("./", import.meta.url);
            const [stateModule, shortcutsModule] =
              await Promise.all([
                import(new URL("./state.runtime.mjs", baseUrl).href),
                import(new URL("./interactionsShortcuts.js", baseUrl).href),
              ]);

            const { createInitialState } = stateModule;
            const { createInteractionShortcutBindings } = shortcutsModule;

            function createClassList() {
              return {
                add() {},
                remove() {},
                toggle() {},
              };
            }

            function createEvent({
              key,
              altKey = false,
              ctrlKey = false,
              metaKey = false,
              shiftKey = false,
            }) {
              return {
                key,
                altKey,
                ctrlKey,
                metaKey,
                shiftKey,
                preventDefaultCalls: 0,
                preventDefault() {
                  this.preventDefaultCalls += 1;
                },
                target: null,
              };
            }

            const shortcutCalls = [];
            const statusCalls = [];
            let textInputActive = false;
            const ctx = {
              state: createInitialState(),
              constants: {
                TENSOR_WIDTH: 140,
                TENSOR_HEIGHT: 84,
                MIN_TENSOR_WIDTH: 96,
                MIN_TENSOR_HEIGHT: 60,
                INDEX_RADIUS: 10,
                INDEX_PADDING: 6,
                HISTORY_LIMIT: 100,
                REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
                DEFAULT_INDEX_SLOTS: [],
              },
              dom: {
                statusMessage: { textContent: "", classList: createClassList() },
                propertiesPanel: {},
                generatedCode: {},
                engineSelect: { options: [], value: "tensornetwork" },
                connectButton: {},
                loadInput: { click() {} },
                undoButton: {},
                redoButton: {},
                helpCloseButton: { focus() {} },
                helpModal: { classList: createClassList() },
              },
              document: {
                activeElement: null,
              },
              isTextInput() {
                return textInputActive;
              },
              getSelectedIdsByKind(kind) {
                return kind === "tensor" ? ["tensor_a", "tensor_b"] : [];
              },
              setStatus(message, level) {
                statusCalls.push({ message, level });
              },
              completeEditor() {
                shortcutCalls.push({ kind: "complete-editor" });
              },
            };

            Object.assign(
              ctx,
              createInteractionShortcutBindings({
                ctx,
                state: ctx.state,
                dom: ctx.dom,
                runtime: {},
                shortcutActions: {
                  setLinearPeriodicMode(enabled) {
                    shortcutCalls.push({ kind: "linear", enabled });
                  },
                  setGridPeriodicMode(enabled) {
                    shortcutCalls.push({ kind: "grid", enabled });
                  },
                  setTreePeriodicMode(enabled) {
                    shortcutCalls.push({ kind: "tree", enabled });
                  },
                  setBenchmarkMode(enabled) {
                    shortcutCalls.push({ kind: "benchmark", enabled });
                  },
                  openSessionTemplatePicker() {
                    shortcutCalls.push({ kind: "load-template" });
                  },
                  exportSelectedSubnetwork() {
                    shortcutCalls.push({ kind: "export-subnetwork" });
                  },
                  openCanvasNameSearch() {
                    shortcutCalls.push({ kind: "open-search" });
                  },
                  openCanvasMetadataFilter() {
                    shortcutCalls.push({ kind: "open-filter" });
                  },
                  addIndexToSelectedTensors(payload) {
                    shortcutCalls.push({
                      kind: "add-index",
                      tensorIds: payload.tensorIds,
                      selectionIds: payload.selectionIds,
                      primaryId: payload.primaryId,
                    });
                    return true;
                  },
                  toggleReflowLayoutPopover() {
                    shortcutCalls.push({ kind: "open-reflow" });
                  },
                },
              })
            );
            ctx.state.selectionIds = ["tensor_a", "tensor_b"];
            ctx.state.primarySelectionId = "tensor_b";

            const dEvent = createEvent({ key: "d" });
            ctx.handleKeydown(dEvent);
            if (dEvent.preventDefaultCalls !== 1) {
              throw new Error("D should prevent the browser default.");
            }
            if (
              JSON.stringify(shortcutCalls.splice(0)) !==
              JSON.stringify([
                { kind: "benchmark", enabled: false },
                { kind: "grid", enabled: true },
              ])
            ) {
              throw new Error("D should switch to For bidimensional mode.");
            }

            const bEvent = createEvent({ key: "b" });
            ctx.handleKeydown(bEvent);
            if (bEvent.preventDefaultCalls !== 1) {
              throw new Error("B should prevent the browser default.");
            }
            if (
              JSON.stringify(shortcutCalls.splice(0)) !==
              JSON.stringify([
                { kind: "linear", enabled: false },
                { kind: "grid", enabled: false },
                { kind: "tree", enabled: false },
                { kind: "benchmark", enabled: true },
              ])
            ) {
              throw new Error("B should switch to Benchmark mode.");
            }

            const shiftSEvent = createEvent({ key: "S", shiftKey: true });
            ctx.handleKeydown(shiftSEvent);
            if (shiftSEvent.preventDefaultCalls !== 1) {
              throw new Error("Shift+S should prevent the browser default.");
            }
            if (
              JSON.stringify(shortcutCalls.splice(0)) !==
              JSON.stringify([
                { kind: "benchmark", enabled: false },
                { kind: "linear", enabled: false },
                { kind: "grid", enabled: false },
                { kind: "tree", enabled: false },
              ])
            ) {
              throw new Error("Shift+S should switch to Single mode.");
            }

            const lEvent = createEvent({ key: "l" });
            ctx.handleKeydown(lEvent);
            if (lEvent.preventDefaultCalls !== 1) {
              throw new Error("L should prevent the browser default.");
            }
            if (
              JSON.stringify(shortcutCalls.splice(0)) !==
              JSON.stringify([{ kind: "load-template" }])
            ) {
              throw new Error("L should open the template picker.");
            }

            const shiftEEvent = createEvent({ key: "E", shiftKey: true });
            ctx.handleKeydown(shiftEEvent);
            if (shiftEEvent.preventDefaultCalls !== 1) {
              throw new Error("Shift+E should prevent the browser default.");
            }
            if (
              JSON.stringify(shortcutCalls.splice(0)) !==
              JSON.stringify([{ kind: "export-subnetwork" }])
            ) {
              throw new Error("Shift+E should save the selected subnetwork.");
            }

            const iEvent = createEvent({ key: "i" });
            ctx.handleKeydown(iEvent);
            if (iEvent.preventDefaultCalls !== 1) {
              throw new Error("I should prevent the browser default.");
            }
            if (
              JSON.stringify(shortcutCalls.splice(0)) !==
              JSON.stringify([
                {
                  kind: "add-index",
                  tensorIds: ["tensor_a", "tensor_b"],
                  selectionIds: ["tensor_a", "tensor_b"],
                  primaryId: "tensor_b",
                },
              ])
            ) {
              throw new Error("I should add one index to each selected tensor.");
            }

            const rEvent = createEvent({ key: "r" });
            ctx.handleKeydown(rEvent);
            if (rEvent.preventDefaultCalls !== 1) {
              throw new Error("R should prevent the browser default.");
            }
            if (
              JSON.stringify(shortcutCalls.splice(0)) !==
              JSON.stringify([{ kind: "open-reflow" }])
            ) {
              throw new Error("R should open the Reflow controls.");
            }

            const ctrlEnterEvent = createEvent({ key: "Enter", ctrlKey: true });
            ctx.handleKeydown(ctrlEnterEvent);
            if (ctrlEnterEvent.preventDefaultCalls !== 1) {
              throw new Error("Ctrl+Enter should prevent the browser default.");
            }
            if (
              JSON.stringify(shortcutCalls.splice(0)) !==
              JSON.stringify([{ kind: "complete-editor" }])
            ) {
              throw new Error("Ctrl+Enter should finish the editor session.");
            }

            textInputActive = true;
            const ctrlEnterInputEvent = createEvent({ key: "Enter", ctrlKey: true });
            ctx.handleKeydown(ctrlEnterInputEvent);
            textInputActive = false;
            if (ctrlEnterInputEvent.preventDefaultCalls !== 1) {
              throw new Error("Ctrl+Enter should prevent the browser default from text inputs.");
            }
            if (
              JSON.stringify(shortcutCalls.splice(0)) !==
              JSON.stringify([{ kind: "complete-editor" }])
            ) {
              throw new Error("Ctrl+Enter should finish the editor session from text inputs.");
            }

            const ctrlFEvent = createEvent({ key: "f", ctrlKey: true });
            ctx.handleKeydown(ctrlFEvent);
            if (ctrlFEvent.preventDefaultCalls !== 1) {
              throw new Error("Ctrl+F should prevent the browser search.");
            }
            if (
              JSON.stringify(shortcutCalls.splice(0)) !==
              JSON.stringify([{ kind: "open-search" }])
            ) {
              throw new Error("Ctrl+F should open the canvas search.");
            }

            const ctrlShiftFEvent = createEvent({
              key: "F",
              ctrlKey: true,
              shiftKey: true,
            });
            ctx.handleKeydown(ctrlShiftFEvent);
            if (ctrlShiftFEvent.preventDefaultCalls !== 1) {
              throw new Error("Ctrl+Shift+F should prevent the browser search.");
            }
            if (
              JSON.stringify(shortcutCalls.splice(0)) !==
              JSON.stringify([{ kind: "open-filter" }])
            ) {
              throw new Error("Ctrl+Shift+F should open the canvas filters.");
            }

            const eEvent = createEvent({ key: "e" });
            ctx.handleKeydown(eEvent);
            if (eEvent.preventDefaultCalls !== 1) {
              throw new Error("E should prevent the browser default.");
            }
            if (
              JSON.stringify(shortcutCalls.splice(0)) !==
              JSON.stringify([
                { kind: "benchmark", enabled: false },
                { kind: "tree", enabled: true },
              ])
            ) {
              throw new Error(
                `E should switch to For Tree mode, received ${JSON.stringify(shortcutCalls)}.`
              );
            }
            """
        ),
        encoding="utf-8",
    )
    return script_path


def _write_keyboard_navigation_and_nudge_runtime_regression_script(
    tmp_path: Path,
) -> Path:
    script_path = tmp_path / "keyboard_navigation_and_nudge_runtime_regression.mjs"
    _copy_runtime_bundle(
        tmp_path,
        {
            "state.runtime.mjs": "state/state.js",
            "interactionsShortcuts.js": "interactions/interactionsShortcuts.js",
            "interactionsEditor.js": "interactions/interactionsEditor.js",
            "selectionEntries.js": "state/selectionEntries.js",
            "notes.js": "graph/notes.js",
        },
        _RUNTIME_EDITOR_SUPPORT_MODULES,
    )

    script_path.write_text(
        textwrap.dedent(
            """
            const baseUrl = new URL("./", import.meta.url);
            const [stateModule, shortcutsModule, editorModule, selectionEntriesModule, notesModule] =
              await Promise.all([
                import(new URL("./state.runtime.mjs", baseUrl).href),
                import(new URL("./interactionsShortcuts.js", baseUrl).href),
                import(new URL("./interactionsEditor.js", baseUrl).href),
                import(new URL("./selectionEntries.js", baseUrl).href),
                import(new URL("./notes.js", baseUrl).href),
              ]);

            const { createInitialState } = stateModule;
            const { createInteractionShortcutBindings } = shortcutsModule;
            const { createInteractionEditorBindings } = editorModule;
            const { createSelectionEntrySupport } = selectionEntriesModule;
            const { registerNotesFeature } = notesModule;

            function createClassList() {
              return {
                add() {},
                remove() {},
                toggle() {},
              };
            }

            function createEvent({
              key,
              altKey = false,
              ctrlKey = false,
              metaKey = false,
              shiftKey = false,
              target = null,
            }) {
              return {
                key,
                altKey,
                ctrlKey,
                metaKey,
                shiftKey,
                target,
                preventDefaultCalls: 0,
                preventDefault() {
                  this.preventDefaultCalls += 1;
                },
              };
            }

            const state = createInitialState();
            const shortcutCalls = [];
            const committedSnapshots = [];
            let snapshotCounter = 0;
            const ctx = {
              state,
              constants: {
                TENSOR_WIDTH: 140,
                TENSOR_HEIGHT: 84,
                MIN_TENSOR_WIDTH: 96,
                MIN_TENSOR_HEIGHT: 60,
                INDEX_RADIUS: 10,
                INDEX_PADDING: 6,
                NOTE_WIDTH: 220,
                NOTE_HEIGHT: 120,
                NOTE_MIN_WIDTH: 120,
                NOTE_MIN_HEIGHT: 90,
                NOTE_COLLAPSED_SIZE: 40,
                HISTORY_LIMIT: 100,
                REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
                DEFAULT_INDEX_SLOTS: [
                  { x: -38, y: 0 },
                  { x: 38, y: 0 },
                  { x: 0, y: -24 },
                  { x: 0, y: 24 },
                ],
              },
              dom: {
                statusMessage: { textContent: "", classList: createClassList() },
                propertiesPanel: {},
                generatedCode: {},
                engineSelect: { options: [], value: "tensornetwork" },
                connectButton: {},
                loadInput: { click() {} },
                undoButton: {},
                redoButton: {},
                helpCloseButton: { focus() {} },
                helpModal: { classList: createClassList() },
                addNoteButton: {},
                notesLayer: {},
              },
              document: {
                activeElement: null,
              },
              window: {
                structuredClone: globalThis.structuredClone,
                crypto: globalThis.crypto,
                setTimeout,
                clearTimeout,
              },
              isTextInput(element) {
                return Boolean(element) && ["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName);
              },
              setStatus() {},
              makeId(prefix) {
                snapshotCounter += 1;
                return `${prefix}_${snapshotCounter}`;
              },
              nextName(prefix, usedNames = []) {
                let counter = 1;
                let candidate = `${prefix}${counter}`;
                while (usedNames.includes(candidate)) {
                  counter += 1;
                  candidate = `${prefix}${counter}`;
                }
                return candidate;
              },
              tensorWidth(tensor) {
                return tensor?.size?.width ?? 140;
              },
              tensorHeight(tensor) {
                return tensor?.size?.height ?? 84;
              },
              defaultIndexOffsetForOrder(indexPosition) {
                return [
                  { x: -38, y: 0 },
                  { x: 38, y: 0 },
                  { x: 0, y: -24 },
                  { x: 0, y: 24 },
                ][indexPosition] || { x: 0, y: 0 };
              },
              runWithTensorSync(action) {
                action();
              },
              syncIndexNodePositions() {},
              renderOverlayDecorations() {},
              renderMinimap() {},
              updateToolbarState() {},
              renderProperties() {},
              createHistorySnapshot() {
                snapshotCounter += 1;
                return { id: `snapshot_${snapshotCounter}` };
              },
              commitHistorySnapshot(snapshot) {
                committedSnapshots.push(snapshot.id);
              },
              clientPointToWorldPoint(clientX, clientY) {
                return { x: clientX, y: clientY };
              },
              findVisibleTensorById(tensorId) {
                return ctx.findTensorById(tensorId);
              },
              canEditCurrentContractionStage() {
                return false;
              },
              updateCurrentStageOperandLayout() {},
              toggleSidebarCollapsed() {},
              setActiveSidebarTab() {},
              syncPendingInteractionClasses() {},
              removeNote(noteId) {
                state.spec.notes = state.spec.notes.filter((note) => note.id !== noteId);
              },
              clearSelection() {},
            };

            state.spec = {
              id: "network_keyboard",
              name: "keyboard",
              tensors: [
                {
                  id: "tensor_a",
                  name: "A",
                  position: { x: 100, y: 120 },
                  size: { width: 140, height: 84 },
                  metadata: {},
                  indices: [
                    {
                      id: "index_a",
                      name: "a",
                      dimension: 2,
                      offset: { x: 14, y: 0 },
                      metadata: {},
                    },
                  ],
                },
                {
                  id: "tensor_b",
                  name: "B",
                  position: { x: 220, y: 160 },
                  size: { width: 140, height: 84 },
                  metadata: {},
                  indices: [],
                },
              ],
              edges: [
                {
                  id: "edge_ab",
                  name: "ab",
                  left: { tensor_id: "tensor_a", index_id: "index_a" },
                  right: { tensor_id: "tensor_b", index_id: "missing_index" },
                  metadata: {},
                },
              ],
              groups: [
                {
                  id: "group_ab",
                  name: "AB",
                  tensor_ids: ["tensor_a", "tensor_b"],
                  metadata: {},
                },
              ],
              notes: [
                {
                  id: "note_a",
                  text: "Note",
                  position: { x: 340, y: 280 },
                  size: { width: 220, height: 120 },
                  metadata: {},
                },
              ],
              contraction_plan: null,
              metadata: {},
            };
            state.noteById = {
              note_a: state.spec.notes[0],
            };

            ctx.findTensorById = (tensorId) =>
              state.spec.tensors.find((tensor) => tensor.id === tensorId) || null;
            ctx.findGroupById = (groupId) =>
              state.spec.groups.find((group) => group.id === groupId) || null;
            ctx.findEdgeById = (edgeId) =>
              state.spec.edges.find((edge) => edge.id === edgeId) || null;
            ctx.findNoteById = (noteId) =>
              state.spec.notes.find((note) => note.id === noteId) || null;
            ctx.findIndexOwner = (indexId) => {
              for (const tensor of state.spec.tensors) {
                const index = (tensor.indices || []).find((candidate) => candidate.id === indexId);
                if (index) {
                  return { tensor, index };
                }
              }
              return null;
            };
            ctx.getVisibleTensors = () => state.spec.tensors;
            ctx.isContractionSceneVisible = () => false;
            ctx.isInspectingPastStage = () => false;
            ctx.isPlannerOperandAvailable = () => false;

            const selectionSupport = createSelectionEntrySupport({
              state,
              findGroupById: ctx.findGroupById,
              findTensorById: ctx.findTensorById,
              findVisibleTensorById: ctx.findVisibleTensorById,
              findIndexOwner: ctx.findIndexOwner,
              findEdgeById: ctx.findEdgeById,
              findNoteById: ctx.findNoteById,
              getVisibleTensors: ctx.getVisibleTensors,
              isContractionSceneVisible: ctx.isContractionSceneVisible,
              isInspectingPastStage: ctx.isInspectingPastStage,
              isPlannerOperandAvailable: ctx.isPlannerOperandAvailable,
              renderSelectionUi() {},
            });
            Object.assign(ctx, selectionSupport);

            registerNotesFeature(ctx);
            ctx.findNoteById = (noteId) =>
              state.spec.notes.find((note) => note.id === noteId) || null;
            Object.assign(
              ctx,
              createInteractionEditorBindings({
                ctx,
                state,
                runtime: {},
              })
            );
            let currentMode = "linear";
            Object.assign(
              ctx,
              createInteractionShortcutBindings({
                ctx,
                state,
                dom: ctx.dom,
                runtime: {},
                shortcutActions: {
                  switchLinearPeriodicCell(direction) {
                    shortcutCalls.push({ kind: "linear-nav", direction });
                  },
                  switchGridPeriodicCell(direction) {
                    shortcutCalls.push({ kind: "grid-nav", direction });
                  },
                  switchTreePeriodicCell(direction) {
                    shortcutCalls.push({ kind: "tree-nav", direction });
                  },
                  switchBenchmarkPosition(direction) {
                    shortcutCalls.push({ kind: "benchmark-nav", direction });
                  },
                },
              })
            );

            ctx.isLinearPeriodicMode = () => currentMode === "linear";
            ctx.isGridPeriodicMode = () => currentMode === "grid";
            ctx.isTreePeriodicMode = () => currentMode === "tree";
            ctx.isBenchmarkMode = () => currentMode === "benchmark";

            currentMode = "linear";
            const linearEvent = createEvent({ key: "ArrowRight", altKey: true });
            ctx.handleKeydown(linearEvent);
            if (linearEvent.preventDefaultCalls !== 1) {
              throw new Error("Alt+ArrowRight should prevent default in linear For mode.");
            }
            if (JSON.stringify(shortcutCalls.splice(0)) !== JSON.stringify([{ kind: "linear-nav", direction: 1 }])) {
              throw new Error("Alt+ArrowRight should switch the linear For cell.");
            }

            currentMode = "grid";
            const gridEvent = createEvent({ key: "ArrowUp", altKey: true });
            ctx.handleKeydown(gridEvent);
            if (gridEvent.preventDefaultCalls !== 1) {
              throw new Error("Alt+ArrowUp should prevent default in grid For mode.");
            }
            if (JSON.stringify(shortcutCalls.splice(0)) !== JSON.stringify([{ kind: "grid-nav", direction: "up" }])) {
              throw new Error("Alt+ArrowUp should switch the grid For cell.");
            }

            currentMode = "tree";
            const treeEvent = createEvent({ key: "ArrowDown", altKey: true });
            ctx.handleKeydown(treeEvent);
            if (treeEvent.preventDefaultCalls !== 1) {
              throw new Error("Alt+ArrowDown should prevent default in tree For mode.");
            }
            if (JSON.stringify(shortcutCalls.splice(0)) !== JSON.stringify([{ kind: "tree-nav", direction: "down" }])) {
              throw new Error("Alt+ArrowDown should switch the tree For cell.");
            }

            currentMode = "benchmark";
            const benchmarkEvent = createEvent({ key: "ArrowLeft", altKey: true });
            ctx.handleKeydown(benchmarkEvent);
            if (benchmarkEvent.preventDefaultCalls !== 1) {
              throw new Error("Alt+ArrowLeft should prevent default in benchmark mode.");
            }
            if (JSON.stringify(shortcutCalls.splice(0)) !== JSON.stringify([{ kind: "benchmark-nav", direction: -1 }])) {
              throw new Error("Alt+ArrowLeft should switch the benchmark position.");
            }

            state.isHelpOpen = true;
            const blockedAltEvent = createEvent({ key: "ArrowRight", altKey: true });
            ctx.handleKeydown(blockedAltEvent);
            if (blockedAltEvent.preventDefaultCalls !== 0 || shortcutCalls.length !== 0) {
              throw new Error("Alt+Arrow navigation should stay inactive while a blocking modal is open.");
            }
            state.isHelpOpen = false;

            ctx.document.activeElement = { tagName: "INPUT" };
            const typingAltEvent = createEvent({ key: "ArrowRight", altKey: true });
            ctx.handleKeydown(typingAltEvent);
            if (typingAltEvent.preventDefaultCalls !== 0 || shortcutCalls.length !== 0) {
              throw new Error("Alt+Arrow navigation should stay inactive while typing.");
            }
            ctx.document.activeElement = null;

            state.selectionIds = ["tensor_a"];
            state.primarySelectionId = "tensor_a";
            const tensorEvent = createEvent({ key: "ArrowRight" });
            ctx.handleKeydown(tensorEvent);
            if (tensorEvent.preventDefaultCalls !== 1) {
              throw new Error("ArrowRight should prevent default when nudging a selected tensor.");
            }
            if (ctx.findTensorById("tensor_a").position.x !== 120) {
              throw new Error(`Expected ArrowRight to move the tensor by 20, received ${ctx.findTensorById("tensor_a").position.x}.`);
            }
            if (state.selectionIds.join(",") !== "tensor_a" || committedSnapshots.length !== 1) {
              throw new Error("Tensor nudging should preserve selection and commit one snapshot.");
            }

            state.selectionIds = ["note_a"];
            state.primarySelectionId = "note_a";
            const noteEvent = createEvent({ key: "ArrowDown", shiftKey: true });
            ctx.handleKeydown(noteEvent);
            if (noteEvent.preventDefaultCalls !== 1) {
              throw new Error("Shift+ArrowDown should prevent default when nudging a selected note.");
            }
            if (ctx.findNoteById("note_a").position.y !== 340) {
              throw new Error(`Expected Shift+ArrowDown to move the note by 60, received ${ctx.findNoteById("note_a").position.y}.`);
            }

            state.selectionIds = ["index_a"];
            state.primarySelectionId = "index_a";
            const indexEvent = createEvent({ key: "ArrowLeft" });
            ctx.handleKeydown(indexEvent);
            if (indexEvent.preventDefaultCalls !== 1) {
              throw new Error("ArrowLeft should prevent default when nudging a selected index.");
            }
            if (ctx.findIndexOwner("index_a").index.offset.x !== -6) {
              throw new Error(`Expected ArrowLeft to move the selected index by 20, received ${ctx.findIndexOwner("index_a").index.offset.x}.`);
            }

            state.selectionIds = ["group_ab"];
            state.primarySelectionId = "group_ab";
            const groupEvent = createEvent({ key: "ArrowUp" });
            ctx.handleKeydown(groupEvent);
            if (groupEvent.preventDefaultCalls !== 1) {
              throw new Error("ArrowUp should prevent default when nudging a selected group.");
            }
            if (ctx.findTensorById("tensor_a").position.y !== 100 || ctx.findTensorById("tensor_b").position.y !== 140) {
              throw new Error("Group nudging should move every tensor inside the selected group.");
            }

            const commitCountBeforeEdge = committedSnapshots.length;
            state.selectionIds = ["edge_ab"];
            state.primarySelectionId = "edge_ab";
            const edgeEvent = createEvent({ key: "ArrowRight" });
            ctx.handleKeydown(edgeEvent);
            if (edgeEvent.preventDefaultCalls !== 0) {
              throw new Error("Arrow nudging should no-op for edge-only selections.");
            }
            if (committedSnapshots.length !== commitCountBeforeEdge) {
              throw new Error("Edge-only selections should not create history snapshots.");
            }
            """
        ),
        encoding="utf-8",
    )
    return script_path


def _write_shift_only_shortcut_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "shift_only_shortcut_runtime_regression.mjs"
    _copy_js_modules(tmp_path, _SHORTCUT_RUNTIME_DEPENDENCY_MODULES)

    script_path.write_text(
        textwrap.dedent(
            """
            const baseUrl = new URL("./", import.meta.url);
            const [stateModule, shortcutsModule] =
              await Promise.all([
                import(new URL("./state.runtime.mjs", baseUrl).href),
                import(new URL("./interactionsShortcuts.js", baseUrl).href),
              ]);

            const { createInitialState } = stateModule;
            const { createInteractionShortcutBindings } = shortcutsModule;

            function createClassList() {
              return {
                add() {},
                remove() {},
                toggle() {},
              };
            }

            function createEvent({
              key,
              altKey = false,
              ctrlKey = false,
              metaKey = false,
              shiftKey = false,
            }) {
              return {
                key,
                altKey,
                ctrlKey,
                metaKey,
                shiftKey,
                preventDefaultCalls: 0,
                preventDefault() {
                  this.preventDefaultCalls += 1;
                },
                target: null,
              };
            }

            const shortcutCalls = [];
            const ctx = {
              state: createInitialState(),
              constants: {
                TENSOR_WIDTH: 140,
                TENSOR_HEIGHT: 84,
                MIN_TENSOR_WIDTH: 96,
                MIN_TENSOR_HEIGHT: 60,
                INDEX_RADIUS: 10,
                INDEX_PADDING: 6,
                HISTORY_LIMIT: 100,
                REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
                DEFAULT_INDEX_SLOTS: [],
              },
              dom: {
                statusMessage: { textContent: "", classList: createClassList() },
                propertiesPanel: {},
                generatedCode: {},
                engineSelect: { options: [], value: "tensornetwork" },
                connectButton: {},
                loadInput: { click() {} },
                undoButton: {},
                redoButton: {},
                helpCloseButton: { focus() {} },
                helpModal: { classList: createClassList() },
              },
              document: {
                activeElement: null,
              },
              isTextInput() {
                return false;
              },
              setStatus() {},
              generateCode() {
                shortcutCalls.push({ kind: "generate-code" });
              },
            };
            ctx.state.spec = { contraction_plan: { id: "plan_demo" } };

            Object.assign(
              ctx,
              createInteractionShortcutBindings({
                ctx,
                state: ctx.state,
                dom: ctx.dom,
                runtime: {},
                shortcutActions: {
                  toggleMinimapVisibility() {
                    shortcutCalls.push({ kind: "toggle-minimap" });
                  },
                  trimContractionPlan(stepCount) {
                    shortcutCalls.push({ kind: "trim-plan", stepCount });
                  },
                },
              })
            );

            const ctrlShiftMEvent = createEvent({ key: "M", ctrlKey: true, shiftKey: true });
            ctx.handleKeydown(ctrlShiftMEvent);
            if (ctrlShiftMEvent.preventDefaultCalls !== 0) {
              throw new Error("Ctrl+Shift+M should not hijack the exact Shift+M minimap shortcut.");
            }

            const altShiftGEvent = createEvent({ key: "G", altKey: true, shiftKey: true });
            ctx.handleKeydown(altShiftGEvent);
            if (altShiftGEvent.preventDefaultCalls !== 0) {
              throw new Error("Alt+Shift+G should not hijack the exact Shift+G generate shortcut.");
            }

            const metaShiftREvent = createEvent({ key: "R", metaKey: true, shiftKey: true });
            ctx.handleKeydown(metaShiftREvent);
            if (metaShiftREvent.preventDefaultCalls !== 0) {
              throw new Error("Cmd+Shift+R should not hijack the exact Shift+R reset shortcut.");
            }

            const altShiftSEvent = createEvent({ key: "S", altKey: true, shiftKey: true });
            ctx.handleKeydown(altShiftSEvent);
            if (altShiftSEvent.preventDefaultCalls !== 0) {
              throw new Error("Alt+Shift+S should not hijack the exact Shift+S single-mode shortcut.");
            }

            const altShiftEEvent = createEvent({ key: "E", altKey: true, shiftKey: true });
            ctx.handleKeydown(altShiftEEvent);
            if (altShiftEEvent.preventDefaultCalls !== 0) {
              throw new Error("Alt+Shift+E should not hijack the exact Shift+E subnetwork shortcut.");
            }

            if (shortcutCalls.length !== 0) {
              throw new Error(
                `Shift-only shortcuts should ignore extra modifiers, received ${JSON.stringify(shortcutCalls)}.`
              );
            }
            """
        ),
        encoding="utf-8",
    )
    return script_path


def _write_additive_selection_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "additive_selection_runtime_regression.mjs"
    _copy_runtime_bundle(
        tmp_path,
        {
            "state.runtime.mjs": "state/state.js",
            "utilities.runtime.mjs": "utils/utilities.js",
            "graphRender.js": "graph/graphRender.js",
            "cytoscapeGraphAdapter.js": "views/cytoscapeGraphAdapter.js",
            "graphDescriptors.js": "views/graphDescriptors.js",
            "graphModelDiff.js": "views/graphModelDiff.js",
            "graphElementModel.js": "views/graphElementModel.js",
            "notes.js": "graph/notes.js",
            "overlaysLayoutTemplates.js": "graph/overlaysLayoutTemplates.js",
        },
        _RUNTIME_EDITOR_SUPPORT_MODULES,
    )

    script_path.write_text(
        textwrap.dedent(
            """
            const baseUrl = new URL("./", import.meta.url);
            const [stateModule, utilitiesModule, graphRenderModule, notesModule, overlaysModule] =
              await Promise.all([
                import(new URL("./state.runtime.mjs", baseUrl).href),
                import(new URL("./utilities.runtime.mjs", baseUrl).href),
                import(new URL("./graphRender.js", baseUrl).href),
                import(new URL("./notes.js", baseUrl).href),
                import(new URL("./overlaysLayoutTemplates.js", baseUrl).href),
              ]);

            const { createInitialState } = stateModule;
            const { registerUtilities } = utilitiesModule;
            const { registerGraphRender } = graphRenderModule;
            const { registerNotesFeature } = notesModule;
            const { registerOverlaysLayoutTemplates } = overlaysModule;

            function createClassList() {
              return {
                add() {},
                remove() {},
                toggle() {},
              };
            }

            const cyHandlers = {};
            const fakeCy = {
              on(eventName, selectorOrHandler, maybeHandler) {
                const key = `${eventName}:${typeof maybeHandler === "function" ? selectorOrHandler : "*"}`;
                cyHandlers[key] = typeof maybeHandler === "function" ? maybeHandler : selectorOrHandler;
              },
              batch(action) {
                action();
              },
              fit() {},
              center() {},
              width() {
                return 1000;
              },
              height() {
                return 800;
              },
              getElementById() {
                return { length: 0 };
              },
              edges() {
                return {
                  forEach() {},
                };
              },
              pan() {
                return { x: 0, y: 0 };
              },
              zoom() {
                return 1;
              },
            };

            const state = createInitialState();
            const selectionCalls = [];
            const selectionSyncCalls = [];
            let renderedCyFactoryCalls = 0;
            const documentElements = {};
            const ctx = {
              state,
              constants: {
                TENSOR_WIDTH: 140,
                TENSOR_HEIGHT: 84,
                MIN_TENSOR_WIDTH: 96,
                MIN_TENSOR_HEIGHT: 60,
                INDEX_RADIUS: 10,
                INDEX_PADDING: 6,
                NOTE_WIDTH: 220,
                NOTE_HEIGHT: 120,
                NOTE_MIN_WIDTH: 120,
                NOTE_MIN_HEIGHT: 90,
                NOTE_COLLAPSED_SIZE: 40,
                HISTORY_LIMIT: 100,
                REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
                DEFAULT_INDEX_SLOTS: [
                  { x: -38, y: 0 },
                  { x: 38, y: 0 },
                  { x: 0, y: -24 },
                  { x: 0, y: 24 },
                ],
              },
              dom: {
                canvasShell: {
                  getBoundingClientRect() {
                    return { left: 0, top: 0, width: 1000, height: 800 };
                  },
                },
                statusMessage: { textContent: "", classList: createClassList() },
                propertiesPanel: {},
                engineSelect: { value: "tensornetwork", options: [] },
                connectButton: {},
                loadInput: {},
                undoButton: {},
                redoButton: {},
                templateSelect: { value: "" },
                insertTemplateButton: {},
                createGroupButton: {},
                helpButton: {},
                helpModal: { classList: createClassList() },
                helpBackdrop: {},
                helpCloseButton: { focus() {} },
                groupLayer: {},
                resizeLayer: {},
                selectionBox: { classList: createClassList(), style: {} },
                minimapCanvas: {},
                addNoteButton: {},
                notesLayer: {},
              },
              apiGet: async () => null,
              apiPost: async () => null,
              window: {
                structuredClone: globalThis.structuredClone,
                crypto: globalThis.crypto,
                setTimeout,
                clearTimeout,
                requestAnimationFrame(callback) {
                  callback();
                  return 1;
                },
                cancelAnimationFrame() {},
              },
              document: {
                getElementById(id) {
                  return documentElements[id] || null;
                },
                createElement() {
                  return {
                    className: "",
                    dataset: {},
                    style: {},
                    classList: createClassList(),
                    appendChild() {},
                    addEventListener() {},
                    setAttribute() {},
                    removeAttribute() {},
                    innerHTML: "",
                  };
                },
              },
              cytoscape() {
                renderedCyFactoryCalls += 1;
                return fakeCy;
              },
              setStatus() {},
              toggleSidebarCollapsed() {},
              setActiveSidebarTab() {},
              closeCanvasContextMenu() {},
              isInspectingPastStage() {
                return false;
              },
              handlePlannerOperandClick() {},
              handleConnectClick() {},
              bringTensorToFront() {},
              renderOverlayDecorations() {},
              renderMinimap() {},
              renderPlanner() {},
              renderSidebarTabs() {},
              renderNotes() {},
              updateToolbarState() {},
              renderProperties() {},
              syncCySelection() {
                selectionSyncCalls.push([...state.selectionIds]);
              },
              isContractionSceneVisible() {
                return false;
              },
              canEditCurrentContractionStage() {
                return false;
              },
              updateCurrentStageOperandLayout() {},
              createHistorySnapshot() {
                return { id: "snapshot_1" };
              },
              clientPointToWorldPoint(clientX, clientY) {
                return { x: clientX, y: clientY };
              },
              getMetadataColor(metadata, fallbackColor) {
                return metadata && metadata.color ? metadata.color : fallbackColor;
              },
              shiftColor(color) {
                return color;
              },
              readableTextColor() {
                return "#111111";
              },
              tensorLayerRank() {
                return 0;
              },
              getIndexColor() {
                return "#456cbf";
              },
              indexAbsolutePosition(tensor, index) {
                return {
                  x: tensor.position.x + index.offset.x,
                  y: tensor.position.y + index.offset.y,
                };
              },
              indexLabelNodeId(indexId) {
                return `${indexId}__label`;
              },
              indexLabelPosition(position) {
                return position;
              },
              getMetadataFilterEntityState() {
                return "";
              },
              getMetadataFilterHighlight() {
                return null;
              },
              findEdgeByIndexId() {
                return null;
              },
              tensorWidth(tensor) {
                return tensor?.size?.width ?? 140;
              },
              tensorHeight(tensor) {
                return tensor?.size?.height ?? 84;
              },
              noteCanvasBounds(note) {
                return {
                  x1: note.position.x,
                  y1: note.position.y,
                  x2: note.position.x + note.size.width,
                  y2: note.position.y + note.size.height,
                  width: note.size.width,
                  height: note.size.height,
                };
              },
              setSelection(selectionIds, options = {}) {
                state.selectionIds = [...selectionIds];
                state.primarySelectionId =
                  options.primaryId || selectionIds[selectionIds.length - 1] || null;
                selectionCalls.push([...selectionIds]);
              },
              selectElement(kind, id, options = {}) {
                if (options.additive) {
                  if (state.selectionIds.includes(id)) {
                    ctx.setSelection(
                      state.selectionIds.filter((selectionId) => selectionId !== id),
                      {
                        primaryId:
                          state.primarySelectionId === id
                            ? state.selectionIds[state.selectionIds.length - 2] || null
                            : state.primarySelectionId,
                      }
                    );
                    return;
                  }
                  ctx.setSelection([...state.selectionIds, id], { primaryId: id });
                  return;
                }
                ctx.setSelection([id], { primaryId: id });
              },
              getSelectedEntries() {
                return state.selectionIds.map((selectionId) => {
                  if (selectionId === "group_ab") {
                    return { kind: "group", id: "group_ab", group: ctx.findGroupById("group_ab") };
                  }
                  if (selectionId === "note_a") {
                    return { kind: "note", id: "note_a", note: ctx.findNoteById("note_a") };
                  }
                  if (selectionId === "tensor_a" || selectionId === "tensor_b") {
                    return { kind: "tensor", id: selectionId, tensor: ctx.findTensorById(selectionId) };
                  }
                  if (selectionId === "index_a" || selectionId === "index_b") {
                    return { kind: "index", id: selectionId, located: ctx.findIndexOwner(selectionId) };
                  }
                  return null;
                }).filter(Boolean);
              },
            };

            documentElements.canvas = { id: "canvas" };
            state.spec = {
              id: "network_additive",
              name: "additive",
              tensors: [
                {
                  id: "tensor_a",
                  name: "A",
                  position: { x: 100, y: 100 },
                  size: { width: 140, height: 84 },
                  metadata: {},
                  indices: [
                    {
                      id: "index_a",
                      name: "a",
                      dimension: 2,
                      offset: { x: -38, y: 0 },
                      metadata: {},
                    },
                  ],
                },
                {
                  id: "tensor_b",
                  name: "B",
                  position: { x: 260, y: 100 },
                  size: { width: 140, height: 84 },
                  metadata: {},
                  indices: [
                    {
                      id: "index_b",
                      name: "b",
                      dimension: 2,
                      offset: { x: -38, y: 0 },
                      metadata: {},
                    },
                  ],
                },
              ],
              groups: [
                {
                  id: "group_ab",
                  name: "AB",
                  tensor_ids: ["tensor_a", "tensor_b"],
                  metadata: {},
                },
              ],
              edges: [],
              notes: [
                {
                  id: "note_a",
                  text: "note",
                  position: { x: 340, y: 280 },
                  size: { width: 220, height: 120 },
                  metadata: {},
                },
              ],
              contraction_plan: null,
              metadata: {},
            };
            ctx.findTensorById = (tensorId) =>
              state.spec.tensors.find((tensor) => tensor.id === tensorId) || null;
            ctx.findVisibleTensorById = ctx.findTensorById;
            ctx.findIndexOwner = (indexId) => {
              for (const tensor of state.spec.tensors) {
                const index = tensor.indices.find((candidate) => candidate.id === indexId);
                if (index) {
                  return { tensor, index };
                }
              }
              return null;
            };
            ctx.findEdgeById = () => null;
            ctx.findNoteById = (noteId) =>
              state.spec.notes.find((note) => note.id === noteId) || null;
            ctx.findGroupById = (groupId) =>
              state.spec.groups.find((group) => group.id === groupId) || null;
            ctx.buildCanvasSelectionDragState = (anchorId) => ({
              snapshot: { id: `snapshot_${anchorId}` },
              tensorIds: anchorId === "group_ab" ? ["tensor_a", "tensor_b"] : [],
              noteIds: anchorId === "note_a" ? ["note_a"] : [],
              tensorStartPositions: {},
              noteStartPositions: {},
            });
            ctx.applyCanvasSelectionDragDelta = () => {};

            registerUtilities(ctx);
            registerNotesFeature(ctx);
            registerOverlaysLayoutTemplates(ctx);
            registerGraphRender(ctx);
            ctx.initGraph();

            if (renderedCyFactoryCalls !== 1) {
              throw new Error("Graph render should initialize the Cytoscape canvas once.");
            }

            state.selectionIds = ["tensor_a"];
            state.primarySelectionId = "tensor_a";
            cyHandlers["grab:node[kind = 'tensor']"]({
              target: {
                id() {
                  return "tensor_b";
                },
              },
              originalEvent: {
                button: 0,
                ctrlKey: true,
              },
            });
            if (typeof cyHandlers["free:node[kind = 'tensor']"] !== "function") {
              throw new Error("Tensor nodes should register a free handler to finish additive clicks without dragging.");
            }
            cyHandlers["tap:node, edge"]({
              target: {
                id() {
                  return "tensor_b";
                },
                data(name) {
                  return name === "kind" ? "tensor" : null;
                },
              },
              originalEvent: {
                button: 0,
                ctrlKey: true,
              },
            });
            if (state.selectionIds.join(",") !== "tensor_a,tensor_b") {
              throw new Error(`Ctrl+click on a tensor should add it to the selection, received ${state.selectionIds.join(",")}.`);
            }
            cyHandlers["free:node[kind = 'tensor']"]({
              target: {
                id() {
                  return "tensor_b";
                },
              },
            });
            if (
              JSON.stringify(selectionSyncCalls.slice(-1)[0] || []) !==
              JSON.stringify(["tensor_a", "tensor_b"])
            ) {
              throw new Error(`Ctrl+click on a tensor should resync the visual selection on release, received ${JSON.stringify(selectionSyncCalls)}.`);
            }

            state.selectionIds = ["index_a"];
            state.primarySelectionId = "index_a";
            cyHandlers["grab:node[kind = 'index']"]({
              target: {
                id() {
                  return "index_b";
                },
              },
              originalEvent: {
                button: 0,
                ctrlKey: true,
              },
            });
            if (state.selectionIds.join(",") !== "index_a,index_b") {
              throw new Error(`Ctrl+grab on an index should add it to the selection even before dragging, received ${state.selectionIds.join(",")}.`);
            }
            if (typeof cyHandlers["free:node[kind = 'index']"] !== "function") {
              throw new Error("Index nodes should register a free handler to finish additive clicks without dragging.");
            }
            cyHandlers["tap:node, edge"]({
              target: {
                id() {
                  return "index_b";
                },
                data(name) {
                  return name === "kind" ? "index" : null;
                },
              },
              originalEvent: {
                button: 0,
                ctrlKey: true,
              },
            });
            if (state.selectionIds.join(",") !== "index_a,index_b") {
              throw new Error(`Ctrl+click on an index should keep additive selection instead of toggling it off, received ${state.selectionIds.join(",")}.`);
            }
            cyHandlers["free:node[kind = 'index']"]({
              target: {
                id() {
                  return "index_b";
                },
              },
            });
            if (
              JSON.stringify(selectionSyncCalls.slice(-1)[0] || []) !==
              JSON.stringify(["index_a", "index_b"])
            ) {
              throw new Error(`Ctrl+click on an index should resync the visual selection on release, received ${JSON.stringify(selectionSyncCalls)}.`);
            }

            state.selectionIds = ["tensor_a"];
            state.primarySelectionId = "tensor_a";
            ctx.startNoteDrag(
              {
                button: 0,
                ctrlKey: true,
                preventDefault() {},
                stopPropagation() {},
                clientX: 0,
                clientY: 0,
              },
              "note_a"
            );
            if (state.selectionIds.join(",") !== "tensor_a,note_a") {
              throw new Error(`Ctrl+drag on a note should preserve additive selection, received ${state.selectionIds.join(",")}.`);
            }

            state.selectionIds = ["tensor_a"];
            state.primarySelectionId = "tensor_a";
            ctx.startGroupDrag(
              {
                button: 0,
                ctrlKey: true,
                preventDefault() {},
                stopPropagation() {},
                clientX: 0,
                clientY: 0,
              },
              "group_ab"
            );
            if (state.selectionIds.join(",") !== "tensor_a,group_ab") {
              throw new Error(`Ctrl+drag on a group should preserve additive selection, received ${state.selectionIds.join(",")}.`);
            }

            state.selectionIds = ["tensor_a"];
            state.primarySelectionId = "tensor_a";
            cyHandlers["grab:node[kind = 'tensor']"]({
              target: {
                id() {
                  return "tensor_b";
                },
              },
              originalEvent: {
                button: 0,
                metaKey: true,
              },
            });
            cyHandlers["tap:node, edge"]({
              target: {
                id() {
                  return "tensor_b";
                },
                data(name) {
                  return name === "kind" ? "tensor" : null;
                },
              },
              originalEvent: {
                button: 0,
                metaKey: true,
              },
            });
            if (state.selectionIds.join(",") !== "tensor_a,tensor_b") {
              throw new Error("Cmd+click should behave like additive selection too.");
            }
            """
        ),
        encoding="utf-8",
    )
    return script_path


def _write_metadata_properties_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "metadata_properties_runtime_regression.mjs"
    state_runtime_path = tmp_path / "state.runtime.mjs"
    utilities_runtime_path = tmp_path / "utilities.runtime.mjs"
    history_runtime_path = tmp_path / "historySelection.runtime.mjs"
    properties_runtime_path = tmp_path / "properties.runtime.mjs"
    _copy_runtime_bundle(
        tmp_path,
        {
            "state.runtime.mjs": "state/state.js",
            "utilities.runtime.mjs": "utils/utilities.js",
            "historySelection.runtime.mjs": "graph/historySelection.js",
            "properties.runtime.mjs": "properties/properties.js",
            "propertiesSupport.js": "properties/propertiesSupport.js",
            "propertiesRenderers.js": "properties/propertiesRenderers.js",
        },
        _RUNTIME_EDITOR_SUPPORT_MODULES,
    )
    script_body = textwrap.dedent(
        """
        import { pathToFileURL } from "node:url";

        function createClassList() {
          return {
            add() {},
            remove() {},
            toggle() {},
          };
        }

        function toDatasetKey(attributeName) {
          return String(attributeName || "")
            .replace(/^data-/, "")
            .replace(/-([a-z])/g, (_, character) => character.toUpperCase());
        }

        function createFakeElement(id = null, tagName = "div", initialAttributes = {}) {
          const attributes = { ...initialAttributes };
          const dataset = {};
          Object.entries(attributes).forEach(([name, value]) => {
            if (name.startsWith("data-")) {
              dataset[toDatasetKey(name)] = String(value);
            }
          });
          return {
            id,
            tagName: String(tagName || "div").toUpperCase(),
            value: "",
            textContent: "",
            dataset,
            attributes,
            checked: false,
            disabled: false,
            classList: createClassList(),
            style: {},
            listeners: {},
            ownerDocument: null,
            addEventListener(eventName, listener) {
              if (!this.listeners[eventName]) {
                this.listeners[eventName] = [];
              }
              this.listeners[eventName].push(listener);
            },
            dispatchEvent(eventName, event = {}) {
              (this.listeners[eventName] || []).forEach((listener) => {
                listener({
                  preventDefault() {},
                  target: this,
                  ...event,
                });
              });
            },
            click() {
              this.dispatchEvent("click");
            },
            focus() {},
            getAttribute(name) {
              return Object.prototype.hasOwnProperty.call(this.attributes, name)
                ? this.attributes[name]
                : null;
            },
            setAttribute(name, value) {
              this.attributes[name] = String(value);
              if (name === "id") {
                this.id = String(value);
              }
              if (name.startsWith("data-")) {
                this.dataset[toDatasetKey(name)] = String(value);
              }
            },
            removeAttribute(name) {
              delete this.attributes[name];
              if (name.startsWith("data-")) {
                delete this.dataset[toDatasetKey(name)];
              }
            },
            appendChild() {},
          };
        }

        function createFakeDocument() {
          const elements = new Map();
          const toggleElements = [];
          return {
            activeElement: null,
            toggleElements,
            registerHtml(html) {
              elements.clear();
              toggleElements.length = 0;

              const tagPattern =
                /<(div|input|textarea|button|select)([^>]*)id="([^"]+)"([^>]*)>/g;
              let tagMatch = tagPattern.exec(html);
              while (tagMatch) {
                const attributes = {};
                const attributeSource = `${tagMatch[2]} id="${tagMatch[3]}"${tagMatch[4]}`;
                attributeSource.replace(
                  /([a-zA-Z_:][-a-zA-Z0-9_:.]*)="([^"]*)"/g,
                  (_, name, value) => {
                    attributes[name] = value;
                    return "";
                  }
                );
                elements.set(
                  tagMatch[3],
                  createFakeElement(tagMatch[3], tagMatch[1], attributes)
                );
                tagMatch = tagPattern.exec(html);
              }

              const togglePattern = /<button[\\s\\S]*?data-index-toggle="([^"]+)"[\\s\\S]*?>/g;
              let toggleMatch = togglePattern.exec(html);
              while (toggleMatch) {
                const element = createFakeElement(null, "button");
                element.dataset.indexToggle = toggleMatch[1];
                toggleElements.push(element);
                toggleMatch = togglePattern.exec(html);
              }
            },
            getElementById(id) {
              return elements.get(id) || null;
            },
            createElement(tagName) {
              return createFakeElement(null, tagName);
            },
            querySelectorAll() {
              return [];
            },
          };
        }

        function createPropertiesPanel(document) {
          let html = "";
          return {
            get innerHTML() {
              return html;
            },
            set innerHTML(value) {
              html = value;
              document.registerHtml(value);
            },
            querySelectorAll(selector) {
              if (selector === "[data-index-toggle]") {
                return document.toggleElements;
              }
              return [];
            },
          };
        }

        function createButton() {
          return createFakeElement(null, "button");
        }

        function createSpec() {
          return {
            id: "network_metadata_regression",
            name: "metadata regression",
            tensors: [
              {
                id: "tensor_a",
                name: "Tensor A",
                position: { x: 120, y: 120 },
                size: { width: 140, height: 84 },
                metadata: {
                  color: "#123456",
                  collapsed: true,
                  tags: ["seed"],
                  source: "sim",
                },
                indices: [
                  {
                    id: "index_a",
                    name: "left",
                    dimension: 2,
                    offset: { x: -38, y: 0 },
                    metadata: {},
                  },
                  {
                    id: "index_b",
                    name: "right",
                    dimension: 3,
                    offset: { x: 38, y: 0 },
                    metadata: {},
                  },
                ],
              },
              {
                id: "tensor_b",
                name: "Tensor B",
                position: { x: 320, y: 140 },
                size: { width: 140, height: 84 },
                metadata: {
                  color: "#654321",
                  tags: ["paired"],
                },
                indices: [
                  {
                    id: "index_c",
                    name: "bond",
                    dimension: 3,
                    offset: { x: -38, y: 0 },
                    metadata: {},
                  },
                ],
              },
            ],
            edges: [
              {
                id: "edge_ab",
                name: "bond_ab",
                left: { tensor_id: "tensor_a", index_id: "index_b" },
                right: { tensor_id: "tensor_b", index_id: "index_c" },
                metadata: {},
              },
            ],
            groups: [
              {
                id: "group_a",
                name: "Group A",
                tensor_ids: ["tensor_a"],
                metadata: {},
              },
            ],
            notes: [
              {
                id: "note_a",
                text: "check metadata",
                position: { x: 40, y: 40 },
                size: { width: 220, height: 120 },
                metadata: {},
              },
            ],
            contraction_plan: null,
            metadata: {},
          };
        }

        function commitField(element, nextValue) {
          if (!element) {
            throw new Error("Missing editable field.");
          }
          element.value = nextValue;
          element.dispatchEvent("input");
          element.dispatchEvent("blur");
        }

        function assertLastRenderDidNotInvalidateGraph(renderCalls, graphCount, minimapCount) {
          const lastRender = renderCalls[renderCalls.length - 1];
          if (!lastRender) {
            throw new Error("Expected a render call after committing metadata.");
          }
          if (lastRender.graph || lastRender.minimap) {
            throw new Error(
              `Expected metadata edits to avoid graph/minimap invalidation, received ${JSON.stringify(lastRender)}.`
            );
          }
          if (graphCount() !== 0 || minimapCount() !== 0) {
            throw new Error(
              `Expected no graph/minimap render work, received graph=${graphCount()} minimap=${minimapCount()}.`
            );
          }
        }

        const [stateModule, utilitiesModule, historyModule, propertiesModule] =
          await Promise.all([
            import(pathToFileURL(__STATE_PATH__).href),
            import(pathToFileURL(__UTILITIES_PATH__).href),
            import(pathToFileURL(__HISTORY_PATH__).href),
            import(pathToFileURL(__PROPERTIES_PATH__).href),
          ]);
        const { createInitialState } = stateModule;
        const { registerUtilities } = utilitiesModule;
        const { registerHistorySelection } = historyModule;
        const { registerProperties } = propertiesModule;

        const document = createFakeDocument();
        const propertiesPanel = createPropertiesPanel(document);
        const renderCalls = [];
        let graphRenderCount = 0;
        let minimapRenderCount = 0;
        const ctx = {
          state: createInitialState(),
          constants: {
            TENSOR_WIDTH: 140,
            TENSOR_HEIGHT: 84,
            MIN_TENSOR_WIDTH: 96,
            MIN_TENSOR_HEIGHT: 60,
            INDEX_RADIUS: 10,
            INDEX_PADDING: 6,
            NOTE_WIDTH: 220,
            NOTE_HEIGHT: 120,
            NOTE_MIN_WIDTH: 120,
            NOTE_MIN_HEIGHT: 90,
            HISTORY_LIMIT: 100,
            REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
            DEFAULT_INDEX_SLOTS: [
              { x: -38, y: 0 },
              { x: 38, y: 0 },
              { x: 0, y: -24 },
              { x: 0, y: 24 },
            ],
          },
          dom: {
            workspace: {},
            statusMessage: { textContent: "", classList: createClassList() },
            propertiesPanel,
            generatedCode: { value: "" },
            engineSelect: { options: [], value: "tensornetwork" },
            collectionFormatSelect: { options: [], value: "list" },
            exportFormatSelect: { value: "py" },
            addNoteButton: createButton(),
            connectButton: createButton(),
            loadInput: {},
            undoButton: createButton(),
            redoButton: createButton(),
            exportButton: createButton(),
            toggleLinearPeriodicButton: createButton(),
            linearPeriodicPreviousCellButton: createButton(),
            linearPeriodicCellLabel: { textContent: "" },
            linearPeriodicNextCellButton: createButton(),
            templateSelect: { value: "" },
            templateParameterPanel: { hidden: true },
            templateGraphSizeLabel: { textContent: "" },
            templateGraphSizeInput: { value: "2", min: "1" },
            templateBondDimensionInput: { value: "3", min: "1" },
            templatePhysicalDimensionInput: { value: "2", min: "1" },
            insertTemplateButton: createButton(),
            createGroupButton: createButton(),
            helpButton: createButton(),
            helpModal: { classList: createClassList() },
            helpBackdrop: createButton(),
            helpCloseButton: createButton(),
            canvasShell: {
              getBoundingClientRect() {
                return { left: 0, top: 0, width: 1000, height: 800 };
              },
            },
            groupLayer: {},
            resizeLayer: {},
            notesLayer: {},
            selectionBox: {},
            minimapCanvas: {},
            sidebar: {},
            plannerPanel: {},
            generateButton: createButton(),
          },
          apiGet: async () => null,
          apiPost: async () => null,
          window: {
            structuredClone: globalThis.structuredClone,
            crypto: globalThis.crypto,
            setTimeout,
            clearTimeout,
            confirm: () => true,
          },
          document,
          cytoscape: null,
          tensorWidth: (tensor) => tensor?.size?.width ?? 140,
          tensorHeight: (tensor) => tensor?.size?.height ?? 84,
          renderOverlayDecorations: () => {},
          renderMinimap: () => {
            minimapRenderCount += 1;
          },
          renderPlanner: () => {},
          renderSidebarTabs: () => {},
          refreshContractionAnalysis: () => {},
          repairContractionPlan: () => {},
          updateToolbarState: () => {},
        };

        registerUtilities(ctx);
        registerHistorySelection(ctx);
        registerProperties(ctx);
        ctx.findNoteById = (noteId) =>
          (Array.isArray(ctx.state.spec?.notes)
            ? ctx.state.spec.notes.find((note) => note.id === noteId)
            : null) || null;

        ctx.captureEditableFocus = () => null;
        ctx.restoreEditableFocus = () => {};
        ctx.render = (options = {}) => {
          const resolvedOptions = {
            graph: true,
            properties: true,
            code: true,
            toolbar: true,
            overlays: true,
            planner: true,
            sidebarTabs: true,
            minimap: true,
            syncSelection: false,
            ...options,
          };
          renderCalls.push(resolvedOptions);
          if (resolvedOptions.graph) {
            graphRenderCount += 1;
          }
          if (resolvedOptions.properties) {
            ctx.renderProperties();
          }
          if (resolvedOptions.minimap) {
            ctx.renderMinimap();
          }
        };

        ctx.state.selectedEngine = "tensornetwork";
        ctx.state.selectedCollectionFormat = "list";
        ctx.state.annotationDefinitions = {
          tensor: [
            {
              key: "role",
              label: "Tensor role",
              placeholder: "observable",
              suggestions: ["state", "operator", "observable"],
            },
            {
              key: "state",
              label: "State",
              placeholder: "ground",
              suggestions: ["ground", "excited"],
            },
          ],
          index: [
            {
              key: "leg_kind",
              label: "Leg kind",
              placeholder: "physical",
              suggestions: ["physical", "logical"],
            },
          ],
        };
        ctx.state.spec = ctx.normalizeSpec(createSpec());
        ctx.renderProperties();

        renderCalls.length = 0;
        graphRenderCount = 0;
        minimapRenderCount = 0;
        commitField(document.getElementById("network-tags-input"), "physical, observable, physical");
        if (JSON.stringify(ctx.state.spec.metadata.tags) !== JSON.stringify(["physical", "observable"])) {
          throw new Error(`Expected normalised network tags, received ${JSON.stringify(ctx.state.spec.metadata.tags)}.`);
        }
        assertLastRenderDidNotInvalidateGraph(
          renderCalls,
          () => graphRenderCount,
          () => minimapRenderCount
        );

        ctx.setSelection(["tensor_a"], { primaryId: "tensor_a" });
        if (!propertiesPanel.innerHTML.includes(">Metadata</summary>")) {
          throw new Error("Selecting a tensor should render the metadata disclosure.");
        }
        if (propertiesPanel.innerHTML.includes("<details open")) {
          throw new Error("Tensor metadata disclosures should be collapsed by default.");
        }
        if (propertiesPanel.innerHTML.includes("Suggested annotations")) {
          throw new Error("Tensor metadata should no longer render guided annotation fields.");
        }
        if (!propertiesPanel.innerHTML.includes('rows="1"')) {
          throw new Error("Custom metadata should start with a single visible row.");
        }
        if (propertiesPanel.innerHTML.includes("Tensor values")) {
          throw new Error("Tensor values should no longer render as a separate disclosure.");
        }
        if (propertiesPanel.innerHTML.includes('id="tensor-values-disclosure"')) {
          throw new Error("Tensor values should no longer use a disclosure container.");
        }
        if (!propertiesPanel.innerHTML.includes(">Initialization<")) {
          throw new Error("Selecting a tensor should expose the inline Initialization controls.");
        }
        if (propertiesPanel.innerHTML.includes("Current initializer:")) {
          throw new Error("Tensor values should no longer render the current initializer helper text.");
        }
        if (
          propertiesPanel.innerHTML.includes(
            "Use JSON numbers that match the tensor shape exactly."
          )
        ) {
          throw new Error("Tensor values should no longer render the redundant JSON helper text.");
        }
        renderCalls.length = 0;
        graphRenderCount = 0;
        minimapRenderCount = 0;
        const tensorDataModeSelect = document.getElementById("tensor-data-mode-select");
        if (!tensorDataModeSelect) {
          throw new Error("Selecting a tensor should expose the tensor data mode selector.");
        }
        const tensorDataModeField = document.getElementById("tensor-data-mode-field");
        if (!tensorDataModeField) {
          throw new Error("Selecting a tensor should expose the initialization chevron field.");
        }
        tensorDataModeSelect.dispatchEvent("mousedown");
        if (tensorDataModeField.getAttribute("data-expanded") !== "true") {
          throw new Error(
            `Expected initialization select mouse down to expand the chevron, received ${tensorDataModeField.getAttribute("data-expanded")}.`
          );
        }
        tensorDataModeSelect.value = "fill";
        tensorDataModeSelect.dispatchEvent("change");
        if (tensorDataModeField.getAttribute("data-expanded") !== "false") {
          throw new Error(
            `Expected initialization select change to collapse the chevron, received ${tensorDataModeField.getAttribute("data-expanded")}.`
          );
        }
        if (
          JSON.stringify(ctx.state.spec.tensors[0].tensor_data)
          !== JSON.stringify({ mode: "fill", fill_value: 0 })
        ) {
          throw new Error(
            `Expected fill mode to initialize tensor data, received ${JSON.stringify(ctx.state.spec.tensors[0].tensor_data)}.`
          );
        }
        assertLastRenderDidNotInvalidateGraph(
          renderCalls,
          () => graphRenderCount,
          () => minimapRenderCount
        );

        renderCalls.length = 0;
        graphRenderCount = 0;
        minimapRenderCount = 0;
        commitField(document.getElementById("tensor-data-fill-input"), "3.5");
        if (
          JSON.stringify(ctx.state.spec.tensors[0].tensor_data)
          !== JSON.stringify({ mode: "fill", fill_value: 3.5 })
        ) {
          throw new Error(
            `Expected fill value edits to update tensor data, received ${JSON.stringify(ctx.state.spec.tensors[0].tensor_data)}.`
          );
        }
        assertLastRenderDidNotInvalidateGraph(
          renderCalls,
          () => graphRenderCount,
          () => minimapRenderCount
        );

        renderCalls.length = 0;
        commitField(document.getElementById("tensor-data-fill-input"), "not-a-number");
        if (
          JSON.stringify(ctx.state.spec.tensors[0].tensor_data)
          !== JSON.stringify({ mode: "fill", fill_value: 3.5 })
        ) {
          throw new Error("Invalid fill edits should not mutate tensor data.");
        }
        if (renderCalls.length !== 0) {
          throw new Error(
            `Invalid fill edits should not trigger a rerender, received ${JSON.stringify(renderCalls)}.`
          );
        }

        renderCalls.length = 0;
        graphRenderCount = 0;
        minimapRenderCount = 0;
        const literalModeSelect = document.getElementById("tensor-data-mode-select");
        literalModeSelect.value = "literal";
        literalModeSelect.dispatchEvent("change");
        if (
          JSON.stringify(ctx.state.spec.tensors[0].tensor_data)
          !== JSON.stringify({
            mode: "literal",
            values: [
              [3.5, 3.5, 3.5],
              [3.5, 3.5, 3.5],
            ],
          })
        ) {
          throw new Error(
            `Expected literal mode to seed explicit values from the current fill value, received ${JSON.stringify(ctx.state.spec.tensors[0].tensor_data)}.`
          );
        }
        if (!propertiesPanel.innerHTML.includes("Expected shape: [2, 3]")) {
          throw new Error("Literal tensor values should expose the expected shape helper.");
        }
        assertLastRenderDidNotInvalidateGraph(
          renderCalls,
          () => graphRenderCount,
          () => minimapRenderCount
        );

        renderCalls.length = 0;
        graphRenderCount = 0;
        minimapRenderCount = 0;
        commitField(
          document.getElementById("tensor-data-values-input"),
          "[[1, 2, 3], [4, 5, 6]]"
        );
        if (
          JSON.stringify(ctx.state.spec.tensors[0].tensor_data)
          !== JSON.stringify({
            mode: "literal",
            values: [
              [1, 2, 3],
              [4, 5, 6],
            ],
          })
        ) {
          throw new Error(
            `Expected explicit tensor values to commit, received ${JSON.stringify(ctx.state.spec.tensors[0].tensor_data)}.`
          );
        }
        assertLastRenderDidNotInvalidateGraph(
          renderCalls,
          () => graphRenderCount,
          () => minimapRenderCount
        );

        renderCalls.length = 0;
        commitField(document.getElementById("tensor-data-values-input"), "[[1, 2], [3]]");
        if (
          JSON.stringify(ctx.state.spec.tensors[0].tensor_data)
          !== JSON.stringify({
            mode: "literal",
            values: [
              [1, 2, 3],
              [4, 5, 6],
            ],
          })
        ) {
          throw new Error("Invalid explicit tensor values should not mutate tensor data.");
        }
        if (renderCalls.length !== 0) {
          throw new Error(
            `Invalid literal edits should not trigger a rerender, received ${JSON.stringify(renderCalls)}.`
          );
        }
        renderCalls.length = 0;
        graphRenderCount = 0;
        minimapRenderCount = 0;
        const externalModeSelect = document.getElementById("tensor-data-mode-select");
        externalModeSelect.value = "external";
        externalModeSelect.dispatchEvent("change");
        if (!document.getElementById("tensor-data-external-path-input")) {
          throw new Error("External tensor data should expose a file path input.");
        }
        if (
          JSON.stringify(ctx.state.spec.tensors[0].tensor_data)
          !== JSON.stringify({ mode: "external", file_path: "" })
        ) {
          throw new Error(
            `Expected external mode to initialize tensor data, received ${JSON.stringify(ctx.state.spec.tensors[0].tensor_data)}.`
          );
        }
        commitField(
          document.getElementById("tensor-data-external-path-input"),
          "data/a.npz"
        );
        if (!document.getElementById("tensor-data-external-array-key-input")) {
          throw new Error("External .npz tensor data should expose an array key input.");
        }
        commitField(
          document.getElementById("tensor-data-external-array-key-input"),
          "left"
        );
        if (
          JSON.stringify(ctx.state.spec.tensors[0].tensor_data)
          !== JSON.stringify({
            mode: "external",
            file_path: "data/a.npz",
            array_key: "left",
          })
        ) {
          throw new Error(
            `Expected external path and key edits to update tensor data, received ${JSON.stringify(ctx.state.spec.tensors[0].tensor_data)}.`
          );
        }
        commitField(
          document.getElementById("tensor-data-external-path-input"),
          "data/a.pt"
        );
        if (!document.getElementById("tensor-data-external-array-key-input")) {
          throw new Error("External .pt tensor data should expose an optional key input.");
        }
        commitField(
          document.getElementById("tensor-data-external-array-key-input"),
          "weights"
        );
        if (
          JSON.stringify(ctx.state.spec.tensors[0].tensor_data)
          !== JSON.stringify({
            mode: "external",
            file_path: "data/a.pt",
            array_key: "weights",
          })
        ) {
          throw new Error(
            `Expected external .pt path and key edits to update tensor data, received ${JSON.stringify(ctx.state.spec.tensors[0].tensor_data)}.`
          );
        }
        assertLastRenderDidNotInvalidateGraph(
          renderCalls,
          () => graphRenderCount,
          () => minimapRenderCount
        );
        ctx.window.setTimeout = (callback) => {
          callback();
          return 1;
        };
        ctx.window.clearTimeout = () => {};
        const fluidTagsInput = document.getElementById("tensor-tags-input");
        fluidTagsInput.value = "alpha, ";
        fluidTagsInput.dispatchEvent("input");
        if (fluidTagsInput.value !== "alpha, ") {
          throw new Error("Typing a comma should not strip the active tag separator.");
        }
        if (JSON.stringify(ctx.state.spec.tensors[0].metadata.tags) !== JSON.stringify(["seed"])) {
          throw new Error("Typing in the tags field should not commit metadata until the field is confirmed.");
        }
        renderCalls.length = 0;
        graphRenderCount = 0;
        minimapRenderCount = 0;
        commitField(document.getElementById("tensor-tags-input"), "alpha, beta, alpha");
        const tensorMetadataAfterTags = ctx.state.spec.tensors[0].metadata;
        if (JSON.stringify(tensorMetadataAfterTags.tags) !== JSON.stringify(["alpha", "beta"])) {
          throw new Error(`Expected tensor tags to be updated, received ${JSON.stringify(tensorMetadataAfterTags.tags)}.`);
        }
        if (tensorMetadataAfterTags.color !== "#123456" || tensorMetadataAfterTags.collapsed !== true) {
          throw new Error("Reserved tensor metadata keys were not preserved after editing tags.");
        }
        if (tensorMetadataAfterTags.source !== "sim") {
          throw new Error("Custom tensor metadata should survive a tags-only edit.");
        }
        assertLastRenderDidNotInvalidateGraph(
          renderCalls,
          () => graphRenderCount,
          () => minimapRenderCount
        );

        renderCalls.length = 0;
        graphRenderCount = 0;
        minimapRenderCount = 0;
        commitField(
          document.getElementById("tensor-custom-metadata-input"),
          '{"source":"imported","role":"observable","color":"#ffffff","tags":["ignored"]}'
        );
        const tensorMetadataAfterCustomEdit = ctx.state.spec.tensors[0].metadata;
        if (tensorMetadataAfterCustomEdit.color !== "#123456") {
          throw new Error("The advanced metadata editor should preserve the existing reserved color.");
        }
        if (tensorMetadataAfterCustomEdit.collapsed !== true) {
          throw new Error("The advanced metadata editor should preserve the reserved collapsed flag.");
        }
        if (JSON.stringify(tensorMetadataAfterCustomEdit.tags) !== JSON.stringify(["alpha", "beta"])) {
          throw new Error("The advanced metadata editor should preserve tags edited in the dedicated field.");
        }
        if (tensorMetadataAfterCustomEdit.role !== "observable") {
          throw new Error("Custom metadata should now keep former guided keys editable in JSON.");
        }
        if (tensorMetadataAfterCustomEdit.source !== "imported") {
          throw new Error("The advanced metadata editor did not apply the custom metadata payload.");
        }
        if (!document.getElementById("tensor-custom-metadata-input").value.includes('"role": "observable"')) {
          throw new Error("The custom metadata editor should surface former guided keys in the JSON payload.");
        }
        assertLastRenderDidNotInvalidateGraph(
          renderCalls,
          () => graphRenderCount,
          () => minimapRenderCount
        );

        ctx.performUndo();
        const tensorMetadataAfterUndo = ctx.state.spec.tensors[0].metadata;
        if (tensorMetadataAfterUndo.source !== "sim" || Object.prototype.hasOwnProperty.call(tensorMetadataAfterUndo, "role")) {
          throw new Error(`Undo should restore the previous tensor custom metadata, received ${JSON.stringify(tensorMetadataAfterUndo)}.`);
        }
        ctx.performRedo();
        const tensorMetadataAfterRedo = ctx.state.spec.tensors[0].metadata;
        if (tensorMetadataAfterRedo.role !== "observable" || tensorMetadataAfterRedo.source !== "imported") {
          throw new Error(`Redo should restore the advanced metadata edit, received ${JSON.stringify(tensorMetadataAfterRedo)}.`);
        }

        ctx.setSelection(["index_a"], { primaryId: "index_a" });
        if (!propertiesPanel.innerHTML.includes(">Metadata</summary>")) {
          throw new Error("Selecting an index should keep metadata inside a disclosure.");
        }
        if (propertiesPanel.innerHTML.includes("Suggested annotations")) {
          throw new Error("Index metadata should no longer render guided annotation fields.");
        }
        const indexTagsInput = document.getElementById("index-tags-input-index_a");
        if (!indexTagsInput) {
          throw new Error("Selecting an index should expose the tags field in the properties sidebar.");
        }
        renderCalls.length = 0;
        graphRenderCount = 0;
        minimapRenderCount = 0;
        commitField(indexTagsInput, "left-leg, physical");
        const indexMetadata = ctx.state.spec.tensors[0].indices[0].metadata;
        if (JSON.stringify(indexMetadata.tags) !== JSON.stringify(["left-leg", "physical"])) {
          throw new Error(`Expected index tags to update, received ${JSON.stringify(indexMetadata.tags)}.`);
        }
        assertLastRenderDidNotInvalidateGraph(
          renderCalls,
          () => graphRenderCount,
          () => minimapRenderCount
        );

        ctx.setSelection(["edge_ab"], { primaryId: "edge_ab" });
        if (!propertiesPanel.innerHTML.includes(">Metadata</summary>")) {
          throw new Error("Selecting a connection should render the metadata disclosure.");
        }
        ctx.setSelection(["tensor_a", "tensor_b", "index_a", "edge_ab"], { primaryId: "tensor_b" });
        const normalizedSelectionMarkup = propertiesPanel.innerHTML.replace(/\\s+/g, " ");
        if (!propertiesPanel.innerHTML.includes('id="add-index-to-selection-button"')) {
          throw new Error("A mixed selection with multiple tensors should still expose Add index.");
        }
        if (!/id="add-index-to-selection-button"[^>]*>\\s*Add index\\s*<\\/button>/.test(normalizedSelectionMarkup)) {
          throw new Error("A mixed selection with multiple tensors should label the bulk index action as Add index.");
        }
        if (!propertiesPanel.innerHTML.includes('id="extract-selection-button"')) {
          throw new Error("A mixed selection with multiple tensors should still expose Extract.");
        }
        if (!/id="extract-selection-button"[^>]*>\\s*Extract\\s*<\\/button>/.test(normalizedSelectionMarkup)) {
          throw new Error("A mixed selection with multiple tensors should label extraction as Extract.");
        }
        if (!propertiesPanel.innerHTML.includes('id="promote-selection-template-button"')) {
          throw new Error("A mixed selection with multiple tensors should still expose To Template.");
        }
        if (!/id="promote-selection-template-button"[^>]*>\\s*To Template\\s*<\\/button>/.test(normalizedSelectionMarkup)) {
          throw new Error("A mixed selection with multiple tensors should label template promotion as To Template.");
        }
        if (!propertiesPanel.innerHTML.includes('id="group-selection-button"')) {
          throw new Error("A mixed selection with multiple tensors should still expose Group.");
        }
        if (propertiesPanel.innerHTML.includes('id="align-selection-left-button"')) {
          throw new Error("The Selection panel should no longer render tensor reflow controls.");
        }

        ctx.state.spec.tensors.push({
          id: "tensor_boundary",
          name: "Next cell",
          linear_periodic_role: "next",
          position: { x: 420, y: 180 },
          size: { width: 140, height: 84 },
          metadata: {},
          indices: [
            {
              id: "index_boundary",
              name: "slot_1",
              dimension: 7,
              metadata: {},
            },
          ],
        });
        const editableTensorBefore = ctx.state.spec.tensors.find(
          (candidate) => candidate.id === "tensor_a"
        ).indices.length;
        ctx.setSelection(["tensor_boundary"], { primaryId: "tensor_boundary" });
        if (propertiesPanel.innerHTML.includes('id="add-index-button"')) {
          throw new Error("Boundary tensors should not expose the add-index action in properties.");
        }
        if (propertiesPanel.innerHTML.includes('id="index-dimension-input-index_boundary"')) {
          throw new Error("Boundary tensor ports should not expose a dimension editor.");
        }
        if (propertiesPanel.innerHTML.includes('id="move-index-up-button-index_boundary"')) {
          throw new Error("Boundary tensor ports should not expose reordering controls.");
        }
        if (propertiesPanel.innerHTML.includes('id="delete-index-button-index_boundary"')) {
          throw new Error("Boundary tensor ports should not expose deletion controls.");
        }
        if (!propertiesPanel.innerHTML.includes(">Virtual tensor<")) {
          throw new Error("Boundary tensors should render the read-only virtual summary.");
        }

        ctx.setSelection(["tensor_a", "tensor_boundary"], { primaryId: "tensor_a" });
        if (!propertiesPanel.innerHTML.includes('id="add-index-to-selection-button"')) {
          throw new Error("Mixed selections should keep the bulk Add index action when editable tensors remain.");
        }
        document.getElementById("add-index-to-selection-button").click();
        const editableTensorAfter = ctx.state.spec.tensors.find(
          (candidate) => candidate.id === "tensor_a"
        ).indices.length;
        const boundaryTensorAfter = ctx.state.spec.tensors.find(
          (candidate) => candidate.id === "tensor_boundary"
        ).indices.length;
        if (editableTensorAfter !== editableTensorBefore + 1) {
          throw new Error(
            `Expected bulk Add index to keep working for editable tensors, received ${editableTensorAfter} indices.`
          );
        }
        if (boundaryTensorAfter !== 1) {
          throw new Error(
            `Expected bulk Add index to ignore boundary tensors, received ${boundaryTensorAfter} boundary indices.`
          );
        }

        ctx.renderNoteProperties("note_a");
        if (!propertiesPanel.innerHTML.includes(">Metadata</summary>")) {
          throw new Error("Selecting a note should keep metadata inside a disclosure.");
        }
        const originalNoteText = ctx.state.spec.notes[0].text;
        const noteTextInput = document.getElementById("note-text-input");
        if (!noteTextInput) {
          throw new Error("Selecting a note should expose the note text editor.");
        }
        ctx.window.setTimeout = (callback) => {
          callback();
          return 1;
        };
        ctx.window.clearTimeout = () => {};
        noteTextInput.value = "Draft note text";
        noteTextInput.dispatchEvent("input");
        if (ctx.state.spec.notes[0].text !== originalNoteText) {
          throw new Error("Typing in a note should not autosave until leaving the field.");
        }
        noteTextInput.dispatchEvent("blur");
        if (ctx.state.spec.notes[0].text !== "Draft note text") {
          throw new Error("Leaving the note field should commit the note text.");
        }
        """
    )
    script_body = script_body.replace(
        "__STATE_PATH__", json.dumps(str(state_runtime_path))
    )
    script_body = script_body.replace(
        "__UTILITIES_PATH__", json.dumps(str(utilities_runtime_path))
    )
    script_body = script_body.replace(
        "__HISTORY_PATH__", json.dumps(str(history_runtime_path))
    )
    script_body = script_body.replace(
        "__PROPERTIES_PATH__", json.dumps(str(properties_runtime_path))
    )
    script_path.write_text(script_body, encoding="utf-8")
    return script_path


def _write_metadata_filter_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "metadata_filter_runtime_regression.mjs"
    state_runtime_path = tmp_path / "state.runtime.mjs"
    utilities_runtime_path = tmp_path / "utilities.runtime.mjs"
    metadata_filters_runtime_path = tmp_path / "metadataFilters.runtime.mjs"
    _copy_runtime_bundle(
        tmp_path,
        {
            "metadataFilters.runtime.mjs": _js_source_name("metadataFilters.js"),
            "state.runtime.mjs": _js_source_name("state.js"),
            "utilities.runtime.mjs": _js_source_name("utilities.js"),
        },
        _RUNTIME_EDITOR_SUPPORT_MODULES,
    )
    script_body = textwrap.dedent(
        """
        import { pathToFileURL } from "node:url";

        function createClassList() {
          return {
            add() {},
            remove() {},
            toggle() {},
          };
        }

        function createFakeElement(id = null, tagName = "div") {
          return {
            id,
            tagName: String(tagName || "div").toUpperCase(),
            value: "",
            textContent: "",
            dataset: {},
            disabled: false,
            classList: createClassList(),
            style: {},
            listeners: {},
            addEventListener(eventName, listener) {
              if (!this.listeners[eventName]) {
                this.listeners[eventName] = [];
              }
              this.listeners[eventName].push(listener);
            },
            dispatchEvent(eventName, event = {}) {
              (this.listeners[eventName] || []).forEach((listener) => {
                listener({
                  preventDefault() {},
                  target: this,
                  ...event,
                });
              });
            },
            click() {
              this.dispatchEvent("click");
            },
            focus() {
              if (this.ownerDocument) {
                this.ownerDocument.activeElement = this;
              }
            },
            setAttribute() {},
            removeAttribute() {},
            appendChild() {},
          };
        }

        function createFakeDocument() {
          const elements = new Map();
          return {
            activeElement: null,
            registerHtml(html) {
              elements.clear();
              const tagPattern = /<(input|select|button)[^>]*id="([^"]+)"[^>]*>/g;
              let tagMatch = tagPattern.exec(html);
              while (tagMatch) {
                const element = createFakeElement(tagMatch[2], tagMatch[1]);
                element.ownerDocument = this;
                elements.set(tagMatch[2], element);
                tagMatch = tagPattern.exec(html);
              }
            },
            getElementById(id) {
              return elements.get(id) || null;
            },
            createElement(tagName) {
              const element = createFakeElement(null, tagName);
              element.ownerDocument = this;
              return element;
            },
          };
        }

        function createPanel(document) {
          let html = "";
          return {
            get innerHTML() {
              return html;
            },
            set innerHTML(value) {
              html = value;
              document.registerHtml(value);
            },
          };
        }

        function createButton() {
          return createFakeElement(null, "button");
        }

        function createSpec() {
          return {
            id: "network_filter_regression",
            name: "filter regression",
            tensors: [
              {
                id: "tensor_a",
                name: "Tensor A",
                position: { x: 120, y: 120 },
                size: { width: 140, height: 84 },
                metadata: {
                  tags: ["block"],
                  role: "state",
                },
                indices: [
                  {
                    id: "index_a",
                    name: "phys",
                    dimension: 2,
                    offset: { x: -38, y: 0 },
                    metadata: { leg_kind: "physical" },
                  },
                  {
                    id: "index_b",
                    name: "bond",
                    dimension: 3,
                    offset: { x: 38, y: 0 },
                    metadata: { leg_kind: "physical" },
                  },
                ],
              },
              {
                id: "tensor_b",
                name: "Tensor B",
                position: { x: 320, y: 120 },
                size: { width: 140, height: 84 },
                metadata: {
                  tags: ["environment"],
                  role: "operator",
                },
                indices: [
                  {
                    id: "index_c",
                    name: "bond",
                    dimension: 3,
                    offset: { x: -38, y: 0 },
                    metadata: { leg_kind: "logical" },
                  },
                ],
              },
            ],
            edges: [
              {
                id: "edge_ab",
                name: "bond_ab",
                left: { tensor_id: "tensor_a", index_id: "index_b" },
                right: { tensor_id: "tensor_b", index_id: "index_c" },
                metadata: {},
              },
            ],
            groups: [],
            notes: [],
            contraction_plan: null,
            metadata: {},
          };
        }

        function commitInput(element, nextValue) {
          if (!element) {
            throw new Error("Missing filter input.");
          }
          element.value = nextValue;
          element.dispatchEvent("input");
          element.dispatchEvent("blur");
        }

        function commitSelect(element, nextValue) {
          if (!element) {
            throw new Error("Missing filter select.");
          }
          element.value = nextValue;
          element.dispatchEvent("change");
        }

        function commitCheckbox(element, nextValue) {
          if (!element) {
            throw new Error("Missing filter checkbox.");
          }
          element.checked = Boolean(nextValue);
          element.dispatchEvent("change");
        }

        const [stateModule, utilitiesModule, metadataFiltersModule] =
          await Promise.all([
            import(pathToFileURL(__STATE_PATH__).href),
            import(pathToFileURL(__UTILITIES_PATH__).href),
            import(pathToFileURL(__METADATA_FILTERS_PATH__).href),
          ]);
        const { createInitialState } = stateModule;
        const { registerUtilities } = utilitiesModule;
        const { registerMetadataFilters } = metadataFiltersModule;

        const document = createFakeDocument();
        const canvasTools = createPanel(document);
        const renderCalls = [];
        const ctx = {
          state: createInitialState(),
          constants: {
            TENSOR_WIDTH: 140,
            TENSOR_HEIGHT: 84,
            MIN_TENSOR_WIDTH: 96,
            MIN_TENSOR_HEIGHT: 60,
            INDEX_RADIUS: 10,
            INDEX_PADDING: 6,
            NOTE_WIDTH: 220,
            NOTE_HEIGHT: 120,
            NOTE_MIN_WIDTH: 120,
            NOTE_MIN_HEIGHT: 90,
            HISTORY_LIMIT: 100,
            REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
            DEFAULT_INDEX_SLOTS: [
              { x: -38, y: 0 },
              { x: 38, y: 0 },
              { x: 0, y: -24 },
              { x: 0, y: 24 },
            ],
          },
          dom: {
            workspace: {},
            canvasShell: {
              getBoundingClientRect() {
                return { left: 0, top: 0, width: 1000, height: 800 };
              },
            },
            canvasTools,
            canvasContextMenuRoot: createPanel(document),
            propertiesPanel: createPanel(document),
            statusMessage: { textContent: "", classList: createClassList() },
            engineSelect: { options: [], value: "tensornetwork" },
            collectionFormatSelect: { options: [], value: "list" },
            exportFormatSelect: { value: "py" },
            addNoteButton: createButton(),
            connectButton: createButton(),
            loadInput: {},
            undoButton: createButton(),
            redoButton: createButton(),
            exportButton: createButton(),
            toggleLinearPeriodicButton: createButton(),
            linearPeriodicPreviousCellButton: createButton(),
            linearPeriodicCellLabel: { textContent: "" },
            linearPeriodicNextCellButton: createButton(),
            templateSelect: { value: "" },
            templateParameterPanel: { hidden: true },
            templateGraphSizeLabel: { textContent: "" },
            templateGraphSizeInput: { value: "2", min: "1" },
            templateBondDimensionInput: { value: "3", min: "1" },
            templatePhysicalDimensionInput: { value: "2", min: "1" },
            insertTemplateButton: createButton(),
            createGroupButton: createButton(),
            helpButton: createButton(),
            helpModal: { classList: createClassList() },
            helpBackdrop: createButton(),
            helpCloseButton: createButton(),
            groupLayer: {},
            resizeLayer: {},
            notesLayer: {},
            selectionBox: {},
            minimapCanvas: {},
            sidebar: {},
            plannerPanel: {},
            generateButton: createButton(),
          },
          apiGet: async () => null,
          apiPost: async () => null,
          window: {
            structuredClone: globalThis.structuredClone,
            crypto: globalThis.crypto,
            setTimeout,
            clearTimeout,
          },
          document,
          cytoscape: null,
        };

        registerUtilities(ctx);
        registerMetadataFilters(ctx);

        ctx.render = (options = {}) => {
          renderCalls.push(options);
        };

        ctx.state.annotationDefinitions = {
          tensor: [
            {
              key: "role",
              label: "Tensor role",
              placeholder: "observable",
              suggestions: ["state", "operator"],
            },
          ],
          index: [
            {
              key: "leg_kind",
              label: "Leg kind",
              placeholder: "physical",
              suggestions: ["physical", "logical"],
            },
          ],
        };
        ctx.state.spec = ctx.normalizeSpec(createSpec());
        ctx.state.selectionIds = ["tensor_b"];
        ctx.state.primarySelectionId = "tensor_b";

        const originalSpec = JSON.stringify(ctx.state.spec);
        const originalSelection = JSON.stringify(ctx.state.selectionIds);
        const originalSpecRevision = ctx.state.specRevision;
        const originalUndoLength = ctx.state.undoStack.length;

        ctx.renderMetadataFilters();
        if (!document.getElementById("canvas-metadata-filter-button")) {
          throw new Error("Expected the floating canvas tools to render.");
        }
        document.getElementById("canvas-metadata-filter-button").click();
        if (!document.getElementById("canvas-metadata-filter-scope-select")) {
          throw new Error("Expected the floating metadata filter popover to open.");
        }
        if (!document.getElementById("canvas-metadata-filter-clear-button")) {
          throw new Error("Expected the floating metadata filter popover to expose a clear action.");
        }
        if (!document.getElementById("canvas-metadata-filter-tag-not-specified")) {
          throw new Error("Expected every filter scope to expose the Not specified checkbox.");
        }
        document.getElementById("canvas-metadata-filter-scope-select").dispatchEvent("mousedown");
        if (
          document.getElementById("canvas-metadata-filter-scope-select").dataset.expanded
          !== "true"
        ) {
          throw new Error("Expected the metadata filter scope chevron to expand while the select is opening.");
        }

        commitCheckbox(document.getElementById("canvas-metadata-filter-tag-block"), true);
        if (JSON.stringify(ctx.state.metadataFilters.selectedTags) !== JSON.stringify(["block"])) {
          throw new Error(`Expected the tensor tag filter to update, received ${JSON.stringify(ctx.state.metadataFilters)}.`);
        }
        if (JSON.stringify(ctx.state.spec) !== originalSpec) {
          throw new Error("Applying metadata filters should not mutate the spec.");
        }
        if (JSON.stringify(ctx.state.selectionIds) !== originalSelection) {
          throw new Error("Applying metadata filters should not change the selection.");
        }
        if (ctx.state.specRevision !== originalSpecRevision || ctx.state.undoStack.length !== originalUndoLength) {
          throw new Error("Metadata filters should not participate in spec history.");
        }
        const tensorHighlight = ctx.getMetadataFilterHighlight();
        if (ctx.getMetadataFilterEntityState("tensor", "tensor_a", tensorHighlight) !== "match") {
          throw new Error("Expected tensor_a to match the tensor metadata filter.");
        }
        if (ctx.getMetadataFilterEntityState("index", "index_a", tensorHighlight) !== "match") {
          throw new Error("Expected index_a to stay bright with its matched tensor.");
        }
        if (ctx.getMetadataFilterEntityState("edge", "edge_ab", tensorHighlight) !== "dim") {
          throw new Error("Expected edge_ab to dim when only one tensor matches.");
        }
        if (!renderCalls.length) {
          throw new Error("Changing filters should trigger a lightweight render.");
        }

        document.getElementById("canvas-metadata-filter-select-none-button").click();
        const noneHighlight = ctx.getMetadataFilterHighlight();
        if (ctx.getMetadataFilterEntityState("tensor", "tensor_a", noneHighlight) !== "dim") {
          throw new Error("Deselecting every tag should leave the filter active and dim all entities.");
        }

        commitSelect(document.getElementById("canvas-metadata-filter-scope-select"), "index");
        if (
          document.getElementById("canvas-metadata-filter-scope-select").dataset.expanded
          !== "false"
        ) {
          throw new Error("Expected the metadata filter scope chevron to collapse after changing the scope.");
        }
        commitCheckbox(document.getElementById("canvas-metadata-filter-tag-not-specified"), true);
        const unspecifiedHighlight = ctx.getMetadataFilterHighlight();
        if (ctx.getMetadataFilterEntityState("index", "index_c", unspecifiedHighlight) !== "match") {
          throw new Error("Expected the Not specified checkbox to match indices without tags.");
        }
        if (ctx.getMetadataFilterEntityState("tensor", "tensor_b", unspecifiedHighlight) !== "context") {
          throw new Error("Expected the owner tensor to stay in context for a Not specified index match.");
        }
        document.getElementById("canvas-metadata-filter-clear-button").click();
        if (ctx.state.metadataFilters.enabled) {
          throw new Error("Clearing the filter should disable the metadata filter state.");
        }
        if (ctx.getMetadataFilterHighlight() !== null) {
          throw new Error("Clearing the filter should restore the neutral highlight state.");
        }

        document.getElementById("canvas-name-search-button").click();
        document.getElementById("canvas-name-search-scope-select").dispatchEvent("mousedown");
        if (
          document.getElementById("canvas-name-search-scope-select").dataset.expanded
          !== "true"
        ) {
          throw new Error("Expected the name search scope chevron to expand while the select is opening.");
        }
        commitSelect(document.getElementById("canvas-name-search-scope-select"), "bond");
        if (
          document.getElementById("canvas-name-search-scope-select").dataset.expanded
          !== "false"
        ) {
          throw new Error("Expected the name search scope chevron to collapse after changing the scope.");
        }
        const searchInput = document.getElementById("canvas-name-search-input");
        if (!searchInput) {
          throw new Error("Expected the search popover to expose its input.");
        }
        searchInput.value = "b";
        searchInput.dispatchEvent("input");
        if (document.getElementById("canvas-name-search-input") !== searchInput) {
          throw new Error("Typing in the name search should not recreate the input and steal focus.");
        }
        searchInput.value = "bond_ab";
        searchInput.dispatchEvent("input");
        document.activeElement = null;
        searchInput.dispatchEvent("blur");
        if (document.activeElement) {
          throw new Error("Leaving the name search input should not immediately steal focus back.");
        }
        const searchHighlight = ctx.getMetadataFilterHighlight();
        if (ctx.getMetadataFilterEntityState("edge", "edge_ab", searchHighlight) !== "match") {
          throw new Error("Expected the bond search to match the edge by exact name.");
        }
        if (ctx.getMetadataFilterEntityState("tensor", "tensor_a", searchHighlight) !== "context") {
          throw new Error("Expected the left tensor to remain in context for a matched bond search.");
        }
        if (ctx.getMetadataFilterEntityState("index", "index_b", searchHighlight) !== "context") {
          throw new Error("Expected the incident indices to remain in context for a matched bond search.");
        }
        if (ctx.getMetadataFilterEntityState("tensor", "tensor_b", searchHighlight) !== "context") {
          throw new Error("Expected the right tensor to remain in context for a matched bond search.");
        }
        """
    )
    script_body = script_body.replace(
        "__STATE_PATH__", json.dumps(str(state_runtime_path))
    )
    script_body = script_body.replace(
        "__UTILITIES_PATH__", json.dumps(str(utilities_runtime_path))
    )
    script_body = script_body.replace(
        "__METADATA_FILTERS_PATH__", json.dumps(str(metadata_filters_runtime_path))
    )
    script_path.write_text(script_body, encoding="utf-8")
    return script_path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_for_mode_dimension_updates_keep_working_after_first_change(
    tmp_path: Path,
) -> None:
    script_path = _write_for_mode_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The frontend runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


def _write_grid_for_mode_runtime_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "grid_for_mode_runtime.mjs"
    state_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "state/state.js"
    )
    base_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utils/utilitiesBase.js"
    )
    geometry_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utils/utilitiesGeometry.js"
    )
    layout_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utils/utilitiesLayout.js"
    )
    spec_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utils/utilitiesSpec.js"
    )
    linear_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utils/utilitiesLinearPeriodic.js"
    )
    grid_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utils/utilitiesGridPeriodic.js"
    )
    script_path.write_text(
        textwrap.dedent(
            f"""
            import {{ pathToFileURL }} from "node:url";

            const [
              stateModule,
              baseModule,
              geometryModule,
              layoutModule,
              specModule,
              linearModule,
              gridModule,
            ] = await Promise.all([
              import(pathToFileURL({json.dumps(str(state_module_path))}).href),
              import(pathToFileURL({json.dumps(str(base_module_path))}).href),
              import(pathToFileURL({json.dumps(str(geometry_module_path))}).href),
              import(pathToFileURL({json.dumps(str(layout_module_path))}).href),
              import(pathToFileURL({json.dumps(str(spec_module_path))}).href),
              import(pathToFileURL({json.dumps(str(linear_module_path))}).href),
              import(pathToFileURL({json.dumps(str(grid_module_path))}).href),
            ]);

            const {{ createInitialState }} = stateModule;
            const {{ createUtilityBaseBindings }} = baseModule;
            const {{ createUtilityGeometryBindings }} = geometryModule;
            const {{ createUtilityLayoutBindings }} = layoutModule;
            const {{ createUtilitySpecBindings }} = specModule;
            const {{ createUtilityLinearPeriodicBindings }} = linearModule;
            const {{ createUtilityGridPeriodicBindings }} = gridModule;

            const state = createInitialState();
            const runtime = {{}};
            const events = [];
            const ctx = {{
              state,
              constants: {{
                TENSOR_WIDTH: 140,
                TENSOR_HEIGHT: 84,
                MIN_TENSOR_WIDTH: 96,
                MIN_TENSOR_HEIGHT: 60,
                INDEX_RADIUS: 10,
                INDEX_PADDING: 6,
                NOTE_WIDTH: 220,
                NOTE_HEIGHT: 120,
                NOTE_MIN_WIDTH: 120,
                NOTE_MIN_HEIGHT: 90,
                DEFAULT_INDEX_SLOTS: [
                  {{ x: -38, y: 0 }},
                  {{ x: 38, y: 0 }},
                  {{ x: 0, y: -24 }},
                  {{ x: 0, y: 24 }},
                ],
              }},
              dom: {{
                engineSelect: {{ options: [], value: "tensornetwork" }},
              }},
              window: {{
                confirm() {{
                  return true;
                }},
              }},
              document: {{
                activeElement: null,
                querySelectorAll() {{
                  return [];
                }},
              }},
              render() {{
                events.push("render");
              }},
              setStatus(message, level = "info") {{
                events.push(`status:${{level}}:${{message}}`);
              }},
              clearGeneratedCodePreview() {{
                events.push("clear-preview");
              }},
              refreshContractionAnalysis() {{
                events.push("refresh-analysis");
              }},
              bumpSpecRevision() {{
                state.specRevision += 1;
              }},
              resetDerivedStateCaches() {{}},
              ensureSpecLookups() {{
                state.tensorById = Object.fromEntries(
                  (Array.isArray(state.spec && state.spec.tensors) ? state.spec.tensors : [])
                    .map((tensor) => [tensor.id, tensor])
                );
              }},
            }};

            const env = {{
              ctx,
              state,
              constants: ctx.constants,
              dom: ctx.dom,
              runtime,
            }};
            Object.assign(runtime, createUtilityBaseBindings(env));
            Object.assign(runtime, createUtilityGeometryBindings(env));
            Object.assign(runtime, createUtilityLayoutBindings(env));
            Object.assign(runtime, createUtilitySpecBindings(env));
            Object.assign(runtime, createUtilityLinearPeriodicBindings(env));
            Object.assign(runtime, createUtilityGridPeriodicBindings(env));
            Object.assign(ctx, runtime);

            state.spec = runtime.normalizeSpec({{
              id: "network_grid_demo",
              name: "Grid Demo",
              tensors: [
                {{
                  id: "tensor_center",
                  name: "Center",
                  position: {{ x: 180, y: 160 }},
                  size: {{ width: 140, height: 84 }},
                  metadata: {{}},
                  indices: [
                    {{
                      id: "center_open_a",
                      name: "a",
                      dimension: 2,
                      offset: {{ x: -38, y: 0 }},
                      metadata: {{}},
                    }},
                    {{
                      id: "center_open_b",
                      name: "b",
                      dimension: 3,
                      offset: {{ x: 38, y: 0 }},
                      metadata: {{}},
                    }},
                  ],
                }},
              ],
              groups: [],
              edges: [],
              notes: [],
              contraction_plan: null,
              metadata: {{}},
            }});

            runtime.setGridPeriodicMode(true);
            if (!runtime.isGridPeriodicMode()) {{
              throw new Error("Grid periodic mode should be enabled.");
            }}
            if (runtime.getActiveGridPeriodicCellName() !== "center") {{
              throw new Error(`Expected center to be active after enabling grid mode, received ${{runtime.getActiveGridPeriodicCellName()}}.`);
            }}
            const activeBoundaryRoles = state.spec.tensors
              .filter((tensor) => tensor.grid_periodic_role)
              .map((tensor) => tensor.grid_periodic_role)
              .sort();
            if (JSON.stringify(activeBoundaryRoles) !== JSON.stringify(["down", "left", "right", "up"])) {{
              throw new Error(`Expected four center boundary tensors, received ${{JSON.stringify(activeBoundaryRoles)}}.`);
            }}
            if (
              !state.spec.grid_periodic_grid.center_cell.tensors.some(
                (tensor) => tensor.id === "tensor_center"
              )
            ) {{
              throw new Error("The original graph should seed the center cell.");
            }}

            runtime.switchGridPeriodicCell("up");
            if (runtime.getActiveGridPeriodicCellName() !== "top") {{
              throw new Error(`Expected top to be active after moving up, received ${{runtime.getActiveGridPeriodicCellName()}}.`);
            }}
            state.spec.tensors.push({{
              id: "tensor_top",
              name: "Top",
              position: {{ x: 180, y: 120 }},
              size: {{ width: 140, height: 84 }},
              metadata: {{}},
              indices: [
                {{
                  id: "top_open",
                  name: "t",
                  dimension: 5,
                  offset: {{ x: 0, y: -24 }},
                  metadata: {{}},
                }},
              ],
            }});
            runtime.syncGridPeriodicBoundaryTensors();
            runtime.setGridPeriodicMode(false);
            if (runtime.isGridPeriodicMode()) {{
              throw new Error("Grid periodic mode should be disabled.");
            }}
            if (state.spec.grid_periodic_grid !== null) {{
              throw new Error("Grid payload should be cleared after leaving grid mode.");
            }}
            if (!state.spec.tensors.some((tensor) => tensor.id === "tensor_top")) {{
              throw new Error("Leaving grid mode should preserve the active cell.");
            }}
            if (state.spec.tensors.some((tensor) => tensor.grid_periodic_role)) {{
              throw new Error("Boundary tensors should be stripped when returning to single mode.");
            }}
            if (!events.includes("refresh-analysis")) {{
              throw new Error(`Expected grid mode transitions to refresh analysis, received ${{JSON.stringify(events)}}.`);
            }}
            """,
        ),
        encoding="utf-8",
    )
    return script_path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_grid_for_mode_can_seed_navigate_and_restore_active_cell(
    tmp_path: Path,
) -> None:
    script_path = _write_grid_for_mode_runtime_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The grid for-mode runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


def _write_tree_for_mode_runtime_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "tree_for_mode_runtime.mjs"
    state_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "state/state.js"
    )
    linear_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utils/utilitiesLinearPeriodic.js"
    )
    grid_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utils/utilitiesGridPeriodic.js"
    )
    tree_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utils/utilitiesTreePeriodic.js"
    )
    utilities_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utils/utilities.js"
    )

    script_path.write_text(
        textwrap.dedent(
            f"""
            import {{ pathToFileURL }} from "node:url";

            const stateUrl = pathToFileURL({str(state_module_path)!r}).href;
            const linearUrl = pathToFileURL({str(linear_module_path)!r}).href;
            const gridUrl = pathToFileURL({str(grid_module_path)!r}).href;
            const treeUrl = pathToFileURL({str(tree_module_path)!r}).href;
            const utilitiesUrl = pathToFileURL({str(utilities_module_path)!r}).href;

            const [stateModule, linearModule, gridModule, treeModule, utilitiesModule] = await Promise.all([
              import(stateUrl),
              import(linearUrl),
              import(gridUrl),
              import(treeUrl),
              import(utilitiesUrl),
            ]);

            function createClassList() {{
              return {{
                add() {{}},
                remove() {{}},
                toggle() {{}},
              }};
            }}

            function createButton() {{
              return {{
                disabled: false,
                hidden: false,
                textContent: "",
                classList: createClassList(),
                addEventListener() {{}},
              }};
            }}

            const {{ createInitialState }} = stateModule;
            const {{ createUtilityLinearPeriodicBindings }} = linearModule;
            const {{ createUtilityGridPeriodicBindings }} = gridModule;
            const {{ createUtilityTreePeriodicBindings }} = treeModule;
            const {{ registerUtilities }} = utilitiesModule;
            const state = createInitialState();
            const events = [];
            const ctx = {{
              state,
              constants: {{
                TENSOR_WIDTH: 140,
                TENSOR_HEIGHT: 84,
                MIN_TENSOR_WIDTH: 96,
                MIN_TENSOR_HEIGHT: 60,
                NOTE_WIDTH: 220,
                NOTE_HEIGHT: 120,
                NOTE_MIN_WIDTH: 120,
                NOTE_MIN_HEIGHT: 90,
                HISTORY_LIMIT: 100,
                REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
                DEFAULT_INDEX_SLOTS: [
                  {{ x: -38, y: 0 }},
                  {{ x: 38, y: 0 }},
                  {{ x: 0, y: -24 }},
                  {{ x: 0, y: 24 }},
                ],
              }},
              dom: {{
                workspace: {{}},
                statusMessage: {{ textContent: "", classList: createClassList() }},
                propertiesPanel: {{ innerHTML: "" }},
                generatedCode: {{ value: "" }},
                engineSelect: {{ options: [], value: "tensornetwork" }},
                collectionFormatSelect: {{ options: [], value: "list" }},
                exportFormatSelect: {{ value: "py" }},
                addNoteButton: createButton(),
                connectButton: {{ classList: createClassList() }},
                loadInput: {{}},
                undoButton: createButton(),
                redoButton: createButton(),
                exportButton: createButton(),
                toggleLinearPeriodicButton: {{ classList: createClassList() }},
                linearPeriodicPreviousCellButton: createButton(),
                linearPeriodicCellLabel: {{ textContent: "" }},
                linearPeriodicNextCellButton: createButton(),
                gridPeriodicUpCellButton: createButton(),
                gridPeriodicDownCellButton: createButton(),
                singleModeMenuItem: createButton(),
                linearPeriodicModeMenuItem: createButton(),
                gridPeriodicModeMenuItem: createButton(),
                treeModeMenuItem: createButton(),
                benchmarkModeMenuItem: createButton(),
                toolbarModeControls: {{ hidden: true }},
                templateSelect: {{ value: "" }},
                templateParameterPanel: {{ hidden: true }},
                templateGraphSizeLabel: {{ textContent: "" }},
                templateGraphSizeInput: {{ value: "2", min: "1" }},
                templateBondDimensionInput: {{ value: "3", min: "1" }},
                templatePhysicalDimensionInput: {{ value: "2", min: "1" }},
                insertTemplateButton: createButton(),
                createGroupButton: createButton(),
                helpButton: createButton(),
                helpModal: {{ classList: createClassList() }},
                helpBackdrop: createButton(),
                helpCloseButton: createButton(),
                canvasShell: {{
                  getBoundingClientRect() {{
                    return {{ left: 0, top: 0, width: 1000, height: 800 }};
                  }},
                }},
                groupLayer: {{}},
                resizeLayer: {{}},
                notesLayer: {{}},
                selectionBox: {{}},
                minimapCanvas: {{}},
                sidebar: {{}},
                plannerPanel: {{}},
                generateButton: createButton(),
              }},
              apiGet: async () => null,
              apiPost: async () => null,
              window: {{
                structuredClone: globalThis.structuredClone,
                crypto: globalThis.crypto,
                setTimeout,
                clearTimeout,
                confirm: () => true,
                prompt: () => "3",
              }},
              document: {{
                getElementById() {{
                  return createButton();
                }},
                querySelectorAll() {{
                  return [];
                }},
              }},
              cytoscape: null,
              tensorWidth: (tensor) => tensor?.size?.width ?? 140,
              tensorHeight: (tensor) => tensor?.size?.height ?? 84,
              render: () => {{}},
              renderOverlayDecorations: () => {{}},
              renderMinimap: () => {{}},
              renderPlanner: () => {{}},
              renderSidebarTabs: () => {{}},
              refreshContractionAnalysis: () => events.push("refresh-analysis"),
              syncPendingInteractionClasses: () => {{}},
              setActiveSidebarTab: () => {{}},
              clearGeneratedCodePreview: () => false,
              resetDerivedStateCaches: () => {{}},
              ensureTensorIndexOffsets: () => {{}},
              buildHistorySnapshotSpec: () => state.spec,
              buildSerializedSpec: () => state.spec,
              bumpSpecRevision: () => {{
                state.specRevision += 1;
              }},
              resetDerivedStateCaches: () => {{}},
            }};

            const runtime = {{}};
            const env = {{
              ctx,
              state,
              constants: ctx.constants,
              dom: ctx.dom,
              runtime,
            }};
            Object.assign(runtime, createUtilityLinearPeriodicBindings(env));
            Object.assign(runtime, createUtilityGridPeriodicBindings(env));
            Object.assign(runtime, createUtilityTreePeriodicBindings(env));
            registerUtilities(ctx);
            Object.assign(runtime, ctx);

            state.spec = ctx.normalizeSpec({{
              id: "network_tree_seed",
              name: "tree-seed",
              tensors: [
                {{
                  id: "tensor_root",
                  name: "Root",
                  position: {{ x: 180, y: 160 }},
                  size: {{ width: 140, height: 84 }},
                  metadata: {{}},
                  indices: [
                    {{
                      id: "root_open_0",
                      name: "a",
                      dimension: 2,
                      offset: {{ x: -38, y: 0 }},
                      metadata: {{}},
                    }},
                    {{
                      id: "root_open_1",
                      name: "b",
                      dimension: 3,
                      offset: {{ x: 38, y: 0 }},
                      metadata: {{}},
                    }},
                  ],
                }},
              ],
              groups: [],
              edges: [],
              notes: [],
              contraction_plan: null,
              metadata: {{}},
            }});

            runtime.setTreePeriodicMode(true);
            if (!runtime.isTreePeriodicMode()) {{
              throw new Error("Tree periodic mode should be enabled.");
            }}
            if (runtime.getActiveTreePeriodicCellName() !== "root") {{
              throw new Error(`Expected root to be active after enabling tree mode, received ${{runtime.getActiveTreePeriodicCellName()}}.`);
            }}
            if (!state.spec.tree_periodic_tree || state.spec.tree_periodic_tree.branching_factor !== 3) {{
              throw new Error("Tree payload should persist the prompted branching factor.");
            }}
            const rootChildren = state.spec.tensors
              .filter((tensor) => tensor.tree_periodic_role === "child")
              .map((tensor) => tensor.tree_periodic_child_index)
              .sort();
            if (JSON.stringify(rootChildren) !== JSON.stringify([0, 1, 2])) {{
              throw new Error(`Expected three ordered child boundaries in root, received ${{JSON.stringify(rootChildren)}}.`);
            }}
            const rootChildDimensions = state.spec.tensors
              .filter((tensor) => tensor.tree_periodic_role === "child")
              .map((tensor) => tensor.indices.map((index) => index.dimension));
            if (
              JSON.stringify(rootChildDimensions) !==
              JSON.stringify([[2, 3], [2, 3], [2, 3]])
            ) {{
              throw new Error(
                `Expected root child boundaries to inherit the root free ports, received ${{JSON.stringify(rootChildDimensions)}}.`
              );
            }}
            if (
              !state.spec.tree_periodic_tree.root_cell.tensors.some(
                (tensor) => tensor.id === "tensor_root"
              )
            ) {{
              throw new Error("The original graph should seed the root cell.");
            }}

            runtime.switchTreePeriodicCell("down");
            if (runtime.getActiveTreePeriodicCellName() !== "branch") {{
              throw new Error(`Expected branch to be active after moving down, received ${{runtime.getActiveTreePeriodicCellName()}}.`);
            }}
            const branchRoles = state.spec.tensors
              .filter((tensor) => tensor.tree_periodic_role)
              .map((tensor) => `${{tensor.tree_periodic_role}}:${{tensor.tree_periodic_child_index ?? "parent"}}`)
              .sort();
            if (
              JSON.stringify(branchRoles) !==
              JSON.stringify(["child:0", "child:1", "child:2", "parent:parent"])
            ) {{
              throw new Error(`Expected one parent and three child boundaries in branch, received ${{JSON.stringify(branchRoles)}}.`);
            }}
            const branchParentBoundary = state.spec.tensors.find(
              (tensor) => tensor.tree_periodic_role === "parent"
            );
            if (!branchParentBoundary) {{
              throw new Error("Expected the branch cell to expose a parent boundary.");
            }}
            const branchParentDimensions = branchParentBoundary.indices.map(
              (index) => index.dimension
            );
            if (JSON.stringify(branchParentDimensions) !== JSON.stringify([2, 3])) {{
              throw new Error(
                `Expected the branch parent boundary to inherit the root interface, received ${{JSON.stringify(branchParentDimensions)}}.`
              );
            }}

            state.spec.tensors.push({{
              id: "tensor_branch",
              name: "Branch tensor",
              position: {{ x: 180, y: 160 }},
              size: {{ width: 140, height: 84 }},
              metadata: {{}},
              indices: [
                {{
                  id: "branch_open",
                  name: "c",
                  dimension: 5,
                  offset: {{ x: 0, y: -24 }},
                  metadata: {{}},
                }},
              ],
            }});
            runtime.syncTreePeriodicBoundaryTensors();
            const branchChildDimensions = state.spec.tensors
              .filter((tensor) => tensor.tree_periodic_role === "child")
              .map((tensor) => tensor.indices.map((index) => index.dimension));
            if (
              JSON.stringify(branchChildDimensions) !==
              JSON.stringify([[5], [5], [5]])
            ) {{
              throw new Error(
                `Expected branch child boundaries to reflect the remaining branch free ports, received ${{JSON.stringify(branchChildDimensions)}}.`
              );
            }}
            runtime.switchTreePeriodicCell("down");
            if (runtime.getActiveTreePeriodicCellName() !== "leaf") {{
              throw new Error(`Expected leaf to be active after moving down twice, received ${{runtime.getActiveTreePeriodicCellName()}}.`);
            }}
            const leafRoles = state.spec.tensors
              .filter((tensor) => tensor.tree_periodic_role)
              .map((tensor) => tensor.tree_periodic_role);
            if (JSON.stringify(leafRoles) !== JSON.stringify(["parent"])) {{
              throw new Error(`Expected only a parent boundary in leaf, received ${{JSON.stringify(leafRoles)}}.`);
            }}
            const leafParentDimensions = state.spec.tensors[0].indices.map(
              (index) => index.dimension
            );
            if (JSON.stringify(leafParentDimensions) !== JSON.stringify([5])) {{
              throw new Error(
                `Expected the leaf parent boundary to inherit the branch interface, received ${{JSON.stringify(leafParentDimensions)}}.`
              );
            }}

            runtime.switchTreePeriodicCell("up");
            if (!state.spec.tensors.some((tensor) => tensor.id === "tensor_branch")) {{
              throw new Error("Returning to branch should restore the branch cell graph.");
            }}

            runtime.setTreePeriodicMode(false);
            if (runtime.isTreePeriodicMode()) {{
              throw new Error("Tree periodic mode should be disabled.");
            }}
            if (state.spec.tree_periodic_tree !== null) {{
              throw new Error("Tree payload should be cleared after leaving tree mode.");
            }}
            if (!state.spec.tensors.some((tensor) => tensor.id === "tensor_branch")) {{
              throw new Error("Leaving tree mode should preserve the active cell.");
            }}
            if (state.spec.tensors.some((tensor) => tensor.tree_periodic_role)) {{
              throw new Error("Tree boundary tensors should be stripped when returning to single mode.");
            }}
            if (!events.includes("refresh-analysis")) {{
              throw new Error(`Expected tree mode transitions to refresh analysis, received ${{JSON.stringify(events)}}.`);
            }}
            """,
        ),
        encoding="utf-8",
    )
    return script_path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_tree_for_mode_can_seed_navigate_and_restore_active_cell(
    tmp_path: Path,
) -> None:
    script_path = _write_tree_for_mode_runtime_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The tree for-mode runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_for_mode_reserved_operands_survive_cell_switches_and_scene_updates(
    tmp_path: Path,
) -> None:
    script_path = _write_for_mode_reserved_operand_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The reserved-operand frontend runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


def _write_tree_mutation_sync_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "tree_mutation_sync_runtime_regression.mjs"
    _copy_runtime_bundle(
        tmp_path,
        {
            "state.runtime.mjs": "state/state.js",
            "utilities.runtime.mjs": "utils/utilities.js",
            "historySelection.runtime.mjs": "graph/historySelection.js",
            "properties.runtime.mjs": "properties/properties.js",
            "propertiesSupport.js": "properties/propertiesSupport.js",
            "propertiesRenderers.js": "properties/propertiesRenderers.js",
            "interactionsEditor.js": "interactions/interactionsEditor.js",
            "interactionsSession.js": "interactions/interactionsSession.js",
            "notes.js": "graph/notes.js",
            "state/editorSelectors.js": "state/editorSelectors.js",
            "state/editorStore.js": "state/editorStore.js",
        },
        _RUNTIME_EDITOR_SUPPORT_MODULES,
    )

    script_body = textwrap.dedent(
        """
        import { pathToFileURL } from "node:url";

        const baseUrl = new URL("./", import.meta.url);
        const [stateModule, utilitiesModule, historyModule, propertiesModule, editorModule, sessionModule, notesModule, selectorsModule, storeModule] =
          await Promise.all([
            import(new URL("./state.runtime.mjs", baseUrl).href),
            import(new URL("./utilities.runtime.mjs", baseUrl).href),
            import(new URL("./historySelection.runtime.mjs", baseUrl).href),
            import(new URL("./properties.runtime.mjs", baseUrl).href),
            import(new URL("./interactionsEditor.js", baseUrl).href),
            import(new URL("./interactionsSession.js", baseUrl).href),
            import(new URL("./notes.js", baseUrl).href),
            import(new URL("./state/editorSelectors.js", baseUrl).href),
            import(new URL("./state/editorStore.js", baseUrl).href),
          ]);

        const { createInitialState } = stateModule;
        const { registerUtilities } = utilitiesModule;
        const { registerHistorySelection } = historyModule;
        const { registerProperties } = propertiesModule;
        const { createInteractionEditorBindings } = editorModule;
        const { createInteractionSessionBindings } = sessionModule;
        const { registerNotesFeature } = notesModule;
        const { createEditorSelectors } = selectorsModule;
        const { createEditorStore } = storeModule;

        function createClassList() {
          return {
            add() {},
            remove() {},
            toggle() {},
          };
        }

        function createButton() {
          return {
            disabled: false,
            hidden: false,
            value: "",
            classList: createClassList(),
            dataset: {},
            addEventListener() {},
            click() {},
            focus() {},
            appendChild() {},
            setAttribute() {},
            removeAttribute() {},
          };
        }

        const state = createInitialState();
        const ctx = {
          state,
          constants: {
            TENSOR_WIDTH: 140,
            TENSOR_HEIGHT: 84,
            MIN_TENSOR_WIDTH: 96,
            MIN_TENSOR_HEIGHT: 60,
            INDEX_RADIUS: 10,
            INDEX_PADDING: 6,
            NOTE_WIDTH: 220,
            NOTE_HEIGHT: 120,
            NOTE_MIN_WIDTH: 120,
            NOTE_MIN_HEIGHT: 90,
            NOTE_COLLAPSED_SIZE: 40,
            HISTORY_LIMIT: 100,
            REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
            DEFAULT_INDEX_SLOTS: [
              { x: -38, y: 0 },
              { x: 38, y: 0 },
              { x: 0, y: -24 },
              { x: 0, y: 24 },
            ],
          },
          dom: {
            workspace: {},
            statusMessage: { textContent: "", classList: createClassList() },
            propertiesPanel: { innerHTML: "" },
            generatedCode: { value: "" },
            generatedCodeView: { textContent: "", dataset: {} },
            engineSelect: { options: [], value: "tensornetwork", appendChild() {} },
            collectionFormatSelect: { options: [], value: "list", appendChild() {} },
            exportFormatSelect: { value: "py" },
            connectButton: createButton(),
            loadInput: createButton(),
            undoButton: createButton(),
            redoButton: createButton(),
            templateSelect: { value: "mps", appendChild() {} },
            templateParameterPanel: { hidden: true, innerHTML: "" },
            templateGraphSizeLabel: { textContent: "" },
            templateGraphSizeInput: { value: "2", min: "1", addEventListener() {} },
            templateBondDimensionInput: { value: "3", min: "1", addEventListener() {} },
            templatePhysicalDimensionInput: { value: "2", min: "1", addEventListener() {} },
            insertTemplateButton: createButton(),
            insertSubnetworkButton: createButton(),
            createGroupButton: createButton(),
            helpButton: createButton(),
            helpModal: { classList: createClassList() },
            helpBackdrop: createButton(),
            helpCloseButton: createButton(),
            canvasShell: {
              getBoundingClientRect() {
                return { left: 0, top: 0, width: 1000, height: 800 };
              },
            },
            groupLayer: {},
            resizeLayer: {},
            notesLayer: {},
            selectionBox: { classList: createClassList(), style: {} },
            minimapCanvas: {},
            subnetworkLoadInput: { value: "", click() {} },
            addNoteButton: createButton(),
          },
          apiGet: async () => null,
          apiPost: async () => null,
          window: {
            structuredClone: globalThis.structuredClone,
            crypto: globalThis.crypto,
            setTimeout,
            clearTimeout,
            confirm: () => true,
            prompt: () => "3",
            Prism: {
              highlightElement() {},
            },
          },
          document: {
            activeElement: null,
            createElement() {
              return {
                value: "",
                textContent: "",
                selected: false,
                dataset: {},
                style: {},
                classList: createClassList(),
                appendChild() {},
                addEventListener() {},
                setAttribute() {},
                removeAttribute() {},
              };
            },
            querySelectorAll() {
              return [];
            },
          },
          cytoscape: null,
          render() {},
          renderOverlayDecorations() {},
          renderMinimap() {},
          renderPlanner() {},
          renderSidebarTabs() {},
          refreshContractionAnalysis() {},
          syncPendingInteractionClasses() {},
          clearGeneratedCodePreview() {
            return false;
          },
          setStatus() {},
          isTextInput() {
            return false;
          },
        };

        ctx.store = createEditorStore(state);
        ctx.selectors = createEditorSelectors({ store: ctx.store });
        ctx.services = {
          session: {
            async buildTemplate() {
              return {
                ok: true,
                spec: {
                  network: {
                    id: "template_fragment",
                    name: "Template fragment",
                    tensors: [
                      {
                        id: "template_tensor",
                        name: "Template tensor",
                        position: { x: 0, y: 0 },
                        size: { width: 140, height: 84 },
                        metadata: {},
                        indices: [
                          {
                            id: "template_index",
                            name: "tmpl",
                            dimension: 7,
                            offset: { x: -38, y: 0 },
                            metadata: {},
                          },
                        ],
                      },
                    ],
                    edges: [],
                    groups: [],
                    notes: [],
                    metadata: {},
                  },
                },
              };
            },
          },
          templateCatalog: {},
          subnetwork: {
            async extractSubnetwork() {
              return null;
            },
          },
        };

        registerUtilities(ctx);
        registerHistorySelection(ctx);
        registerProperties(ctx);
        registerNotesFeature(ctx);
        ctx.uniquifyImportedSpec = (spec) => ctx.normalizeSpec(structuredClone(spec));
        ctx.translateImportedSpec = (spec) => ctx.normalizeSpec(structuredClone(spec));
        Object.assign(
          ctx,
          createInteractionEditorBindings({
            ctx,
            state,
            runtime: {},
          })
        );
        Object.assign(
          ctx,
          createInteractionSessionBindings({
            ctx,
            state,
            dom: ctx.dom,
            store: ctx.store,
            selectors: ctx.selectors,
            services: ctx.services,
            sessionUi: {
              async copyText() {},
              downloadText() {},
              downloadBlob() {},
              requestFileText: async () => "",
              openFilePicker() {},
              schedule(callback) {
                callback();
              },
              closeWindow() {},
              promptText: () => null,
              confirmAction: () => true,
            },
            sessionActions: {
              ensureCodePanelVisible() {},
              syncCodeGenerationWarning() {},
              getTensorKrowchManualPlanIssueMessage() {
                return "";
              },
              getSelectedTensorIds() {
                return ctx.getSelectedIdsByKind("tensor");
              },
              findGroupById(groupId) {
                return ctx.findGroupById(groupId);
              },
              isLinearPeriodicMode() {
                return false;
              },
              isForMode() {
                return ctx.isForMode();
              },
              syncGeneratedCodePreview() {},
              setStatus(message, level) {
                ctx.setStatus(message, level);
              },
              serializeCurrentSpec() {
                return ctx.serializeCurrentSpec();
              },
              formatIssues() {
                return ctx.formatIssues();
              },
              stripImportLines(code) {
                return ctx.stripImportLines(code);
              },
              sanitizeFilename(value) {
                return ctx.sanitizeFilename(value);
              },
              resetDesignState() {},
              downloadPngExport() {},
              downloadSvgExport() {},
              applyTemplateCatalogPayload() {},
              normalizeSpec(spec) {
                return ctx.normalizeSpec(spec);
              },
              applyDesignChange(mutate, options) {
                return ctx.applyDesignChange(mutate, options);
              },
              bringTensorToFront(tensorId) {
                return ctx.bringTensorToFront(tensorId);
              },
              formatTemplateLabel(templateName) {
                return ctx.formatTemplateLabel(templateName);
              },
              getTemplateSource(templateName) {
                return ctx.getTemplateSource(templateName);
              },
              getTemplateSpec(templateName) {
                return ctx.getTemplateSpec(templateName);
              },
              listTemplateEntries() {
                return ctx.listTemplateEntries();
              },
              hasTemplateDisplayName(displayName, excludedTemplateName) {
                return ctx.hasTemplateDisplayName(displayName, excludedTemplateName);
              },
              getNextSessionTemplateDisplayName(baseDisplayName) {
                return ctx.getNextSessionTemplateDisplayName(baseDisplayName);
              },
              addSessionTemplate(payload) {
                return ctx.addSessionTemplate(payload);
              },
              updateSessionTemplateDisplayNames(updates) {
                return ctx.updateSessionTemplateDisplayNames(updates);
              },
              removeSessionTemplate(templateName) {
                return ctx.removeSessionTemplate(templateName);
              },
              toggleTemplateManager(forceOpen) {
                return ctx.toggleTemplateManager(forceOpen);
              },
              syncTemplateManagerModalState() {},
              setTemplateManagerValidationMessage() {},
              persistTemplateParametersFromControls() {
                return {};
              },
              uniquifyImportedSpec(spec, prefix) {
                return ctx.uniquifyImportedSpec(spec, prefix);
              },
              makeId(prefix) {
                return ctx.makeId(prefix);
              },
              translateImportedSpec(spec, targetCenter) {
                return ctx.translateImportedSpec(spec, targetCenter);
              },
              suggestTensorPosition(position) {
                return ctx.suggestTensorPosition(position);
              },
              viewportCenterPosition() {
                return ctx.viewportCenterPosition();
              },
            },
          })
        );

        state.selectedEngine = "tensornetwork";
        state.selectedCollectionFormat = "list";
        state.availableTemplates = ["mps"];
        state.templateDefinitions = {
          mps: {
            name: "mps",
            display_name: "MPS",
            source: "global",
          },
        };
        state.spec = ctx.normalizeSpec({
          id: "network_tree_mutation",
          name: "tree mutation",
          tensors: [
            {
              id: "tensor_root",
              name: "Root",
              position: { x: 180, y: 160 },
              size: { width: 140, height: 84 },
              metadata: {},
              indices: [
                {
                  id: "root_open_0",
                  name: "a",
                  dimension: 2,
                  offset: { x: -38, y: 0 },
                  metadata: {},
                },
                {
                  id: "root_open_1",
                  name: "b",
                  dimension: 3,
                  offset: { x: 38, y: 0 },
                  metadata: {},
                },
              ],
            },
          ],
          groups: [],
          edges: [],
          notes: [],
          contraction_plan: null,
          metadata: {},
        });

        function getChildBoundaryLengths() {
          return state.spec.tensors
            .filter((tensor) => tensor.tree_periodic_role === "child")
            .map((tensor) => tensor.indices.length);
        }

        ctx.setTreePeriodicMode(true);
        if (JSON.stringify(getChildBoundaryLengths()) !== JSON.stringify([2, 2, 2])) {
          throw new Error("Tree mode should start with one child boundary per root free index.");
        }

        ctx.propertyCommands.addIndexToSelectedTensors({
          tensorIds: ["tensor_root"],
          selectionIds: ["tensor_root"],
          primaryId: "tensor_root",
          statusMessage: "Added one index to Root.",
        });
        if (JSON.stringify(getChildBoundaryLengths()) !== JSON.stringify([3, 3, 3])) {
          throw new Error(`Adding an index in the root cell should refresh child boundaries immediately, received ${JSON.stringify(getChildBoundaryLengths())}.`);
        }

        ctx.addTensorAtCenter();
        if (JSON.stringify(getChildBoundaryLengths()) !== JSON.stringify([5, 5, 5])) {
          throw new Error(`Adding a tensor in the root cell should refresh child boundaries immediately, received ${JSON.stringify(getChildBoundaryLengths())}.`);
        }

        await ctx.insertTemplate();
        if (JSON.stringify(getChildBoundaryLengths()) !== JSON.stringify([6, 6, 6])) {
          throw new Error(`Inserting a template in the root cell should refresh child boundaries immediately, received ${JSON.stringify(getChildBoundaryLengths())}.`);
        }
        """
    )
    script_path.write_text(script_body, encoding="utf-8")
    return script_path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_tree_mutations_refresh_child_boundaries_without_switching_cell(
    tmp_path: Path,
) -> None:
    script_path = _write_tree_mutation_sync_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The tree mutation sync runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_benchmark_mode_keeps_temporary_schemes_session_local_and_promotes_active_one_on_exit(
    tmp_path: Path,
) -> None:
    script_path = _write_benchmark_mode_runtime_regression_script(tmp_path)

    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The benchmark mode runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_planner_visible_design_changes_refresh_auto_paths_immediately(
    tmp_path: Path,
) -> None:
    script_path = _write_planner_auto_paths_immediate_refresh_runtime_regression_script(
        tmp_path
    )

    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The planner immediate auto-path refresh runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_benchmark_compare_exports_csv_text_and_latex(
    tmp_path: Path,
) -> None:
    script_path = _write_benchmark_compare_export_runtime_regression_script(tmp_path)

    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The benchmark compare export runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_manual_contraction_anchor_follows_the_second_selected_tensor(
    tmp_path: Path,
) -> None:
    script_path = _write_manual_contraction_anchor_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The manual-contraction anchor runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_engine_picker_uses_the_requested_display_order(tmp_path: Path) -> None:
    script_path = _write_engine_order_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The engine-order runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_tensor_index_move_refreshes_properties_menu_order_and_disclosure(
    tmp_path: Path,
) -> None:
    script_path = _write_tensor_index_move_properties_runtime_regression_script(
        tmp_path
    )
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The tensor-index move properties regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_sidebar_can_be_resized_and_keeps_custom_width(tmp_path: Path) -> None:
    script_path = _write_sidebar_resize_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The sidebar resize runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_shift_m_hides_the_minimap_without_recursive_shortcut_calls(
    tmp_path: Path,
) -> None:
    script_path = _write_minimap_shortcut_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The minimap shortcut runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_svg_export_escapes_xml_attributes_and_keeps_index_labels_clean(
    tmp_path: Path,
) -> None:
    script_path = _write_svg_export_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    stdout = completed_process.stdout.decode("utf-8")
    stderr = completed_process.stderr.decode("utf-8")

    assert completed_process.returncode == 0, (
        "The SVG export runtime regression script failed.\n"
        f"STDOUT:\n{stdout}\n"
        f"STDERR:\n{stderr}"
    )
    assert 'font-family=""' not in stdout
    assert "&quot;Segoe UI Variable Text&quot;" in stdout
    assert "right \u00b7 3" in stdout
    ET.fromstring(stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_planner_auto_shortcuts_keep_ctrl_a_for_canvas_tensor_selection(
    tmp_path: Path,
) -> None:
    script_path = _write_planner_auto_shortcut_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The planner auto-shortcut runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_mode_and_template_shortcuts_dispatch_the_requested_actions(
    tmp_path: Path,
) -> None:
    script_path = _write_mode_and_template_shortcut_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The mode/template shortcut runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_hyperedge_shortcut_and_index_selection_context_menu_share_creation_logic(
    tmp_path: Path,
) -> None:
    script_path = _write_hyperedge_shortcut_and_context_menu_runtime_regression_script(
        tmp_path
    )
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The hyperedge shortcut/context-menu runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_multi_index_dimension_edits_apply_from_selection_and_context_menu(
    tmp_path: Path,
) -> None:
    script_path = _write_multi_index_dimension_batch_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The multi-index dimension batch-edit runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_graph_model_layers_open_ports_below_front_tensors(
    tmp_path: Path,
) -> None:
    script_path = _write_port_layering_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The port layering runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_copy_shortcut_prefers_native_text_selection_over_graph_copy(
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "copy_shortcut_native_selection_runtime.mjs"
    script_path.write_text(
        textwrap.dedent(
            f"""
            import {{ pathToFileURL }} from "node:url";

            const shortcutsUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "interactions" / "interactionsShortcuts.js")!r}).href;
            const {{ createInteractionShortcutBindings }} = await import(shortcutsUrl);

            let copiedSubgraphCount = 0;
            let selectedText = "Copy this label";
            let selectedTensorIds = [];
            let selectionNode = null;

            function createElement(name, parent = null) {{
              return {{
                name,
                parentNode: parent,
                contains(node) {{
                  let current = node;
                  while (current) {{
                    if (current === this) {{
                      return true;
                    }}
                    current = current.parentNode || null;
                  }}
                  return false;
                }},
              }};
            }}

            const workspace = createElement("workspace");
            const canvasShell = createElement("canvas-shell", workspace);
            const sidebar = createElement("sidebar", workspace);
            const canvasTextNode = createElement("canvas-text", canvasShell);
            const sidebarTextNode = createElement("sidebar-text", sidebar);

            function createEvent() {{
              return {{
                key: "c",
                altKey: false,
                ctrlKey: true,
                metaKey: false,
                shiftKey: false,
                target: {{ tagName: "DIV" }},
                preventDefaultCalls: 0,
                preventDefault() {{
                  this.preventDefaultCalls += 1;
                }},
              }};
            }}

            const bindings = createInteractionShortcutBindings({{
              ctx: {{
                getSelectedIdsByKind(kind) {{
                  return kind === "tensor" ? [...selectedTensorIds] : [];
                }},
                isTextInput() {{
                  return false;
                }},
                document: {{
                  activeElement: null,
                }},
                window: {{
                  getSelection() {{
                    return {{
                      anchorNode: selectionNode,
                      focusNode: selectionNode,
                      toString() {{
                        return selectedText;
                      }},
                    }};
                  }},
                }},
              }},
              state: {{}},
              dom: {{
                canvasShell,
                engineSelect: {{ value: "", options: [] }},
                generatedCode: {{}},
                loadInput: {{ click() {{}} }},
              }},
              runtime: {{}},
              shortcutActions: {{
                copySelectedSubgraphToClipboard() {{
                  copiedSubgraphCount += 1;
                }},
              }},
            }});

            selectedTensorIds = ["tensor_a"];
            selectionNode = canvasTextNode;
            const nativeCopyEvent = createEvent();
            bindings.handleKeydown(nativeCopyEvent);
            if (nativeCopyEvent.preventDefaultCalls !== 1) {{
              throw new Error("Ctrl+C should keep copying the selected tensor subgraph when the selection lives inside the drawing area.");
            }}
            if (copiedSubgraphCount !== 1) {{
              throw new Error("Ctrl+C should copy the selected tensor subgraph when the selected text is inside the drawing area.");
            }}

            selectionNode = sidebarTextNode;
            const tensorCopyEvent = createEvent();
            bindings.handleKeydown(tensorCopyEvent);
            if (tensorCopyEvent.preventDefaultCalls !== 0) {{
              throw new Error("Ctrl+C should preserve the native copy shortcut when the selected text is outside the drawing area.");
            }}
            if (copiedSubgraphCount !== 1) {{
              throw new Error(`Expected outside-canvas text selection to avoid extra tensor copies, received ${{copiedSubgraphCount}}.`);
            }}
            """
        ),
        encoding="utf-8",
    )
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The copy-shortcut native text selection regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_alt_arrow_navigation_and_arrow_nudging_follow_mode_and_selection_rules(
    tmp_path: Path,
) -> None:
    script_path = _write_keyboard_navigation_and_nudge_runtime_regression_script(
        tmp_path
    )
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The keyboard navigation and nudge runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_shift_only_shortcuts_ignore_extra_modifiers(tmp_path: Path) -> None:
    script_path = _write_shift_only_shortcut_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The shift-only shortcut runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_ctrl_and_cmd_additive_selection_match_shift_for_canvas_interactions(
    tmp_path: Path,
) -> None:
    script_path = _write_additive_selection_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The additive selection runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_metadata_properties_edits_preserve_reserved_keys_and_skip_graph_rerenders(
    tmp_path: Path,
) -> None:
    script_path = _write_metadata_properties_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The metadata-properties frontend runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_metadata_filters_are_local_and_classify_entities_for_highlighting(
    tmp_path: Path,
) -> None:
    script_path = _write_metadata_filter_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The metadata-filter frontend runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


def _write_utility_runtime_contract_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "utility_runtime_contract.mjs"
    _copy_js_modules(tmp_path, _UTILITY_RUNTIME_DEPENDENCY_MODULES)

    script_path.write_text(
        textwrap.dedent(
            """
            import { pathToFileURL } from "node:url";

            function createClassList() {
              const names = new Set();
              return {
                add(name) {
                  names.add(name);
                },
                remove(name) {
                  names.delete(name);
                },
                toggle(name, force) {
                  if (force === true) {
                    names.add(name);
                    return true;
                  }
                  if (force === false) {
                    names.delete(name);
                    return false;
                  }
                  if (names.has(name)) {
                    names.delete(name);
                    return false;
                  }
                  names.add(name);
                  return true;
                },
                contains(name) {
                  return names.has(name);
                },
              };
            }

            function createStyleObject() {
              return {
                values: {},
                setProperty(name, value) {
                  this.values[name] = value;
                },
                getPropertyValue(name) {
                  return this.values[name] || "";
                },
              };
            }

            function createButton(rect = { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }) {
              return {
                disabled: false,
                hidden: false,
                classList: createClassList(),
                style: createStyleObject(),
                attributes: {},
                dataset: {},
                focusCalls: 0,
                setAttribute(name, value) {
                  this.attributes[name] = String(value);
                },
                getAttribute(name) {
                  return this.attributes[name] ?? null;
                },
                focus() {
                  this.focusCalls += 1;
                },
                getBoundingClientRect() {
                  return rect;
                },
              };
            }

            const baseUrl = new URL("./", import.meta.url);
            const [stateModule, utilitiesModule, baseModule, geometryModule, gridPeriodicModule, layoutModule, linearPeriodicModule, specModule, uiModule] =
              await Promise.all([
                import(new URL("./state.runtime.mjs", baseUrl).href),
                import(new URL("./utilities.runtime.mjs", baseUrl).href),
                import(new URL("./utilitiesBase.js", baseUrl).href),
                import(new URL("./utilitiesGeometry.js", baseUrl).href),
                import(new URL("./utilitiesGridPeriodic.js", baseUrl).href),
                import(new URL("./utilitiesLayout.js", baseUrl).href),
                import(new URL("./utilitiesLinearPeriodic.js", baseUrl).href),
                import(new URL("./utilitiesSpec.js", baseUrl).href),
                import(new URL("./utilitiesUi.js", baseUrl).href),
              ]);

            const { createInitialState } = stateModule;
            const { registerUtilities } = utilitiesModule;
            const { createUtilityBaseBindings } = baseModule;
            const { createUtilityGeometryBindings } = geometryModule;
            const { createUtilityGridPeriodicBindings } = gridPeriodicModule;
            const { createUtilityLayoutBindings } = layoutModule;
            const { createUtilityLinearPeriodicBindings } = linearPeriodicModule;
            const { createUtilitySpecBindings } = specModule;
            const { createUtilityUiBindings } = uiModule;

            const requiredFactories = [
              createUtilityBaseBindings,
              createUtilityGeometryBindings,
              createUtilityLayoutBindings,
              createUtilityLinearPeriodicBindings,
              createUtilitySpecBindings,
              createUtilityUiBindings,
            ];
            if (requiredFactories.some((candidate) => typeof candidate !== "function")) {
              throw new Error("One or more utility helper factories were not exported.");
            }

            const primaryToolbarGroup = createButton();
            const primaryToolbarDivider = createButton();
            primaryToolbarGroup.nextElementSibling = primaryToolbarDivider;
            const templateToolbarGroup = createButton();
            const templateSelectField = createButton();
            templateSelectField.parentElement = templateToolbarGroup;
            const templateSettingsShell = createButton();
            templateSettingsShell.parentElement = templateToolbarGroup;
            const reflowLayoutShell = createButton();
            reflowLayoutShell.parentElement = templateToolbarGroup;

            const ctx = {
              state: createInitialState(),
              constants: {
                TENSOR_WIDTH: 140,
                TENSOR_HEIGHT: 84,
                MIN_TENSOR_WIDTH: 96,
                MIN_TENSOR_HEIGHT: 60,
                INDEX_RADIUS: 10,
                INDEX_PADDING: 6,
                NOTE_WIDTH: 220,
                NOTE_HEIGHT: 120,
                NOTE_MIN_WIDTH: 120,
                NOTE_MIN_HEIGHT: 90,
                HISTORY_LIMIT: 100,
                REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
                DEFAULT_INDEX_SLOTS: [
                  { x: -38, y: 0 },
                  { x: 38, y: 0 },
                  { x: 0, y: -24 },
                  { x: 0, y: 24 },
                ],
              },
              dom: {
                workspace: {},
                statusMessage: {
                  textContent: "",
                  classList: createClassList(),
                },
                propertiesPanel: { innerHTML: "" },
                generatedCode: { value: "" },
                generatedCodeView: { textContent: "", dataset: {} },
                generatedCodeModalView: { textContent: "", dataset: {} },
                engineSelect: { options: [], value: "tensornetwork" },
                collectionFormatSelect: { options: [], value: "list" },
                exportFormatSelect: { value: "py" },
                fileMenuButton: createButton({
                  left: 24,
                  top: 8,
                  right: 68,
                  bottom: 34,
                  width: 44,
                  height: 26,
                }),
                fileMenuPanel: createButton({
                  left: 0,
                  top: 0,
                  right: 240,
                  bottom: 220,
                  width: 240,
                  height: 220,
                }),
                themeMenuButton: createButton({
                  left: 76,
                  top: 8,
                  right: 136,
                  bottom: 34,
                  width: 60,
                  height: 26,
                }),
                themeMenuPanel: createButton({
                  left: 0,
                  top: 0,
                  right: 240,
                  bottom: 260,
                  width: 240,
                  height: 260,
                }),
                modesMenuButton: createButton(),
                modesMenuPanel: createButton(),
                templatesMenuButton: createButton(),
                templatesMenuPanel: createButton(),
                helpMenuButton: createButton(),
                helpMenuPanel: createButton(),
                themeDarkMenuItem: createButton(),
                themeLightMenuItem: createButton(),
                themeContrastMenuItem: createButton(),
                themeColorblindMenuItem: createButton(),
                themeShinyMenuItem: createButton(),
                addNoteButton: {
                  ...createButton(),
                  parentElement: primaryToolbarGroup,
                },
                connectButton: {
                  ...createButton(),
                  parentElement: primaryToolbarGroup,
                },
                loadInput: {},
                undoButton: createButton(),
                redoButton: createButton(),
                exportButton: createButton(),
                toggleLinearPeriodicButton: createButton(),
                benchmarkModeMenuItem: createButton(),
                toolbarModeControls: createButton(),
                linearPeriodicPreviousCellButton: createButton(),
                linearPeriodicCellLabel: { textContent: "" },
                gridPeriodicUpCellButton: createButton(),
                gridPeriodicDownCellButton: createButton(),
                linearPeriodicNextCellButton: createButton(),
                copyCodeButton: createButton(),
                expandGeneratedCodeButton: createButton(),
                generatedCodeModal: {
                  ...createButton(),
                  classList: createClassList(),
                },
                generatedCodeModalBackdrop: createButton(),
                generatedCodeModalCloseButton: createButton(),
                benchmarkSchemeNameInput: createButton(),
                benchmarkCompareButton: createButton(),
                templateSelect: {
                  value: "",
                  disabled: false,
                  hidden: false,
                  parentElement: templateSelectField,
                },
                templateParameterPanel: { hidden: true },
                templateGraphSizeLabel: { textContent: "" },
                templateGraphSizeInput: { value: "2", min: "1" },
                templateBondDimensionInput: { value: "3", min: "1" },
                templatePhysicalDimensionInput: { value: "2", min: "1" },
                templateSettingsButton: {
                  ...createButton({
                    left: 720,
                    top: 132,
                    right: 756,
                    bottom: 164,
                    width: 36,
                    height: 32,
                  }),
                  parentElement: templateSettingsShell,
                },
                templateSettingsPopover: createButton({
                  left: 0,
                  top: 0,
                  right: 280,
                  bottom: 220,
                  width: 280,
                  height: 220,
                }),
                reflowLayoutPopover: createButton({
                  left: 0,
                  top: 0,
                  right: 360,
                  bottom: 280,
                  width: 360,
                  height: 280,
                }),
                insertTemplateButton: {
                  ...createButton(),
                  parentElement: templateToolbarGroup,
                },
                reflowImportedButton: {
                  ...createButton({
                    left: 812,
                    top: 132,
                    right: 876,
                    bottom: 164,
                    width: 64,
                    height: 32,
                  }),
                  parentElement: reflowLayoutShell,
                },
                reflowAlignLeftButton: createButton(),
                reflowAlignRightButton: createButton(),
                reflowAlignTopButton: createButton(),
                reflowAlignMiddleButton: createButton(),
                reflowAlignBottomButton: createButton(),
                reflowIndicesLeftButton: createButton(),
                reflowIndicesRightButton: createButton(),
                reflowIndicesTopButton: createButton(),
                reflowIndicesResetButton: createButton(),
                reflowIndicesBottomButton: createButton(),
                reflowArrangeChainButton: createButton(),
                reflowArrangeTreeButton: createButton(),
                reflowArrangeGridButton: createButton(),
                reflowAutoLayoutButton: createButton(),
                reflowDistributeHorizontalButton: createButton(),
                reflowDistributeVerticalButton: createButton(),
                reflowSnapGridButton: createButton(),
                insertSubnetworkButton: createButton(),
                createGroupButton: {
                  ...createButton(),
                  parentElement: primaryToolbarGroup,
                },
                helpButton: createButton(),
                helpModal: { classList: createClassList() },
                helpBackdrop: createButton(),
                helpCloseButton: createButton(),
                helpSharedHeader: { hidden: false },
                helpTitle: { textContent: "", hidden: false },
                helpNote: { textContent: "", hidden: false },
                helpInfoSection: { hidden: false },
                helpShortcutsSection: { hidden: true },
                helpAboutSection: { hidden: true },
                aboutRepositoryLink: {
                  textContent: "",
                  href: "",
                },
                aboutVersion: { textContent: "" },
                aboutLicense: { textContent: "" },
                aboutAuthor: { textContent: "" },
                canvasShell: {
                  getBoundingClientRect() {
                    return { left: 0, top: 0, width: 1000, height: 800 };
                  },
                },
                groupLayer: {},
                resizeLayer: {},
                notesLayer: {},
                selectionBox: {
                  classList: createClassList(),
                  style: {},
                },
                minimapCanvas: {},
                sidebar: {},
                plannerPanel: {},
                generateButton: createButton(),
                codeGenerationWarning: {
                  textContent: "",
                  title: "",
                  hidden: true,
                },
                subnetworkLoadInput: {
                  value: "",
                  click() {},
                },
              },
              apiGet: async () => null,
              apiPost: async () => null,
              window: {
                structuredClone: globalThis.structuredClone,
                crypto: globalThis.crypto,
                setTimeout,
                clearTimeout,
                confirm: () => true,
                innerWidth: 1280,
                innerHeight: 720,
                localStorage: {
                  values: {},
                  getItem(name) {
                    return Object.prototype.hasOwnProperty.call(this.values, name)
                      ? this.values[name]
                      : null;
                  },
                  setItem(name, value) {
                    this.values[name] = String(value);
                  },
                },
                Prism: {
                  highlightElement(element) {
                    if (element?.dataset) {
                      element.dataset.highlighted = "true";
                    }
                  },
                },
              },
              document: {
                activeElement: null,
                documentElement: {
                  dataset: {},
                  style: {},
                },
                createElement() {
                  return {
                    value: "",
                    textContent: "",
                    selected: false,
                    appendChild() {},
                    click() {},
                  };
                },
                querySelectorAll() {
                  return [];
                },
              },
              cytoscape: null,
              getSelectedIdsByKind(kind) {
                return kind === "tensor" ? [...ctx.state.selectionIds] : [];
              },
              getSelectedEntries() {
                return [];
              },
              findTensorById(tensorId) {
                return (
                  ctx.state.spec?.tensors?.find((tensor) => tensor.id === tensorId) || null
                );
              },
              renderOverlayDecorations() {},
              renderMinimap() {},
              renderPlanner() {},
              refreshContractionAnalysis() {},
              renderSidebarTabs() {},
              render() {},
              repairContractionPlan() {},
              clearGeneratedCodePreview() {},
            };

            const runtime = {};
            const env = {
              ctx,
              state: ctx.state,
              constants: ctx.constants,
              dom: ctx.dom,
              runtime,
            };
            Object.assign(runtime, createUtilityBaseBindings(env));
            if (runtime.sanitizeFilename("Tensor Network!") !== "tensor-network") {
              throw new Error("sanitizeFilename no longer normalizes names as expected.");
            }
            Object.assign(runtime, createUtilityGeometryBindings(env));
            if (runtime.formatColorHex({ red: 1, green: 35, blue: 255 }) !== "#0123ff") {
              throw new Error("formatColorHex returned an unexpected value.");
            }
            Object.assign(runtime, createUtilityLayoutBindings(env));
            Object.assign(runtime, createUtilitySpecBindings(env));
            Object.assign(runtime, createUtilityLinearPeriodicBindings(env));
            Object.assign(runtime, createUtilityUiBindings(env));
            if (
              runtime.formatIssues([
                { message: "first" },
                { message: "second" },
                { message: "third" },
                { message: "fourth" },
              ]) !== "first second third"
            ) {
              throw new Error("formatIssues should keep the first three messages.");
            }
            ctx.state.spec = {
              id: "network_demo",
              name: "demo",
              tensors: [
                {
                  id: "tensor_a",
                  name: "A",
                  position: { x: 100, y: 100 },
                  size: { width: 140, height: 84 },
                  indices: [
                    {
                      id: "tensor_a_i",
                      name: "i",
                      dimension: 2,
                      offset: { x: 0, y: 0 },
                      metadata: {},
                    },
                  ],
                  metadata: {},
                },
              ],
              edges: [],
              groups: [],
              notes: [],
              contraction_plan: null,
              metadata: {},
            };
            ctx.dom.templateSelect.value = "mps";
            ctx.state.availableTemplates = ["mps"];
            ctx.state.selectionIds = ["tensor_a"];
            ctx.state.primarySelectionId = "tensor_a";
            runtime.updateToolbarState();
            if (ctx.dom.reflowImportedButton.disabled) {
              throw new Error("Reflow should stay enabled when one tensor is selected so indices can be reflowed.");
            }
            if (!ctx.dom.copyCodeButton.disabled || !ctx.dom.expandGeneratedCodeButton.disabled) {
              throw new Error("Generated-code actions should stay disabled until code exists.");
            }
            if (
              ctx.dom.reflowImportedButton.dataset.shortcutDescription
              !== "Reflow indices for the selected tensor."
            ) {
              throw new Error(
                `Expected the single-selection Reflow tooltip to mention indices, received ${ctx.dom.reflowImportedButton.dataset.shortcutDescription}.`
              );
            }
            ctx.state.selectionIds = [];
            runtime.updateToolbarState();
            if (!ctx.dom.reflowImportedButton.disabled) {
              throw new Error("Reflow should disable again when no tensor is selected.");
            }
            ctx.state.spec.tensors.push({
              id: "tensor_b",
              name: "B",
              position: { x: 320, y: 220 },
              size: { width: 140, height: 84 },
              indices: [],
              metadata: {},
            });
            runtime.updateToolbarState();
            if (ctx.dom.reflowImportedButton.disabled) {
              throw new Error("Reflow should stay enabled when the graph has at least two tensors, even with no active tensor selection.");
            }
            if (
              ctx.dom.reflowImportedButton.dataset.shortcutDescription
              !== "Open layout tools. Auto layout will arrange the whole graph."
            ) {
              throw new Error(
                `Expected the empty-selection Reflow tooltip to explain whole-graph auto layout, received ${ctx.dom.reflowImportedButton.dataset.shortcutDescription}.`
              );
            }
            if (ctx.dom.reflowAutoLayoutButton.disabled) {
              throw new Error("Auto layout should stay enabled when the whole graph can be arranged.");
            }
            ctx.state.selectionIds = ["tensor_a"];
            runtime.isBenchmarkMode = () => true;
            runtime.getBenchmarkSession = () => ({
              activePosition: 0,
              schemes: [{ name: "Scheme A" }],
            });
            runtime.getActiveBenchmarkScheme = () => null;
            runtime.canOpenBenchmarkCompare = () => true;
            runtime.getBenchmarkNextButtonLabel = () => ">";
            runtime.getBenchmarkBaseLabel = () => "Tensor network";
            runtime.updateToolbarState();
            if (!ctx.dom.linearPeriodicCellLabel.hidden) {
              throw new Error("The benchmark base view should not show the redundant tensor-network label.");
            }
            runtime.isBenchmarkMode = () => true;
            runtime.getBenchmarkSession = () => ({
              activePosition: 1,
              schemes: [{ name: "Scheme A" }],
            });
            runtime.getActiveBenchmarkScheme = () => ({ name: "Scheme A" });
            runtime.canOpenBenchmarkCompare = () => true;
            runtime.getBenchmarkNextButtonLabel = () => ">";
            runtime.getBenchmarkBaseLabel = () => "Tensor network";
            ctx.state.isTemplateSettingsOpen = true;
            ctx.state.isReflowLayoutOpen = true;
            runtime.updateToolbarState();
            if (!templateToolbarGroup.hidden) {
              throw new Error("The template toolbar group should disappear while viewing a benchmark scheme.");
            }
            if (!primaryToolbarGroup.hidden || !primaryToolbarDivider.hidden) {
              throw new Error("The primary toolbar controls should disappear while viewing a benchmark scheme.");
            }
            if (!ctx.dom.templateSelect.disabled || !ctx.dom.templateSelect.parentElement.hidden) {
              throw new Error("Template selection should disappear while viewing a benchmark scheme.");
            }
            if (
              !ctx.dom.templateSettingsButton.disabled
              || !ctx.dom.templateSettingsButton.parentElement.hidden
            ) {
              throw new Error("Template settings should disappear while viewing a benchmark scheme.");
            }
            if (!ctx.dom.insertTemplateButton.hidden || !ctx.dom.insertTemplateButton.disabled) {
              throw new Error("Insert template should disappear while viewing a benchmark scheme.");
            }
            if (
              !ctx.dom.reflowImportedButton.disabled
              || !ctx.dom.reflowImportedButton.parentElement.hidden
            ) {
              throw new Error("Reflow should disappear while viewing a benchmark scheme.");
            }
            if (ctx.dom.benchmarkSchemeNameInput.hidden || ctx.dom.benchmarkSchemeNameInput.disabled) {
              throw new Error("The benchmark scheme name should stay available on saved schemes.");
            }
            if (ctx.dom.benchmarkCompareButton.hidden || ctx.dom.benchmarkCompareButton.disabled) {
              throw new Error("Compare should stay available on saved benchmark schemes.");
            }
            if (!ctx.dom.linearPeriodicCellLabel.hidden) {
              throw new Error("The base tensor-network label should be hidden while viewing a saved benchmark scheme.");
            }
            if (ctx.state.isTemplateSettingsOpen || ctx.state.isReflowLayoutOpen) {
              throw new Error("Benchmark schemes should close template and reflow popovers.");
            }
            runtime.renderGeneratedCodePreview("result = 1");
            if (ctx.dom.generatedCode.value !== "result = 1") {
              throw new Error(`Expected the raw generated code buffer to stay in sync, received ${ctx.dom.generatedCode.value}.`);
            }
            if (ctx.dom.generatedCodeView.textContent !== "result = 1") {
              throw new Error(`Expected the inline generated-code preview to stay in sync, received ${ctx.dom.generatedCodeView.textContent}.`);
            }
            if (ctx.dom.generatedCodeModalView.textContent !== "result = 1") {
              throw new Error(`Expected the full-size generated-code preview to stay in sync, received ${ctx.dom.generatedCodeModalView.textContent}.`);
            }
            if (ctx.dom.copyCodeButton.disabled || ctx.dom.expandGeneratedCodeButton.disabled) {
              throw new Error("Generated-code actions should enable once code exists.");
            }
            runtime.toggleGeneratedCodeModal(true);
            if (!ctx.state.isGeneratedCodeModalOpen || ctx.dom.generatedCodeModal.hidden) {
              throw new Error("Expected the generated-code modal to open when requested.");
            }
            if (ctx.dom.generatedCodeModalCloseButton.focusCalls !== 1) {
              throw new Error("Opening the generated-code modal should focus its close button.");
            }
            runtime.renderGeneratedCodePreview("");
            if (ctx.state.isGeneratedCodeModalOpen || !ctx.dom.generatedCodeModal.hidden) {
              throw new Error("Clearing generated code should close the full-size generated-code modal.");
            }
            if (!ctx.dom.copyCodeButton.disabled || !ctx.dom.expandGeneratedCodeButton.disabled) {
              throw new Error("Generated-code actions should disable again when the preview is cleared.");
            }
            runtime.isBenchmarkMode = () => false;
            runtime.updateToolbarState();
            if (templateToolbarGroup.hidden) {
              throw new Error("The template toolbar group should reappear when leaving the benchmark scheme view.");
            }
            if (primaryToolbarGroup.hidden || primaryToolbarDivider.hidden) {
              throw new Error("The primary toolbar controls should reappear when leaving the benchmark scheme view.");
            }
            runtime.openToolbarMenu("file");
            if (ctx.dom.fileMenuPanel.hidden !== false) {
              throw new Error("Opening a toolbar menu should reveal the floating menu panel.");
            }
            if (
              ctx.dom.fileMenuPanel.style.getPropertyValue("--toolbar-menu-left") !== "24px"
            ) {
              throw new Error(
                `Expected the floating toolbar menu to anchor to its button, received ${ctx.dom.fileMenuPanel.style.getPropertyValue("--toolbar-menu-left")}.`
              );
            }
            if (
              ctx.dom.fileMenuPanel.style.getPropertyValue("--toolbar-menu-top") !== "38px"
            ) {
              throw new Error(
                `Expected the floating toolbar menu to sit below its button, received ${ctx.dom.fileMenuPanel.style.getPropertyValue("--toolbar-menu-top")}.`
              );
            }
            runtime.openToolbarMenu("theme");
            if (ctx.dom.themeMenuPanel.hidden !== false) {
              throw new Error("Opening the theme menu should reveal its floating menu panel.");
            }
            if (
              ctx.dom.themeMenuPanel.style.getPropertyValue("--toolbar-menu-left") !== "76px"
            ) {
              throw new Error(
                `Expected the theme menu to anchor to its button, received ${ctx.dom.themeMenuPanel.style.getPropertyValue("--toolbar-menu-left")}.`
              );
            }
            runtime.setEditorTheme("light", { announce: false });
            if (ctx.state.selectedTheme !== "light") {
              throw new Error(`Expected the selected theme to update, received ${ctx.state.selectedTheme}.`);
            }
            if (ctx.document.documentElement.dataset.theme !== "light") {
              throw new Error(
                `Expected the root dataset theme to update, received ${ctx.document.documentElement.dataset.theme}.`
              );
            }
            if (ctx.document.documentElement.style.colorScheme !== "light") {
              throw new Error(
                `Expected the root color-scheme to update, received ${ctx.document.documentElement.style.colorScheme}.`
              );
            }
            if (ctx.window.localStorage.values["tensor-network-editor.theme"] !== "light") {
              throw new Error(
                `Expected the theme preference to persist, received ${ctx.window.localStorage.values["tensor-network-editor.theme"]}.`
              );
            }
            if (
              ctx.dom.themeLightMenuItem.getAttribute("aria-checked") !== "true"
              || !ctx.dom.themeLightMenuItem.classList.contains("is-checked")
            ) {
              throw new Error("The light theme menu item should be marked as checked.");
            }
            if (ctx.dom.themeDarkMenuItem.getAttribute("aria-checked") !== "false") {
              throw new Error("The dark theme menu item should be unchecked after changing the theme.");
            }
            runtime.toggleTemplateSettingsPopover();
            if (ctx.dom.templateSettingsPopover.hidden !== false) {
              throw new Error("Opening the template settings popover should reveal its floating overlay.");
            }
            if (
              ctx.dom.templateSettingsPopover.style.getPropertyValue("--template-settings-popover-left")
              !== "476px"
            ) {
              throw new Error(
                `Expected the template settings popover to anchor to the three-dot button, received ${ctx.dom.templateSettingsPopover.style.getPropertyValue("--template-settings-popover-left")}.`
              );
            }
            if (
              ctx.dom.templateSettingsPopover.style.getPropertyValue("--template-settings-popover-top")
              !== "168px"
            ) {
              throw new Error(
                `Expected the template settings popover to sit below the three-dot button, received ${ctx.dom.templateSettingsPopover.style.getPropertyValue("--template-settings-popover-top")}.`
              );
            }
            runtime.toggleReflowLayoutPopover();
            if (ctx.dom.reflowLayoutPopover.hidden !== false) {
              throw new Error("Opening the Reflow popover should reveal its floating overlay.");
            }
            if (
              ctx.dom.reflowLayoutPopover.style.getPropertyValue("--reflow-layout-popover-left")
              !== "516px"
            ) {
              throw new Error(
                `Expected the Reflow popover to anchor to the Reflow button, received ${ctx.dom.reflowLayoutPopover.style.getPropertyValue("--reflow-layout-popover-left")}.`
              );
            }
            if (
              ctx.dom.reflowLayoutPopover.style.getPropertyValue("--reflow-layout-popover-top")
              !== "168px"
            ) {
              throw new Error(
                `Expected the Reflow popover to sit below the Reflow button, received ${ctx.dom.reflowLayoutPopover.style.getPropertyValue("--reflow-layout-popover-top")}.`
              );
            }
            runtime.openHelpSection("shortcuts");
            if (
              ctx.dom.helpSharedHeader.hidden !== false
              || ctx.dom.helpTitle.hidden !== false
            ) {
              throw new Error("Shortcuts should keep the shared help header visible.");
            }
            if (ctx.dom.helpTitle.textContent !== "Shortcuts" || ctx.dom.helpNote.hidden !== true) {
              throw new Error("Shortcuts should show its title and hide the help note.");
            }
            if (ctx.dom.helpShortcutsSection.hidden !== false || ctx.dom.helpInfoSection.hidden !== true) {
              throw new Error("Shortcuts should only show the shortcuts section.");
            }
            runtime.openHelpSection("about");
            if (
              ctx.dom.helpSharedHeader.hidden !== false
              || ctx.dom.helpTitle.hidden !== false
            ) {
              throw new Error("About should keep the shared help header visible.");
            }
            if (ctx.dom.helpTitle.textContent !== "About" || ctx.dom.helpNote.hidden !== true) {
              throw new Error("About should show its title and hide the help note.");
            }
            if (ctx.dom.helpAboutSection.hidden !== false || ctx.dom.helpShortcutsSection.hidden !== true) {
              throw new Error("About should only show the about section.");
            }
            runtime.openHelpSection("info");
            if (
              ctx.dom.helpSharedHeader.hidden !== false
              || ctx.dom.helpTitle.hidden !== false
              || ctx.dom.helpNote.hidden !== false
            ) {
              throw new Error("Info should keep the shared help header visible.");
            }
            if (ctx.dom.helpTitle.textContent !== "Info") {
              throw new Error("Info should set the shared help title to Info.");
            }
            if (ctx.dom.helpInfoSection.hidden !== false || ctx.dom.helpAboutSection.hidden !== true) {
              throw new Error("Info should only show the information section.");
            }

            registerUtilities(ctx);
            const requiredCtxBindings = [
              "serializeCurrentSpec",
              "toggleLinearPeriodicMode",
              "computeDesignBounds",
              "renderGeneratedCodePreview",
              "setStatus",
              "sanitizeFilename",
            ];
            for (const bindingName of requiredCtxBindings) {
              if (typeof ctx[bindingName] !== "function") {
                throw new Error(`registerUtilities did not expose ${bindingName}.`);
              }
            }
            """
        ),
        encoding="utf-8",
    )
    return script_path


def _write_interaction_runtime_contract_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "interaction_runtime_contract.mjs"
    _copy_js_modules(tmp_path, _INTERACTION_RUNTIME_CONTRACT_DEPENDENCY_MODULES)

    script_path.write_text(
        textwrap.dedent(
            """
            import { pathToFileURL } from "node:url";

            function createClassList() {
              return {
                add() {},
                remove() {},
                toggle() {},
              };
            }

            function createButton() {
              return {
                disabled: false,
                classList: createClassList(),
                click() {},
                focus() {},
              };
            }

            const baseUrl = new URL("./", import.meta.url);
            const [stateModule, interactionsModule, canvasModule, editorModule, sessionModule, shortcutsModule, selectorsModule, storeModule] =
              await Promise.all([
                import(new URL("./state.runtime.mjs", baseUrl).href),
                import(new URL("./interactions.runtime.mjs", baseUrl).href),
                import(new URL("./interactionsCanvas.js", baseUrl).href),
                import(new URL("./interactionsEditor.js", baseUrl).href),
                import(new URL("./interactionsSession.js", baseUrl).href),
                import(new URL("./interactionsShortcuts.js", baseUrl).href),
                import(new URL("./state/editorSelectors.js", baseUrl).href),
                import(new URL("./state/editorStore.js", baseUrl).href),
              ]);

            const { createInitialState } = stateModule;
            const { registerInteractions } = interactionsModule;
            const { createInteractionCanvasBindings } = canvasModule;
            const { createInteractionEditorBindings } = editorModule;
            const { createInteractionSessionBindings } = sessionModule;
            const { createInteractionShortcutBindings } = shortcutsModule;
            const { createEditorSelectors } = selectorsModule;
            const { createEditorStore } = storeModule;

            const requiredFactories = [
              createInteractionCanvasBindings,
              createInteractionEditorBindings,
              createInteractionSessionBindings,
              createInteractionShortcutBindings,
            ];
            if (requiredFactories.some((candidate) => typeof candidate !== "function")) {
              throw new Error("One or more interaction helper factories were not exported.");
            }

            const selectionBox = {
              classList: createClassList(),
              style: {},
            };
            let minimapDragCleared = 0;
            let minimapViewportUpdates = 0;
            const ctx = {
              state: createInitialState(),
              constants: {
                TENSOR_WIDTH: 140,
                TENSOR_HEIGHT: 84,
                MIN_TENSOR_WIDTH: 96,
                MIN_TENSOR_HEIGHT: 60,
                INDEX_RADIUS: 10,
                INDEX_PADDING: 6,
                HISTORY_LIMIT: 100,
                REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
                DEFAULT_INDEX_SLOTS: [
                  { x: -38, y: 0 },
                  { x: 38, y: 0 },
                  { x: 0, y: -24 },
                  { x: 0, y: 24 },
                ],
              },
              dom: {
                workspace: {},
                statusMessage: {
                  textContent: "",
                  classList: createClassList(),
                },
                propertiesPanel: {},
                generatedCode: { value: "" },
                generatedCodeView: { textContent: "", dataset: {} },
                engineSelect: { options: [], value: "tensornetwork" },
                collectionFormatSelect: { options: [], value: "list" },
                exportFormatSelect: { value: "py" },
                connectButton: createButton(),
                loadInput: createButton(),
                undoButton: createButton(),
                redoButton: createButton(),
                templateSelect: { value: "" },
                insertTemplateButton: createButton(),
                insertSubnetworkButton: createButton(),
                createGroupButton: createButton(),
                helpButton: createButton(),
                helpModal: { classList: createClassList() },
                helpBackdrop: createButton(),
                helpCloseButton: createButton(),
                canvasShell: {
                  getBoundingClientRect() {
                    return { left: 0, top: 0, width: 1000, height: 800 };
                  },
                },
                groupLayer: {},
                resizeLayer: {},
                selectionBox,
                minimapCanvas: {
                  classList: {
                    add() {},
                    remove() {
                      minimapDragCleared += 1;
                    },
                    toggle() {},
                  },
                },
                subnetworkLoadInput: {
                  value: "",
                  click() {},
                },
              },
              apiGet: async () => null,
              apiPost: async () => ({
                ok: true,
                engine: "tensornetwork",
                code: "import x\\nresult = 1\\n",
              }),
              window: {
                structuredClone: globalThis.structuredClone,
                crypto: globalThis.crypto,
                Prism: {
                  highlightElement(element) {
                    element.dataset.highlighted = "true";
                  },
                },
                setTimeout,
                clearTimeout,
                confirm: () => true,
                close() {},
              },
              document: {
                activeElement: null,
              },
              cytoscape: null,
              clamp: (value, min, max) => Math.min(max, Math.max(min, value)),
              clientPointToCanvasPoint(clientX, clientY) {
                return { x: clientX, y: clientY };
              },
              normalizedBox(startPoint, currentPoint) {
                return {
                  left: Math.min(startPoint.x, currentPoint.x),
                  top: Math.min(startPoint.y, currentPoint.y),
                  width: Math.abs(currentPoint.x - startPoint.x),
                  height: Math.abs(currentPoint.y - startPoint.y),
                };
              },
              boxesIntersect() {
                return false;
              },
              setSelection() {},
              render() {},
              renderOverlayDecorations() {},
              renderMinimap() {},
              renderPlanner() {},
              updateViewportFromMinimapClientPoint() {
                minimapViewportUpdates += 1;
              },
              updateToolbarState() {},
              setStatus() {},
              clearSelection() {},
              isTextInput() {
                return false;
              },
              stripImportLines(code) {
                return code.replace(/^import .*\\n/, "");
              },
              serializeCurrentSpec() {
                return {
                  schema_version: "1.0",
                  network: {
                    id: "network_demo",
                    name: "demo",
                    tensors: [],
                    groups: [],
                    edges: [],
                    notes: [],
                    contraction_plan: null,
                    metadata: {},
                  },
                };
              },
              formatIssues() {
                return "bad spec";
              },
              sanitizeFilename(value) {
                return value;
              },
              downloadBlob() {},
              removeNote() {},
              getSelectedEntries() {
                return [];
              },
              getSelectedIdsByKind() {
                return [];
              },
              findEdgeByIndexId() {
                return null;
              },
              findIndexOwner() {
                return null;
              },
              resolveConnectableIndexOwner() {
                return null;
              },
              isLinearPeriodicBoundaryTensor() {
                return false;
              },
              nextName(prefix) {
                return prefix;
              },
              makeId(prefix) {
                return `${prefix}_1`;
              },
              removeTensor() {},
              removeIndex() {},
              removeEdge() {},
              findEdgeById() {
                return null;
              },
              findTensorById() {
                return null;
              },
              tensorWidth() {
                return 140;
              },
              tensorHeight() {
                return 84;
              },
              createTensor(x, y) {
                return {
                  id: "tensor_1",
                  name: "T1",
                  position: { x, y },
                  size: { width: 140, height: 84 },
                  indices: [],
                  metadata: {},
                };
              },
              applyDesignChange(mutator) {
                mutator();
              },
              bringTensorToFront() {},
              reconcileTensorOrder() {},
              normalizeSpec(spec) {
                return spec;
              },
              syncCodeGenerationWarning() {},
              persistTemplateParametersFromControls() {
                return {};
              },
              formatTemplateLabel(value) {
                return value;
              },
              uniquifyImportedSpec(spec) {
                return spec;
              },
              translateImportedSpec(spec) {
                return spec;
              },
              viewportCenterPosition() {
                return { x: 500, y: 400 };
              },
              suggestTensorPosition(position) {
                return position;
              },
            };
            ctx.store = createEditorStore(ctx.state);
            ctx.selectors = createEditorSelectors({ store: ctx.store });
            ctx.services = {
              session: {
                async generateCode(payload) {
                  return {
                    ok: true,
                    engine: payload.engine,
                    code: "import x\\nresult = 1\\n",
                  };
                },
                async completeSession() {
                  return { ok: true };
                },
                async cancelSession() {
                  return { ok: true };
                },
                async validatePythonCode() {
                  return { ok: true, spec: { network: { tensors: [], edges: [], groups: [], notes: [], metadata: {} }, schema_version: "1.0" } };
                },
                async validateSerializedSpec(spec) {
                  return { ok: true, spec };
                },
                async buildTemplate() {
                  return {
                    ok: true,
                    spec: {
                      network: {
                        id: "template_demo",
                        name: "Template",
                        tensors: [],
                        edges: [],
                        groups: [],
                        notes: [],
                        metadata: {},
                      },
                    },
                  };
                },
              },
              templateCatalog: {
                async promoteTemplate() {
                  return { ok: true, templates: [], template_definitions: {}, selected_template: null, template_catalog_warnings: [] };
                },
                async renameTemplate() {
                  return { ok: true, templates: [], template_definitions: {}, selected_template: null, template_catalog_warnings: [] };
                },
                async deleteTemplate() {
                  return { ok: true, templates: [], template_definitions: {}, selected_template: null, template_catalog_warnings: [] };
                },
              },
              subnetwork: {
                async extractSubnetwork() {
                  return { ok: true, spec: { network: { id: "subnetwork", name: "subnetwork", tensors: [], edges: [], groups: [], notes: [], metadata: {} } } };
                },
                async prepareSubnetworkForInsert() {
                  return { ok: true, spec: { network: { id: "subnetwork", name: "subnetwork", tensors: [], edges: [], groups: [], notes: [], metadata: {} } } };
                },
              },
            };

            const runtime = {};
            const env = {
              ctx,
              state: ctx.state,
              constants: ctx.constants,
              dom: ctx.dom,
              runtime,
            };
            Object.assign(runtime, createInteractionCanvasBindings(env));
            runtime.startBoxSelection({ clientX: 10, clientY: 20, shiftKey: false });
            if (!ctx.state.boxSelection || ctx.state.boxSelection.start.x !== 10) {
              throw new Error("Canvas bindings no longer seed box selection correctly.");
            }
            ctx.state.boxSelection = null;

            const rightMouseEvent = {
              button: 2,
              shiftKey: false,
              clientX: 30,
              clientY: 40,
              preventDefaultCalls: 0,
              stopPropagationCalls: 0,
              preventDefault() {
                this.preventDefaultCalls += 1;
              },
              stopPropagation() {
                this.stopPropagationCalls += 1;
              },
              target: {
                closest() {
                  return null;
                },
              },
            };
            runtime.handleCanvasMouseDown(rightMouseEvent);
            if (!ctx.state.pendingBoxSelection || ctx.state.pendingBoxSelection.startPoint.x !== 30) {
              throw new Error("Right mouse down should still arm box selection for empty canvas drags.");
            }
            if (rightMouseEvent.preventDefaultCalls !== 0 || rightMouseEvent.stopPropagationCalls !== 0) {
              throw new Error("Right mouse down on the canvas should not swallow the Cytoscape context-menu path.");
            }

            ctx.state.pendingBoxSelection = null;
            runtime.handleCanvasMouseDown({
              button: 2,
              shiftKey: false,
              clientX: 70,
              clientY: 80,
              target: {
                closest(selector) {
                  return selector === ".minimap-shell" ? {} : null;
                },
              },
            });
            if (ctx.state.pendingBoxSelection) {
              throw new Error("Right mouse down inside the minimap should not arm box selection.");
            }

            ctx.state.minimapDrag = { active: true };
            runtime.handleGlobalMouseMove({
              clientX: 88,
              clientY: 99,
              buttons: 0,
            });
            if (ctx.state.minimapDrag !== null) {
              throw new Error("Mouse move without the primary button should clear a stale minimap drag.");
            }
            if (minimapDragCleared !== 1) {
              throw new Error(`Expected stale minimap drags to remove the dragging class once, received ${minimapDragCleared}.`);
            }
            if (minimapViewportUpdates !== 0) {
              throw new Error(`Expected stale minimap drags to stop before updating the viewport, received ${minimapViewportUpdates} updates.`);
            }

            let closeCanvasContextMenuCalls = 0;
            ctx.closeCanvasContextMenu = () => {
              closeCanvasContextMenuCalls += 1;
            };
            runtime.handleCanvasMouseDown({
              button: 0,
              clientX: 44,
              clientY: 55,
              target: {
                closest(selector) {
                  return selector === ".canvas-context-menu" ? {} : null;
                },
              },
            });
            if (closeCanvasContextMenuCalls !== 0) {
              throw new Error("Mouse interactions inside the canvas context menu should not close it.");
            }

            Object.assign(runtime, createInteractionShortcutBindings(env));
            Object.assign(runtime, createInteractionEditorBindings(env));
            Object.assign(
              runtime,
              createInteractionSessionBindings({
                ...env,
                store: ctx.store,
                selectors: ctx.selectors,
                services: ctx.services,
                sessionUi: {
                  async copyText() {},
                  downloadText() {},
                  downloadBlob() {},
                  requestFileText: async () => "",
                  openFilePicker() {},
                  schedule(callback) {
                    callback();
                  },
                  closeWindow() {},
                  promptText: () => null,
                  confirmAction: () => true,
                },
                sessionActions: {
                  ensureCodePanelVisible() {},
                  syncCodeGenerationWarning() {},
                  getTensorKrowchManualPlanIssueMessage() {
                    return "";
                  },
                  getSelectedTensorIds() {
                    return [];
                  },
                  findGroupById() {
                    return null;
                  },
                  isLinearPeriodicMode() {
                    return false;
                  },
                  syncGeneratedCodePreview: (code) => {
                    ctx.dom.generatedCode.value = code;
                    ctx.dom.generatedCodeView.textContent = code;
                    ctx.window.Prism.highlightElement(ctx.dom.generatedCodeView);
                  },
                  setStatus: (message, level) => ctx.setStatus(message, level),
                  serializeCurrentSpec: () => ctx.serializeCurrentSpec(),
                  formatIssues: () => ctx.formatIssues(),
                  stripImportLines: (code) => ctx.stripImportLines(code),
                  sanitizeFilename: (value) => ctx.sanitizeFilename(value),
                  resetDesignState() {},
                  downloadPngExport() {},
                  downloadSvgExport() {},
                  applyTemplateCatalogPayload() {},
                  normalizeSpec: (spec) => ctx.normalizeSpec(spec),
                  applyDesignChange: (mutate, options) =>
                    ctx.applyDesignChange(mutate, options),
                  bringTensorToFront: (tensorId) => ctx.bringTensorToFront(tensorId),
                  formatTemplateLabel: (value) => ctx.formatTemplateLabel(value),
                  getTemplateSource: (templateName) => ctx.getTemplateSource(templateName),
                  getTemplateSpec: (templateName) => ctx.getTemplateSpec(templateName),
                  listTemplateEntries: () => ctx.listTemplateEntries(),
                  hasTemplateDisplayName: (displayName, excludedTemplateName) =>
                    ctx.hasTemplateDisplayName(displayName, excludedTemplateName),
                  getNextSessionTemplateDisplayName: (baseDisplayName) =>
                    ctx.getNextSessionTemplateDisplayName(baseDisplayName),
                  addSessionTemplate: (payload) => ctx.addSessionTemplate(payload),
                  updateSessionTemplateDisplayNames: (updates) =>
                    ctx.updateSessionTemplateDisplayNames(updates),
                  removeSessionTemplate: (templateName) =>
                    ctx.removeSessionTemplate(templateName),
                  toggleTemplateManager: (forceOpen) =>
                    ctx.toggleTemplateManager(forceOpen),
                  syncTemplateManagerModalState: () =>
                    ctx.syncTemplateManagerModalState(),
                  setTemplateManagerValidationMessage: (message) =>
                    ctx.setTemplateManagerValidationMessage(message),
                  persistTemplateParametersFromControls: () =>
                    ctx.persistTemplateParametersFromControls(),
                  uniquifyImportedSpec: (spec, prefix) =>
                    ctx.uniquifyImportedSpec(spec, prefix),
                  makeId: (prefix) => ctx.makeId(prefix),
                  translateImportedSpec: (spec, targetCenter) =>
                    ctx.translateImportedSpec(spec, targetCenter),
                  suggestTensorPosition: (position) =>
                    ctx.suggestTensorPosition(position),
                  viewportCenterPosition: () => ctx.viewportCenterPosition(),
                },
              })
            );

            registerInteractions(ctx);
            const requiredCtxBindings = [
              "handleCanvasWheel",
              "handleKeydown",
              "toggleConnectMode",
              "generateCode",
              "insertTemplate",
              "openSubnetworkPicker",
              "loadSubnetworkFromFile",
            ];
            for (const bindingName of requiredCtxBindings) {
              if (typeof ctx[bindingName] !== "function") {
                throw new Error(`registerInteractions did not expose ${bindingName}.`);
              }
            }

            await ctx.generateCode();
            if (ctx.dom.generatedCode.value.trim() !== "result = 1") {
              throw new Error(`Generated code text was not stripped as expected: ${ctx.dom.generatedCode.value}`);
            }
            if (ctx.dom.generatedCodeView.textContent.trim() !== "result = 1") {
              throw new Error(`Generated code preview did not receive the stripped source: ${ctx.dom.generatedCodeView.textContent}`);
            }
            if (ctx.dom.generatedCodeView.dataset.highlighted !== "true") {
              throw new Error("Generated code preview was not highlighted through Prism.");
            }
            """
        ),
        encoding="utf-8",
    )
    return script_path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_runtime_utility_helper_modules_preserve_facade_contract(
    tmp_path: Path,
) -> None:
    script_path = _write_utility_runtime_contract_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The utility helper contract runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


def _write_interaction_session_dependency_injection_runtime_script(
    tmp_path: Path,
) -> Path:
    script_path = tmp_path / "interaction_session_dependency_injection.mjs"
    _copy_js_modules(tmp_path, _INTERACTION_SESSION_BINDING_DEPENDENCY_MODULES)

    script_path.write_text(
        textwrap.dedent(
            """
            import { pathToFileURL } from "node:url";

            const baseUrl = new URL("./", import.meta.url);
            const [selectorsModule, storeModule, sessionModule] = await Promise.all([
              import(new URL("./state/editorSelectors.js", baseUrl).href),
              import(new URL("./state/editorStore.js", baseUrl).href),
              import(new URL("./interactionsSession.js", baseUrl).href),
            ]);

            const { createEditorStore } = storeModule;
            const { createEditorSelectors } = selectorsModule;
            const { createInteractionSessionBindings } = sessionModule;

            const state = {
              schemaVersion: 4,
              spec: {
                id: "network_demo",
                name: "demo_network",
                tensors: [],
                groups: [],
                edges: [],
                notes: [],
                contraction_plan: null,
                metadata: {},
              },
              generatedCode: "",
              selectedEngine: "quimb",
              selectedCollectionFormat: "dict",
              templateDefinitions: {},
              availableTemplates: [],
              templateCatalogWarnings: [],
              templateParametersByTemplate: {},
            };
            const store = createEditorStore(state);
            const selectors = createEditorSelectors({ store });
            const calls = [];
            const dom = {
              generatedCode: { value: "" },
              generatedCodeView: { textContent: "", dataset: {} },
              exportFormatSelect: { value: "py" },
              loadInput: { value: "" },
              subnetworkLoadInput: { value: "" },
              templateSelect: { value: "" },
            };
            const ctx = {
              apiGet: async () => {
                throw new Error("Fallback apiGet should not be used when services are injected.");
              },
              apiPost: async () => {
                throw new Error("Fallback apiPost should not be used when services are injected.");
              },
              setStatus(message, level = "info") {
                calls.push({ type: "status", message, level });
              },
            };
            const bindings = createInteractionSessionBindings({
              ctx,
              state,
              dom,
              store,
              selectors,
              services: {
                session: {
                  async generateCode(payload) {
                    calls.push({ type: "generateCode", payload });
                    return {
                      ok: true,
                      engine: payload.engine,
                      code: "import tensor_network\\nresult = 1\\n",
                    };
                  },
                },
                templateCatalog: {},
                subnetwork: {},
              },
              sessionActions: {
                ensureCodePanelVisible() {},
                syncCodeGenerationWarning() {},
                getTensorKrowchManualPlanIssueMessage() {
                  return "";
                },
                serializeCurrentSpec: () => ({
                  schema_version: "4.0",
                  network: {
                    id: "network_demo",
                    name: "demo_network",
                  },
                }),
                stripImportLines: (code) => code.replace(/^import .*\\n/, ""),
                syncGeneratedCodePreview: (code) => {
                  dom.generatedCode.value = code;
                  dom.generatedCodeView.textContent = code;
                },
                setStatus: (message, level = "info") =>
                  calls.push({ type: "actionStatus", message, level }),
                sanitizeFilename: (value) => value,
                applyTemplateCatalogPayload() {},
                normalizeSpec: (spec) => spec,
                applyDesignChange(mutate) {
                  mutate();
                },
                bringTensorToFront() {},
              },
              sessionUi: {
                async copyText(text) {
                  calls.push({ type: "copyText", text });
                },
                downloadText(filename, text, contentType) {
                  calls.push({ type: "downloadText", filename, text, contentType });
                },
              },
            });

            await bindings.generateCode();
            await bindings.copyGeneratedCode();
            bindings.saveDesign();

            const generateCall = calls.find((entry) => entry.type === "generateCode");
            if (!generateCall) {
              throw new Error(`Expected generateCode to use the injected session service, received ${JSON.stringify(calls)}.`);
            }
            if (generateCall.payload.engine !== "quimb" || generateCall.payload.collectionFormat !== "dict") {
              throw new Error(`Unexpected generate payload: ${JSON.stringify(generateCall.payload)}.`);
            }
            if (dom.generatedCode.value.trim() !== "result = 1") {
              throw new Error(`Expected injected preview sync to receive stripped code, received ${dom.generatedCode.value}.`);
            }
            const copyCall = calls.find((entry) => entry.type === "copyText");
            if (!copyCall || copyCall.text.trim() !== "result = 1") {
              throw new Error(`Expected copyGeneratedCode to use the injected UI adapter, received ${JSON.stringify(calls)}.`);
            }
            const downloadCall = calls.find((entry) => entry.type === "downloadText");
            if (!downloadCall || downloadCall.filename !== "demo_network.json") {
              throw new Error(`Expected saveDesign to use the injected download adapter, received ${JSON.stringify(calls)}.`);
            }
          """
        ),
        encoding="utf-8",
    )
    return script_path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_interaction_session_bindings_use_injected_services_and_ui_adapters(
    tmp_path: Path,
) -> None:
    script_path = _write_interaction_session_dependency_injection_runtime_script(
        tmp_path
    )
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The interaction-session dependency injection runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


def _write_session_editor_draft_autosave_runtime_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "session_editor_draft_autosave.mjs"
    _copy_js_modules(tmp_path, _SESSION_EDITOR_FLOWS_DEPENDENCY_MODULES)

    script_path.write_text(
        textwrap.dedent(
            """
            const baseUrl = new URL("./", import.meta.url);
            const { createSessionEditorFlows } = await import(
              new URL("./session/sessionEditorFlows.js", baseUrl).href
            );

            const calls = [];
            let scheduledCallback = null;
            const state = {
              spec: { name: "draft demo" },
              generatedCode: "",
              editorFinished: false,
              draftAutosaveReady: true,
              draftAutosaveTimer: null,
              draftAutosaveDirty: false,
              draftAutosaveSaving: false,
            };
            const flows = createSessionEditorFlows({
              dom: {
                exportFormatSelect: { value: "py" },
                generatedCode: { value: "" },
                loadInput: null,
              },
              state,
              store: {
                setGeneratedCode(code) {
                  state.generatedCode = code;
                },
                setEditorFinished(value) {
                  state.editorFinished = value;
                },
              },
              selectors: {
                getSelectedEngine: () => "einsum_numpy",
                getSelectedCollectionFormat: () => "dict",
              },
              services: {
                session: {
                  async saveDraft(payload) {
                    calls.push({ type: "saveDraft", payload });
                    return { ok: true };
                  },
                  async clearDraft() {
                    calls.push({ type: "clearDraft" });
                    return { ok: true };
                  },
                  async completeSession(payload) {
                    calls.push({ type: "completeSession", payload });
                    return { ok: true };
                  },
                  async cancelSession() {
                    calls.push({ type: "cancelSession" });
                    return { ok: true };
                  },
                  async renderSpec(payload) {
                    calls.push({ type: "renderSpec", payload });
                    if (payload.format === "svg") {
                      return {
                        ok: true,
                        format: "svg",
                        text: "<?xml version='1.0'?><svg />",
                        content_type: "image/svg+xml;charset=utf-8",
                      };
                    }
                    if (payload.format === "png") {
                      return {
                        ok: true,
                        format: "png",
                        base64: "iVBORw0KGgo=",
                        content_type: "image/png",
                      };
                    }
                    if (payload.format === "pdf") {
                      return {
                        ok: true,
                        format: "pdf",
                        base64: "JVBERi0xLjQ=",
                        content_type: "application/pdf",
                      };
                    }
                    return {
                      ok: true,
                      format: payload.format,
                      text:
                        payload.format === "tikz"
                          ? "\\\\begin{tikzpicture}"
                          : "graph demo {}",
                      content_type:
                        payload.format === "tikz"
                          ? "text/x-tex;charset=utf-8"
                          : "text/vnd.graphviz;charset=utf-8",
                    };
                  },
                },
              },
              commands: {
                syncGeneratedCodePreview() {},
              },
              sessionUi: {
                schedule(callback) {
                  scheduledCallback = callback;
                  return 7;
                },
                downloadText(filename, text, contentType) {
                  calls.push({ type: "downloadText", filename, text, contentType });
                },
                downloadBlob(filename, blob) {
                  calls.push({ type: "downloadBlob", filename, contentType: blob.type });
                },
                closeWindow() {},
              },
              actions: {
                serializeCurrentSpec({ persistViewSnapshots }) {
                  return {
                    schema_version: 2,
                    persistViewSnapshots,
                    network: { id: "network_draft" },
                  };
                },
                sanitizeFilename: (value) => value.replace(/\\s+/g, "_"),
                setStatus(message, level = "info") {
                  calls.push({ type: "status", message, level });
                },
                ensureCodePanelVisible() {},
                syncCodeGenerationWarning() {},
                getTensorKrowchManualPlanIssueMessage: () => "",
                stripImportLines: (code) => code,
                formatIssues: () => "issues",
              },
            });

            flows.scheduleDraftAutosave();
            if (typeof scheduledCallback !== "function") {
              throw new Error("Expected draft autosave to schedule a debounced save.");
            }
            await scheduledCallback();
            const saveCall = calls.find((entry) => entry.type === "saveDraft");
            if (!saveCall) {
              throw new Error(`Expected autosave to call saveDraft, received ${JSON.stringify(calls)}.`);
            }
            if (
              saveCall.payload.engine !== "einsum_numpy" ||
              saveCall.payload.collectionFormat !== "dict" ||
              saveCall.payload.spec.network.id !== "network_draft"
            ) {
              throw new Error(`Unexpected autosave payload: ${JSON.stringify(saveCall.payload)}.`);
            }

            flows.saveDesign();
            await Promise.resolve();
            if (!calls.some((entry) => entry.type === "clearDraft")) {
              throw new Error(`Expected explicit JSON save to clear the draft, received ${JSON.stringify(calls)}.`);
            }
            if (!state.draftAutosaveReady) {
              throw new Error("Explicit JSON save should resume future autosaves.");
            }

            await flows.downloadExportAs("svg");
            await flows.downloadExportAs("png");
            await flows.downloadExportAs("pdf");
            await flows.downloadExportAs("tikz");
            await flows.downloadExportAs("dot");
            const svgRenderCall = calls.find(
              (entry) => entry.type === "renderSpec" && entry.payload.format === "svg"
            );
            const pngRenderCall = calls.find(
              (entry) => entry.type === "renderSpec" && entry.payload.format === "png"
            );
            const pdfRenderCall = calls.find(
              (entry) => entry.type === "renderSpec" && entry.payload.format === "pdf"
            );
            const tikzRenderCall = calls.find(
              (entry) => entry.type === "renderSpec" && entry.payload.format === "tikz"
            );
            const dotRenderCall = calls.find(
              (entry) => entry.type === "renderSpec" && entry.payload.format === "dot"
            );
            const svgDownloadCall = calls.find(
              (entry) => entry.type === "downloadText" && entry.filename === "draft_demo.svg"
            );
            const pngDownloadCall = calls.find(
              (entry) => entry.type === "downloadBlob" && entry.filename === "draft_demo.png"
            );
            const pdfDownloadCall = calls.find(
              (entry) => entry.type === "downloadBlob" && entry.filename === "draft_demo.pdf"
            );
            const tikzDownloadCall = calls.find(
              (entry) => entry.type === "downloadText" && entry.filename === "draft_demo.tex"
            );
            const dotDownloadCall = calls.find(
              (entry) => entry.type === "downloadText" && entry.filename === "draft_demo.dot"
            );
            if (!svgRenderCall || !pngRenderCall || !pdfRenderCall || !tikzRenderCall || !dotRenderCall) {
              throw new Error(`Expected academic exports to call renderSpec, received ${JSON.stringify(calls)}.`);
            }
            if (
              !svgRenderCall.payload.spec.persistViewSnapshots ||
              !pngRenderCall.payload.spec.persistViewSnapshots ||
              !pdfRenderCall.payload.spec.persistViewSnapshots ||
              !tikzRenderCall.payload.spec.persistViewSnapshots ||
              !dotRenderCall.payload.spec.persistViewSnapshots
            ) {
              throw new Error(`Academic exports should persist view snapshots, received ${JSON.stringify(calls)}.`);
            }
            if (!svgDownloadCall || svgDownloadCall.contentType !== "image/svg+xml;charset=utf-8") {
              throw new Error(`Expected SVG export to download a .svg file, received ${JSON.stringify(calls)}.`);
            }
            if (!pngDownloadCall || pngDownloadCall.contentType !== "image/png") {
              throw new Error(`Expected PNG export to download a .png file, received ${JSON.stringify(calls)}.`);
            }
            if (!pdfDownloadCall || pdfDownloadCall.contentType !== "application/pdf") {
              throw new Error(`Expected PDF export to download a .pdf file, received ${JSON.stringify(calls)}.`);
            }
            if (!tikzDownloadCall || tikzDownloadCall.contentType !== "text/x-tex;charset=utf-8") {
              throw new Error(`Expected TikZ export to download a .tex file, received ${JSON.stringify(calls)}.`);
            }
            if (!dotDownloadCall || dotDownloadCall.contentType !== "text/vnd.graphviz;charset=utf-8") {
              throw new Error(`Expected DOT export to download a .dot file, received ${JSON.stringify(calls)}.`);
            }

            const callCountBeforeComplete = calls.length;
            await flows.completeEditor();
            const completeCalls = calls.slice(callCountBeforeComplete);
            const clearIndex = completeCalls.findIndex((entry) => entry.type === "clearDraft");
            const completeIndex = completeCalls.findIndex((entry) => entry.type === "completeSession");
            if (clearIndex < 0 || completeIndex < 0) {
              throw new Error(`Expected Done to finish and clear, received ${JSON.stringify(completeCalls)}.`);
            }
            if (state.draftAutosaveReady) {
              throw new Error("Done should stop further draft autosaves.");
            }
          """
        ),
        encoding="utf-8",
    )
    return script_path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_session_editor_flows_autosave_and_clear_project_drafts(
    tmp_path: Path,
) -> None:
    script_path = _write_session_editor_draft_autosave_runtime_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The session-editor draft autosave runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


def _write_tensor_initializer_parsing_runtime_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "tensor_initializer_parsing.mjs"
    _copy_js_modules(
        tmp_path,
        {
            "properties/tensorPropertiesStandardData.js": (
                "properties/tensorPropertiesStandardData.js"
            )
        },
    )

    script_path.write_text(
        textwrap.dedent(
            """
            const baseUrl = new URL("./", import.meta.url);
            const { createStandardTensorDataSupport } = await import(
              new URL("./properties/tensorPropertiesStandardData.js", baseUrl).href
            );

            const dataSupport = createStandardTensorDataSupport();
            const fill = dataSupport.analyzeTensorDataFillInput("1+2j");
            if (
              !fill.ok ||
              JSON.stringify(fill.tensorData.fill_value) !==
                JSON.stringify({ real: 1, imag: 2 })
            ) {
              throw new Error(`Expected friendly complex fill parsing, received ${JSON.stringify(fill)}.`);
            }

            const literal = dataSupport.analyzeTensorLiteralInput(
              '[["1+2j", {"real": 3, "imag": -4}]]',
              [1, 2]
            );
            if (
              !literal.ok ||
              JSON.stringify(literal.tensorData.values) !==
                JSON.stringify([[{ real: 1, imag: 2 }, { real: 3, imag: -4 }]])
            ) {
              throw new Error(`Expected literal complex values to normalize, received ${JSON.stringify(literal)}.`);
            }

            const tensor = {
              indices: [{ dimension: 2 }, { dimension: 2 }],
              tensor_data: {
                mode: "random",
                dtype: "complex64",
                seed: 123,
                distribution: "uniform",
              },
            };
            if (dataSupport.getTensorDataMode(tensor) !== "random") {
              throw new Error("Expected random tensor-data mode.");
            }
            if (dataSupport.getTensorDataDType(tensor) !== "complex64") {
              throw new Error("Expected complex64 dtype.");
            }
            if (dataSupport.getTensorRandomSeed(tensor) !== 123) {
              throw new Error("Expected seeded random initializer.");
            }
            if (dataSupport.getTensorRandomDistribution(tensor) !== "uniform") {
              throw new Error("Expected uniform random distribution.");
            }
            const externalTensor = {
              indices: [{ dimension: 2 }, { dimension: 3 }],
              tensor_data: {
                mode: "external",
                file_path: "data/a.npz",
                array_key: "a",
                dtype: "float64",
              },
            };
            if (dataSupport.getTensorDataMode(externalTensor) !== "external") {
              throw new Error("Expected external tensor-data mode.");
            }
            if (dataSupport.getTensorExternalFilePath(externalTensor) !== "data/a.npz") {
              throw new Error("Expected external file path.");
            }
            if (dataSupport.getTensorExternalArrayKey(externalTensor) !== "a") {
              throw new Error("Expected external array key.");
            }
          """
        ),
        encoding="utf-8",
    )
    return script_path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_tensor_initializer_ui_parses_complex_dtype_and_random_fields(
    tmp_path: Path,
) -> None:
    script_path = _write_tensor_initializer_parsing_runtime_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The tensor-initializer parsing runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


def _write_session_editor_live_python_import_runtime_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "session_editor_live_python_import.mjs"
    _copy_js_modules(tmp_path, _SESSION_EDITOR_FLOWS_DEPENDENCY_MODULES)

    script_path.write_text(
        textwrap.dedent(
            """
            const baseUrl = new URL("./", import.meta.url);
            const { createSessionEditorFlows } = await import(
              new URL("./session/sessionEditorFlows.js", baseUrl).href
            );

            const validateCalls = [];
            const statusCalls = [];
            const resetCalls = [];
            const confirmMessages = [];
            const promptMessages = [];
            const confirmQueue = [false, true, true];
            const promptQueue = ["named_network", ""];

            const flows = createSessionEditorFlows({
              dom: {
                exportFormatSelect: { value: "py" },
                generatedCode: { value: "" },
                loadInput: { value: "pending" },
              },
              state: {
                generatedCode: "",
                spec: { name: "demo_network" },
              },
              store: {
                setGeneratedCode() {},
                setEditorFinished() {},
              },
              selectors: {
                getSelectedEngine() {
                  return "quimb";
                },
                getSelectedCollectionFormat() {
                  return "dict";
                },
              },
              services: {
                session: {
                  async validatePythonCode(payload) {
                    validateCalls.push(payload);
                    return {
                      ok: true,
                      spec: {
                        schema_version: 6,
                        network: {
                          id: "imported_network",
                          name: "Imported Python Network",
                          tensors: [],
                          groups: [],
                          edges: [],
                          notes: [],
                          metadata: {},
                        },
                      },
                      warnings:
                        payload.pythonImportMode === "live"
                          ? ["Dropped tensor data for tensor A."]
                          : [],
                    };
                  },
                  async validateSerializedSpec() {
                    throw new Error("Serialized validation should not run for Python files.");
                  },
                },
              },
              commands: {
                syncGeneratedCodePreview() {},
              },
              sessionUi: {
                async requestFileText(file) {
                  return file.text();
                },
                confirmAction(message) {
                  confirmMessages.push(message);
                  return confirmQueue.shift();
                },
                promptText(message, defaultValue = "") {
                  promptMessages.push({ message, defaultValue });
                  return promptQueue.shift() ?? null;
                },
                downloadText() {},
                copyText() {},
                schedule() {},
                closeWindow() {},
              },
              actions: {
                ensureCodePanelVisible() {},
                syncCodeGenerationWarning() {},
                getTensorKrowchManualPlanIssueMessage() {
                  return "";
                },
                serializeCurrentSpec() {
                  return {};
                },
                formatIssues(issues) {
                  return JSON.stringify(issues || []);
                },
                stripImportLines(code) {
                  return code;
                },
                sanitizeFilename(value) {
                  return value;
                },
                resetDesignState(spec, message, schemaVersion) {
                  resetCalls.push({ spec, message, schemaVersion });
                },
                setStatus(message, level = "info") {
                  statusCalls.push({ message, level });
                },
                downloadPngExport() {},
                downloadSvgExport() {},
              },
            });

            const buildEvent = (name) => ({
              target: {
                files: [
                  {
                    name,
                    async text() {
                      return "network = object()";
                    },
                  },
                ],
              },
            });

            await flows.loadDesignFromFile(buildEvent("static_import.py"));
            await flows.loadDesignFromFile(buildEvent("live_named.py"));
            await flows.loadDesignFromFile(buildEvent("live_auto.py"));

            if (validateCalls.length !== 3) {
              throw new Error(`Expected three Python validation calls, received ${JSON.stringify(validateCalls)}.`);
            }
            if (validateCalls[0].pythonImportMode !== "static") {
              throw new Error(`Expected the first Python load to stay in static mode, received ${JSON.stringify(validateCalls[0])}.`);
            }
            if (validateCalls[0].pythonReconstructionLevel !== "auto") {
              throw new Error(`Expected the first Python load to request automatic reconstruction, received ${JSON.stringify(validateCalls[0])}.`);
            }
            if (validateCalls[0].pythonObjectName !== null) {
              throw new Error(`Static Python loads should not set an object override, received ${JSON.stringify(validateCalls[0])}.`);
            }
            if (validateCalls[1].pythonImportMode !== "live" || validateCalls[1].pythonObjectName !== "named_network") {
              throw new Error(`Expected the second Python load to use live mode with the prompted object name, received ${JSON.stringify(validateCalls[1])}.`);
            }
            if (validateCalls[1].pythonReconstructionLevel !== "auto") {
              throw new Error(`Expected live Python loads to request automatic reconstruction, received ${JSON.stringify(validateCalls[1])}.`);
            }
            if (validateCalls[2].pythonImportMode !== "live" || validateCalls[2].pythonObjectName !== null) {
              throw new Error(`Expected blank live-object prompts to fall back to auto-discovery, received ${JSON.stringify(validateCalls[2])}.`);
            }
            if (validateCalls[2].pythonReconstructionLevel !== "auto") {
              throw new Error(`Expected blank live-object prompts to keep automatic reconstruction, received ${JSON.stringify(validateCalls[2])}.`);
            }
            if (confirmMessages.length !== 3) {
              throw new Error(`Expected the Python load flow to ask about live execution every time, received ${JSON.stringify(confirmMessages)}.`);
            }
            if (promptMessages.length !== 2) {
              throw new Error(`Expected object-name prompts only for live imports, received ${JSON.stringify(promptMessages)}.`);
            }
            if (!resetCalls.every((entry) => entry.message.includes("Loaded design from"))) {
              throw new Error(`Expected each successful Python load to reset the design state, received ${JSON.stringify(resetCalls)}.`);
            }
            if (!statusCalls.some((entry) => entry.message.includes("Dropped tensor data for tensor A."))) {
              throw new Error(`Expected live-import warnings to surface as a non-blocking status message, received ${JSON.stringify(statusCalls)}.`);
            }
          """
        ),
        encoding="utf-8",
    )
    return script_path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_session_editor_flows_support_live_python_import_runtime(
    tmp_path: Path,
) -> None:
    script_path = _write_session_editor_live_python_import_runtime_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The session-editor live Python import runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


def _write_editor_session_service_validate_python_runtime_script(
    tmp_path: Path,
) -> Path:
    script_path = tmp_path / "editor_session_service_validate_python_runtime.mjs"
    _copy_js_modules(
        tmp_path,
        {"services/editorSessionService.js": "services/editorSessionService.js"},
    )
    script_path.write_text(
        textwrap.dedent(
            """
            const baseUrl = new URL("./", import.meta.url);
            const { createEditorSessionService } = await import(
              new URL("./services/editorSessionService.js", baseUrl).href
            );

            const apiCalls = [];
            const service = createEditorSessionService({
              async apiGet(path) {
                apiCalls.push({ path, method: "GET" });
                return { ok: true, draft: null };
              },
              async apiPost(path, payload) {
                apiCalls.push({ path, method: "POST", payload });
                return { ok: true };
              },
            });

            await service.validatePythonCode("network = object()");
            await service.validatePythonCode({
              pythonCode: "network = other_object()",
              sourceProfile: "quimb",
              pythonImportMode: "live",
              pythonReconstructionLevel: "simple",
              pythonObjectName: "network",
            });
            await service.loadDraft();
            await service.saveDraft({
              spec: { schema_version: 2, network: { id: "network_draft" } },
              engine: "einsum_numpy",
              collectionFormat: "dict",
            });
            await service.renderSpec({
              format: "dot",
              spec: { schema_version: 2, network: { id: "network_draft" } },
            });
            await service.clearDraft();

            if (apiCalls.length !== 6) {
              throw new Error(`Expected validate and draft calls, received ${JSON.stringify(apiCalls)}.`);
            }
            if (apiCalls[0].path !== "/api/validate") {
              throw new Error(`Expected validatePythonCode to target /api/validate, received ${JSON.stringify(apiCalls[0])}.`);
            }
            if (apiCalls[0].payload.python_code !== "network = object()") {
              throw new Error(`Expected string requests to be wrapped as python_code, received ${JSON.stringify(apiCalls[0])}.`);
            }
            if (apiCalls[0].payload.python_reconstruction_level !== "auto") {
              throw new Error(`Expected string requests to default to automatic reconstruction, received ${JSON.stringify(apiCalls[0])}.`);
            }
            if (apiCalls[0].payload.python_import_mode !== "static") {
              throw new Error(`Expected string requests to default to static import, received ${JSON.stringify(apiCalls[0])}.`);
            }
            if (apiCalls[1].payload.python_code !== "network = other_object()") {
              throw new Error(`Expected object requests to preserve python_code, received ${JSON.stringify(apiCalls[1])}.`);
            }
            if (apiCalls[1].payload.source_profile !== "quimb") {
              throw new Error(`Expected object requests to preserve source_profile, received ${JSON.stringify(apiCalls[1])}.`);
            }
            if (apiCalls[1].payload.python_import_mode !== "live") {
              throw new Error(`Expected object requests to preserve python_import_mode, received ${JSON.stringify(apiCalls[1])}.`);
            }
            if (apiCalls[1].payload.python_reconstruction_level !== "simple") {
              throw new Error(`Expected object requests to preserve python_reconstruction_level, received ${JSON.stringify(apiCalls[1])}.`);
            }
            if (apiCalls[1].payload.python_object_name !== "network") {
              throw new Error(`Expected object requests to preserve python_object_name, received ${JSON.stringify(apiCalls[1])}.`);
            }
            if (apiCalls[2].path !== "/api/draft" || apiCalls[2].method !== "GET") {
              throw new Error(`Expected loadDraft to GET /api/draft, received ${JSON.stringify(apiCalls[2])}.`);
            }
            if (apiCalls[3].path !== "/api/draft" || apiCalls[3].method !== "POST") {
              throw new Error(`Expected saveDraft to POST /api/draft, received ${JSON.stringify(apiCalls[3])}.`);
            }
            if (apiCalls[3].payload.collection_format !== "dict") {
              throw new Error(`Expected saveDraft to normalize collection_format, received ${JSON.stringify(apiCalls[3])}.`);
            }
            if (apiCalls[4].path !== "/api/render" || apiCalls[4].method !== "POST") {
              throw new Error(`Expected renderSpec to POST /api/render before clearing, received ${JSON.stringify(apiCalls[4])}.`);
            }
            if (apiCalls[4].payload.format !== "dot" || apiCalls[4].payload.spec.network.id !== "network_draft") {
              throw new Error(`Expected renderSpec to keep format and spec payloads, received ${JSON.stringify(apiCalls[4])}.`);
            }
            if (apiCalls[5].path !== "/api/draft/clear" || apiCalls[5].method !== "POST") {
              throw new Error(`Expected clearDraft to POST /api/draft/clear, received ${JSON.stringify(apiCalls[5])}.`);
            }
          """
        ),
        encoding="utf-8",
    )
    return script_path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_editor_session_service_validate_python_runtime_contract(
    tmp_path: Path,
) -> None:
    script_path = _write_editor_session_service_validate_python_runtime_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The editor-session-service validatePythonCode runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_runtime_interaction_helper_modules_preserve_facade_contract(
    tmp_path: Path,
) -> None:
    script_path = _write_interaction_runtime_contract_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The interaction helper contract runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


def _write_layout_subnetwork_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "layout_subnetwork_runtime_regression.mjs"
    _copy_js_modules(tmp_path, _LAYOUT_SUBNETWORK_RUNTIME_DEPENDENCY_MODULES)

    script_path.write_text(
        textwrap.dedent(
            """
            import { pathToFileURL } from "node:url";

            function createClassList() {
              return {
                add() {},
                remove() {},
                toggle() {},
              };
            }

            function createButton() {
              return {
                disabled: false,
                classList: createClassList(),
                click() {},
                addEventListener() {},
              };
            }

            function createSelect(value = "") {
              return {
                value,
                options: [],
                innerHTML: "",
                appendChild(option) {
                  this.options.push(option);
                  if (option && option.selected) {
                    this.value = option.value;
                  }
                },
                addEventListener() {},
              };
            }

            function createPromptQueue(values) {
              const queue = [...values];
              return () => (queue.length ? queue.shift() : null);
            }

            const baseUrl = new URL("./", import.meta.url);
            const [stateModule, utilitiesModule, historyModule, sessionModule, selectorsModule, storeModule] =
              await Promise.all([
                import(new URL("./state.runtime.mjs", baseUrl).href),
                import(new URL("./utilities.runtime.mjs", baseUrl).href),
                import(new URL("./historySelection.runtime.mjs", baseUrl).href),
                import(new URL("./interactionsSession.js", baseUrl).href),
                import(new URL("./state/editorSelectors.js", baseUrl).href),
                import(new URL("./state/editorStore.js", baseUrl).href),
              ]);

            const { createInitialState } = stateModule;
            const { registerUtilities } = utilitiesModule;
            const { registerHistorySelection } = historyModule;
            const { createInteractionSessionBindings } = sessionModule;
            const { createEditorSelectors } = selectorsModule;
            const { createEditorStore } = storeModule;

            const apiCalls = [];

            const ctx = {
              state: createInitialState(),
              constants: {
                TENSOR_WIDTH: 180,
                TENSOR_HEIGHT: 108,
                MIN_TENSOR_WIDTH: 140,
                MIN_TENSOR_HEIGHT: 84,
                INDEX_RADIUS: 15,
                INDEX_PADDING: 8,
                NOTE_WIDTH: 220,
                NOTE_HEIGHT: 152,
                NOTE_MIN_WIDTH: 176,
                NOTE_MIN_HEIGHT: 152,
                HISTORY_LIMIT: 100,
                REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
                GRID_SNAP_SIZE: 20,
                DEFAULT_INDEX_SLOTS: [
                  { x: -58, y: -20 },
                  { x: 58, y: -20 },
                  { x: -58, y: 20 },
                  { x: 58, y: 20 },
                ],
              },
              dom: {
                workspace: {},
                statusMessage: {
                  textContent: "",
                  classList: createClassList(),
                },
                propertiesPanel: { innerHTML: "" },
                generatedCode: { value: "" },
                generatedCodeView: { textContent: "", dataset: {} },
                engineSelect: createSelect("tensornetwork"),
                collectionFormatSelect: createSelect("list"),
                exportFormatSelect: createSelect("py"),
                addNoteButton: createButton(),
                connectButton: createButton(),
                loadInput: { value: "", click() {}, addEventListener() {} },
                subnetworkLoadInput: {
                  value: "",
                  click() {},
                  addEventListener() {},
                },
                undoButton: createButton(),
                redoButton: createButton(),
                exportButton: createButton(),
                toggleLinearPeriodicButton: createButton(),
                linearPeriodicPreviousCellButton: createButton(),
                linearPeriodicCellLabel: { textContent: "" },
                linearPeriodicNextCellButton: createButton(),
                templateSelect: createSelect(""),
                templateParameterPanel: { hidden: true },
                templateGraphSizeLabel: { textContent: "" },
                templateGraphSizeInput: { value: "2", min: "1" },
                templateBondDimensionInput: { value: "3", min: "1" },
                templatePhysicalDimensionInput: { value: "2", min: "1" },
                insertTemplateButton: createButton(),
                insertSubnetworkButton: createButton(),
                reflowImportedButton: createButton(),
                createGroupButton: createButton(),
                helpButton: createButton(),
                helpModal: { classList: createClassList() },
                helpBackdrop: createButton(),
                helpCloseButton: createButton(),
                canvasShell: {
                  getBoundingClientRect() {
                    return { left: 0, top: 0, width: 1000, height: 800 };
                  },
                  addEventListener() {},
                },
                groupLayer: {},
                resizeLayer: {},
                notesLayer: {},
                selectionBox: {
                  classList: createClassList(),
                  style: {},
                },
                minimapCanvas: {
                  classList: createClassList(),
                  addEventListener() {},
                },
                sidebar: {},
                plannerPanel: {},
                generateButton: createButton(),
                codeGenerationWarning: {
                  textContent: "",
                  title: "",
                  hidden: true,
                },
              },
              apiGet: async () => null,
              apiPost: async (path, payload) => {
                apiCalls.push({ path, payload });
                if (path === "/api/subnetwork/extract") {
                  return {
                    ok: true,
                    spec: {
                      schema_version: "1.0",
                      network: {
                        id: "fragment_network",
                        name: "Fragment",
                        tensors: [
                          {
                            id: "fragment_tensor",
                            name: "Fragment",
                            position: { x: 240, y: 220 },
                            size: { width: 180, height: 108 },
                            indices: [],
                            metadata: {},
                          },
                        ],
                        groups: [],
                        edges: [],
                        notes: [],
                        contraction_plan: null,
                        metadata: {},
                      },
                    },
                  };
                }
                if (path === "/api/template/promote") {
                  return {
                    ok: true,
                    templates: ["selection_fragment", "mps"],
                    template_definitions: {
                      selection_fragment: {
                        display_name: "Selection Fragment",
                        graph_size_label: "Tensors",
                        defaults: {
                          graph_size: 3,
                          bond_dimension: 1,
                          physical_dimension: 1,
                        },
                        minimums: {
                          graph_size: 1,
                          bond_dimension: 1,
                          physical_dimension: 1,
                        },
                        supports_parameters: false,
                      },
                      mps: {
                        display_name: "MPS",
                        graph_size_label: "Sites",
                        defaults: {
                          graph_size: 4,
                          bond_dimension: 3,
                          physical_dimension: 2,
                        },
                        minimums: {
                          graph_size: 2,
                          bond_dimension: 1,
                          physical_dimension: 1,
                        },
                        supports_parameters: true,
                      },
                    },
                    selected_template: "selection_fragment",
                    template_catalog_warnings: [],
                  };
                }
                if (path === "/api/template") {
                  return {
                    ok: true,
                    spec: {
                      schema_version: "1.0",
                      network: {
                        id: "template_selection_fragment",
                        name: "Selection Fragment",
                        tensors: [
                          {
                            id: "template_tensor_a",
                            name: "TA",
                            position: { x: 620, y: 480 },
                            size: { width: 180, height: 108 },
                            indices: [
                              {
                                id: "template_tensor_a_x",
                                name: "x",
                                dimension: 3,
                                offset: { x: 0, y: 0 },
                                metadata: {},
                              },
                            ],
                            metadata: {},
                          },
                          {
                            id: "template_tensor_b",
                            name: "TB",
                            position: { x: 760, y: 360 },
                            size: { width: 180, height: 108 },
                            indices: [
                              {
                                id: "template_tensor_b_x",
                                name: "x",
                                dimension: 3,
                                offset: { x: 0, y: 0 },
                                metadata: {},
                              },
                            ],
                            metadata: {},
                          },
                          {
                            id: "template_tensor_c",
                            name: "TC",
                            position: { x: 900, y: 500 },
                            size: { width: 180, height: 108 },
                            indices: [],
                            metadata: {},
                          },
                        ],
                        groups: [],
                        edges: [
                          {
                            id: "template_edge_ab",
                            name: "bond_ab",
                            left: {
                              tensor_id: "template_tensor_a",
                              index_id: "template_tensor_a_x",
                            },
                            right: {
                              tensor_id: "template_tensor_b",
                              index_id: "template_tensor_b_x",
                            },
                            metadata: {},
                          },
                        ],
                        notes: [],
                        contraction_plan: null,
                        metadata: {},
                      },
                    },
                  };
                }
                throw new Error(`Unexpected API path: ${path}`);
              },
              window: {
                structuredClone: globalThis.structuredClone,
                crypto: globalThis.crypto,
                setTimeout,
                clearTimeout,
                confirm: () => true,
                prompt: createPromptQueue([
                  "selection_fragment_export",
                  "selection_fragment",
                ]),
                Prism: null,
              },
              document: {
                activeElement: null,
                createElement() {
                  return {
                    href: "",
                    download: "",
                    click() {},
                    appendChild() {},
                  };
                },
                querySelectorAll() {
                  return [];
                },
              },
              cytoscape: null,
              render() {},
              renderOverlayDecorations() {},
              renderMinimap() {},
              renderPlanner() {},
              renderSidebarTabs() {},
              refreshContractionAnalysis() {},
              repairContractionPlan() {},
            };

            registerUtilities(ctx);
            registerHistorySelection(ctx);
            ctx.store = createEditorStore(ctx.state);
            ctx.selectors = createEditorSelectors({ store: ctx.store });
            ctx.services = {
              session: {
                async buildTemplate(payload) {
                  return ctx.apiPost("/api/template", {
                    template: payload.templateName,
                    parameters: payload.parameters,
                  });
                },
              },
              templateCatalog: {
                async promoteTemplate(payload) {
                  return ctx.apiPost("/api/template/promote", {
                    spec: payload.serializedSpec,
                    tensor_ids: payload.tensorIds,
                    template_name: payload.templateName,
                    overwrite: payload.overwrite,
                  });
                },
                async renameTemplate(payload) {
                  return ctx.apiPost("/api/template/rename", {
                    template_name: payload.templateName,
                    new_template_name: payload.newTemplateName,
                    overwrite: payload.overwrite,
                  });
                },
                async deleteTemplate(payload) {
                  return ctx.apiPost("/api/template/delete", {
                    template_name: payload.templateName,
                  });
                },
              },
              subnetwork: {
                async extractSubnetwork(payload) {
                  return ctx.apiPost("/api/subnetwork/extract", {
                    spec: payload.serializedSpec,
                    tensor_ids: payload.tensorIds,
                  });
                },
                async prepareSubnetworkForInsert(payload) {
                  return ctx.apiPost("/api/subnetwork/prepare-insert", {
                    spec: payload.serializedSpec,
                    target_center: payload.targetCenter,
                  });
                },
              },
            };
            const env = {
              ctx,
              state: ctx.state,
              dom: ctx.dom,
            };
            const downloadEvents = [];
            Object.assign(
              ctx,
              createInteractionSessionBindings({
                ...env,
                store: ctx.store,
                selectors: ctx.selectors,
                services: ctx.services,
                sessionUi: {
                  async copyText() {},
                  downloadText(filename, text, contentType) {
                    downloadEvents.push({ filename, text, contentType });
                  },
                  downloadBlob() {},
                  requestFileText: async (file) => file.text(),
                  openFilePicker(input) {
                    input.click();
                  },
                  schedule(callback) {
                    callback();
                  },
                  closeWindow() {},
                  promptText: (...args) => ctx.window.prompt(...args),
                  confirmAction: (...args) => ctx.window.confirm(...args),
                },
                sessionActions: {
                  ensureCodePanelVisible() {},
                  syncCodeGenerationWarning() {},
                  getTensorKrowchManualPlanIssueMessage() {
                    return "";
                  },
                  getSelectedTensorIds: () => ctx.getSelectedIdsByKind("tensor"),
                  findGroupById: (groupId) => ctx.findGroupById(groupId),
                  isLinearPeriodicMode: () => ctx.isLinearPeriodicMode(),
                  syncGeneratedCodePreview: (code) =>
                    ctx.renderGeneratedCodePreview(code),
                  setStatus: (message, level) => ctx.setStatus(message, level),
                  serializeCurrentSpec: (options) => ctx.serializeCurrentSpec(options),
                  formatIssues: (issues) => ctx.formatIssues(issues),
                  stripImportLines: (code) => ctx.stripImportLines(code),
                  sanitizeFilename: (value) => ctx.sanitizeFilename(value),
                  resetDesignState: (spec, message, schemaVersion) =>
                    ctx.resetDesignState(spec, message, schemaVersion),
                  downloadPngExport: () => ctx.downloadPngExport(),
                  downloadSvgExport: () => ctx.downloadSvgExport(),
                  applyTemplateCatalogPayload: (payload) =>
                    ctx.applyTemplateCatalogPayload(payload),
                  normalizeSpec: (spec) => ctx.normalizeSpec(spec),
                  applyDesignChange: (mutate, options) =>
                    ctx.applyDesignChange(mutate, options),
                  bringTensorToFront: (tensorId) => ctx.bringTensorToFront(tensorId),
                  formatTemplateLabel: (value) => ctx.formatTemplateLabel(value),
                  getTemplateSource: (templateName) => ctx.getTemplateSource(templateName),
                  getTemplateSpec: (templateName) => ctx.getTemplateSpec(templateName),
                  listTemplateEntries: () => ctx.listTemplateEntries(),
                  hasTemplateDisplayName: (displayName, excludedTemplateName) =>
                    ctx.hasTemplateDisplayName(displayName, excludedTemplateName),
                  getNextSessionTemplateDisplayName: (baseDisplayName) =>
                    ctx.getNextSessionTemplateDisplayName(baseDisplayName),
                  addSessionTemplate: (payload) => ctx.addSessionTemplate(payload),
                  updateSessionTemplateDisplayNames: (updates) =>
                    ctx.updateSessionTemplateDisplayNames(updates),
                  removeSessionTemplate: (templateName) =>
                    ctx.removeSessionTemplate(templateName),
                  toggleTemplateManager: (forceOpen) =>
                    ctx.toggleTemplateManager(forceOpen),
                  syncTemplateManagerModalState: () =>
                    ctx.syncTemplateManagerModalState(),
                  setTemplateManagerValidationMessage: (message) =>
                    ctx.setTemplateManagerValidationMessage(message),
                  persistTemplateParametersFromControls: () =>
                    ctx.persistTemplateParametersFromControls(),
                  uniquifyImportedSpec: (spec, prefix) =>
                    ctx.uniquifyImportedSpec(spec, prefix),
                  makeId: (prefix) => ctx.makeId(prefix),
                  translateImportedSpec: (spec, targetCenter) =>
                    ctx.translateImportedSpec(spec, targetCenter),
                  suggestTensorPosition: (position) =>
                    ctx.suggestTensorPosition(position),
                  viewportCenterPosition: () => ctx.viewportCenterPosition(),
                },
              })
            );
            ctx.uniquifyImportedSpec = (spec) => ctx.normalizeSpec(spec);
            ctx.translateImportedSpec = (spec) => ctx.normalizeSpec(spec);
            ctx.viewportCenterPosition = () => ({ x: 500, y: 400 });
            ctx.suggestTensorPosition = (center) => center;
            ctx.state.availableTemplates = ["mps"];
            ctx.state.templateDefinitions = {
              mps: {
                display_name: "MPS",
                graph_size_label: "Sites",
                defaults: {
                  graph_size: 4,
                  bond_dimension: 3,
                  physical_dimension: 2,
                },
                minimums: {
                  graph_size: 2,
                  bond_dimension: 1,
                  physical_dimension: 1,
                },
                supports_parameters: true,
              },
            };
            ctx.state.templateParametersByTemplate = ctx.buildTemplateParameterState(
              ctx.state.availableTemplates,
              ctx.state.templateDefinitions
            );
            ctx.dom.templateSelect.value = "mps";

            ctx.state.spec = ctx.normalizeSpec({
              id: "network_demo",
              name: "demo",
              tensors: [
                {
                  id: "tensor_a",
                  name: "A",
                  position: { x: 83, y: 101 },
                  size: { width: 180, height: 108 },
                  indices: [
                    {
                      id: "tensor_a_x",
                      name: "x",
                      dimension: 3,
                      offset: { x: 0, y: 0 },
                      metadata: {},
                    },
                  ],
                  metadata: {},
                },
                {
                  id: "tensor_b",
                  name: "B",
                  position: { x: 247, y: 162 },
                  size: { width: 180, height: 108 },
                  indices: [
                    {
                      id: "tensor_b_x",
                      name: "x",
                      dimension: 3,
                      offset: { x: 0, y: 0 },
                      metadata: {},
                    },
                    {
                      id: "tensor_b_y",
                      name: "y",
                      dimension: 5,
                      offset: { x: 0, y: 0 },
                      metadata: {},
                    },
                  ],
                  metadata: {},
                },
                {
                  id: "tensor_c",
                  name: "C",
                  position: { x: 431, y: 227 },
                  size: { width: 180, height: 108 },
                  indices: [
                    {
                      id: "tensor_c_y",
                      name: "y",
                      dimension: 5,
                      offset: { x: 0, y: 0 },
                      metadata: {},
                    },
                  ],
                  metadata: {},
                },
              ],
              groups: [],
              edges: [
                {
                  id: "edge_ab",
                  name: "bond_ab",
                  left: { tensor_id: "tensor_a", index_id: "tensor_a_x" },
                  right: { tensor_id: "tensor_b", index_id: "tensor_b_x" },
                  metadata: {},
                },
                {
                  id: "edge_bc",
                  name: "bond_bc",
                  left: { tensor_id: "tensor_b", index_id: "tensor_b_y" },
                  right: { tensor_id: "tensor_c", index_id: "tensor_c_y" },
                  metadata: {},
                },
              ],
              notes: [],
              contraction_plan: null,
              linear_periodic_chain: null,
              metadata: {},
            });
            ctx.state.selectionIds = ["tensor_a", "tensor_b", "tensor_c"];
            ctx.state.primarySelectionId = "tensor_c";

            ctx.state.selectionIds = ["tensor_a", "tensor_b"];
            ctx.state.primarySelectionId = "tensor_b";
            ctx.state.spec.tensors[0].indices[0].offset = { x: 12, y: -4 };
            ctx.state.spec.tensors[1].indices[0].offset = { x: 16, y: -8 };
            ctx.state.spec.tensors[1].indices[1].offset = { x: 20, y: 10 };
            ctx.state.spec.tensors[1].indices.push(
              {
                id: "tensor_b_extra_1",
                name: "z1",
                dimension: 7,
                offset: { x: 0, y: 0 },
                metadata: {},
              },
              {
                id: "tensor_b_extra_2",
                name: "z2",
                dimension: 11,
                offset: { x: 0, y: 0 },
                metadata: {},
              },
              {
                id: "tensor_b_extra_3",
                name: "z3",
                dimension: 13,
                offset: { x: 0, y: 0 },
                metadata: {},
              }
            );
            ctx.applyReflowIndicesAction("bottom");
            const tensorAAfterIndexBottom = ctx.findTensorById("tensor_a");
            const tensorBAfterIndexBottom = ctx.findTensorById("tensor_b");
            const expectedBottomOffsetA =
              tensorAAfterIndexBottom.size.height / 2
              - ctx.constants.INDEX_RADIUS
              - ctx.constants.INDEX_PADDING;
            const expectedBottomOffsetB =
              tensorBAfterIndexBottom.size.height / 2
              - ctx.constants.INDEX_RADIUS
              - ctx.constants.INDEX_PADDING;
            if (!tensorAAfterIndexBottom.indices.every((index) => index.offset.y === expectedBottomOffsetA)) {
              throw new Error("Bottom index reflow should pin tensor A indices to the lower edge.");
            }
            const uniqueTensorBIndexRows = new Set(
              tensorBAfterIndexBottom.indices.map((index) => index.offset.y)
            );
            if (uniqueTensorBIndexRows.size < 2) {
              throw new Error(
                "Crowded bottom index reflow should use multiple rows when one edge is too cramped."
              );
            }
            if (ctx.state.selectionIds.join(",") !== "tensor_a,tensor_b") {
              throw new Error("Index reflow should preserve the selected tensors.");
            }

            ctx.state.selectionIds = ["tensor_b"];
            ctx.state.primarySelectionId = "tensor_b";
            ctx.applyReflowIndicesAction("reset");
            const tensorBAfterIndexReset = ctx.findTensorById("tensor_b");
            const expectedResetOffsets = tensorBAfterIndexReset.indices.map((index, indexPosition) =>
              ctx.defaultIndexOffsetForOrder(indexPosition, tensorBAfterIndexReset)
            );
            const actualResetOffsets = tensorBAfterIndexReset.indices.map((index) => index.offset);
            if (JSON.stringify(actualResetOffsets) !== JSON.stringify(expectedResetOffsets)) {
              throw new Error(
                `Reset index reflow should restore the balanced default offsets, received ${JSON.stringify(actualResetOffsets)}.`
              );
            }

            ctx.state.selectionIds = ["tensor_a", "tensor_b", "tensor_c"];
            ctx.state.primarySelectionId = "tensor_c";

            ctx.state.spec.tensors[0].position = { x: 83, y: 118 };
            ctx.state.spec.tensors[1].position = { x: 247, y: 124 };
            ctx.state.spec.tensors[2].position = { x: 431, y: 130 };
            ctx.alignSelectedTensors("left");
            const leftEdges = ctx.state.spec.tensors.map((tensor) => tensor.position.x - tensor.size.width / 2);
            if (!leftEdges.every((value) => value === leftEdges[0])) {
              throw new Error(`Expected aligned left edges, received ${leftEdges.join(", ")}`);
            }
            const leftAlignedTensors = [...ctx.state.spec.tensors].sort(
              (leftTensor, rightTensor) => leftTensor.position.y - rightTensor.position.y
            );
            for (let index = 1; index < leftAlignedTensors.length; index += 1) {
              const previousTensor = leftAlignedTensors[index - 1];
              const currentTensor = leftAlignedTensors[index];
              const previousBottom =
                previousTensor.position.y + previousTensor.size.height / 2;
              const currentTop =
                currentTensor.position.y - currentTensor.size.height / 2;
              if (currentTop - previousBottom < 32) {
                throw new Error(
                  `Left alignment should keep a visible vertical gap, received ${leftAlignedTensors.map((tensor) => tensor.position.y).join(", ")}.`
                );
              }
            }
            if (ctx.state.selectionIds.join(",") !== "tensor_a,tensor_b,tensor_c") {
              throw new Error("Alignment should preserve the tensor selection.");
            }

            ctx.state.spec.tensors[0].position = { x: 120, y: 100 };
            ctx.state.spec.tensors[1].position = { x: 126, y: 170 };
            ctx.state.spec.tensors[2].position = { x: 132, y: 240 };
            ctx.alignSelectedTensors("middle");
            const middleCenters = ctx.state.spec.tensors.map((tensor) => tensor.position.y);
            if (!middleCenters.every((value) => value === middleCenters[0])) {
              throw new Error(
                `Middle alignment should align vertical centers, received ${middleCenters.join(", ")}.`
              );
            }
            const middleAlignedTensors = [...ctx.state.spec.tensors].sort(
              (leftTensor, rightTensor) => leftTensor.position.x - rightTensor.position.x
            );
            for (let index = 1; index < middleAlignedTensors.length; index += 1) {
              const previousTensor = middleAlignedTensors[index - 1];
              const currentTensor = middleAlignedTensors[index];
              const previousRight =
                previousTensor.position.x + previousTensor.size.width / 2;
              const currentLeft =
                currentTensor.position.x - currentTensor.size.width / 2;
              if (currentLeft - previousRight < 32) {
                throw new Error(
                  `Middle alignment should keep a visible horizontal gap, received ${middleAlignedTensors.map((tensor) => tensor.position.x).join(", ")}.`
                );
              }
            }

            ctx.state.spec.tensors[0].position.x = 100;
            ctx.state.spec.tensors[1].position.x = 260;
            ctx.state.spec.tensors[2].position.x = 460;
            ctx.distributeSelectedTensors("horizontal");
            const centers = ctx.state.spec.tensors.map((tensor) => tensor.position.x);
            const spacing = centers[1] - centers[0];
            if (Math.abs((centers[2] - centers[1]) - spacing) > 1e-9) {
              throw new Error(`Expected even horizontal spacing, received ${centers.join(", ")}`);
            }

            ctx.snapSelectedTensorsToGrid();
            if (!ctx.state.spec.tensors.every((tensor) => tensor.position.x % 20 === 0 && tensor.position.y % 20 === 0)) {
              throw new Error("Snap to grid should move every selected tensor onto the 20px grid.");
            }
            if (ctx.state.undoStack.length < 3) {
              throw new Error(`Expected layout actions to create undo history, received ${ctx.state.undoStack.length} snapshots.`);
            }

            ctx.state.spec.tensors[0].position = { x: 80, y: 140 };
            ctx.state.spec.tensors[1].position = { x: 260, y: 240 };
            ctx.state.spec.tensors[2].position = { x: 420, y: 120 };
            ctx.arrangeSelectedTensors("chain");
            const chainYPositions = ctx.state.spec.tensors.map((tensor) => tensor.position.y);
            if (!chainYPositions.every((value) => value === chainYPositions[0])) {
              throw new Error(`Arrange Chain should align centers horizontally, received ${chainYPositions.join(", ")}.`);
            }

            ctx.state.primarySelectionId = "tensor_b";
            ctx.arrangeSelectedTensors("tree");
            const tensorB = ctx.findTensorById("tensor_b");
            const tensorA = ctx.findTensorById("tensor_a");
            const tensorC = ctx.findTensorById("tensor_c");
            if (!(tensorB.position.y < tensorA.position.y && tensorB.position.y < tensorC.position.y)) {
              throw new Error("Arrange Tree should place the primary tensor above the remaining path tensors.");
            }

            ctx.applyDesignChange(
              () => {
                const tensorCForBranch = ctx.findTensorById("tensor_c");
                tensorCForBranch.indices.push({
                  id: "tensor_c_z",
                  name: "z",
                  dimension: 7,
                  offset: { x: 0, y: 0 },
                  metadata: {},
                });
                ctx.state.spec.tensors.push({
                  id: "tensor_d",
                  name: "D",
                  position: { x: 540, y: 360 },
                  size: { width: 180, height: 108 },
                  indices: [
                    {
                      id: "tensor_d_z",
                      name: "z",
                      dimension: 7,
                      offset: { x: 0, y: 0 },
                      metadata: {},
                    },
                  ],
                  metadata: {},
                });
                ctx.state.spec.edges.push({
                  id: "edge_cd",
                  name: "bond_cd",
                  left: { tensor_id: "tensor_c", index_id: "tensor_c_z" },
                  right: { tensor_id: "tensor_d", index_id: "tensor_d_z" },
                  metadata: {},
                });
              },
              {
                selectionIds: ["tensor_a", "tensor_b", "tensor_c", "tensor_d"],
                primaryId: "tensor_b",
                statusMessage: "Added tensor D for tree testing.",
              }
            );
            ctx.arrangeSelectedTensors("tree");
            const branchedTensorB = ctx.findTensorById("tensor_b");
            const branchedTensorA = ctx.findTensorById("tensor_a");
            const branchedTensorC = ctx.findTensorById("tensor_c");
            const branchedTensorD = ctx.findTensorById("tensor_d");
            if (
              !(
                branchedTensorB.position.y < branchedTensorA.position.y
                && branchedTensorB.position.y < branchedTensorC.position.y
                && branchedTensorC.position.y < branchedTensorD.position.y
              )
            ) {
              throw new Error(
                "Arrange Tree should keep branched descendants below their parent tensors."
              );
            }
            if (
              !(
                Math.abs(branchedTensorD.position.x - branchedTensorC.position.x)
                < Math.abs(branchedTensorD.position.x - branchedTensorA.position.x)
              )
            ) {
              throw new Error(
                "Arrange Tree should keep a descendant closer to its parent branch than to an unrelated sibling branch."
              );
            }

            ctx.arrangeSelectedTensors("grid");
            const gridSelection = ctx.state.selectionIds.map((tensorId) => ctx.findTensorById(tensorId));
            const uniqueGridXs = new Set(gridSelection.map((tensor) => tensor.position.x));
            const uniqueGridYs = new Set(gridSelection.map((tensor) => tensor.position.y));
            if (!(uniqueGridXs.size === 2 && uniqueGridYs.size === 2)) {
              throw new Error("Arrange Grid should place four tensors on a 2x2 grid.");
            }

            ctx.state.spec.edges = [];
            ctx.state.spec.tensors[0].position = { x: 140, y: 180 };
            ctx.state.spec.tensors[1].position = { x: 280, y: 120 };
            ctx.state.spec.tensors[2].position = { x: 430, y: 260 };
            ctx.state.spec.tensors[3].position = { x: 610, y: 200 };
            ctx.state.selectionIds = ["tensor_a", "tensor_b", "tensor_c", "tensor_d"];
            ctx.state.primarySelectionId = "tensor_b";
            ctx.arrangeSelectedTensors("tree");
            const noBondTreeTensors = ctx.state.selectionIds.map((tensorId) => ctx.findTensorById(tensorId));
            const noBondTreeYs = noBondTreeTensors.map((tensor) => tensor.position.y);
            const noBondTreeRoot = ctx.findTensorById("tensor_b");
            if (!(new Set(noBondTreeYs).size > 1)) {
              throw new Error(
                `Arrange Tree without bonds should still create multiple levels, received ${noBondTreeYs.join(", ")}.`
              );
            }
            if (!noBondTreeTensors.every((tensor) => tensor.id === "tensor_b" || noBondTreeRoot.position.y < tensor.position.y)) {
              throw new Error(
                "Arrange Tree without bonds should place the primary tensor above the remaining tensors."
              );
            }

            ctx.state.spec.tensors[0].position = { x: 120, y: 300 };
            ctx.state.spec.tensors[1].position = { x: 270, y: 110 };
            ctx.state.spec.tensors[2].position = { x: 520, y: 280 };
            ctx.state.spec.tensors[3].position = { x: 700, y: 190 };
            ctx.arrangeSelectedTensors("grid");
            const noBondGridSelection = ctx.state.selectionIds.map((tensorId) => ctx.findTensorById(tensorId));
            const noBondGridXs = new Set(noBondGridSelection.map((tensor) => tensor.position.x));
            const noBondGridYs = new Set(noBondGridSelection.map((tensor) => tensor.position.y));
            if (!(noBondGridXs.size === 2 && noBondGridYs.size === 2)) {
              throw new Error(
                `Arrange Grid without bonds should still place four tensors on a 2x2 grid, received x=${[...noBondGridXs].join(",")} y=${[...noBondGridYs].join(",")}.`
              );
            }

            await ctx.exportSelectedSubnetwork();
            if (!apiCalls.some((call) => call.path === "/api/subnetwork/extract")) {
              throw new Error("Selection export did not call the extract subnetwork API.");
            }
            if (downloadEvents.length !== 1) {
              throw new Error(`Expected one subnetwork export download, received ${downloadEvents.length}.`);
            }
            if (downloadEvents[0].filename !== "selection_fragment_export.json") {
              throw new Error(
                `Expected the exported subnetwork filename to use the prompted name, received ${downloadEvents[0].filename}.`
              );
            }
            const exportedSubnetworkPayload = JSON.parse(downloadEvents[0].text);
            if (exportedSubnetworkPayload.network.name !== "selection_fragment_export") {
              throw new Error(
                `Expected the exported subnetwork payload to use the prompted name, received ${JSON.stringify(exportedSubnetworkPayload.network)}.`
              );
            }

            await ctx.promoteSelectedSubnetworkToTemplate();
            const promotedSessionTemplate = ctx.state.availableTemplates.find((templateName) =>
              templateName.startsWith("session::")
            );
            if (!promotedSessionTemplate) {
              throw new Error("Promote Selection to Template did not add a session template.");
            }
            const promotedEntry = ctx.listTemplateEntries().find(
              (entry) => entry.templateName === promotedSessionTemplate
            );
            if (!promotedEntry || promotedEntry.displayName !== "selection_fragment") {
              throw new Error(`Expected the promoted session template to use the prompted name, received ${JSON.stringify(promotedEntry)}.`);
            }
            if (ctx.dom.templateSelect.value !== promotedSessionTemplate) {
              throw new Error(`Expected promoted template to become selected, received ${ctx.dom.templateSelect.value}.`);
            }

            downloadEvents.length = 0;
            ctx.window.prompt = createPromptQueue(["selection_fragment_file"]);
            await ctx.exportSelectedTemplateSpec();
            if (downloadEvents.length !== 1) {
              throw new Error(`Expected one template export download, received ${downloadEvents.length}.`);
            }
            if (downloadEvents[0].filename !== "selection_fragment_file.json") {
              throw new Error(`Expected the exported template filename to use the prompted name, received ${downloadEvents[0].filename}.`);
            }
            const exportedTemplatePayload = JSON.parse(downloadEvents[0].text);
            if (exportedTemplatePayload.templates[0].display_name !== "selection_fragment_file") {
              throw new Error(
                `Expected the exported template payload to preserve the prompted display name, received ${JSON.stringify(exportedTemplatePayload.templates[0])}.`
              );
            }

            ctx.insertPreparedSubnetwork(
              {
                id: "prepared_fragment",
                name: "Prepared fragment",
                tensors: [
                  {
                    id: "prepared_tensor",
                    name: "Prepared",
                    position: { x: 520, y: 340 },
                    size: { width: 180, height: 108 },
                    indices: [],
                    metadata: {},
                  },
                ],
                groups: [],
                edges: [],
                notes: [],
                contraction_plan: null,
                metadata: {},
              },
              "Prepared fragment"
            );
            if (!ctx.state.spec.tensors.some((tensor) => tensor.id === "prepared_tensor")) {
              throw new Error("Prepared subnetwork insertion did not append the new tensor.");
            }
            if (ctx.state.selectionIds.join(",") !== "prepared_tensor") {
              throw new Error(`Prepared subnetwork insertion should select the new tensors, received ${ctx.state.selectionIds.join(",")}.`);
            }
            if (!/Inserted Prepared fragment/.test(ctx.dom.statusMessage.textContent)) {
              throw new Error("Prepared subnetwork insertion did not report a success status.");
            }
            if (ctx.state.lastImportedTensorIds.join(",") !== "prepared_tensor") {
              throw new Error(`Prepared subnetwork insertion should track the imported tensor ids, received ${ctx.state.lastImportedTensorIds.join(",")}.`);
            }

            await ctx.insertTemplate();
            const importedSessionTemplateIds = [...ctx.state.lastImportedTensorIds];
            if (importedSessionTemplateIds.length !== 1) {
              throw new Error(`Expected session template insertion to track one imported tensor, received ${importedSessionTemplateIds.join(",")}.`);
            }

            ctx.dom.templateSelect.value = "mps";
            ctx.handleTemplateSelectionChange({ target: ctx.dom.templateSelect });
            await ctx.insertTemplate();
            const importedTemplateIds = [...ctx.state.lastImportedTensorIds];
            if (importedTemplateIds.length !== 3) {
              throw new Error(`Expected template insertion to track three imported tensors, received ${importedTemplateIds.join(",")}.`);
            }
            if (ctx.state.selectionIds.join(",") !== importedTemplateIds.join(",")) {
              throw new Error("Template insertion should select the imported tensors.");
            }

            ctx.reflowLastImportedTensors();
            const reflowedTemplateTensors = importedTemplateIds.map((tensorId) => ctx.findTensorById(tensorId));
            const reflowedTemplateYs = reflowedTemplateTensors.map((tensor) => tensor.position.y);
            if (!reflowedTemplateYs.every((value) => value === reflowedTemplateYs[0])) {
              throw new Error("Reflow Imported should arrange the imported path tensors into a horizontal chain.");
            }

            ctx.state.spec = ctx.normalizeSpec({
              id: "network_auto_layout",
              name: "auto-layout",
              tensors: [
                {
                  id: "auto_a",
                  name: "A",
                  position: { x: 120, y: 160 },
                  size: { width: 180, height: 108 },
                  indices: [],
                  metadata: {},
                },
                {
                  id: "auto_b",
                  name: "B",
                  position: { x: 360, y: 120 },
                  size: { width: 180, height: 108 },
                  indices: [],
                  metadata: {},
                },
                {
                  id: "auto_c",
                  name: "C",
                  position: { x: 280, y: 360 },
                  size: { width: 180, height: 108 },
                  indices: [],
                  metadata: {},
                },
                {
                  id: "auto_d",
                  name: "D",
                  position: { x: 520, y: 300 },
                  size: { width: 180, height: 108 },
                  indices: [],
                  metadata: {},
                },
              ],
              groups: [],
              edges: [
                {
                  id: "edge_ab",
                  name: "ab",
                  left: { tensor_id: "auto_a", index_id: null },
                  right: { tensor_id: "auto_b", index_id: null },
                  metadata: {},
                },
                {
                  id: "edge_bc",
                  name: "bc",
                  left: { tensor_id: "auto_b", index_id: null },
                  right: { tensor_id: "auto_c", index_id: null },
                  metadata: {},
                },
                {
                  id: "edge_cd",
                  name: "cd",
                  left: { tensor_id: "auto_c", index_id: null },
                  right: { tensor_id: "auto_d", index_id: null },
                  metadata: {},
                },
                {
                  id: "edge_da",
                  name: "da",
                  left: { tensor_id: "auto_d", index_id: null },
                  right: { tensor_id: "auto_a", index_id: null },
                  metadata: {},
                },
              ],
              notes: [],
              contraction_plan: null,
              metadata: {},
            });
            ctx.state.selectionIds = [];
            ctx.state.primarySelectionId = null;
            ctx.applyReflowLayoutAction("auto");
            const autoLayoutTensors = ctx.state.spec.tensors.map((tensor) => ({
              id: tensor.id,
              x: tensor.position.x,
              y: tensor.position.y,
              width: tensor.size.width,
              height: tensor.size.height,
            }));
            if (ctx.state.selectionIds.length !== 0 || ctx.state.primarySelectionId !== null) {
              throw new Error("Whole-graph auto layout should preserve an empty tensor selection.");
            }
            if (new Set(autoLayoutTensors.map((tensor) => tensor.y)).size < 2) {
              throw new Error(
                `Whole-graph auto layout should create multiple layers for cyclic graphs, received ${JSON.stringify(autoLayoutTensors)}.`
              );
            }
            for (let leftIndex = 0; leftIndex < autoLayoutTensors.length; leftIndex += 1) {
              for (let rightIndex = leftIndex + 1; rightIndex < autoLayoutTensors.length; rightIndex += 1) {
                const leftTensor = autoLayoutTensors[leftIndex];
                const rightTensor = autoLayoutTensors[rightIndex];
                const overlapX =
                  Math.abs(leftTensor.x - rightTensor.x)
                  < (leftTensor.width + rightTensor.width) / 2;
                const overlapY =
                  Math.abs(leftTensor.y - rightTensor.y)
                  < (leftTensor.height + rightTensor.height) / 2;
                if (overlapX && overlapY) {
                  throw new Error(
                    `Whole-graph auto layout should avoid tensor overlap, received ${JSON.stringify(autoLayoutTensors)}.`
                  );
                }
              }
            }
            """
        ),
        encoding="utf-8",
    )
    return script_path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_layout_and_subnetwork_runtime_regression(tmp_path: Path) -> None:
    script_path = _write_layout_subnetwork_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The layout and subnetwork runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


def _write_template_catalog_management_runtime_regression_script(
    tmp_path: Path,
) -> Path:
    script_path = tmp_path / "template_catalog_management_runtime_regression.mjs"
    _copy_runtime_bundle(
        tmp_path,
        {
            "state.runtime.mjs": "state/state.js",
            "utilities.runtime.mjs": "utils/utilities.js",
            "historySelection.runtime.mjs": "graph/historySelection.js",
        },
        _RUNTIME_EDITOR_SUPPORT_MODULES,
    )
    script_body = textwrap.dedent(
        """
        import { pathToFileURL } from "node:url";

        function createClassList() {
          return {
            add() {},
            remove() {},
            toggle() {},
          };
        }

        function createButton() {
          return {
            disabled: false,
            title: "",
            classList: createClassList(),
            focus() {},
            addEventListener() {},
          };
        }

        function createSelect(value) {
          return {
            value,
            options: [],
            innerHTML: "",
            appendChild(option) {
              this.options.push(option);
              if (option && option.selected) {
                this.value = option.value;
              }
            },
            addEventListener() {},
          };
        }

        function createFakeNode(tagName = "div", ownerDocument = null) {
          return {
            tagName: String(tagName || "div").toUpperCase(),
            ownerDocument,
            value: "",
            textContent: "",
            disabled: false,
            hidden: false,
            type: "",
            className: "",
            dataset: {},
            attributes: {},
            children: [],
            listeners: {},
            classList: createClassList(),
            addEventListener(eventName, listener) {
              this.listeners[eventName] = listener;
            },
            append(...items) {
              items.forEach((item) => this.appendChild(item));
            },
            appendChild(item) {
              if (!item) {
                return item;
              }
              this.children.push(item);
              return item;
            },
            setAttribute(name, value) {
              this.attributes[name] = String(value);
            },
            focus() {
              if (this.ownerDocument) {
                this.ownerDocument.activeElement = this;
              }
            },
            querySelector(selector) {
              const inputMatch = /^input\\[data-template-name=\"([^\"]+)\"\\]$/.exec(selector);
              if (!inputMatch) {
                return null;
              }
              const templateName = inputMatch[1];
              const stack = [...this.children];
              while (stack.length) {
                const node = stack.shift();
                if (
                  node
                  && node.tagName === "INPUT"
                  && node.dataset
                  && node.dataset.templateName === templateName
                ) {
                  return node;
                }
                if (node && Array.isArray(node.children) && node.children.length) {
                  stack.unshift(...node.children);
                }
              }
              return null;
            },
          };
        }

        function createFakeDocument() {
          return {
            activeElement: null,
            createElement(tagName) {
              return createFakeNode(tagName, this);
            },
            querySelectorAll() {
              return [];
            },
          };
        }

        function createFakeList(ownerDocument) {
          const list = createFakeNode("div", ownerDocument);
          let html = "";
          Object.defineProperty(list, "innerHTML", {
            get() {
              return html;
            },
            set(value) {
              html = value;
              list.children = [];
            },
          });
          return list;
        }

        function createPromptQueue(values) {
          const queue = [...values];
          return () => (queue.length ? queue.shift() : null);
        }

        const baseUrl = new URL("./", import.meta.url);
        const [stateModule, utilitiesModule, historyModule, sessionModule, selectorsModule, storeModule] =
          await Promise.all([
            import(new URL("./state.runtime.mjs", baseUrl).href),
            import(new URL("./utilities.runtime.mjs", baseUrl).href),
            import(new URL("./historySelection.runtime.mjs", baseUrl).href),
            import(new URL("./interactionsSession.js", baseUrl).href),
            import(new URL("./state/editorSelectors.js", baseUrl).href),
            import(new URL("./state/editorStore.js", baseUrl).href),
          ]);

        const { createInitialState } = stateModule;
        const { registerUtilities } = utilitiesModule;
        const { registerHistorySelection } = historyModule;
        const { createInteractionSessionBindings } = sessionModule;
        const { createEditorSelectors } = selectorsModule;
        const { createEditorStore } = storeModule;

        const apiCalls = [];
        const confirmMessages = [];
        let deleteResponseUsed = false;

        const fakeDocument = createFakeDocument();
        const templateManagerList = createFakeList(fakeDocument);
        const subnetworkLibraryList = createFakeList(fakeDocument);

        const ctx = {
          state: createInitialState(),
          constants: {
            TENSOR_WIDTH: 180,
            TENSOR_HEIGHT: 108,
            MIN_TENSOR_WIDTH: 140,
            MIN_TENSOR_HEIGHT: 84,
            INDEX_RADIUS: 15,
            INDEX_PADDING: 8,
            NOTE_WIDTH: 220,
            NOTE_HEIGHT: 152,
            NOTE_MIN_WIDTH: 176,
            NOTE_MIN_HEIGHT: 152,
            HISTORY_LIMIT: 100,
            REDO_SHORTCUT_LABEL: "Ctrl+Shift+Z",
            GRID_SNAP_SIZE: 20,
            DEFAULT_INDEX_SLOTS: [
              { x: -58, y: -20 },
              { x: 58, y: -20 },
              { x: -58, y: 20 },
              { x: 58, y: 20 },
            ],
          },
          dom: {
            workspace: {},
            statusMessage: {
              textContent: "",
              classList: createClassList(),
            },
            propertiesPanel: { innerHTML: "" },
            generatedCode: { value: "" },
            generatedCodeView: { textContent: "", dataset: {} },
            engineSelect: createSelect("tensornetwork"),
            collectionFormatSelect: createSelect("list"),
            exportFormatSelect: createSelect("py"),
            addNoteButton: createButton(),
            connectButton: createButton(),
            loadInput: { value: "", click() {}, addEventListener() {} },
            subnetworkLoadInput: { value: "", click() {}, addEventListener() {} },
            undoButton: createButton(),
            redoButton: createButton(),
            exportButton: createButton(),
            toggleLinearPeriodicButton: createButton(),
            linearPeriodicPreviousCellButton: createButton(),
            linearPeriodicCellLabel: { textContent: "" },
            linearPeriodicNextCellButton: createButton(),
            templateSelect: createSelect(""),
            templateParameterPanel: { hidden: true },
            templateGraphSizeField: { hidden: false },
            templateGraphSizeLabel: { textContent: "" },
            templateGraphSizeInput: { value: "2", min: "1" },
            templateBondDimensionField: { hidden: false },
            templateBondDimensionInput: { value: "3", min: "1" },
            templatePhysicalDimensionField: { hidden: false },
            templatePhysicalDimensionInput: { value: "2", min: "1" },
            insertTemplateButton: createButton(),
            insertSubnetworkButton: createButton(),
            renameTemplateButton: createButton(),
            deleteTemplateButton: createButton(),
            templateManagerModal: { classList: createClassList() },
            templateManagerBackdrop: createButton(),
            templateManagerSaveButton: createButton(),
            templateManagerDiscardButton: createButton(),
            templateManagerError: {
              hidden: true,
              textContent: "",
            },
            templateManagerList,
            subnetworkLibraryModal: { classList: createClassList() },
            subnetworkLibraryBackdrop: createButton(),
            subnetworkLibraryCloseButton: createButton(),
            subnetworkLibrarySearchInput: createFakeNode("input", fakeDocument),
            subnetworkLibraryTagFilter: createSelect(""),
            subnetworkLibrarySelectAllInput: createFakeNode("input", fakeDocument),
            subnetworkLibrarySelectionSummary: createFakeNode("span", fakeDocument),
            subnetworkLibraryAddSelectedButton: createFakeNode("button", fakeDocument),
            subnetworkLibraryWarning: createFakeNode("p", fakeDocument),
            subnetworkLibraryList,
            templateCatalogWarning: {
              hidden: true,
              textContent: "",
              title: "",
            },
            reflowImportedButton: createButton(),
            createGroupButton: createButton(),
            helpButton: createButton(),
            helpModal: { classList: createClassList() },
            helpBackdrop: createButton(),
            helpCloseButton: createButton(),
            canvasShell: {
              getBoundingClientRect() {
                return { left: 0, top: 0, width: 1000, height: 800 };
              },
              addEventListener() {},
            },
            groupLayer: {},
            resizeLayer: {},
            notesLayer: {},
            selectionBox: {
              classList: createClassList(),
              style: {},
            },
            minimapCanvas: {
              classList: createClassList(),
              addEventListener() {},
            },
            sidebar: {},
            plannerPanel: {},
            generateButton: createButton(),
            codeGenerationWarning: {
              textContent: "",
              title: "",
              hidden: true,
            },
          },
          apiGet: async () => null,
          apiPost: async (path, payload) => {
            apiCalls.push({ path, payload });
            if (path === "/api/subnetwork/extract") {
              return {
                ok: true,
                spec: {
                  schema_version: 4,
                  network: {
                    id: "selection_fragment",
                    name: "Selection Fragment",
                    tensors: [
                      {
                        id: "fragment_a",
                        name: "Fragment A",
                        position: { x: 140, y: 180 },
                        size: { width: 180, height: 108 },
                        indices: [],
                        metadata: {},
                      },
                    ],
                    groups: [],
                    edges: [],
                    notes: [],
                    contraction_plan: null,
                    metadata: {},
                  },
                },
              };
            }
            if (path === "/api/template/promote") {
              return {
                ok: true,
                templates: ["project_fragment", "mps"],
                template_definitions: {
                  project_fragment: {
                    display_name: "Project Fragment",
                    graph_size_label: "Tensors",
                    defaults: {
                      graph_size: 2,
                      bond_dimension: 1,
                      physical_dimension: 1,
                    },
                    minimums: {
                      graph_size: 1,
                      bond_dimension: 1,
                      physical_dimension: 1,
                    },
                    supports_parameters: false,
                    source: "project",
                  },
                  mps: {
                    display_name: "MPS",
                    graph_size_label: "Sites",
                    defaults: {
                      graph_size: 4,
                      bond_dimension: 3,
                      physical_dimension: 2,
                    },
                    minimums: {
                      graph_size: 2,
                      bond_dimension: 1,
                      physical_dimension: 1,
                    },
                    supports_parameters: true,
                    source: "global",
                  },
                },
                selected_template: "project_fragment",
                template_catalog_warnings: [],
              };
            }
            if (path === "/api/template/rename") {
              return {
                ok: true,
                templates: ["renamed_fragment", "project_second", "mps"],
                template_definitions: {
                  renamed_fragment: {
                    display_name: "Renamed Fragment",
                    graph_size_label: "Tensors",
                    defaults: {
                      graph_size: 2,
                      bond_dimension: 1,
                      physical_dimension: 1,
                    },
                    minimums: {
                      graph_size: 1,
                      bond_dimension: 1,
                      physical_dimension: 1,
                    },
                    supports_parameters: false,
                    source: "project",
                  },
                  project_second: {
                    display_name: "Project Second",
                    graph_size_label: "Tensors",
                    defaults: {
                      graph_size: 1,
                      bond_dimension: 1,
                      physical_dimension: 1,
                    },
                    minimums: {
                      graph_size: 1,
                      bond_dimension: 1,
                      physical_dimension: 1,
                    },
                    supports_parameters: false,
                    source: "project",
                  },
                  mps: {
                    display_name: "MPS",
                    graph_size_label: "Sites",
                    defaults: {
                      graph_size: 4,
                      bond_dimension: 3,
                      physical_dimension: 2,
                    },
                    minimums: {
                      graph_size: 2,
                      bond_dimension: 1,
                      physical_dimension: 1,
                    },
                    supports_parameters: true,
                    source: "global",
                  },
                },
                selected_template: "renamed_fragment",
                template_catalog_warnings: [],
              };
            }
            if (path === "/api/template/delete") {
              deleteResponseUsed = true;
              return {
                ok: true,
                templates: ["project_second", "mps"],
                template_definitions: {
                  project_second: {
                    display_name: "Project Second",
                    graph_size_label: "Tensors",
                    defaults: {
                      graph_size: 1,
                      bond_dimension: 1,
                      physical_dimension: 1,
                    },
                    minimums: {
                      graph_size: 1,
                      bond_dimension: 1,
                      physical_dimension: 1,
                    },
                    supports_parameters: false,
                    source: "project",
                  },
                  mps: {
                    display_name: "MPS",
                    graph_size_label: "Sites",
                    defaults: {
                      graph_size: 4,
                      bond_dimension: 3,
                      physical_dimension: 2,
                    },
                    minimums: {
                      graph_size: 2,
                      bond_dimension: 1,
                      physical_dimension: 1,
                    },
                    supports_parameters: true,
                    source: "global",
                  },
                },
                selected_template: "project_second",
                template_catalog_warnings: [],
              };
            }
            throw new Error(`Unexpected API path: ${path}`);
          },
          window: {
            structuredClone: globalThis.structuredClone,
            crypto: globalThis.crypto,
            setTimeout,
            clearTimeout,
            confirm(message) {
              confirmMessages.push(message);
              return true;
            },
            prompt: createPromptQueue(["Session Fragment", "Session Fragment 2"]),
            Prism: null,
          },
          document: fakeDocument,
          cytoscape: null,
          render() {},
          renderOverlayDecorations() {},
          renderMinimap() {},
          renderPlanner() {},
          renderSidebarTabs() {},
          refreshContractionAnalysis() {},
          repairContractionPlan() {},
        };

        registerUtilities(ctx);
        registerHistorySelection(ctx);
        ctx.store = createEditorStore(ctx.state);
        ctx.selectors = createEditorSelectors({ store: ctx.store });
        ctx.services = {
          session: {
            async generateCode(payload) {
              return ctx.apiPost("/api/generate", {
                engine: payload.engine,
                collection_format: payload.collectionFormat,
                spec: payload.spec,
              });
            },
            async completeSession(payload) {
              return ctx.apiPost("/api/complete", {
                engine: payload.engine,
                collection_format: payload.collectionFormat,
                spec: payload.spec,
              });
            },
            async cancelSession() {
              return ctx.apiPost("/api/cancel", {});
            },
            async validatePythonCode(payload) {
              return ctx.apiPost("/api/validate", {
                python_code: payload,
              });
            },
            async validateSerializedSpec(payload) {
              return ctx.apiPost("/api/validate", {
                spec: payload,
              });
            },
            async buildTemplate(payload) {
              return ctx.apiPost("/api/template", {
                template: payload.templateName,
                parameters: payload.parameters,
              });
            },
          },
          templateCatalog: {
            async promoteTemplate(payload) {
              return ctx.apiPost("/api/template/promote", {
                spec: payload.serializedSpec,
                tensor_ids: payload.tensorIds,
                template_name: payload.templateName,
                overwrite: payload.overwrite,
              });
            },
            async renameTemplate(payload) {
              return ctx.apiPost("/api/template/rename", {
                template_name: payload.templateName,
                new_template_name: payload.newTemplateName,
                overwrite: payload.overwrite,
              });
            },
            async deleteTemplate(payload) {
              return ctx.apiPost("/api/template/delete", {
                template_name: payload.templateName,
              });
            },
          },
          subnetwork: {
            async extractSubnetwork(payload) {
              return ctx.apiPost("/api/subnetwork/extract", {
                spec: payload.serializedSpec,
                tensor_ids: payload.tensorIds,
              });
            },
            async prepareSubnetworkForInsert(payload) {
              return ctx.apiPost("/api/subnetwork/prepare-insert", {
                spec: payload.serializedSpec,
                target_center: payload.targetCenter,
              });
            },
          },
        };
        Object.assign(
          ctx,
          createInteractionSessionBindings({
            ctx,
            state: ctx.state,
            dom: ctx.dom,
            store: ctx.store,
            selectors: ctx.selectors,
            services: ctx.services,
            sessionUi: {
              async copyText() {},
              downloadText() {},
              downloadBlob() {},
              requestFileText: async (file) => file.text(),
              openFilePicker(input) {
                input.click();
              },
              schedule(callback) {
                callback();
              },
              closeWindow() {},
              promptText: (...args) => ctx.window.prompt(...args),
              confirmAction: (...args) => ctx.window.confirm(...args),
            },
            sessionActions: {
              ensureCodePanelVisible() {},
              syncCodeGenerationWarning() {},
              getTensorKrowchManualPlanIssueMessage() {
                return "";
              },
              getSelectedTensorIds: () => ctx.getSelectedIdsByKind("tensor"),
              findGroupById: (groupId) => ctx.findGroupById(groupId),
              isLinearPeriodicMode: () => ctx.isLinearPeriodicMode(),
              syncGeneratedCodePreview: (code) =>
                ctx.renderGeneratedCodePreview(code),
              setStatus: (message, level) => ctx.setStatus(message, level),
              serializeCurrentSpec: (options) => ctx.serializeCurrentSpec(options),
              formatIssues: (issues) => ctx.formatIssues(issues),
              stripImportLines: (code) => ctx.stripImportLines(code),
              sanitizeFilename: (value) => ctx.sanitizeFilename(value),
              resetDesignState: (spec, message, schemaVersion) =>
                ctx.resetDesignState(spec, message, schemaVersion),
              downloadPngExport: () => ctx.downloadPngExport(),
              downloadSvgExport: () => ctx.downloadSvgExport(),
              applyTemplateCatalogPayload: (payload) =>
                ctx.applyTemplateCatalogPayload(payload),
              normalizeSpec: (spec) => ctx.normalizeSpec(spec),
              applyDesignChange: (mutate, options) =>
                ctx.applyDesignChange(mutate, options),
              bringTensorToFront: (tensorId) => ctx.bringTensorToFront(tensorId),
              formatTemplateLabel: (value) => ctx.formatTemplateLabel(value),
              getTemplateSource: (templateName) => ctx.getTemplateSource(templateName),
              getTemplateSpec: (templateName) => ctx.getTemplateSpec(templateName),
              listTemplateEntries: () => ctx.listTemplateEntries(),
              hasTemplateDisplayName: (displayName, excludedTemplateName) =>
                ctx.hasTemplateDisplayName(displayName, excludedTemplateName),
              getNextSessionTemplateDisplayName: (baseDisplayName) =>
                ctx.getNextSessionTemplateDisplayName(baseDisplayName),
              addSessionTemplate: (payload) => ctx.addSessionTemplate(payload),
              updateSessionTemplateDisplayNames: (updates) =>
                ctx.updateSessionTemplateDisplayNames(updates),
              removeSessionTemplate: (templateName) =>
                ctx.removeSessionTemplate(templateName),
              toggleSubnetworkLibrary: (forceOpen) =>
                ctx.toggleSubnetworkLibrary(forceOpen),
              toggleTemplateManager: (forceOpen) =>
                ctx.toggleTemplateManager(forceOpen),
              updateToolbarState: () => ctx.updateToolbarState(),
              syncTemplateManagerModalState: () =>
                ctx.syncTemplateManagerModalState(),
              syncSubnetworkLibraryModalState: () =>
                ctx.syncSubnetworkLibraryModalState(),
              setTemplateManagerValidationMessage: (message) =>
                ctx.setTemplateManagerValidationMessage(message),
              persistTemplateParametersFromControls: () =>
                ctx.persistTemplateParametersFromControls(),
              uniquifyImportedSpec: (spec, prefix) =>
                ctx.uniquifyImportedSpec(spec, prefix),
              makeId: (prefix) => ctx.makeId(prefix),
              translateImportedSpec: (spec, targetCenter) =>
                ctx.translateImportedSpec(spec, targetCenter),
              suggestTensorPosition: (position) =>
                ctx.suggestTensorPosition(position),
              viewportCenterPosition: () => ctx.viewportCenterPosition(),
            },
          })
        );

        ctx.state.spec = ctx.normalizeSpec({
          id: "network_demo",
          name: "demo",
          tensors: [
            {
              id: "tensor_a",
              name: "A",
              position: { x: 120, y: 160 },
              size: { width: 180, height: 108 },
              indices: [],
              metadata: {},
            },
            {
              id: "tensor_b",
              name: "B",
              position: { x: 320, y: 160 },
              size: { width: 180, height: 108 },
              indices: [],
              metadata: {},
            },
          ],
          groups: [],
          edges: [],
          notes: [],
          contraction_plan: null,
          linear_periodic_chain: null,
          metadata: {},
        });
        ctx.state.selectionIds = ["tensor_a", "tensor_b"];
        ctx.state.primarySelectionId = "tensor_b";

        ctx.applyTemplateCatalogPayload({
          templateNames: ["project_fragment", "mps"],
          templateDefinitions: {
            project_fragment: {
              display_name: "Project Fragment",
              graph_size_label: "Tensors",
              defaults: {
                graph_size: 2,
                bond_dimension: 1,
                physical_dimension: 1,
              },
              minimums: {
                graph_size: 1,
                bond_dimension: 1,
                physical_dimension: 1,
              },
              supports_parameters: false,
              source: "project",
            },
            mps: {
              display_name: "MPS",
              graph_size_label: "Sites",
              defaults: {
                graph_size: 4,
                bond_dimension: 3,
                physical_dimension: 2,
              },
              minimums: {
                graph_size: 2,
                bond_dimension: 1,
                physical_dimension: 1,
              },
              supports_parameters: true,
              source: "global",
            },
          },
          selectedTemplate: "mps",
          templateCatalogWarnings: ["First warning", "Second warning"],
        });

        if (ctx.dom.templateCatalogWarning.hidden) {
          throw new Error("Template catalog warning should be visible when warnings are present.");
        }
        if (!ctx.dom.templateCatalogWarning.textContent.includes("First warning")) {
          throw new Error(`Expected the first template warning to be shown, received ${ctx.dom.templateCatalogWarning.textContent}.`);
        }
        if (ctx.dom.templateCatalogWarning.dataset.tooltipEnabled !== "true") {
          throw new Error("Template catalog warning should expose the shared tooltip behavior.");
        }
        if (!ctx.dom.templateCatalogWarning.dataset.shortcutDescription.includes("Second warning")) {
          throw new Error("Template catalog warning tooltip should include the full warning list.");
        }
        await ctx.promoteSelectedSubnetworkToTemplate();
        const firstSessionTemplate = ctx.state.availableTemplates.find((templateName) =>
          templateName.startsWith("session::")
        );
        if (!firstSessionTemplate) {
          throw new Error("Promote Selection should add a session template.");
        }
        const firstSessionEntry = ctx.listTemplateEntries().find(
          (entry) => entry.templateName === firstSessionTemplate
        );
        if (!firstSessionEntry || firstSessionEntry.displayName !== "Session Fragment") {
          throw new Error(`Expected the first session template to use the prompted name, received ${JSON.stringify(firstSessionEntry)}.`);
        }
        await ctx.promoteSelectedSubnetworkToTemplate();
        const secondSessionEntry = ctx.listTemplateEntries().find(
          (entry) => entry.displayName === "Session Fragment 2"
        );
        if (!secondSessionEntry) {
          throw new Error("Saving the same selection twice should keep the prompted names.");
        }
        ctx.toggleTemplateManager(true);
        if (ctx.dom.templateManagerList.children.length !== 2) {
          throw new Error(
            `Expected the template manager to show only session templates, received ${ctx.dom.templateManagerList.children.length} rows.`
          );
        }
        const firstManagerRow = ctx.dom.templateManagerList.children[0];
        if (!firstManagerRow) {
          throw new Error("Expected the template manager to render at least one editable row.");
        }
        if (!String(firstManagerRow.className || "").includes("subnetwork-library-row")) {
          throw new Error("Template manager rows should reuse the library card layout.");
        }
        const firstManagerPreview = firstManagerRow.children[0];
        if (
          !firstManagerPreview
          || !String(firstManagerPreview.className || "").includes("subnetwork-library-preview")
        ) {
          throw new Error("Template manager rows should include a thumbnail preview.");
        }
        const firstManagerContent = firstManagerRow.children[1];
        if (
          !firstManagerContent
          || !String(firstManagerContent.className || "").includes("subnetwork-library-content")
        ) {
          throw new Error("Template manager rows should include the shared library content area.");
        }
        const firstManagerSourceBadge = firstManagerContent.children[0]?.children[1];
        if (
          !firstManagerSourceBadge
          || firstManagerSourceBadge.textContent !== "Session"
        ) {
          throw new Error("Template manager rows should label session templates clearly.");
        }
        const firstManagerDeleteButton = firstManagerRow.children[2]?.children[0];
        if (!firstManagerDeleteButton) {
          throw new Error("Expected the template manager row to include a delete action.");
        }
        if (!String(firstManagerDeleteButton.innerHTML || "").includes("<svg")) {
          throw new Error("Template manager delete actions should render a trash icon.");
        }
        if (String(firstManagerDeleteButton.innerHTML || "").includes("Delete")) {
          throw new Error("Template manager delete actions should keep only the trash icon.");
        }
        if (
          !String(firstManagerDeleteButton.attributes["aria-label"] || "").includes("Delete")
        ) {
          throw new Error("Template manager delete actions should keep an accessible label.");
        }
        const firstManagerInput = ctx.dom.templateManagerList.querySelector(
          `input[data-template-name="${firstSessionTemplate}"]`
        );
        if (!firstManagerInput) {
          throw new Error("Expected the session template row to remain editable in the manager.");
        }
        firstManagerInput.value = "Discarded Rename";
        if (ctx.discardTemplateManagerChanges() !== false) {
          throw new Error("Discarding template manager edits should close the manager without saving.");
        }
        const discardedEntry = ctx.listTemplateEntries().find(
          (entry) => entry.templateName === firstSessionTemplate
        );
        if (!discardedEntry || discardedEntry.displayName !== "Session Fragment") {
          throw new Error(
            `Discarding template manager edits should keep the original session template name, received ${JSON.stringify(discardedEntry)}.`
          );
        }

        ctx.toggleTemplateManager(true);
        const reopenedFirstManagerInput = ctx.dom.templateManagerList.querySelector(
          `input[data-template-name="${firstSessionTemplate}"]`
        );
        if (!reopenedFirstManagerInput) {
          throw new Error("Expected the template manager to re-open with the session template row.");
        }
        reopenedFirstManagerInput.value = "Project Fragment";
        if (ctx.saveTemplateManagerChanges() !== true) {
          throw new Error("Saving the template manager with a duplicate locked name should be blocked.");
        }
        if (!ctx.state.isTemplateManagerOpen) {
          throw new Error("The template manager should stay open when validation fails.");
        }
        if (
          ctx.dom.templateManagerError.textContent
          !== "Template name 'Project Fragment' is already in use."
        ) {
          throw new Error(
            `Expected duplicate-name validation against hidden locked templates, received ${ctx.dom.templateManagerError.textContent}.`
          );
        }
        reopenedFirstManagerInput.value = "Manager Rename";
        if (ctx.saveTemplateManagerChanges() !== false) {
          throw new Error("Saving the template manager with valid session names should succeed.");
        }
        const managerRenamedEntry = ctx.listTemplateEntries().find(
          (entry) => entry.templateName === firstSessionTemplate
        );
        if (!managerRenamedEntry || managerRenamedEntry.displayName !== "Manager Rename") {
          throw new Error(
            `Expected template manager rename to persist for session templates, received ${JSON.stringify(managerRenamedEntry)}.`
          );
        }

        ctx.store.setSubnetworkCatalogData({
          subnetworkNames: ["project_fragment_copy", "shared_pair", "other_block"],
          subnetworkDefinitions: {
            project_fragment_copy: {
              display_name: "Project Fragment",
              source: "project",
              tags: ["bundle"],
              tensor_count: 1,
              edge_count: 0,
              spec: {
                schema_version: 4,
                network: {
                  id: "project_fragment_copy",
                  name: "Project Fragment",
                  tensors: [
                    {
                      id: "project_fragment_tensor",
                      name: "Project Fragment Tensor",
                      position: { x: 120, y: 160 },
                      size: { width: 180, height: 108 },
                      indices: [],
                      metadata: {},
                    },
                  ],
                  groups: [],
                  edges: [],
                  notes: [],
                  contraction_plan: null,
                  metadata: {},
                },
              },
            },
            shared_pair: {
              display_name: "Shared Pair",
              source: "shared",
              tags: ["bundle", "shared"],
              tensor_count: 2,
              edge_count: 1,
              spec: {
                schema_version: 4,
                network: {
                  id: "shared_pair",
                  name: "Shared Pair",
                  tensors: [
                    {
                      id: "shared_left",
                      name: "Left",
                      position: { x: 120, y: 160 },
                      size: { width: 180, height: 108 },
                      indices: [],
                      metadata: {},
                    },
                    {
                      id: "shared_right",
                      name: "Right",
                      position: { x: 320, y: 160 },
                      size: { width: 180, height: 108 },
                      indices: [],
                      metadata: {},
                    },
                  ],
                  groups: [],
                  edges: [
                    {
                      id: "shared_edge",
                      left: { tensor_id: "shared_left", index_id: "left_index" },
                      right: { tensor_id: "shared_right", index_id: "right_index" },
                    },
                  ],
                  notes: [],
                  contraction_plan: null,
                  metadata: {},
                },
              },
            },
            other_block: {
              display_name: "Other Block",
              source: "shared",
              tags: ["other"],
              tensor_count: 1,
              edge_count: 0,
              spec: {
                schema_version: 4,
                network: {
                  id: "other_block",
                  name: "Other Block",
                  tensors: [
                    {
                      id: "other_tensor",
                      name: "Other",
                      position: { x: 200, y: 200 },
                      size: { width: 180, height: 108 },
                      indices: [],
                      metadata: {},
                    },
                  ],
                  groups: [],
                  edges: [],
                  notes: [],
                  contraction_plan: null,
                  metadata: {},
                },
              },
            },
          },
          subnetworkCatalogWarnings: [],
          selectedSubnetworkName: "shared_pair",
        });
        if (ctx.openSubnetworkLibrary() !== true) {
          throw new Error("Expected the subnetwork library to open in normal graph mode.");
        }
        if (ctx.dom.subnetworkLibraryList.children.length !== 3) {
          throw new Error(
            `Expected the subnetwork library to render all saved entries, received ${ctx.dom.subnetworkLibraryList.children.length} rows.`
          );
        }
        ctx.updateSubnetworkLibraryTagFilter("bundle");
        if (ctx.dom.subnetworkLibraryList.children.length !== 2) {
          throw new Error(
            `Expected the tag filter to limit the visible subnetworks, received ${ctx.dom.subnetworkLibraryList.children.length} rows.`
          );
        }
        ctx.toggleSelectAllVisibleSubnetworks(true);
        if (
          ctx.state.selectedSubnetworkLibraryNames.join(",")
          !== "project_fragment_copy,shared_pair"
        ) {
          throw new Error(
            `Expected Select all visible to track the filtered subnetworks, received ${ctx.state.selectedSubnetworkLibraryNames.join(",")}.`
          );
        }
        if (ctx.dom.subnetworkLibrarySelectionSummary.textContent !== "2 selected") {
          throw new Error(
            `Expected the library selection summary to reflect the batch selection, received ${ctx.dom.subnetworkLibrarySelectionSummary.textContent}.`
          );
        }
        if (ctx.dom.subnetworkLibraryAddSelectedButton.disabled) {
          throw new Error("Expected batch add to enable once subnetworks are selected.");
        }
        if (!String(ctx.dom.subnetworkLibraryAddSelectedButton.textContent || "").includes("(2)")) {
          throw new Error("Expected the batch add button to show how many subnetworks will be added.");
        }
        ctx.addSelectedSubnetworksToSessionTemplates();
        const batchAddedEntry = ctx.listTemplateEntries().find(
          (entry) => entry.displayName === "Project Fragment 2"
        );
        if (!batchAddedEntry) {
          throw new Error("Adding a duplicate library name should suffix the new session template automatically.");
        }
        const sharedPairEntry = ctx.listTemplateEntries().find(
          (entry) => entry.displayName === "Shared Pair"
        );
        if (!sharedPairEntry) {
          throw new Error("Batch add should create a session template for each selected subnetwork.");
        }
        if (ctx.state.selectedSubnetworkLibraryNames.length !== 0) {
          throw new Error("Batch add should clear the current subnetwork-library selection.");
        }
        if (ctx.dom.subnetworkLibrarySelectionSummary.textContent !== "No subnetworks selected.") {
          throw new Error("Clearing the selection should reset the library selection summary.");
        }
        if (!ctx.dom.subnetworkLibraryAddSelectedButton.disabled) {
          throw new Error("Batch add button should be disabled again after the selection is cleared.");
        }

        ctx.dom.templateSelect.value = firstSessionTemplate;
        ctx.handleTemplateSelectionChange({ target: ctx.dom.templateSelect });
        ctx.window.prompt = createPromptQueue(["Renamed Session Fragment"]);
        await ctx.renameSelectedTemplate();
        const renamedEntry = ctx.listTemplateEntries().find(
          (entry) => entry.templateName === firstSessionTemplate
        );
        if (!renamedEntry || renamedEntry.displayName !== "Renamed Session Fragment") {
          throw new Error("Rename Template should update the selected session template locally.");
        }
        if (ctx.dom.templateSelect.value !== firstSessionTemplate) {
          throw new Error(`Expected the renamed template to stay selected, received ${ctx.dom.templateSelect.value}.`);
        }

        await ctx.deleteSelectedTemplate();
        if (ctx.state.availableTemplates.includes(firstSessionTemplate)) {
          throw new Error("Delete Template should remove the selected session template.");
        }
        if (ctx.dom.templateSelect.value !== secondSessionEntry.templateName) {
          throw new Error(`Expected delete to fall back to the next session template, received ${ctx.dom.templateSelect.value}.`);
        }
        if (ctx.state.templateDefinitions.project_fragment.source !== "project") {
          throw new Error("Project template metadata should remain read-only and keep source='project'.");
        }
        """
    )
    script_path.write_text(script_body, encoding="utf-8")
    return script_path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_template_catalog_management_runtime_regression(tmp_path: Path) -> None:
    script_path = _write_template_catalog_management_runtime_regression_script(tmp_path)
    completed_process = subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0, (
        "The template catalog management runtime regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )

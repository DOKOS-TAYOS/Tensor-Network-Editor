from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_runtime_editor_support_modules(tmp_path: Path) -> None:
    js_root = REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js"
    module_names = [
        "utilitiesBase.js",
        "utilitiesGeometry.js",
        "utilitiesLinearPeriodic.js",
        "utilitiesSpec.js",
        "utilitiesUi.js",
        "interactionsCanvas.js",
        "interactionsEditor.js",
        "interactionsSession.js",
        "interactionsShortcuts.js",
        "propertiesRenderersOverview.js",
        "propertiesRenderersTensor.js",
        "propertiesRenderersEntities.js",
    ]
    for module_name in module_names:
        (tmp_path / module_name).write_text(
            (js_root / module_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def _write_for_mode_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "for_mode_runtime_regression.mjs"
    state_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "state.js"
    )
    utilities_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utilities.js"
    )
    utilities_templates_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utilitiesTemplates.js"
    )
    history_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "historySelection.js"
    )
    state_runtime_path = tmp_path / "state.runtime.mjs"
    utilities_runtime_path = tmp_path / "utilities.runtime.mjs"
    utilities_templates_runtime_path = tmp_path / "utilitiesTemplates.js"
    history_runtime_path = tmp_path / "historySelection.runtime.mjs"
    state_runtime_path.write_text(
        state_module_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    utilities_runtime_path.write_text(
        utilities_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    utilities_templates_runtime_path.write_text(
        utilities_templates_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _copy_runtime_editor_support_modules(tmp_path)
    history_runtime_path.write_text(
        history_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
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


def _write_for_mode_reserved_operand_runtime_regression_script(
    tmp_path: Path,
) -> Path:
    script_path = tmp_path / "for_mode_reserved_operands_runtime_regression.mjs"
    state_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "state.js"
    )
    utilities_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utilities.js"
    )
    utilities_templates_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utilitiesTemplates.js"
    )
    planner_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "planner.js"
    )
    planner_support_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "plannerSupport.js"
    )
    planner_renderers_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "plannerRenderers.js"
    )
    contraction_scene_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "contractionScene.js"
    )
    state_runtime_path = tmp_path / "state.runtime.mjs"
    utilities_runtime_path = tmp_path / "utilities.runtime.mjs"
    utilities_templates_runtime_path = tmp_path / "utilitiesTemplates.js"
    planner_runtime_path = tmp_path / "planner.runtime.mjs"
    planner_support_runtime_path = tmp_path / "plannerSupport.js"
    planner_renderers_runtime_path = tmp_path / "plannerRenderers.js"
    contraction_scene_runtime_path = tmp_path / "contractionScene.runtime.mjs"
    state_runtime_path.write_text(
        state_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    utilities_runtime_path.write_text(
        utilities_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    utilities_templates_runtime_path.write_text(
        utilities_templates_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _copy_runtime_editor_support_modules(tmp_path)
    planner_runtime_path.write_text(
        planner_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    planner_support_runtime_path.write_text(
        planner_support_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    planner_renderers_runtime_path.write_text(
        planner_renderers_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    contraction_scene_runtime_path.write_text(
        contraction_scene_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
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
        / "utilities.js"
    )
    utilities_templates_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utilitiesTemplates.js"
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
    state_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "state.js"
    )
    utilities_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utilities.js"
    )
    utilities_templates_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utilitiesTemplates.js"
    )
    history_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "historySelection.js"
    )
    properties_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "properties.js"
    )
    properties_support_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "propertiesSupport.js"
    )
    properties_renderers_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "propertiesRenderers.js"
    )
    state_runtime_path = tmp_path / "state.runtime.mjs"
    utilities_runtime_path = tmp_path / "utilities.runtime.mjs"
    utilities_templates_runtime_path = tmp_path / "utilitiesTemplates.js"
    history_runtime_path = tmp_path / "historySelection.runtime.mjs"
    properties_runtime_path = tmp_path / "properties.runtime.mjs"
    properties_support_runtime_path = tmp_path / "propertiesSupport.js"
    properties_renderers_runtime_path = tmp_path / "propertiesRenderers.js"
    state_runtime_path.write_text(
        state_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    utilities_runtime_path.write_text(
        utilities_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    utilities_templates_runtime_path.write_text(
        utilities_templates_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _copy_runtime_editor_support_modules(tmp_path)
    history_runtime_path.write_text(
        history_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    properties_runtime_path.write_text(
        properties_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    properties_support_runtime_path.write_text(
        properties_support_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    properties_renderers_runtime_path.write_text(
        properties_renderers_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
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
    state_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "state.js"
    )
    sidebar_tabs_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "sidebarTabs.js"
    )
    state_runtime_path = tmp_path / "state.runtime.mjs"
    sidebar_tabs_runtime_path = tmp_path / "sidebarTabs.runtime.mjs"
    state_runtime_path.write_text(
        state_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    sidebar_tabs_runtime_path.write_text(
        sidebar_tabs_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
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
        };

        registerSidebarTabs(ctx);

        if (ctx.state.sidebarWidth !== 360) {
          throw new Error(`Expected default sidebar width 360, received ${ctx.state.sidebarWidth}.`);
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


def _write_utility_runtime_contract_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "utility_runtime_contract.mjs"
    js_root = REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js"
    copied_modules = {
        "state.runtime.mjs": "state.js",
        "utilities.runtime.mjs": "utilities.js",
        "utilitiesTemplates.js": "utilitiesTemplates.js",
        "utilitiesBase.js": "utilitiesBase.js",
        "utilitiesGeometry.js": "utilitiesGeometry.js",
        "utilitiesLinearPeriodic.js": "utilitiesLinearPeriodic.js",
        "utilitiesSpec.js": "utilitiesSpec.js",
        "utilitiesUi.js": "utilitiesUi.js",
    }
    for target_name, source_name in copied_modules.items():
        (tmp_path / target_name).write_text(
            (js_root / source_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

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
              };
            }

            const baseUrl = new URL("./", import.meta.url);
            const [stateModule, utilitiesModule, baseModule, geometryModule, linearPeriodicModule, specModule, uiModule] =
              await Promise.all([
                import(new URL("./state.runtime.mjs", baseUrl).href),
                import(new URL("./utilities.runtime.mjs", baseUrl).href),
                import(new URL("./utilitiesBase.js", baseUrl).href),
                import(new URL("./utilitiesGeometry.js", baseUrl).href),
                import(new URL("./utilitiesLinearPeriodic.js", baseUrl).href),
                import(new URL("./utilitiesSpec.js", baseUrl).href),
                import(new URL("./utilitiesUi.js", baseUrl).href),
              ]);

            const { createInitialState } = stateModule;
            const { registerUtilities } = utilitiesModule;
            const { createUtilityBaseBindings } = baseModule;
            const { createUtilityGeometryBindings } = geometryModule;
            const { createUtilityLinearPeriodicBindings } = linearPeriodicModule;
            const { createUtilitySpecBindings } = specModule;
            const { createUtilityUiBindings } = uiModule;

            const requiredFactories = [
              createUtilityBaseBindings,
              createUtilityGeometryBindings,
              createUtilityLinearPeriodicBindings,
              createUtilitySpecBindings,
              createUtilityUiBindings,
            ];
            if (requiredFactories.some((candidate) => typeof candidate !== "function")) {
              throw new Error("One or more utility helper factories were not exported.");
            }

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
              document: {
                activeElement: null,
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
              getSelectedIdsByKind() {
                return [];
              },
              getSelectedEntries() {
                return [];
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

            registerUtilities(ctx);
            const requiredCtxBindings = [
              "serializeCurrentSpec",
              "toggleLinearPeriodicMode",
              "computeDesignBounds",
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
    js_root = REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js"
    copied_modules = {
        "state.runtime.mjs": "state.js",
        "interactions.runtime.mjs": "interactions.js",
        "interactionsCanvas.js": "interactionsCanvas.js",
        "interactionsEditor.js": "interactionsEditor.js",
        "interactionsSession.js": "interactionsSession.js",
        "interactionsShortcuts.js": "interactionsShortcuts.js",
    }
    for target_name, source_name in copied_modules.items():
        (tmp_path / target_name).write_text(
            (js_root / source_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

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
            const [stateModule, interactionsModule, canvasModule, editorModule, sessionModule, shortcutsModule] =
              await Promise.all([
                import(new URL("./state.runtime.mjs", baseUrl).href),
                import(new URL("./interactions.runtime.mjs", baseUrl).href),
                import(new URL("./interactionsCanvas.js", baseUrl).href),
                import(new URL("./interactionsEditor.js", baseUrl).href),
                import(new URL("./interactionsSession.js", baseUrl).href),
                import(new URL("./interactionsShortcuts.js", baseUrl).href),
              ]);

            const { createInitialState } = stateModule;
            const { registerInteractions } = interactionsModule;
            const { createInteractionCanvasBindings } = canvasModule;
            const { createInteractionEditorBindings } = editorModule;
            const { createInteractionSessionBindings } = sessionModule;
            const { createInteractionShortcutBindings } = shortcutsModule;

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
                engineSelect: { options: [], value: "tensornetwork" },
                collectionFormatSelect: { options: [], value: "list" },
                exportFormatSelect: { value: "py" },
                connectButton: createButton(),
                loadInput: createButton(),
                undoButton: createButton(),
                redoButton: createButton(),
                templateSelect: { value: "" },
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
                selectionBox,
                minimapCanvas: {
                  classList: createClassList(),
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
              uniquifyImportedSpec(spec) {
                return spec;
              },
              translateImportedSpec(spec) {
                return spec;
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
            Object.assign(runtime, createInteractionShortcutBindings(env));
            Object.assign(runtime, createInteractionEditorBindings(env));
            Object.assign(runtime, createInteractionSessionBindings(env));

            registerInteractions(ctx);
            const requiredCtxBindings = [
              "handleCanvasWheel",
              "handleKeydown",
              "toggleConnectMode",
              "generateCode",
              "insertTemplate",
            ];
            for (const bindingName of requiredCtxBindings) {
              if (typeof ctx[bindingName] !== "function") {
                throw new Error(`registerInteractions did not expose ${bindingName}.`);
              }
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

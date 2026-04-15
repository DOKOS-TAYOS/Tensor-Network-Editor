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
        "utilitiesLayout.js",
        "utilitiesLinearPeriodic.js",
        "utilitiesSpec.js",
        "utilitiesTemplates.js",
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


def _write_minimap_shortcut_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "minimap_shortcut_runtime_regression.mjs"
    js_root = REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js"
    copied_modules = {
        "state.runtime.mjs": "state.js",
        "interactionsShortcuts.js": "interactionsShortcuts.js",
        "exportMinimap.js": "exportMinimap.js",
    }
    for target_name, source_name in copied_modules.items():
        (tmp_path / target_name).write_text(
            (js_root / source_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

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


def _write_metadata_properties_runtime_regression_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "metadata_properties_runtime_regression.mjs"
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

              const tagPattern = /<(input|textarea|button)[^>]*id="([^"]+)"[^>]*>/g;
              let tagMatch = tagPattern.exec(html);
              while (tagMatch) {
                elements.set(tagMatch[2], createFakeElement(tagMatch[2], tagMatch[1]));
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
            ],
            edges: [
              {
                id: "edge_ab",
                name: "bond_ab",
                left: { tensor_id: "tensor_a", index_id: "index_a" },
                right: { tensor_id: "tensor_a", index_id: "index_b" },
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
        commitField(document.getElementById("tensor-annotation-role-input"), "observable");
        const tensorMetadataAfterGuidedEdit = ctx.state.spec.tensors[0].metadata;
        if (tensorMetadataAfterGuidedEdit.role !== "observable") {
          throw new Error(`Expected the guided tensor role to update, received ${JSON.stringify(tensorMetadataAfterGuidedEdit)}.`);
        }
        if (document.getElementById("tensor-custom-metadata-input").value.includes('"role"')) {
          throw new Error("The custom metadata editor should hide guided tensor annotations.");
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
          '{"source":"imported","color":"#ffffff","tags":["ignored"]}'
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
          throw new Error("The advanced metadata editor should preserve guided tensor annotations.");
        }
        if (tensorMetadataAfterCustomEdit.source !== "imported") {
          throw new Error("The advanced metadata editor did not apply the custom metadata payload.");
        }
        assertLastRenderDidNotInvalidateGraph(
          renderCalls,
          () => graphRenderCount,
          () => minimapRenderCount
        );

        ctx.performUndo();
        const tensorMetadataAfterUndo = ctx.state.spec.tensors[0].metadata;
        if (tensorMetadataAfterUndo.source !== "sim" || tensorMetadataAfterUndo.role !== "observable") {
          throw new Error(`Undo should restore the previous tensor custom metadata, received ${JSON.stringify(tensorMetadataAfterUndo)}.`);
        }
        ctx.performRedo();
        const tensorMetadataAfterRedo = ctx.state.spec.tensors[0].metadata;
        if (tensorMetadataAfterRedo.role !== "observable" || tensorMetadataAfterRedo.source !== "imported") {
          throw new Error(`Redo should restore the advanced metadata edit, received ${JSON.stringify(tensorMetadataAfterRedo)}.`);
        }

        renderCalls.length = 0;
        graphRenderCount = 0;
        minimapRenderCount = 0;
        document.getElementById("tensor-annotation-role-suggestion-operator").click();
        if (ctx.state.spec.tensors[0].metadata.role !== "operator") {
          throw new Error("Clicking a guided tensor suggestion should update the selected value.");
        }
        assertLastRenderDidNotInvalidateGraph(
          renderCalls,
          () => graphRenderCount,
          () => minimapRenderCount
        );

        ctx.setSelection(["index_a"], { primaryId: "index_a" });
        if (!propertiesPanel.innerHTML.includes(">Metadata</summary>")) {
          throw new Error("Selecting an index should keep metadata inside a disclosure.");
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

        renderCalls.length = 0;
        graphRenderCount = 0;
        minimapRenderCount = 0;
        commitField(
          document.getElementById("index-annotation-leg_kind-input-index_a"),
          "logical"
        );
        if (ctx.state.spec.tensors[0].indices[0].metadata.leg_kind !== "logical") {
          throw new Error("Expected the guided index leg kind to update.");
        }
        if (
          document
            .getElementById("index-custom-metadata-input-index_a")
            .value.includes('"leg_kind"')
        ) {
          throw new Error("The custom metadata editor should hide guided index annotations.");
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
    metadata_filters_module_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "metadataFilters.js"
    )
    state_runtime_path = tmp_path / "state.runtime.mjs"
    utilities_runtime_path = tmp_path / "utilities.runtime.mjs"
    metadata_filters_runtime_path = tmp_path / "metadataFilters.runtime.mjs"
    state_runtime_path.write_text(
        state_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    utilities_runtime_path.write_text(
        utilities_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _copy_runtime_editor_support_modules(tmp_path)
    metadata_filters_runtime_path.write_text(
        metadata_filters_module_path.read_text(encoding="utf-8"),
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
            setAttribute() {},
            removeAttribute() {},
            appendChild() {},
          };
        }

        function createFakeDocument() {
          const elements = new Map();
          return {
            registerHtml(html) {
              elements.clear();
              const tagPattern = /<(input|select|button)[^>]*id="([^"]+)"[^>]*>/g;
              let tagMatch = tagPattern.exec(html);
              while (tagMatch) {
                elements.set(tagMatch[2], createFakeElement(tagMatch[2], tagMatch[1]));
                tagMatch = tagPattern.exec(html);
              }
            },
            getElementById(id) {
              return elements.get(id) || null;
            },
            createElement(tagName) {
              return createFakeElement(null, tagName);
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
        const metadataFiltersPanel = createPanel(document);
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
            metadataFiltersPanel,
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
        if (!document.getElementById("metadata-filter-scope-select")) {
          throw new Error("Expected the metadata filter panel to render.");
        }
        if (!metadataFiltersPanel.innerHTML.includes(">Metadata filters</summary>")) {
          throw new Error("Metadata filters should render inside a disclosure.");
        }
        if (metadataFiltersPanel.innerHTML.includes("<details open")) {
          throw new Error("Metadata filters should be collapsed by default.");
        }

        commitInput(document.getElementById("metadata-filter-tag-input"), "block");
        if (ctx.state.metadataFilters.tag !== "block") {
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

        document.getElementById("clear-metadata-filters-button").click();
        if (ctx.getMetadataFilterHighlight() !== null) {
          throw new Error("Clearing metadata filters should reset the highlight state.");
        }

        commitSelect(document.getElementById("metadata-filter-scope-select"), "index");
        commitSelect(document.getElementById("metadata-filter-key-select"), "leg_kind");
        commitInput(document.getElementById("metadata-filter-value-input"), "physical");
        const indexHighlight = ctx.getMetadataFilterHighlight();
        if (ctx.getMetadataFilterEntityState("index", "index_a", indexHighlight) !== "match") {
          throw new Error("Expected index_a to match the guided index filter.");
        }
        if (ctx.getMetadataFilterEntityState("tensor", "tensor_a", indexHighlight) !== "context") {
          throw new Error("Expected tensor_a to remain as context for a matched index.");
        }
        if (ctx.getMetadataFilterEntityState("edge", "edge_ab", indexHighlight) !== "match") {
          throw new Error("Expected the incident edge to stay bright for a matched index.");
        }
        if (ctx.getMetadataFilterEntityState("tensor", "tensor_b", indexHighlight) !== "dim") {
          throw new Error("Expected non-matching tensors to dim under the index filter.");
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
    js_root = REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js"
    copied_modules = {
        "state.runtime.mjs": "state.js",
        "utilities.runtime.mjs": "utilities.js",
        "utilitiesTemplates.js": "utilitiesTemplates.js",
        "utilitiesBase.js": "utilitiesBase.js",
        "utilitiesGeometry.js": "utilitiesGeometry.js",
        "utilitiesLayout.js": "utilitiesLayout.js",
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
            const [stateModule, utilitiesModule, baseModule, geometryModule, layoutModule, linearPeriodicModule, specModule, uiModule] =
              await Promise.all([
                import(new URL("./state.runtime.mjs", baseUrl).href),
                import(new URL("./utilities.runtime.mjs", baseUrl).href),
                import(new URL("./utilitiesBase.js", baseUrl).href),
                import(new URL("./utilitiesGeometry.js", baseUrl).href),
                import(new URL("./utilitiesLayout.js", baseUrl).href),
                import(new URL("./utilitiesLinearPeriodic.js", baseUrl).href),
                import(new URL("./utilitiesSpec.js", baseUrl).href),
                import(new URL("./utilitiesUi.js", baseUrl).href),
              ]);

            const { createInitialState } = stateModule;
            const { registerUtilities } = utilitiesModule;
            const { createUtilityBaseBindings } = baseModule;
            const { createUtilityGeometryBindings } = geometryModule;
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
                  classList: createClassList(),
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
    js_root = REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js"
    copied_modules = {
        "state.runtime.mjs": "state.js",
        "utilities.runtime.mjs": "utilities.js",
        "historySelection.runtime.mjs": "historySelection.js",
        "interactionsSession.js": "interactionsSession.js",
        "utilitiesTemplates.js": "utilitiesTemplates.js",
        "utilitiesBase.js": "utilitiesBase.js",
        "utilitiesGeometry.js": "utilitiesGeometry.js",
        "utilitiesLayout.js": "utilitiesLayout.js",
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
            const [stateModule, utilitiesModule, historyModule, sessionModule] =
              await Promise.all([
                import(new URL("./state.runtime.mjs", baseUrl).href),
                import(new URL("./utilities.runtime.mjs", baseUrl).href),
                import(new URL("./historySelection.runtime.mjs", baseUrl).href),
                import(new URL("./interactionsSession.js", baseUrl).href),
              ]);

            const { createInitialState } = stateModule;
            const { registerUtilities } = utilitiesModule;
            const { registerHistorySelection } = historyModule;
            const { createInteractionSessionBindings } = sessionModule;

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
                prompt: createPromptQueue(["selection_fragment"]),
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
            const env = {
              ctx,
              state: ctx.state,
              dom: ctx.dom,
            };
            Object.assign(ctx, createInteractionSessionBindings(env));
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

            ctx.alignSelectedTensors("left");
            const leftEdges = ctx.state.spec.tensors.map((tensor) => tensor.position.x - tensor.size.width / 2);
            if (!leftEdges.every((value) => value === leftEdges[0])) {
              throw new Error(`Expected aligned left edges, received ${leftEdges.join(", ")}`);
            }
            if (ctx.state.selectionIds.join(",") !== "tensor_a,tensor_b,tensor_c") {
              throw new Error("Alignment should preserve the tensor selection.");
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
                ctx.state.spec.tensors.push({
                  id: "tensor_d",
                  name: "D",
                  position: { x: 540, y: 360 },
                  size: { width: 180, height: 108 },
                  indices: [],
                  metadata: {},
                });
              },
              {
                selectionIds: ["tensor_a", "tensor_b", "tensor_c", "tensor_d"],
                primaryId: "tensor_d",
                statusMessage: "Added tensor D for grid testing.",
              }
            );
            ctx.arrangeSelectedTensors("grid");
            const gridSelection = ctx.state.selectionIds.map((tensorId) => ctx.findTensorById(tensorId));
            const uniqueGridXs = new Set(gridSelection.map((tensor) => tensor.position.x));
            const uniqueGridYs = new Set(gridSelection.map((tensor) => tensor.position.y));
            if (!(uniqueGridXs.size === 2 && uniqueGridYs.size === 2)) {
              throw new Error("Arrange Grid should place four tensors on a 2x2 grid.");
            }

            await ctx.exportSelectedSubnetwork();
            if (!apiCalls.some((call) => call.path === "/api/subnetwork/extract")) {
              throw new Error("Selection export did not call the extract subnetwork API.");
            }

            await ctx.promoteSelectedSubnetworkToTemplate();
            if (!apiCalls.some((call) => call.path === "/api/template/promote")) {
              throw new Error("Promote Selection to Template did not call the promote template API.");
            }
            if (!ctx.state.availableTemplates.includes("selection_fragment")) {
              throw new Error("Promoted template was not added to the available template list.");
            }
            if (ctx.dom.templateSelect.value !== "selection_fragment") {
              throw new Error(`Expected promoted template to become selected, received ${ctx.dom.templateSelect.value}.`);
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
    history_runtime_path = tmp_path / "historySelection.runtime.mjs"
    state_runtime_path.write_text(
        state_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    utilities_runtime_path.write_text(
        utilities_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _copy_runtime_editor_support_modules(tmp_path)
    history_runtime_path.write_text(
        history_module_path.read_text(encoding="utf-8"),
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

        function createButton() {
          return {
            disabled: false,
            title: "",
            classList: createClassList(),
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

        function createPromptQueue(values) {
          const queue = [...values];
          return () => (queue.length ? queue.shift() : null);
        }

        const baseUrl = new URL("./", import.meta.url);
        const [stateModule, utilitiesModule, historyModule, sessionModule] =
          await Promise.all([
            import(new URL("./state.runtime.mjs", baseUrl).href),
            import(new URL("./utilities.runtime.mjs", baseUrl).href),
            import(new URL("./historySelection.runtime.mjs", baseUrl).href),
            import(new URL("./interactionsSession.js", baseUrl).href),
          ]);

        const { createInitialState } = stateModule;
        const { registerUtilities } = utilitiesModule;
        const { registerHistorySelection } = historyModule;
        const { createInteractionSessionBindings } = sessionModule;

        const apiCalls = [];
        const confirmMessages = [];
        let deleteResponseUsed = false;

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
            prompt: createPromptQueue([]),
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
        Object.assign(
          ctx,
          createInteractionSessionBindings({
            ctx,
            state: ctx.state,
            dom: ctx.dom,
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
        if (!ctx.dom.templateCatalogWarning.title.includes("Second warning")) {
          throw new Error("Template catalog warning title should include the full warning list.");
        }
        if (!ctx.dom.renameTemplateButton.disabled || !ctx.dom.deleteTemplateButton.disabled) {
          throw new Error("Rename/Delete should stay disabled for globally registered templates.");
        }

        ctx.dom.templateSelect.value = "project_fragment";
        ctx.handleTemplateSelectionChange({ target: ctx.dom.templateSelect });
        if (ctx.dom.renameTemplateButton.disabled || ctx.dom.deleteTemplateButton.disabled) {
          throw new Error("Rename/Delete should be enabled for project-local templates.");
        }

        ctx.window.prompt = createPromptQueue(["project_fragment"]);
        await ctx.promoteSelectedSubnetworkToTemplate();
        const overwritePromoteCall = apiCalls.findLast((call) => call.path === "/api/template/promote");
        if (!overwritePromoteCall || overwritePromoteCall.payload.overwrite !== true) {
          throw new Error("Promoting over an existing project template should resend the API request with overwrite=true.");
        }
        if (!confirmMessages.length) {
          throw new Error("Overwriting a project template should require user confirmation.");
        }

        const promoteCallCount = apiCalls.filter((call) => call.path === "/api/template/promote").length;
        ctx.window.prompt = createPromptQueue(["mps"]);
        await ctx.promoteSelectedSubnetworkToTemplate();
        const promoteCallCountAfterGlobal = apiCalls.filter((call) => call.path === "/api/template/promote").length;
        if (promoteCallCountAfterGlobal !== promoteCallCount) {
          throw new Error("Promoting over a global template should be blocked before the API call.");
        }
        if (!ctx.dom.statusMessage.textContent.includes("global")) {
          throw new Error(`Expected a global-template error message, received ${ctx.dom.statusMessage.textContent}.`);
        }

        ctx.dom.templateSelect.value = "project_fragment";
        ctx.handleTemplateSelectionChange({ target: ctx.dom.templateSelect });
        ctx.window.prompt = createPromptQueue(["renamed_fragment"]);
        await ctx.renameSelectedTemplate();
        const renameCall = apiCalls.find((call) => call.path === "/api/template/rename");
        if (!renameCall || renameCall.payload.new_template_name !== "renamed_fragment") {
          throw new Error("Rename Template should call the rename API with the requested new name.");
        }
        if (!ctx.state.availableTemplates.includes("renamed_fragment")) {
          throw new Error("Rename Template should update the available template list.");
        }
        if (ctx.dom.templateSelect.value !== "renamed_fragment") {
          throw new Error(`Expected the renamed template to stay selected, received ${ctx.dom.templateSelect.value}.`);
        }

        await ctx.deleteSelectedTemplate();
        if (!deleteResponseUsed) {
          throw new Error("Delete Template should call the delete API.");
        }
        if (ctx.dom.templateSelect.value !== "project_second") {
          throw new Error(`Expected delete to fall back to the next project template, received ${ctx.dom.templateSelect.value}.`);
        }
        if (ctx.dom.templateCatalogWarning.hidden !== true) {
          throw new Error("Template catalog warning should hide once the catalog reloads without warnings.");
        }
        if (ctx.state.templateDefinitions.project_second.source !== "project") {
          throw new Error("Project-local template metadata should preserve source='project' after updates.");
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

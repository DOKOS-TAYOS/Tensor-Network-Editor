from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


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
        state_module_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    utilities_runtime_path.write_text(
        utilities_module_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
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
    utilities_runtime_path = tmp_path / "utilities.runtime.mjs"
    utilities_runtime_path.write_text(
        utilities_module_path.read_text(encoding="utf-8"),
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

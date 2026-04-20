from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from tensor_network_editor.templates import (
    build_template_spec,
    parse_template_parameters,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER_SELECTORS_PATH = (
    REPO_ROOT
    / "src"
    / "tensor_network_editor"
    / "app"
    / "static"
    / "js"
    / "state"
    / "plannerSelectors.js"
)
GRAPH_RENDER_PATH = (
    REPO_ROOT
    / "src"
    / "tensor_network_editor"
    / "app"
    / "static"
    / "js"
    / "graphRender.js"
)


def _write_runtime_script(tmp_path: Path, filename: str, body: str) -> Path:
    script_path = tmp_path / filename
    script_path.write_text(textwrap.dedent(body), encoding="utf-8")
    return script_path


def _run_runtime_script(script_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _build_runtime_prelude() -> str:
    constants_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "constants.js"
    )
    state_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "state.js"
    )
    utilities_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "utilities.js"
    )
    history_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "historySelection.js"
    )
    planner_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "planner.js"
    )
    contraction_scene_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "contractionScene.js"
    )
    interactions_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "interactions.js"
    )
    notes_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "notes.js"
    )
    graph_render_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "graphRender.js"
    )
    editor_store_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "state"
        / "editorStore.js"
    )
    editor_selectors_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "state"
        / "editorSelectors.js"
    )
    session_service_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "services"
        / "editorSessionService.js"
    )
    template_service_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "services"
        / "templateCatalogService.js"
    )
    subnetwork_service_path = (
        REPO_ROOT
        / "src"
        / "tensor_network_editor"
        / "app"
        / "static"
        / "js"
        / "services"
        / "subnetworkService.js"
    )

    return f"""
    import {{ pathToFileURL }} from "node:url";

    const constantsModuleUrl = pathToFileURL({json.dumps(str(constants_path))}).href;
    const stateModuleUrl = pathToFileURL({json.dumps(str(state_path))}).href;
    const utilitiesModuleUrl = pathToFileURL({json.dumps(str(utilities_path))}).href;
    const historyModuleUrl = pathToFileURL({json.dumps(str(history_path))}).href;
    const plannerModuleUrl = pathToFileURL({json.dumps(str(planner_path))}).href;
    const contractionSceneModuleUrl = pathToFileURL({json.dumps(str(contraction_scene_path))}).href;
    const interactionsModuleUrl = pathToFileURL({json.dumps(str(interactions_path))}).href;
    const notesModuleUrl = pathToFileURL({json.dumps(str(notes_path))}).href;
    const graphRenderModuleUrl = pathToFileURL({json.dumps(str(graph_render_path))}).href;
    const editorStoreModuleUrl = pathToFileURL({json.dumps(str(editor_store_path))}).href;
    const editorSelectorsModuleUrl = pathToFileURL({json.dumps(str(editor_selectors_path))}).href;
    const sessionServiceModuleUrl = pathToFileURL({json.dumps(str(session_service_path))}).href;
    const templateServiceModuleUrl = pathToFileURL({json.dumps(str(template_service_path))}).href;
    const subnetworkServiceModuleUrl = pathToFileURL({json.dumps(str(subnetwork_service_path))}).href;

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
        setAttribute() {{}},
        removeAttribute() {{}},
        dataset: {{}},
        focus() {{}},
      }};
    }}

    function createSelectElement(value = "") {{
      return {{
        value,
        options: [],
        addEventListener() {{}},
        appendChild(option) {{
          this.options.push(option);
          if (option.selected) {{
            this.value = option.value;
          }}
        }},
      }};
    }}

    function createTextAreaElement() {{
      return {{
        value: "",
        addEventListener() {{}},
        setSelectionRange() {{}},
        focus() {{}},
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

    function createDocumentStub() {{
      return {{
        activeElement: null,
        getElementById() {{
          return createButton();
        }},
        createElement(tagName) {{
          if (tagName === "option") {{
            return {{
              value: "",
              textContent: "",
              selected: false,
            }};
          }}
          if (tagName === "textarea") {{
            return createTextAreaElement();
          }}
          return {{
            value: "",
            textContent: "",
            selected: false,
            appendChild() {{}},
            click() {{}},
            addEventListener() {{}},
            setAttribute() {{}},
            removeAttribute() {{}},
            classList: createClassList(),
            dataset: {{}},
            style: {{}},
            focus() {{}},
          }};
        }},
        querySelectorAll() {{
          return [];
        }},
        addEventListener() {{}},
        removeEventListener() {{}},
        body: {{
          appendChild() {{}},
        }},
      }};
    }}

    function createBaseDom() {{
      return {{
        workspace: {{}},
        statusMessage: {{
          textContent: "",
          classList: createClassList(),
        }},
        propertiesPanel: {{ innerHTML: "" }},
        generatedCode: {{ value: "" }},
        engineSelect: createSelectElement(),
        collectionFormatSelect: createSelectElement("list"),
        exportFormatSelect: createSelectElement("py"),
        addNoteButton: createButton(),
        connectButton: createButton(),
        loadInput: {{ addEventListener() {{}}, click() {{}}, value: "" }},
        undoButton: createButton(),
        redoButton: createButton(),
        exportButton: createButton(),
        toggleLinearPeriodicButton: createButton(),
        linearPeriodicPreviousCellButton: createButton(),
        linearPeriodicCellLabel: {{ textContent: "" }},
        linearPeriodicNextCellButton: createButton(),
        templateSelect: createSelectElement(),
        templateParameterPanel: {{ hidden: true }},
        templateGraphSizeLabel: {{ textContent: "" }},
        templateGraphSizeInput: {{ value: "2", min: "1", addEventListener() {{}} }},
        templateBondDimensionInput: {{ value: "3", min: "1", addEventListener() {{}} }},
        templatePhysicalDimensionInput: {{ value: "2", min: "1", addEventListener() {{}} }},
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
          addEventListener() {{}},
        }},
        groupLayer: {{}},
        resizeLayer: {{}},
        notesLayer: null,
        selectionBox: {{
          classList: createClassList(),
          style: {{}},
        }},
        minimapShell: {{
          classList: createClassList(),
        }},
        minimapCanvas: {{
          classList: createClassList(),
          addEventListener() {{}},
        }},
        sidebar: {{
          classList: createClassList(),
        }},
        sidebarPanel: {{
          classList: createClassList(),
        }},
        sidebarToggleButton: createButton(),
        sidebarTabs: {{}},
        sidebarTabSelection: createButton(),
        sidebarTabPlanner: createButton(),
        sidebarTabCode: createButton(),
        sidebarPaneSelection: {{
          classList: createClassList(),
          hidden: false,
        }},
        sidebarPanePlanner: {{
          classList: createClassList(),
          hidden: true,
        }},
        sidebarPaneCode: {{
          classList: createClassList(),
          hidden: true,
        }},
        plannerPanel: null,
        generateButton: createButton(),
        codeGenerationWarning: {{
          textContent: "",
          hidden: true,
          classList: createClassList(),
        }},
      }};
    }}

    function createCyStub() {{
      const stats = {{
        addCalls: 0,
        addIds: [],
        batchCalls: 0,
        bulkUnselectCalls: 0,
        centerCalls: 0,
        dataSetIds: [],
        fitCalls: 0,
        grabbableIds: [],
        nodeQueryCalls: [],
        positionSetIds: [],
        removeAllCalls: 0,
        removedIds: [],
        selectableIds: [],
        selectCalls: [],
        toggleClassIds: [],
        unselectCalls: [],
      }};
      const elementsById = new Map();

      function recordUnique(entries, value) {{
        if (!entries.includes(value)) {{
          entries.push(value);
        }}
      }}

      function normalizeClasses(value) {{
        if (!value) {{
          return new Set();
        }}
        if (Array.isArray(value)) {{
          return new Set(value.filter(Boolean));
        }}
        return new Set(String(value).split(/\\s+/).filter(Boolean));
      }}

      function createEmptyElement(id = null) {{
        return {{
          length: 0,
          id() {{
            return id;
          }},
          data() {{
            return undefined;
          }},
          position() {{
            return undefined;
          }},
          select() {{
            return this;
          }},
          unselect() {{
            return this;
          }},
          toggleClass() {{
            return this;
          }},
          remove() {{
            return this;
          }},
          selectable() {{
            return false;
          }},
          grabbable() {{
            return false;
          }},
          classes() {{
            return "";
          }},
          hasClass() {{
            return false;
          }},
          snapshot() {{
            return null;
          }},
        }};
      }}

      function matchesNodeSelector(element, selector = null) {{
        if (!selector) {{
          return element.group() === "nodes";
        }}
        if (selector === "node[kind = 'tensor']") {{
          return element.group() === "nodes" && element.data("kind") === "tensor";
        }}
        if (selector === "node[kind = 'index']") {{
          return element.group() === "nodes" && element.data("kind") === "index";
        }}
        return element.group() === "nodes";
      }}

      function isEdgeElement(element) {{
        return element.group() === "edges";
      }}

      function createElement(descriptor) {{
        const elementState = {{
          classes: normalizeClasses(descriptor.classes),
          data: {{ ...(descriptor.data || {{}}) }},
          grabbable: Boolean(descriptor.grabbable),
          group: descriptor.group,
          id: descriptor.data.id,
          position: descriptor.position
            ? {{ x: descriptor.position.x, y: descriptor.position.y }}
            : {{ x: 0, y: 0 }},
          selectable: descriptor.selectable !== false,
          selected: false,
        }};

        return {{
          length: 1,
          id() {{
            return elementState.id;
          }},
          group() {{
            return elementState.group;
          }},
          data(key, value) {{
            if (!arguments.length) {{
              return {{ ...elementState.data }};
            }}
            if (arguments.length === 1 && typeof key === "string") {{
              return elementState.data[key];
            }}
            if (arguments.length === 1 && key && typeof key === "object") {{
              Object.assign(elementState.data, key);
              recordUnique(stats.dataSetIds, elementState.id);
              return this;
            }}
            elementState.data[key] = value;
            recordUnique(stats.dataSetIds, elementState.id);
            return this;
          }},
          position(arg, value) {{
            if (!arguments.length) {{
              return {{ ...elementState.position }};
            }}
            if (arguments.length === 1 && typeof arg === "string") {{
              return elementState.position[arg];
            }}
            if (arguments.length === 1 && arg && typeof arg === "object") {{
              elementState.position = {{ x: arg.x, y: arg.y }};
              recordUnique(stats.positionSetIds, elementState.id);
              return this;
            }}
            elementState.position[arg] = value;
            recordUnique(stats.positionSetIds, elementState.id);
            return this;
          }},
          select() {{
            if (!elementState.selected) {{
              elementState.selected = true;
              stats.selectCalls.push(elementState.id);
            }}
            return this;
          }},
          unselect() {{
            if (elementState.selected) {{
              elementState.selected = false;
              stats.unselectCalls.push(elementState.id);
            }}
            return this;
          }},
          toggleClass(className, force) {{
            const shouldHaveClass =
              force === undefined
                ? !elementState.classes.has(className)
                : Boolean(force);
            if (shouldHaveClass) {{
              elementState.classes.add(className);
            }} else {{
              elementState.classes.delete(className);
            }}
            recordUnique(stats.toggleClassIds, elementState.id);
            return this;
          }},
          remove() {{
            if (elementsById.has(elementState.id)) {{
              elementsById.delete(elementState.id);
              stats.removedIds.push(elementState.id);
            }}
            return this;
          }},
          selectable(value) {{
            if (!arguments.length) {{
              return elementState.selectable;
            }}
            elementState.selectable = Boolean(value);
            recordUnique(stats.selectableIds, elementState.id);
            return this;
          }},
          grabbable(value) {{
            if (!arguments.length) {{
              return elementState.grabbable;
            }}
            elementState.grabbable = Boolean(value);
            recordUnique(stats.grabbableIds, elementState.id);
            return this;
          }},
          classes(value) {{
            if (!arguments.length) {{
              return [...elementState.classes].sort().join(" ");
            }}
            elementState.classes = normalizeClasses(value);
            recordUnique(stats.toggleClassIds, elementState.id);
            return this;
          }},
          hasClass(className) {{
            return elementState.classes.has(className);
          }},
          snapshot() {{
            return {{
              classes: [...elementState.classes].sort(),
              data: {{ ...elementState.data }},
              grabbable: elementState.grabbable,
              id: elementState.id,
              position: {{ ...elementState.position }},
              selectable: elementState.selectable,
              selected: elementState.selected,
            }};
          }},
        }};
      }}

      function resetStats() {{
        Object.keys(stats).forEach((key) => {{
          if (Array.isArray(stats[key])) {{
            stats[key].length = 0;
            return;
          }}
          stats[key] = 0;
        }});
      }}

      const cy = {{
        add(collection) {{
          const descriptors = Array.isArray(collection) ? collection : [collection];
          stats.addCalls += 1;
          descriptors.forEach((descriptor) => {{
            stats.addIds.push(descriptor.data.id);
            elementsById.set(descriptor.data.id, createElement(descriptor));
          }});
          return descriptors.map((descriptor) => elementsById.get(descriptor.data.id));
        }},
        batch(callback) {{
          stats.batchCalls += 1;
          return callback();
        }},
        center() {{
          stats.centerCalls += 1;
        }},
        edges() {{
          return [...elementsById.values()].filter((element) => isEdgeElement(element));
        }},
        elements() {{
          return {{
            remove() {{
              stats.removeAllCalls += 1;
              stats.removedIds.push(...elementsById.keys());
              elementsById.clear();
            }},
          }};
        }},
        fit() {{
          stats.fitCalls += 1;
        }},
        getElementById(id) {{
          return elementsById.get(id) || createEmptyElement(id);
        }},
        nodes(selector = null) {{
          stats.nodeQueryCalls.push(selector);
          return [...elementsById.values()].filter((element) =>
            matchesNodeSelector(element, selector)
          );
        }},
        on() {{}},
        $(selector) {{
          if (selector !== ":selected") {{
            return {{
              unselect() {{}},
            }};
          }}
          const selectedElements = [...elementsById.values()].filter(
            (element) => element.snapshot() && element.snapshot().selected
          );
          return {{
            forEach(callback) {{
              selectedElements.forEach(callback);
            }},
            map(callback) {{
              return selectedElements.map(callback);
            }},
            get length() {{
              return selectedElements.length;
            }},
            unselect() {{
              stats.bulkUnselectCalls += 1;
              selectedElements.forEach((element) => {{
                element.unselect();
              }});
            }},
          }};
        }},
      }};

      return {{
        cy,
        getElementIds() {{
          return [...elementsById.keys()].sort();
        }},
        getElementSnapshot(id) {{
          const element = elementsById.get(id);
          return element ? element.snapshot() : null;
        }},
        resetStats,
        stats,
      }};
    }}

    async function buildContext() {{
      const [
        constantsModule,
        stateModule,
        utilitiesModule,
        storeModule,
        selectorsModule,
        sessionServiceModule,
        templateServiceModule,
        subnetworkServiceModule,
      ] = await Promise.all([
        import(constantsModuleUrl),
        import(stateModuleUrl),
        import(utilitiesModuleUrl),
        import(editorStoreModuleUrl),
        import(editorSelectorsModuleUrl),
        import(sessionServiceModuleUrl),
        import(templateServiceModuleUrl),
        import(subnetworkServiceModuleUrl),
      ]);
      const {{ constants }} = constantsModule;
      const {{ createInitialState }} = stateModule;
      const {{ registerUtilities }} = utilitiesModule;
      const {{ createEditorStore }} = storeModule;
      const {{ createEditorSelectors }} = selectorsModule;
      const {{ createEditorSessionService }} = sessionServiceModule;
      const {{ createTemplateCatalogService }} = templateServiceModule;
      const {{ createSubnetworkService }} = subnetworkServiceModule;

      const ctx = {{
        state: createInitialState(),
        constants,
        dom: createBaseDom(),
        apiGet: async () => {{
          throw new Error("apiGet should not be called in this regression test.");
        }},
        apiPost: async () => {{
          throw new Error("apiPost should not be called in this regression test.");
        }},
        window: {{
          structuredClone: globalThis.structuredClone,
          crypto: globalThis.crypto,
          setTimeout,
          clearTimeout,
          confirm: () => true,
          addEventListener() {{}},
          removeEventListener() {{}},
          innerHeight: 900,
          innerWidth: 1400,
        }},
        document: createDocumentStub(),
        cytoscape: null,
      }};

      registerUtilities(ctx);
      ctx.render = () => {{}};
      ctx.renderGraph = () => {{}};
      ctx.renderOverlayDecorations = () => {{}};
      ctx.renderMinimap = () => {{}};
      ctx.renderPlanner = () => {{}};
      ctx.renderSidebarTabs = () => {{}};
      ctx.renderProperties = () => {{}};
      ctx.refreshContractionAnalysis = () => {{}};
      ctx.syncPendingInteractionClasses = () => {{}};
      ctx.setActiveSidebarTab = () => {{}};
      ctx.captureEditableFocus = () => null;
      ctx.restoreEditableFocus = () => {{}};
      ctx.downloadPngExport = () => {{}};
      ctx.downloadSvgExport = () => {{}};
      ctx.handleMinimapMouseDown = () => {{}};
      ctx.initGraph = () => {{}};
      ctx.store = createEditorStore(ctx.state);
      ctx.selectors = createEditorSelectors({{ store: ctx.store }});
      ctx.services = {{
        session: createEditorSessionService({{
          apiGet: (...args) => ctx.apiGet(...args),
          apiPost: (...args) => ctx.apiPost(...args),
        }}),
        templateCatalog: createTemplateCatalogService({{
          apiPost: (...args) => ctx.apiPost(...args),
        }}),
        subnetwork: createSubnetworkService({{
          apiPost: (...args) => ctx.apiPost(...args),
        }}),
      }};
      return ctx;
    }}

    async function registerHistory(ctx) {{
      const {{ registerHistorySelection }} = await import(historyModuleUrl);
      registerHistorySelection(ctx);
    }}

    async function registerGraphRender(ctx) {{
      const {{ registerGraphRender: registerGraphRenderFeature }} = await import(
        graphRenderModuleUrl
      );
      registerGraphRenderFeature(ctx);
    }}

    async function registerPlanner(ctx) {{
      const {{ registerPlannerFeature }} = await import(plannerModuleUrl);
      registerPlannerFeature(ctx);
    }}

    async function registerContractionScene(ctx) {{
      const {{ registerContractionScene }} = await import(contractionSceneModuleUrl);
      registerContractionScene(ctx);
    }}

    async function registerInteractions(ctx) {{
      const {{ registerInteractions }} = await import(interactionsModuleUrl);
      registerInteractions(ctx);
    }}

    async function registerNotes(ctx) {{
      const {{ registerNotesFeature }} = await import(notesModuleUrl);
      registerNotesFeature(ctx);
    }}
    """


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_tensorkrowch_planner_allows_manual_outer_products(tmp_path: Path) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "tensorkrowch_outer_product_allowed.mjs",
        _build_runtime_prelude()
        + """
        function buildOuterProductSpec() {
          return {
            id: "network_outer_product",
            name: "outer-product",
            tensors: [
              {
                id: "tensor_a",
                name: "A",
                position: { x: 120, y: 120 },
                indices: [
                  { id: "tensor_a_i", name: "i", dimension: 2, offset: { x: -58, y: -20 }, metadata: {} },
                  { id: "tensor_a_x", name: "x", dimension: 3, offset: { x: 58, y: -20 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_b",
                name: "B",
                position: { x: 360, y: 120 },
                indices: [
                  { id: "tensor_b_y", name: "y", dimension: 5, offset: { x: -58, y: -20 }, metadata: {} },
                  { id: "tensor_b_j", name: "j", dimension: 7, offset: { x: 58, y: -20 }, metadata: {} },
                ],
                metadata: {},
              },
            ],
            groups: [],
            edges: [],
            notes: [],
            contraction_plan: null,
            metadata: {},
          };
        }

        const ctx = await buildContext();
        await registerHistory(ctx);
        await registerPlanner(ctx);

        ctx.state.selectedEngine = "tensorkrowch";
        ctx.state.selectedCollectionFormat = "list";
        ctx.state.spec = ctx.normalizeSpec(buildOuterProductSpec());
        ctx.state.plannerMode = true;

        ctx.handlePlannerOperandClick("tensor_a");
        ctx.handlePlannerOperandClick("tensor_b");

        if (!ctx.state.spec.contraction_plan || ctx.state.spec.contraction_plan.steps.length !== 1) {
          throw new Error("TensorKrowch should still let the user build an outer-product manual step.");
        }
        if (ctx.state.spec.contraction_plan.steps[0].left_operand_id !== "tensor_a") {
          throw new Error("Expected the saved manual step to keep the original left operand.");
        }
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The TensorKrowch manual outer-product save regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_tensorkrowch_incompatible_plan_warns_only_when_generating(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "tensorkrowch_incompatible_plan_generate_warning.mjs",
        _build_runtime_prelude()
        + """
        function buildOuterProductPlanSpec() {
          return {
            id: "network_outer_product",
            name: "outer-product",
            tensors: [
              {
                id: "tensor_a",
                name: "A",
                position: { x: 120, y: 120 },
                indices: [
                  { id: "tensor_a_i", name: "i", dimension: 2, offset: { x: -58, y: -20 }, metadata: {} },
                  { id: "tensor_a_x", name: "x", dimension: 3, offset: { x: 58, y: -20 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_b",
                name: "B",
                position: { x: 360, y: 120 },
                indices: [
                  { id: "tensor_b_y", name: "y", dimension: 5, offset: { x: -58, y: -20 }, metadata: {} },
                  { id: "tensor_b_j", name: "j", dimension: 7, offset: { x: 58, y: -20 }, metadata: {} },
                ],
                metadata: {},
              },
            ],
            groups: [],
            edges: [],
            notes: [],
            contraction_plan: {
              id: "plan_outer_product",
              name: "Outer product path",
              steps: [
                {
                  id: "step_outer",
                  left_operand_id: "tensor_a",
                  right_operand_id: "tensor_b",
                  metadata: {},
                },
              ],
              metadata: {},
            },
            metadata: {},
          };
        }

        const ctx = await buildContext();
        let apiPostCalls = 0;
        ctx.apiPost = async () => {
          apiPostCalls += 1;
          throw new Error("apiPost should not be reached for incompatible TensorKrowch plans.");
        };
        ctx.state.selectedEngine = "tensorkrowch";
        ctx.state.selectedCollectionFormat = "list";
        ctx.state.spec = ctx.normalizeSpec(buildOuterProductPlanSpec());
        await registerContractionScene(ctx);
        await registerInteractions(ctx);

        ctx.updateToolbarState();

        if (!ctx.state.spec.contraction_plan || ctx.state.spec.contraction_plan.steps.length !== 1) {
          throw new Error("The incompatible manual plan should be preserved when switching to TensorKrowch.");
        }
        if (ctx.dom.generateButton.disabled !== false) {
          throw new Error("Generate button should stay enabled until the user tries to generate code.");
        }
        if (ctx.dom.exportButton.disabled !== false) {
          throw new Error("Python export should stay enabled until the user tries to generate code.");
        }
        if (ctx.dom.codeGenerationWarning.hidden !== false) {
          throw new Error("The code panel should show the TensorKrowch warning next to Generate.");
        }
        if (!ctx.dom.codeGenerationWarning.textContent.includes("TensorKrowch")) {
          throw new Error(`Expected the inline code warning to mention TensorKrowch, received: ${ctx.dom.codeGenerationWarning.textContent}`);
        }

        await ctx.generateCode();

        if (apiPostCalls !== 0) {
          throw new Error("Generate should stop in the frontend before calling the backend.");
        }
        if (!ctx.dom.statusMessage.textContent.includes("TensorKrowch")) {
          throw new Error(`Expected a TensorKrowch generation warning, received: ${ctx.dom.statusMessage.textContent}`);
        }
        if (!ctx.dom.statusMessage.textContent.toLowerCase().includes("shared index")) {
          throw new Error(`Expected the warning to mention a shared index, received: ${ctx.dom.statusMessage.textContent}`);
        }
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The TensorKrowch incompatible-plan generate warning regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
@pytest.mark.parametrize("engine_name", ["quimb", "einsum_numpy", "einsum_torch"])
def test_for_mode_preserves_new_backend_selection_and_keeps_actions_enabled(
    tmp_path: Path,
    engine_name: str,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        f"for_mode_preserves_{engine_name}.mjs",
        _build_runtime_prelude()
        + f"""
        function buildSimpleSpec() {{
          return {{
            id: "network_simple",
            name: "simple",
            tensors: [
              {{
                id: "tensor_a",
                name: "A",
                position: {{ x: 120, y: 120 }},
                indices: [
                  {{ id: "tensor_a_i", name: "i", dimension: 2, offset: {{ x: -58, y: -20 }}, metadata: {{}} }},
                  {{ id: "tensor_a_x", name: "x", dimension: 3, offset: {{ x: 58, y: -20 }}, metadata: {{}} }},
                ],
                metadata: {{}},
              }},
              {{
                id: "tensor_b",
                name: "B",
                position: {{ x: 360, y: 120 }},
                indices: [
                  {{ id: "tensor_b_x", name: "x", dimension: 3, offset: {{ x: -58, y: -20 }}, metadata: {{}} }},
                  {{ id: "tensor_b_j", name: "j", dimension: 5, offset: {{ x: 58, y: -20 }}, metadata: {{}} }},
                ],
                metadata: {{}},
              }},
            ],
            edges: [
              {{
                id: "edge_ax_bx",
                index_ids: ["tensor_a_x", "tensor_b_x"],
                metadata: {{}},
              }},
            ],
            groups: [],
            notes: [],
            contraction_plan: null,
            metadata: {{}},
          }};
        }}

        const ctx = await buildContext();
        ctx.state.selectedEngine = {json.dumps(engine_name)};
        ctx.state.selectedCollectionFormat = "list";
        ctx.populateEngineOptions([
          "tensornetwork",
          "quimb",
          "tensorkrowch",
          "einsum_numpy",
          "einsum_torch",
        ]);
        ctx.state.spec = ctx.normalizeSpec(buildSimpleSpec());
        await registerContractionScene(ctx);
        await registerInteractions(ctx);

        ctx.toggleLinearPeriodicMode();
        ctx.updateToolbarState();

        if (ctx.state.selectedEngine !== {json.dumps(engine_name)}) {{
          throw new Error(
            `Expected for mode to keep ${engine_name} selected, received ${{ctx.state.selectedEngine}}.`
          );
        }}
        if (ctx.dom.engineSelect.value !== {json.dumps(engine_name)}) {{
          throw new Error(
            `Expected the engine picker to keep ${engine_name}, received ${{ctx.dom.engineSelect.value}}.`
          );
        }}
        if (ctx.dom.engineSelect.options.some((option) => option.disabled)) {{
          throw new Error("For mode should not disable the remaining backend options.");
        }}
        if (ctx.dom.generateButton.disabled) {{
          throw new Error("Generate should stay enabled for supported backends in for mode.");
        }}
        if (ctx.dom.exportButton.disabled) {{
          throw new Error("Export should stay enabled for supported backends in for mode.");
        }}
        if (ctx.dom.codeGenerationWarning.hidden !== true) {{
          throw new Error("No inline warning should be shown for compatible backends in for mode.");
        }}
        if (ctx.dom.statusMessage.textContent.includes("TensorNetwork and TensorKrowch")) {{
          throw new Error(
            `The old two-backend warning should not appear anymore: ${{ctx.dom.statusMessage.textContent}}`
          );
        }}
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The for-mode backend preservation regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_for_mode_template_insertion_recomputes_open_interface_from_current_spec(
    tmp_path: Path,
) -> None:
    template_parameters = parse_template_parameters(
        "mps",
        {
            "graph_size": 4,
            "bond_dimension": 3,
            "physical_dimension": 2,
        },
    )
    mps_template_payload = build_template_spec("mps", template_parameters).to_dict()
    script_path = _write_runtime_script(
        tmp_path,
        "for_mode_template_insertion_open_interface_regression.mjs",
        _build_runtime_prelude()
        + f"""
        const mpsTemplate = {json.dumps(mps_template_payload)};

        function buildBoundaryOnlyInitialSpec() {{
          return {{
            id: "network_boundary_only_initial",
            name: "boundary-only-initial",
            tensors: [
              {{
                id: "seed_next_boundary",
                name: "Next cell",
                position: {{ x: 320, y: 140 }},
                linear_periodic_role: "next",
                indices: [
                  {{ id: "seed_slot_1", name: "slot_1", dimension: 3, metadata: {{}} }},
                  {{ id: "seed_slot_2", name: "slot_2", dimension: 3, metadata: {{}} }},
                  {{ id: "seed_slot_3", name: "slot_3", dimension: 3, metadata: {{}} }},
                ],
                metadata: {{}},
              }},
            ],
            edges: [],
            groups: [],
            notes: [],
            contraction_plan: null,
            linear_periodic_chain: {{
              active_cell: "initial",
              initial_cell: {{
                tensors: [],
                edges: [],
                groups: [],
                notes: [],
                contraction_plan: null,
                metadata: {{}},
              }},
              periodic_cell: {{
                tensors: [],
                edges: [],
                groups: [],
                notes: [],
                contraction_plan: null,
                metadata: {{}},
              }},
              final_cell: {{
                tensors: [],
                edges: [],
                groups: [],
                notes: [],
                contraction_plan: null,
                metadata: {{}},
              }},
              metadata: {{}},
            }},
            metadata: {{}},
          }};
        }}

        function buildThreeOpenTensorSpec() {{
          return {{
            id: "network_three_open_tensors",
            name: "three-open-tensors",
            tensors: [
              {{
                id: "seed_a",
                name: "SeedA",
                position: {{ x: 80, y: 140 }},
                indices: [
                  {{ id: "seed_a_open", name: "a", dimension: 3, metadata: {{}} }},
                ],
                metadata: {{}},
              }},
              {{
                id: "seed_b",
                name: "SeedB",
                position: {{ x: 220, y: 140 }},
                indices: [
                  {{ id: "seed_b_open", name: "b", dimension: 5, metadata: {{}} }},
                ],
                metadata: {{}},
              }},
              {{
                id: "seed_c",
                name: "SeedC",
                position: {{ x: 360, y: 140 }},
                indices: [
                  {{ id: "seed_c_open", name: "c", dimension: 7, metadata: {{}} }},
                ],
                metadata: {{}},
              }},
            ],
            edges: [],
            groups: [],
            notes: [],
            contraction_plan: null,
            metadata: {{}},
          }};
        }}

        function insertTemplateThroughDesignChange(ctx, templateSpec) {{
          const normalizedTemplate = ctx.normalizeSpec(templateSpec);
          ctx.ensureSpecLookups();
          ctx.applyDesignChange(
            () => {{
              ctx.state.spec.tensors.push(...normalizedTemplate.tensors);
              ctx.state.spec.edges.push(...normalizedTemplate.edges);
              ctx.state.spec.groups.push(...normalizedTemplate.groups);
            }},
            {{
              invalidate: {{ lookups: true }},
            }}
          );
          const nextBoundary = ctx.state.spec.tensors.find(
            (tensor) => tensor.linear_periodic_role === "next"
          );
          return nextBoundary ? nextBoundary.indices.length : null;
        }}

        const boundaryOnlyContext = await buildContext();
        await registerHistory(boundaryOnlyContext);
        boundaryOnlyContext.state.selectedEngine = "tensornetwork";
        boundaryOnlyContext.state.selectedCollectionFormat = "list";
        boundaryOnlyContext.state.spec = boundaryOnlyContext.normalizeSpec(
          buildBoundaryOnlyInitialSpec()
        );
        const boundaryOnlySlotCount = insertTemplateThroughDesignChange(
          boundaryOnlyContext,
          mpsTemplate
        );
        if (boundaryOnlySlotCount !== 4) {{
          throw new Error(
            `Expected the seeded initial boundary to rebuild to 4 open slots, received ${{boundaryOnlySlotCount}}.`
          );
        }}

        const existingOpenContext = await buildContext();
        await registerHistory(existingOpenContext);
        existingOpenContext.state.selectedEngine = "tensornetwork";
        existingOpenContext.state.selectedCollectionFormat = "list";
        existingOpenContext.state.spec = existingOpenContext.normalizeSpec(
          buildThreeOpenTensorSpec()
        );
        existingOpenContext.toggleLinearPeriodicMode();
        const existingOpenSlotCount = insertTemplateThroughDesignChange(
          existingOpenContext,
          mpsTemplate
        );
        if (existingOpenSlotCount !== 7) {{
          throw new Error(
            `Expected the interface to keep the 3 existing open slots plus 4 new ones, received ${{existingOpenSlotCount}}.`
          );
        }}
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The for-mode template insertion interface regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_for_mode_switching_back_to_initial_preserves_interface_slot_count(
    tmp_path: Path,
) -> None:
    template_parameters = parse_template_parameters(
        "mps",
        {
            "graph_size": 4,
            "bond_dimension": 3,
            "physical_dimension": 2,
        },
    )
    mps_template_payload = build_template_spec("mps", template_parameters).to_dict()
    script_path = _write_runtime_script(
        tmp_path,
        "for_mode_switch_back_initial_interface_regression.mjs",
        _build_runtime_prelude()
        + f"""
        const initialSpec = {json.dumps(mps_template_payload)};

        function getNextBoundarySlotCount(ctx) {{
          const nextBoundary = ctx.state.spec.tensors.find(
            (tensor) => tensor.linear_periodic_role === "next"
          );
          return nextBoundary ? nextBoundary.indices.length : null;
        }}

        const ctx = await buildContext();
        ctx.state.selectedEngine = "tensornetwork";
        ctx.state.selectedCollectionFormat = "list";
        ctx.state.spec = ctx.normalizeSpec(initialSpec);

        ctx.toggleLinearPeriodicMode();
        if (getNextBoundarySlotCount(ctx) !== 4) {{
          throw new Error(
            `Expected the initial For-mode boundary to start with 4 slots, received ${{getNextBoundarySlotCount(ctx)}}.`
          );
        }}

        ctx.ensureSpecLookups();
        ctx.switchLinearPeriodicCell(1);
        if (getNextBoundarySlotCount(ctx) !== 4) {{
          throw new Error(
            `Expected the periodic cell to keep the inherited 4-slot interface, received ${{getNextBoundarySlotCount(ctx)}}.`
          );
        }}

        ctx.ensureSpecLookups();
        ctx.switchLinearPeriodicCell(-1);
        if (getNextBoundarySlotCount(ctx) !== 4) {{
          throw new Error(
            `Expected returning to the initial cell to preserve 4 slots, received ${{getNextBoundarySlotCount(ctx)}}.`
          );
        }}
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The For-mode switch-back interface regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_for_mode_tensorkrowch_generation_surfaces_backend_rejection(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "for_mode_tensorkrowch_generation_error_panel.mjs",
        _build_runtime_prelude()
        + """
        function buildCarryCellSpec() {
          return {
            id: "network_linear_periodic_carry",
            name: "linear-periodic-carry-chain",
            tensors: [],
            groups: [],
            edges: [],
            notes: [],
            contraction_plan: null,
            linear_periodic_chain: {
              active_cell: "periodic",
              initial_cell: {
                tensors: [
                  {
                    id: "initial_tensor",
                    name: "Initial",
                    position: { x: 100, y: 140 },
                    indices: [
                      { id: "initial_phys", name: "phys", dimension: 2, metadata: {} },
                      { id: "initial_bond", name: "bond", dimension: 3, metadata: {} },
                    ],
                    metadata: {},
                  },
                  {
                    id: "initial_next_boundary",
                    name: "Next cell",
                    position: { x: 320, y: 140 },
                    linear_periodic_role: "next",
                    indices: [
                      { id: "initial_next_slot", name: "slot_1", dimension: 3, metadata: {} },
                    ],
                    metadata: {},
                  },
                ],
                groups: [],
                edges: [
                  {
                    id: "initial_edge_to_next",
                    name: "initial_to_next",
                    left: { tensor_id: "initial_tensor", index_id: "initial_bond" },
                    right: { tensor_id: "initial_next_boundary", index_id: "initial_next_slot" },
                    metadata: {},
                  },
                ],
                notes: [],
                contraction_plan: {
                  id: "initial_plan",
                  name: "Initial carry plan",
                  steps: [
                    {
                      id: "initial_carry",
                      left_operand_id: "initial_tensor",
                      right_operand_id: "__linear_next__",
                      metadata: {},
                    },
                  ],
                  view_snapshots: [],
                  metadata: {},
                },
                metadata: {},
              },
              periodic_cell: {
                tensors: [
                  {
                    id: "periodic_left_tensor",
                    name: "PeriodicLeft",
                    position: { x: 140, y: 120 },
                    indices: [
                      { id: "periodic_left_in", name: "left", dimension: 3, metadata: {} },
                      { id: "periodic_left_phys", name: "phys_l", dimension: 2, metadata: {} },
                      { id: "periodic_left_inner", name: "inner", dimension: 5, metadata: {} },
                    ],
                    metadata: {},
                  },
                  {
                    id: "periodic_right_tensor",
                    name: "PeriodicRight",
                    position: { x: 320, y: 120 },
                    indices: [
                      { id: "periodic_right_inner", name: "inner", dimension: 5, metadata: {} },
                      { id: "periodic_right_phys", name: "phys_r", dimension: 2, metadata: {} },
                      { id: "periodic_right_out", name: "right", dimension: 3, metadata: {} },
                    ],
                    metadata: {},
                  },
                  {
                    id: "periodic_previous_boundary",
                    name: "Previous cell",
                    position: { x: 20, y: 120 },
                    linear_periodic_role: "previous",
                    indices: [
                      { id: "periodic_previous_slot", name: "slot_1", dimension: 3, metadata: {} },
                    ],
                    metadata: {},
                  },
                  {
                    id: "periodic_next_boundary",
                    name: "Next cell",
                    position: { x: 460, y: 120 },
                    linear_periodic_role: "next",
                    indices: [
                      { id: "periodic_next_slot", name: "slot_1", dimension: 3, metadata: {} },
                    ],
                    metadata: {},
                  },
                ],
                groups: [],
                edges: [
                  {
                    id: "periodic_edge_from_previous",
                    name: "from_previous",
                    left: { tensor_id: "periodic_previous_boundary", index_id: "periodic_previous_slot" },
                    right: { tensor_id: "periodic_left_tensor", index_id: "periodic_left_in" },
                    metadata: {},
                  },
                  {
                    id: "periodic_edge_inner",
                    name: "inner",
                    left: { tensor_id: "periodic_left_tensor", index_id: "periodic_left_inner" },
                    right: { tensor_id: "periodic_right_tensor", index_id: "periodic_right_inner" },
                    metadata: {},
                  },
                  {
                    id: "periodic_edge_to_next",
                    name: "to_next",
                    left: { tensor_id: "periodic_right_tensor", index_id: "periodic_right_out" },
                    right: { tensor_id: "periodic_next_boundary", index_id: "periodic_next_slot" },
                    metadata: {},
                  },
                ],
                notes: [],
                contraction_plan: {
                  id: "periodic_carry_plan",
                  name: "Periodic carry plan",
                  steps: [
                    {
                      id: "periodic_from_previous",
                      left_operand_id: "__linear_previous__",
                      right_operand_id: "periodic_left_tensor",
                      metadata: {},
                    },
                    {
                      id: "periodic_contract_full",
                      left_operand_id: "periodic_from_previous",
                      right_operand_id: "periodic_right_tensor",
                      metadata: {},
                    },
                    {
                      id: "periodic_carry",
                      left_operand_id: "periodic_contract_full",
                      right_operand_id: "__linear_next__",
                      metadata: {},
                    },
                  ],
                  view_snapshots: [],
                  metadata: {},
                },
                metadata: {},
              },
              final_cell: {
                tensors: [
                  {
                    id: "final_tensor",
                    name: "Final",
                    position: { x: 260, y: 140 },
                    indices: [
                      { id: "final_bond", name: "bond", dimension: 3, metadata: {} },
                      { id: "final_phys", name: "phys", dimension: 7, metadata: {} },
                    ],
                    metadata: {},
                  },
                  {
                    id: "final_previous_boundary",
                    name: "Previous cell",
                    position: { x: 60, y: 140 },
                    linear_periodic_role: "previous",
                    indices: [
                      { id: "final_previous_slot", name: "slot_1", dimension: 3, metadata: {} },
                    ],
                    metadata: {},
                  },
                ],
                groups: [],
                edges: [
                  {
                    id: "final_edge_from_previous",
                    name: "from_previous",
                    left: { tensor_id: "final_previous_boundary", index_id: "final_previous_slot" },
                    right: { tensor_id: "final_tensor", index_id: "final_bond" },
                    metadata: {},
                  },
                ],
                notes: [],
                contraction_plan: {
                  id: "final_plan",
                  name: "Final carry plan",
                  steps: [
                    {
                      id: "final_contract",
                      left_operand_id: "__linear_previous__",
                      right_operand_id: "final_tensor",
                      metadata: {},
                    },
                  ],
                  view_snapshots: [],
                  metadata: {},
                },
                metadata: {},
              },
              metadata: {},
            },
            metadata: {},
          };
        }

        const ctx = await buildContext();
        const apiPostCalls = [];
        ctx.apiPost = async (url, payload) => {
          apiPostCalls.push({ url, payload });
          return {
            ok: false,
            message: "Backend refused TensorKrowch for mode in this regression.",
          };
        };
        ctx.state.selectedEngine = "tensorkrowch";
        ctx.state.selectedCollectionFormat = "list";
        ctx.state.spec = ctx.normalizeSpec(buildCarryCellSpec());
        await registerContractionScene(ctx);
        await registerInteractions(ctx);

        await ctx.generateCode();

        if (apiPostCalls.length !== 1 || apiPostCalls[0].url !== "/api/generate") {
          throw new Error(`Expected one generate call, received ${JSON.stringify(apiPostCalls)}.`);
        }
        const generatedNetwork = apiPostCalls[0].payload.spec.network;
        const periodicSteps =
          generatedNetwork.linear_periodic_chain.periodic_cell.contraction_plan.steps;
        if (periodicSteps.length !== 3 || periodicSteps[0].left_operand_id !== "__linear_previous__") {
          throw new Error("The generate payload did not preserve the periodic carry plan.");
        }
        if (!ctx.dom.statusMessage.textContent.includes("Backend refused TensorKrowch")) {
          throw new Error(`Expected the backend error in the status message, received: ${ctx.dom.statusMessage.textContent}`);
        }
        if (!ctx.dom.generatedCode.value.includes("Backend refused TensorKrowch")) {
          throw new Error(`Expected the backend error in the code panel, received: ${ctx.dom.generatedCode.value}`);
        }
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The TensorKrowch for-mode generation error regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_note_sizes_normalize_and_resize_to_the_real_minimum(tmp_path: Path) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "note_minimum_size_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildSmallNoteSpec() {
          return {
            id: "network_notes",
            name: "notes",
            tensors: [],
            groups: [],
            edges: [],
            notes: [
              {
                id: "note_1",
                text: "Tiny",
                position: { x: 80, y: 60 },
                size: { width: 120, height: 90 },
                metadata: {},
              },
            ],
            contraction_plan: null,
            metadata: {},
          };
        }

        const ctx = await buildContext();
        await registerNotes(ctx);

        ctx.state.spec = ctx.normalizeSpec(buildSmallNoteSpec());

        const note = ctx.findNoteById("note_1");
        if (!note) {
          throw new Error("Expected the note to exist after normalisation.");
        }
        if (note.size.width !== ctx.constants.NOTE_MIN_WIDTH) {
          throw new Error(`Expected note width ${ctx.constants.NOTE_MIN_WIDTH}, received ${note.size.width}.`);
        }
        if (note.size.height !== ctx.constants.NOTE_MIN_HEIGHT) {
          throw new Error(`Expected note height ${ctx.constants.NOTE_MIN_HEIGHT}, received ${note.size.height}.`);
        }

        note.size.width = 260;
        note.size.height = 220;
        ctx.state.activeNoteResize = {
          noteId: "note_1",
          snapshot: null,
          startPointer: { x: 300, y: 240 },
          startSize: { width: 260, height: 220 },
        };

        ctx.updateActiveNoteResize({ clientX: 0, clientY: 0 });

        if (note.size.width !== ctx.constants.NOTE_MIN_WIDTH) {
          throw new Error(`Expected resized note width ${ctx.constants.NOTE_MIN_WIDTH}, received ${note.size.width}.`);
        }
        if (note.size.height !== ctx.constants.NOTE_MIN_HEIGHT) {
          throw new Error(`Expected resized note height ${ctx.constants.NOTE_MIN_HEIGHT}, received ${note.size.height}.`);
        }
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The note minimum-size regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_deleting_a_contracted_result_removes_all_nested_base_tensors(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "delete_contracted_result_and_sources_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildPartialPlanSpec() {
          return {
            id: "network_chain",
            name: "chain",
            tensors: [
              {
                id: "tensor_a",
                name: "A",
                position: { x: 80, y: 120 },
                indices: [
                  { id: "tensor_a_i", name: "i", dimension: 2, offset: { x: -58, y: -20 }, metadata: {} },
                  { id: "tensor_a_x", name: "x", dimension: 3, offset: { x: 58, y: -20 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_b",
                name: "B",
                position: { x: 240, y: 120 },
                indices: [
                  { id: "tensor_b_x", name: "x", dimension: 3, offset: { x: -58, y: -20 }, metadata: {} },
                  { id: "tensor_b_y", name: "y", dimension: 5, offset: { x: 58, y: -20 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_c",
                name: "C",
                position: { x: 400, y: 120 },
                indices: [
                  { id: "tensor_c_y", name: "y", dimension: 5, offset: { x: -58, y: -20 }, metadata: {} },
                  { id: "tensor_c_j", name: "j", dimension: 7, offset: { x: 58, y: -20 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_d",
                name: "D",
                position: { x: 560, y: 120 },
                indices: [
                  { id: "tensor_d_k", name: "k", dimension: 11, offset: { x: -58, y: -20 }, metadata: {} },
                ],
                metadata: {},
              },
            ],
            groups: [],
            edges: [
              {
                id: "edge_x",
                name: "bond_x",
                left: { tensor_id: "tensor_a", index_id: "tensor_a_x" },
                right: { tensor_id: "tensor_b", index_id: "tensor_b_x" },
                metadata: {},
              },
              {
                id: "edge_y",
                name: "bond_y",
                left: { tensor_id: "tensor_b", index_id: "tensor_b_y" },
                right: { tensor_id: "tensor_c", index_id: "tensor_c_y" },
                metadata: {},
              },
            ],
            notes: [],
            contraction_plan: {
              id: "plan_chain",
              name: "Chain path",
              steps: [
                {
                  id: "step_ab",
                  left_operand_id: "tensor_a",
                  right_operand_id: "tensor_b",
                  metadata: {},
                },
                {
                  id: "step_abc",
                  left_operand_id: "step_ab",
                  right_operand_id: "tensor_c",
                  metadata: {},
                },
              ],
              metadata: {},
            },
            metadata: {},
          };
        }

        const ctx = await buildContext();
        await registerContractionScene(ctx);
        await registerHistory(ctx);
        await registerPlanner(ctx);
        await registerInteractions(ctx);

        ctx.state.selectedEngine = "tensornetwork";
        ctx.state.selectedCollectionFormat = "list";
        ctx.state.spec = ctx.normalizeSpec(buildPartialPlanSpec());

        ctx.setSelection(["step_abc"], { primaryId: "step_abc" });
        ctx.deleteSelection();

        if (ctx.state.spec.contraction_plan !== null) {
          throw new Error("Deleting a contracted result should clear the affected contraction history.");
        }
        if (ctx.state.spec.tensors.length !== 1) {
          throw new Error(`Expected only the unrelated tensor to remain, found ${ctx.state.spec.tensors.length} tensors.`);
        }
        if (ctx.state.spec.tensors[0].id !== "tensor_d") {
          throw new Error(`Expected tensor_d to remain, received ${ctx.state.spec.tensors[0].id}.`);
        }
        if (ctx.state.spec.edges.length !== 0) {
          throw new Error("Deleting a contracted result should also remove the edges of the contained tensors.");
        }
        if (!ctx.dom.statusMessage.textContent.includes("deleted")) {
          throw new Error(`Expected a deletion confirmation message, received: ${ctx.dom.statusMessage.textContent}`);
        }
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The contracted-result deletion regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_planner_renders_comparison_summaries(tmp_path: Path) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "planner_comparison_summary_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildComparisonSpec() {
          return {
            id: "network_compare",
            name: "compare",
            tensors: [
              {
                id: "tensor_a",
                name: "A",
                position: { x: 80, y: 120 },
                indices: [
                  { id: "tensor_a_i", name: "i", dimension: 2, offset: { x: -58, y: -20 }, metadata: {} },
                  { id: "tensor_a_x", name: "x", dimension: 3, offset: { x: 58, y: -20 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_b",
                name: "B",
                position: { x: 240, y: 120 },
                indices: [
                  { id: "tensor_b_x", name: "x", dimension: 3, offset: { x: -58, y: -20 }, metadata: {} },
                  { id: "tensor_b_j", name: "j", dimension: 4, offset: { x: 58, y: -20 }, metadata: {} },
                ],
                metadata: {},
              },
            ],
            groups: [],
            edges: [
              {
                id: "edge_x",
                name: "bond_x",
                left: { tensor_id: "tensor_a", index_id: "tensor_a_x" },
                right: { tensor_id: "tensor_b", index_id: "tensor_b_x" },
                metadata: {},
              },
            ],
            notes: [],
            contraction_plan: null,
            metadata: {},
          };
        }

        const ctx = await buildContext();
        ctx.dom.plannerPanel = {
          innerHTML: "",
          querySelectorAll() {
            return [];
          },
        };
        await registerHistory(ctx);
        await registerPlanner(ctx);

        ctx.state.spec = ctx.normalizeSpec(buildComparisonSpec());
        ctx.state.contractionAnalysis = {
          status: "ready",
          payload: {
            memory_dtype: "float64",
            network_output_shape: [2, 4],
            manual: {
              status: "complete",
              steps: [],
              summary: {
                total_estimated_flops: 1600,
                total_estimated_macs: 800,
                peak_intermediate_size: 100,
                peak_intermediate_bytes: 800,
                final_shape: [2, 4],
              },
            },
            automatic_full: {
              status: "complete",
              steps: [],
              summary: {
                total_estimated_flops: 1224,
                total_estimated_macs: 612,
                peak_intermediate_size: 6,
                peak_intermediate_bytes: 48,
              },
            },
            automatic_future: {
              status: "complete",
              steps: [],
              summary: {
                total_estimated_flops: 140,
                total_estimated_macs: 70,
                peak_intermediate_size: 14,
                peak_intermediate_bytes: 112,
              },
            },
            automatic_past: {
              status: "complete",
              steps: [],
              summary: {
                total_estimated_flops: 576,
                total_estimated_macs: 288,
                peak_intermediate_size: 12,
                peak_intermediate_bytes: 96,
              },
            },
            comparisons: {
              manual_vs_automatic_full: {
                status: "complete",
                baseline_label: "manual",
                candidate_label: "automatic_full",
                memory_dtype: "float64",
                baseline_peak_intermediate_bytes: 800,
                candidate_peak_intermediate_bytes: 48,
                delta_total_estimated_flops: -376,
                delta_total_estimated_macs: -188,
                delta_peak_intermediate_size: -94,
                delta_peak_intermediate_bytes: -752,
                baseline_peak_step_id: "step_bcd",
                candidate_peak_step_id: "auto_full_step_1",
                baseline_bottleneck_labels: ["x", "y", "z"],
                candidate_bottleneck_labels: ["i", "j"],
              },
              manual_subtrees_vs_automatic_past: {
                status: "complete",
                baseline_label: "manual_subtrees",
                candidate_label: "automatic_past",
                memory_dtype: "float64",
                baseline_peak_intermediate_bytes: 192,
                candidate_peak_intermediate_bytes: 96,
                delta_total_estimated_flops: -24,
                delta_total_estimated_macs: -12,
                delta_peak_intermediate_size: -12,
                delta_peak_intermediate_bytes: -96,
                baseline_peak_step_id: "step_ab",
                candidate_peak_step_id: "step_ab",
                baseline_bottleneck_labels: ["x", "y"],
                candidate_bottleneck_labels: ["x"],
              },
            },
            automatic_strategy: "greedy",
          },
        };
        ctx.state.plannerDisclosureState.automaticFull = true;
        ctx.state.plannerDisclosureState.automaticFuture = true;
        ctx.state.plannerDisclosureState.automaticPast = true;
        ctx.state.plannerDisclosureState.automaticFullComparison = true;
        ctx.state.plannerDisclosureState.automaticPastComparison = false;
        ctx.state.plannerInspectionStepCount = 0;

        ctx.renderPlanner();

        const html = ctx.dom.plannerPanel.innerHTML;
        const autoFullPosition = html.indexOf("Auto full");
        const manualFullPosition = html.indexOf("Manual vs auto full");
        const autoFuturePosition = html.indexOf("Auto future");
        const autoPastPosition = html.indexOf("Auto past");
        const manualPastPosition = html.indexOf("Manual contractions vs auto past");
        if (autoFullPosition < 0) {
          throw new Error(`Expected an Auto full disclosure, received: ${html}`);
        }
        if (
          manualFullPosition < 0 ||
          !(autoFullPosition < manualFullPosition && manualFullPosition < autoFuturePosition)
        ) {
          throw new Error(`Expected Manual vs auto full inside the Auto full disclosure, received: ${html}`);
        }
        if (
          manualPastPosition < 0 ||
          !(autoPastPosition < manualPastPosition)
        ) {
          throw new Error(`Expected the manual comparison inside the Auto past disclosure, received: ${html}`);
        }
        if (html.includes("Manual subtrees vs auto past")) {
          throw new Error(`The planner should not expose the internal "Manual subtrees" label, received: ${html}`);
        }
        if (html.includes("Viewing the scene before step 1")) {
          throw new Error(`The planner should not render the old inspection helper message, received: ${html}`);
        }
        if (!html.includes("Auto - Manual")) {
          throw new Error(`Expected comparison chips to explain the Auto - Manual delta, received: ${html}`);
        }
        if (!html.includes('data-tooltip-enabled="true"')) {
          throw new Error(`Expected the planner disclosures and actions to opt into shared tooltips, received: ${html}`);
        }
        if (!html.includes("Computes a full automatic contraction path for the whole visible network.")) {
          throw new Error(`Expected Auto full to explain its scope when hovered, received: ${html}`);
        }
        if (!html.includes("Plans the remaining visible operands from the current manual path onward.")) {
          throw new Error(`Expected Auto future to explain its scope when hovered, received: ${html}`);
        }
        if (!html.includes("Replans tensors that are already merged inside the current manual contractions.")) {
          throw new Error(`Expected Auto past to explain its scope when hovered, received: ${html}`);
        }
        if (!html.includes("Compares the current manual path against the full automatic contraction path.")) {
          throw new Error(`Expected Manual vs auto full to explain the comparison, received: ${html}`);
        }
        if (!html.includes("Compares the already contracted manual subtrees against the automatic replanning of that past work.")) {
          throw new Error(`Expected Manual contractions vs auto past to explain the comparison, received: ${html}`);
        }
        if (!html.includes("Toggle a non-destructive preview of this automatic path on the canvas.")) {
          throw new Error(`Expected Preview to explain that it does not replace the manual plan, received: ${html}`);
        }
        if (!html.includes("Replace the current manual path with this automatic contraction plan.")) {
          throw new Error(`Expected Accept to explain that it replaces the current path, received: ${html}`);
        }
        if (!html.includes('id="toggle-planner-mode-button"') || !html.includes('data-shortcut="M"')) {
          throw new Error(`Expected the planner to keep exposing the Contract shortcut, received: ${html}`);
        }
        if (!html.includes("Toggle manual contraction mode, then click two tensors or intermediate results to add a step.")) {
          throw new Error(`Expected Contract to explain how to add a manual step, received: ${html}`);
        }
        if (!html.includes("Estimated floating-point operations across the full contraction path.")) {
          throw new Error(`Expected FLOP metric help text, received: ${html}`);
        }
        if (!html.includes("Estimated multiply-accumulate operations across the full contraction path.")) {
          throw new Error(`Expected MAC metric help text, received: ${html}`);
        }
        if (!html.includes("Largest intermediate tensor reached during the path, measured in elements.")) {
          throw new Error(`Expected Peak metric help text, received: ${html}`);
        }
        if (!html.includes("Estimated memory used by the largest intermediate tensor for the reported dtype.")) {
          throw new Error(`Expected Memory metric help text, received: ${html}`);
        }
        if (!html.includes('class="planner-chip-info"')) {
          throw new Error(`Expected metric chips to render the new help icon, received: ${html}`);
        }
        if (!html.includes("planner-disclosure-state planner-disclosure-state-hide")) {
          throw new Error(`Expected open disclosures to render the pale Hide state, received: ${html}`);
        }
        if (!html.includes("planner-disclosure-state planner-disclosure-state-show")) {
          throw new Error(`Expected closed disclosures to render the pale Show state, received: ${html}`);
        }
        if (!html.includes(">FLOP</span>") || !html.includes("<strong>-376</strong>")) {
          throw new Error(`Expected the FLOP comparison chip to render the raw delta, received: ${html}`);
        }
        if (!html.includes(">Memory</span>") || !html.includes("<strong>-752 bytes</strong>")) {
          throw new Error(`Expected the memory comparison chip to render the raw delta, received: ${html}`);
        }
        if (!html.includes("<strong>800 bytes</strong>")) {
          throw new Error(`Expected the manual summary to include peak memory, received: ${html}`);
        }
        if (!html.includes("<strong>112 bytes</strong>")) {
          throw new Error(`Expected the automatic future summary to include peak memory, received: ${html}`);
        }
        if (!html.includes("<strong>96 bytes</strong>")) {
          throw new Error(`Expected the automatic past summary to include peak memory, received: ${html}`);
        }
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The planner comparison summary regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_planner_hides_automatic_sections_when_opt_einsum_is_missing(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "planner_missing_opt_einsum_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildSimpleSpec() {
          return {
            id: "network_missing_opt_einsum",
            name: "missing-opt-einsum",
            tensors: [
              {
                id: "tensor_a",
                name: "A",
                position: { x: 80, y: 120 },
                indices: [
                  { id: "tensor_a_i", name: "i", dimension: 2, offset: { x: -38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_b",
                name: "B",
                position: { x: 240, y: 120 },
                indices: [
                  { id: "tensor_b_i", name: "i", dimension: 2, offset: { x: -38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
            ],
            groups: [],
            edges: [],
            notes: [],
            contraction_plan: null,
            metadata: {},
          };
        }

        const ctx = await buildContext();
        ctx.dom.plannerPanel = {
          innerHTML: "",
          querySelectorAll() {
            return [];
          },
        };
        await registerHistory(ctx);
        await registerPlanner(ctx);

        ctx.state.spec = ctx.normalizeSpec(buildSimpleSpec());
        ctx.state.contractionAnalysis = {
          status: "ready",
          payload: {
            memory_dtype: "float64",
            network_output_shape: [2, 2],
            manual: {
              status: "complete",
              steps: [],
              summary: {
                total_estimated_flops: 0,
                total_estimated_macs: 0,
                peak_intermediate_size: 0,
                peak_intermediate_bytes: 0,
                final_shape: [2, 2],
              },
            },
            automatic_full: {
              status: "unavailable",
              steps: [],
              summary: {
                total_estimated_flops: 0,
                total_estimated_macs: 0,
                peak_intermediate_size: 0,
                peak_intermediate_bytes: 0,
              },
              message:
                "Install opt_einsum in the current .venv to enable Auto full, Auto future, and Auto past.",
            },
            automatic_future: {
              status: "unavailable",
              steps: [],
              summary: {
                total_estimated_flops: 0,
                total_estimated_macs: 0,
                peak_intermediate_size: 0,
                peak_intermediate_bytes: 0,
              },
              message:
                "Install opt_einsum in the current .venv to enable Auto full, Auto future, and Auto past.",
            },
            automatic_past: {
              status: "unavailable",
              steps: [],
              summary: {
                total_estimated_flops: 0,
                total_estimated_macs: 0,
                peak_intermediate_size: 0,
                peak_intermediate_bytes: 0,
              },
              message:
                "Install opt_einsum in the current .venv to enable Auto full, Auto future, and Auto past.",
            },
            comparisons: {
              manual_vs_automatic_full: {
                status: "unavailable",
                message:
                  "Install opt_einsum in the current .venv to enable Auto full, Auto future, and Auto past.",
              },
              manual_subtrees_vs_automatic_past: {
                status: "unavailable",
                message:
                  "Install opt_einsum in the current .venv to enable Auto full, Auto future, and Auto past.",
              },
            },
            automatic_strategy: "greedy",
          },
        };
        ctx.state.plannerDisclosureState.automaticFull = true;
        ctx.state.plannerDisclosureState.automaticFuture = true;
        ctx.state.plannerDisclosureState.automaticPast = true;

        ctx.renderPlanner();

        const html = ctx.dom.plannerPanel.innerHTML;
        if (
          html.includes('data-disclosure="automaticFull"') ||
          html.includes('data-disclosure="automaticFuture"') ||
          html.includes('data-disclosure="automaticPast"')
        ) {
          throw new Error(`Expected automatic sections to stay hidden when opt_einsum is missing, received: ${html}`);
        }
        if (!html.includes("planner-inline-meta planner-error")) {
          throw new Error(`Expected the missing opt_einsum warning to render in red, received: ${html}`);
        }
        if (!html.includes("Install opt_einsum in the current .venv")) {
          throw new Error(`Expected the missing opt_einsum warning to be visible, received: ${html}`);
        }
        if (!html.includes(">Manual<")) {
          throw new Error(`Expected the manual section to remain visible, received: ${html}`);
        }
      """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The missing-opt-einsum planner regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_contraction_scene_builds_long_manual_chain_without_repeated_rebuilds(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "contraction_scene_long_chain_performance_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildChainSpec(tensorCount) {
          const tensors = [];
          const edges = [];
          for (let index = 0; index < tensorCount; index += 1) {
            tensors.push({
              id: `tensor_${index}`,
              name: `T${index}`,
              position: { x: 120 + index * 160, y: 120 },
              size: { width: 140, height: 84 },
              indices: [
                {
                  id: `tensor_${index}_left`,
                  name: "left",
                  dimension: 2,
                  offset: { x: -38, y: 0 },
                  metadata: {},
                },
                {
                  id: `tensor_${index}_right`,
                  name: "right",
                  dimension: 2,
                  offset: { x: 38, y: 0 },
                  metadata: {},
                },
              ],
              metadata: {},
            });
            if (index > 0) {
              edges.push({
                id: `edge_${index}`,
                name: `bond_${index}`,
                left: {
                  tensor_id: `tensor_${index - 1}`,
                  index_id: `tensor_${index - 1}_right`,
                },
                right: {
                  tensor_id: `tensor_${index}`,
                  index_id: `tensor_${index}_left`,
                },
                metadata: {},
              });
            }
          }

          const steps = [];
          let leftOperandId = "tensor_0";
          for (let index = 1; index < tensorCount; index += 1) {
            const stepId = `step_${index}`;
            steps.push({
              id: stepId,
              left_operand_id: leftOperandId,
              right_operand_id: `tensor_${index}`,
              metadata: {},
            });
            leftOperandId = stepId;
          }

          return {
            id: "network_long_chain",
            name: "long-chain",
            tensors,
            groups: [],
            edges,
            notes: [],
            contraction_plan: {
              id: "plan_long_chain",
              name: "Long chain",
              steps,
              view_snapshots: [],
              metadata: {},
            },
            metadata: {},
          };
        }

        const tensorCount = 80;
        const ctx = await buildContext();
        await registerContractionScene(ctx);
        ctx.state.spec = ctx.normalizeSpec(buildChainSpec(tensorCount));

        let tensorRebuildCount = 0;
        let edgeRebuildCount = 0;
        const originalGetContractibleTensors = ctx.getContractibleTensors;
        const originalGetContractibleEdges = ctx.getContractibleEdges;
        ctx.getContractibleTensors = (...args) => {
          tensorRebuildCount += 1;
          return originalGetContractibleTensors(...args);
        };
        ctx.getContractibleEdges = (...args) => {
          edgeRebuildCount += 1;
          return originalGetContractibleEdges(...args);
        };

        const scene = ctx.buildContractionScene();
        const snapshots = ctx.state.spec.contraction_plan.view_snapshots;

        if (!scene) {
          throw new Error("Expected a contraction scene for the long manual chain.");
        }
        if (scene.validSteps.length !== tensorCount - 1) {
          throw new Error(`Expected ${tensorCount - 1} valid steps, received ${scene.validSteps.length}.`);
        }
        if (scene.tensors.length !== 1) {
          throw new Error(`Expected one final visible operand, received ${scene.tensors.length}.`);
        }
        if (scene.tensors[0].sourceTensorIds.length !== tensorCount) {
          throw new Error(`Expected the final operand to contain ${tensorCount} source tensors.`);
        }
        if (!Array.isArray(snapshots) || snapshots.length !== tensorCount) {
          throw new Error(`Expected ${tensorCount} snapshots, received ${snapshots && snapshots.length}.`);
        }
        if (tensorRebuildCount > 8 || edgeRebuildCount > 8) {
          throw new Error(
            `Expected bounded scene rebuilds, received tensors=${tensorRebuildCount}, edges=${edgeRebuildCount}.`
          );
        }
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The long-chain contraction-scene performance regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_contraction_scene_reuses_cached_scene_until_revision_changes(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "contraction_scene_cache_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildChainSpec(tensorCount) {
          const tensors = [];
          const edges = [];
          for (let index = 0; index < tensorCount; index += 1) {
            tensors.push({
              id: `tensor_${index}`,
              name: `T${index}`,
              position: { x: 120 + index * 160, y: 120 },
              size: { width: 140, height: 84 },
              indices: [
                {
                  id: `tensor_${index}_left`,
                  name: "left",
                  dimension: 2,
                  offset: { x: -38, y: 0 },
                  metadata: {},
                },
                {
                  id: `tensor_${index}_right`,
                  name: "right",
                  dimension: 2,
                  offset: { x: 38, y: 0 },
                  metadata: {},
                },
              ],
              metadata: {},
            });
            if (index > 0) {
              edges.push({
                id: `edge_${index}`,
                name: `bond_${index}`,
                left: {
                  tensor_id: `tensor_${index - 1}`,
                  index_id: `tensor_${index - 1}_right`,
                },
                right: {
                  tensor_id: `tensor_${index}`,
                  index_id: `tensor_${index}_left`,
                },
                metadata: {},
              });
            }
          }

          const steps = [];
          let leftOperandId = "tensor_0";
          for (let index = 1; index < tensorCount; index += 1) {
            const stepId = `step_${index}`;
            steps.push({
              id: stepId,
              left_operand_id: leftOperandId,
              right_operand_id: `tensor_${index}`,
              metadata: {},
            });
            leftOperandId = stepId;
          }

          return {
            id: "network_long_chain",
            name: "long-chain",
            tensors,
            groups: [],
            edges,
            notes: [],
            contraction_plan: {
              id: "plan_long_chain",
              name: "Long chain",
              steps,
              view_snapshots: [],
              metadata: {},
            },
            metadata: {},
          };
        }

        const ctx = await buildContext();
        await registerContractionScene(ctx);
        ctx.state.spec = ctx.normalizeSpec(buildChainSpec(24));
        ctx.bumpSpecRevision();

        let tensorBuildCount = 0;
        let edgeBuildCount = 0;
        const originalGetContractibleTensors = ctx.getContractibleTensors;
        const originalGetContractibleEdges = ctx.getContractibleEdges;
        ctx.getContractibleTensors = (...args) => {
          tensorBuildCount += 1;
          return originalGetContractibleTensors(...args);
        };
        ctx.getContractibleEdges = (...args) => {
          edgeBuildCount += 1;
          return originalGetContractibleEdges(...args);
        };

        const firstScene = ctx.buildContractionScene();
        const tensorCountAfterFirstScene = tensorBuildCount;
        const edgeCountAfterFirstScene = edgeBuildCount;
        const secondScene = ctx.buildContractionScene();
        const visibleTensors = ctx.getVisibleTensors();
        const visibleEdges = ctx.getVisibleEdges();
        const locatedTensor = ctx.findVisibleTensorById(firstScene.tensors[0].id);

        if (!firstScene || !secondScene || !locatedTensor) {
          throw new Error("Expected repeated scene lookups to resolve a visible contraction scene.");
        }
        if (secondScene !== firstScene) {
          throw new Error("Expected repeated scene reads to reuse the cached scene object.");
        }
        if (visibleTensors !== firstScene.tensors || visibleEdges !== firstScene.edges) {
          throw new Error("Expected visible tensors and edges to come from the cached scene.");
        }
        if (tensorBuildCount !== tensorCountAfterFirstScene || edgeBuildCount !== edgeCountAfterFirstScene) {
          throw new Error(
            `Expected repeated latest-scene reads to avoid recomputation, received tensors=${tensorBuildCount}, edges=${edgeBuildCount}.`
          );
        }

        ctx.state.plannerInspectionStepCount = 5;
        const inspectedScene = ctx.buildContractionScene();
        const tensorCountAfterInspection = tensorBuildCount;
        const edgeCountAfterInspection = edgeBuildCount;
        const inspectedSceneAgain = ctx.buildContractionScene();
        if (!inspectedScene || inspectedScene.appliedStepCount !== 5) {
          throw new Error("Expected the inspected scene to use the requested step count.");
        }
        if (inspectedSceneAgain !== inspectedScene) {
          throw new Error("Expected repeated inspected-scene reads to reuse the cached object.");
        }
        if (tensorBuildCount !== tensorCountAfterInspection || edgeBuildCount !== edgeCountAfterInspection) {
          throw new Error(
            `Expected repeated inspected-scene reads to avoid recomputation, received tensors=${tensorBuildCount}, edges=${edgeBuildCount}.`
          );
        }

        ctx.state.spec.contraction_plan.steps.pop();
        ctx.bumpSpecRevision();
        const rebuiltScene = ctx.buildContractionScene();
        if (!rebuiltScene) {
          throw new Error("Expected the scene to rebuild after the spec revision changes.");
        }
        if (tensorBuildCount <= tensorCountAfterInspection || edgeBuildCount <= edgeCountAfterInspection) {
          throw new Error(
            `Expected a spec revision change to invalidate the scene cache, received tensors=${tensorBuildCount}, edges=${edgeBuildCount}.`
          );
        }
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The contraction-scene cache regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_planner_operand_state_reuses_cache_until_revision_changes(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "planner_operand_state_cache_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildPlannerSpec(tensorCount) {
          const tensors = [];
          const edges = [];
          const steps = [];
          let leftOperandId = "tensor_0";
          for (let index = 0; index < tensorCount; index += 1) {
            tensors.push({
              id: `tensor_${index}`,
              name: `T${index}`,
              position: { x: 120 + index * 160, y: 120 },
              size: { width: 140, height: 84 },
              indices: [
                {
                  id: `tensor_${index}_left`,
                  name: "left",
                  dimension: 2,
                  offset: { x: -38, y: 0 },
                  metadata: {},
                },
                {
                  id: `tensor_${index}_right`,
                  name: "right",
                  dimension: 2,
                  offset: { x: 38, y: 0 },
                  metadata: {},
                },
              ],
              metadata: {},
            });
            if (index > 0) {
              edges.push({
                id: `edge_${index}`,
                name: `bond_${index}`,
                left: {
                  tensor_id: `tensor_${index - 1}`,
                  index_id: `tensor_${index - 1}_right`,
                },
                right: {
                  tensor_id: `tensor_${index}`,
                  index_id: `tensor_${index}_left`,
                },
                metadata: {},
              });
            }
            if (index > 0 && index < tensorCount - 1) {
              const stepId = `step_${index}`;
              steps.push({
                id: stepId,
                left_operand_id: leftOperandId,
                right_operand_id: `tensor_${index}`,
                metadata: {},
              });
              leftOperandId = stepId;
            }
          }

          return {
            id: "network_planner_chain",
            name: "planner-chain",
            tensors,
            groups: [],
            edges,
            notes: [],
            contraction_plan: {
              id: "plan_planner_chain",
              name: "Planner chain",
              steps,
              metadata: {},
            },
            metadata: {},
          };
        }

        const ctx = await buildContext();
        await registerPlanner(ctx);
        ctx.state.spec = ctx.normalizeSpec(buildPlannerSpec(9));
        ctx.bumpSpecRevision();

        let tensorBuildCount = 0;
        const originalGetContractibleTensors = ctx.getContractibleTensors;
        ctx.getContractibleTensors = (...args) => {
          tensorBuildCount += 1;
          return originalGetContractibleTensors(...args);
        };

        const firstState = ctx.buildPlannerOperandState();
        const tensorCountAfterFirstState = tensorBuildCount;
        const secondState = ctx.buildPlannerOperandState();
        const remainingOperandIds = ctx.getPlannerRemainingOperandIds();
        const resolvedOperandId = ctx.resolvePlannerOperandId("tensor_0");
        const stepOrdersByTensorId = ctx.buildStepOrdersByTensorId(
          ctx.state.spec.contraction_plan.steps
        );

        if (!firstState || !secondState) {
          throw new Error("Expected planner operand state helpers to resolve a state.");
        }
        if (firstState !== secondState) {
          throw new Error("Expected repeated planner-state reads to reuse the cached state object.");
        }
        if (!Array.isArray(remainingOperandIds) || remainingOperandIds.length !== 2) {
          throw new Error(`Expected two remaining planner operands, received ${remainingOperandIds}.`);
        }
        if (resolvedOperandId !== "step_7") {
          throw new Error(`Expected tensor_0 to resolve to the latest derived operand, received ${resolvedOperandId}.`);
        }
        if (!Array.isArray(stepOrdersByTensorId.tensor_0) || stepOrdersByTensorId.tensor_0.length !== 7) {
          throw new Error("Expected planner step orders to stay available for cached state consumers.");
        }
        if (tensorBuildCount !== tensorCountAfterFirstState) {
          throw new Error(
            `Expected repeated planner-state reads to avoid recomputation, received tensors=${tensorBuildCount}.`
          );
        }

        ctx.state.spec.contraction_plan.steps.pop();
        ctx.bumpSpecRevision();
        const rebuiltState = ctx.buildPlannerOperandState();
        if (!rebuiltState || rebuiltState.validSteps.length !== 6) {
          throw new Error("Expected the planner state to rebuild after the spec revision changes.");
        }
        if (tensorBuildCount <= tensorCountAfterFirstState) {
          throw new Error(
            `Expected a spec revision change to invalidate the planner cache, received tensors=${tensorBuildCount}.`
          );
        }
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The planner operand-state cache regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_planner_first_build_limits_membership_scans(tmp_path: Path) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "planner_first_build_membership_regression.mjs",
        f"""
        import {{ pathToFileURL }} from "node:url";

        const plannerSelectorsModuleUrl = pathToFileURL({json.dumps(str(PLANNER_SELECTORS_PATH))}).href;
        const {{ buildPlannerOperandState, buildPlannerSeedOperands }} = await import(
          plannerSelectorsModuleUrl
        );

        function buildPlannerData(tensorCount) {{
          const tensors = [];
          const specTensors = [];
          const steps = [];
          let leftOperandId = "tensor_0";
          for (let index = 0; index < tensorCount; index += 1) {{
            const tensor = {{
              id: `tensor_${{index}}`,
              linear_periodic_role: null,
            }};
            tensors.push(tensor);
            specTensors.push(tensor);
            if (index > 0 && index < tensorCount - 1) {{
              const stepId = `step_${{index}}`;
              steps.push({{
                id: stepId,
                left_operand_id: leftOperandId,
                right_operand_id: `tensor_${{index}}`,
                metadata: {{}},
              }});
              leftOperandId = stepId;
            }}
          }}
          return {{ tensors, specTensors, steps }};
        }}

        const {{ tensors, specTensors, steps }} = buildPlannerData(80);
        const seedOperands = buildPlannerSeedOperands({{
          tensors,
          specTensors,
          isLinearPeriodicMode: false,
          isLinearPeriodicBoundaryTensor: () => false,
          getLinearPeriodicReservedOperandIdForTensor: () => null,
        }});

        let someCallCount = 0;
        const originalSome = Array.prototype.some;
        Array.prototype.some = function (...args) {{
          someCallCount += 1;
          return originalSome.apply(this, args);
        }};

        try {{
          const plannerOperandState = buildPlannerOperandState({{
            tensors,
            steps,
            seedOperands,
            previousOperandId: "__linear_previous__",
            nextOperandId: "__linear_next__",
          }});

          if (plannerOperandState.validSteps.length !== 78) {{
            throw new Error(
              `Expected 78 valid planner steps, received ${{plannerOperandState.validSteps.length}}.`
            );
          }}
          if (plannerOperandState.activeOperandIds.length !== 2) {{
            throw new Error(
              `Expected two remaining planner operands, received ${{plannerOperandState.activeOperandIds}}.`
            );
          }}
          if (plannerOperandState.representativeByOperandId.tensor_0 !== "step_78") {{
            throw new Error(
              `Expected tensor_0 to resolve to step_78, received ${{plannerOperandState.representativeByOperandId.tensor_0}}.`
            );
          }}
          if (
            !Array.isArray(plannerOperandState.stepOrdersByTensorId.tensor_0) ||
            plannerOperandState.stepOrdersByTensorId.tensor_0.length !== 78
          ) {{
            throw new Error("Expected tensor_0 to keep all planner step orders.");
          }}
          if (someCallCount > 200) {{
            throw new Error(
              `Expected the first planner build to avoid quadratic membership scans, received some()=${{someCallCount}}.`
            );
          }}
        }} finally {{
          Array.prototype.some = originalSome;
        }}
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The planner first-build membership regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_preview_orders_stay_stable_without_visible_tensor_quadratic_scans(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "planner_preview_order_regression.mjs",
        f"""
        import {{ pathToFileURL }} from "node:url";

        const plannerSelectorsModuleUrl = pathToFileURL({json.dumps(str(PLANNER_SELECTORS_PATH))}).href;
        const {{ buildPreviewOrderByVisibleTensorId }} = await import(plannerSelectorsModuleUrl);

        function buildVisibleTensors(tensorCount) {{
          return Array.from({{ length: tensorCount }}, (_, index) => ({{
            id: `tensor_${{index}}`,
            sourceTensorIds: [`tensor_${{index}}`],
          }}));
        }}

        function buildAutomaticSteps(tensorCount) {{
          const steps = [];
          let leftOperandId = "tensor_0";
          for (let index = 1; index < tensorCount; index += 1) {{
            const resultOperandId = `step_${{index}}`;
            steps.push({{
              left_operand_id: leftOperandId,
              right_operand_id: `tensor_${{index}}`,
              result_operand_id: resultOperandId,
            }});
            leftOperandId = resultOperandId;
          }}
          return steps;
        }}

        const visibleTensors = buildVisibleTensors(80);
        const steps = buildAutomaticSteps(80);

        let someCallCount = 0;
        const originalSome = Array.prototype.some;
        Array.prototype.some = function (...args) {{
          someCallCount += 1;
          return originalSome.apply(this, args);
        }};

        try {{
          const previewOrderByTensorId = buildPreviewOrderByVisibleTensorId(
            visibleTensors,
            steps
          );

          if (
            !Array.isArray(previewOrderByTensorId.tensor_0) ||
            previewOrderByTensorId.tensor_0.length !== 79 ||
            previewOrderByTensorId.tensor_0[0] !== 1 ||
            previewOrderByTensorId.tensor_0.at(-1) !== 79
          ) {{
            throw new Error(
              `Expected tensor_0 preview orders to span the whole chain, received ${{previewOrderByTensorId.tensor_0}}.`
            );
          }}
          if (
            !Array.isArray(previewOrderByTensorId.tensor_40) ||
            previewOrderByTensorId.tensor_40.length !== 40 ||
            previewOrderByTensorId.tensor_40[0] !== 40 ||
            previewOrderByTensorId.tensor_40.at(-1) !== 79
          ) {{
            throw new Error(
              `Expected tensor_40 preview orders to begin at step 40, received ${{previewOrderByTensorId.tensor_40}}.`
            );
          }}
          if (
            !Array.isArray(previewOrderByTensorId.tensor_79) ||
            previewOrderByTensorId.tensor_79.length !== 1 ||
            previewOrderByTensorId.tensor_79[0] !== 79
          ) {{
            throw new Error(
              `Expected tensor_79 preview orders to contain only the final step, received ${{previewOrderByTensorId.tensor_79}}.`
            );
          }}
          if (someCallCount > 500) {{
            throw new Error(
              `Expected preview-order calculation to avoid scanning every visible tensor on each step, received some()=${{someCallCount}}.`
            );
          }}
        }} finally {{
          Array.prototype.some = originalSome;
        }}
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The planner preview-order regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_contraction_scene_first_build_limits_layout_map_rebuilds(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "contraction_scene_first_build_layout_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildChainSpec(tensorCount) {
          const tensors = [];
          const edges = [];
          for (let index = 0; index < tensorCount; index += 1) {
            tensors.push({
              id: `tensor_${index}`,
              name: `T${index}`,
              position: { x: 120 + index * 160, y: 120 },
              size: { width: 140, height: 84 },
              indices: [
                {
                  id: `tensor_${index}_left`,
                  name: "left",
                  dimension: 2,
                  offset: { x: -38, y: 0 },
                  metadata: {},
                },
                {
                  id: `tensor_${index}_right`,
                  name: "right",
                  dimension: 2,
                  offset: { x: 38, y: 0 },
                  metadata: {},
                },
              ],
              metadata: {},
            });
            if (index > 0) {
              edges.push({
                id: `edge_${index}`,
                name: `bond_${index}`,
                left: {
                  tensor_id: `tensor_${index - 1}`,
                  index_id: `tensor_${index - 1}_right`,
                },
                right: {
                  tensor_id: `tensor_${index}`,
                  index_id: `tensor_${index}_left`,
                },
                metadata: {},
              });
            }
          }

          const steps = [];
          let leftOperandId = "tensor_0";
          for (let index = 1; index < tensorCount; index += 1) {
            const stepId = `step_${index}`;
            steps.push({
              id: stepId,
              left_operand_id: leftOperandId,
              right_operand_id: `tensor_${index}`,
              metadata: {},
            });
            leftOperandId = stepId;
          }

          return {
            id: "network_long_chain",
            name: "long-chain",
            tensors,
            groups: [],
            edges,
            notes: [],
            contraction_plan: {
              id: "plan_long_chain",
              name: "Long chain",
              steps,
              view_snapshots: [],
              metadata: {},
            },
            metadata: {},
          };
        }

        const ctx = await buildContext();
        await registerContractionScene(ctx);
        ctx.state.spec = ctx.normalizeSpec(buildChainSpec(80));
        ctx.bumpSpecRevision();

        let asFiniteNumberCount = 0;
        const originalAsFiniteNumber = ctx.asFiniteNumber;
        ctx.asFiniteNumber = (...args) => {
          asFiniteNumberCount += 1;
          return originalAsFiniteNumber(...args);
        };

        const scene = ctx.buildContractionScene();
        if (!scene) {
          throw new Error("Expected a contraction scene for the long manual chain.");
        }
        if (scene.validSteps.length !== 79) {
          throw new Error(`Expected 79 valid steps, received ${scene.validSteps.length}.`);
        }
        if (!Array.isArray(ctx.state.spec.contraction_plan.view_snapshots) || ctx.state.spec.contraction_plan.view_snapshots.length !== 80) {
          throw new Error(
            `Expected 80 view snapshots, received ${ctx.state.spec.contraction_plan.view_snapshots && ctx.state.spec.contraction_plan.view_snapshots.length}.`
          );
        }
        if (asFiniteNumberCount > 1000) {
          throw new Error(
            `Expected the first scene build to avoid rebuilding every prior layout map, received asFiniteNumber=${asFiniteNumberCount}.`
          );
        }
        """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The contraction-scene first-build layout regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_automatic_past_preview_keeps_root_group_order_and_earliest_step(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "automatic_past_root_group_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildTensor(id, xPosition) {
          return {
            id,
            name: id.toUpperCase(),
            position: { x: xPosition, y: 120 },
            indices: [],
            metadata: {},
          };
        }

        function buildSpec() {
          return {
            id: "network_auto_past",
            name: "auto-past",
            tensors: [
              buildTensor("tensor_a", 80),
              buildTensor("tensor_b", 180),
              buildTensor("tensor_c", 280),
              buildTensor("tensor_d", 380),
              buildTensor("tensor_e", 480),
            ],
            groups: [],
            edges: [],
            notes: [],
            contraction_plan: {
              id: "plan_auto_past",
              name: "Auto past",
              steps: [
                { id: "step_ab", left_operand_id: "tensor_a", right_operand_id: "tensor_b", metadata: {} },
                { id: "step_abc", left_operand_id: "step_ab", right_operand_id: "tensor_c", metadata: {} },
                { id: "step_de", left_operand_id: "tensor_d", right_operand_id: "tensor_e", metadata: {} },
                { id: "step_root", left_operand_id: "step_abc", right_operand_id: "step_de", metadata: {} },
              ],
              view_snapshots: [],
              metadata: {},
            },
            metadata: {},
          };
        }

        const ctx = await buildContext();
        await registerContractionScene(ctx);
        await registerPlanner(ctx);
        await registerHistory(ctx);

        ctx.state.spec = ctx.normalizeSpec(buildSpec());
        ctx.bumpSpecRevision();
        ctx.state.contractionAnalysis = {
          status: "ready",
          payload: {
            automatic_past: {
              status: "complete",
              steps: [
                {
                  left_operand_id: "tensor_a",
                  right_operand_id: "tensor_c",
                  result_operand_id: "step_abc__auto_past_1",
                },
                {
                  left_operand_id: "step_abc__auto_past_1",
                  right_operand_id: "tensor_b",
                  result_operand_id: "step_abc",
                },
                {
                  left_operand_id: "tensor_d",
                  right_operand_id: "tensor_e",
                  result_operand_id: "step_de",
                },
              ],
            },
          },
        };

        ctx.startAutomaticPreview("automaticPast");

        if (ctx.state.plannerInspectionStepCount !== 0) {
          throw new Error(
            `Expected auto past preview to start from the first affected manual step, received ${ctx.state.plannerInspectionStepCount}.`
          );
        }

        ctx.acceptAutomaticPlan("automaticPast");

        const stepIds = ctx.state.spec.contraction_plan.steps.map((step) => step.id);
        if (stepIds.at(-1) !== "step_root") {
          throw new Error(`Expected the root contraction to remain last, received ${stepIds}.`);
        }
        if (stepIds.indexOf("step_abc") >= stepIds.indexOf("step_de")) {
          throw new Error(`Expected the rewritten auto-past roots to preserve manual root order, received ${stepIds}.`);
        }
      """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The automatic-past root-group regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_graph_render_reuses_existing_cytoscape_elements_for_stable_graph(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "graph_render_stable_diff_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildSimpleGraphSpec() {
          return {
            id: "network_graph_render",
            name: "graph-render",
            tensors: [
              {
                id: "tensor_a",
                name: "A",
                position: { x: 120, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_a_left", name: "left", dimension: 2, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_a_right", name: "right", dimension: 3, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_b",
                name: "B",
                position: { x: 360, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_b_left", name: "left", dimension: 3, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_b_right", name: "right", dimension: 5, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
            ],
            groups: [],
            edges: [
              {
                id: "edge_ab",
                name: "bond",
                left: { tensor_id: "tensor_a", index_id: "tensor_a_right" },
                right: { tensor_id: "tensor_b", index_id: "tensor_b_left" },
                metadata: {},
              },
            ],
            notes: [],
            contraction_plan: null,
            metadata: {},
          };
        }

        const ctx = await buildContext();
        await registerHistory(ctx);
        await registerGraphRender(ctx);
        const cyHarness = createCyStub();
        ctx.state.cy = cyHarness.cy;
        ctx.state.spec = ctx.normalizeSpec(buildSimpleGraphSpec());
        ctx.bumpSpecRevision();

        ctx.renderGraph();
        if (!cyHarness.getElementIds().includes("tensor_a") || !cyHarness.getElementIds().includes("edge_ab")) {
          throw new Error(`Expected the first render to populate graph elements, received ${cyHarness.getElementIds()}.`);
        }

        cyHarness.resetStats();
        ctx.renderGraph();

        if (cyHarness.stats.removeAllCalls !== 0 || cyHarness.stats.addCalls !== 0) {
          throw new Error(
            `Expected the stable graph render to reuse existing Cytoscape elements, received removeAll=${cyHarness.stats.removeAllCalls}, add=${cyHarness.stats.addCalls}.`
          );
        }
      """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The stable graph render regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_graph_render_updates_existing_elements_in_place_for_visual_edits(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "graph_render_in_place_update_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildSimpleGraphSpec() {
          return {
            id: "network_graph_update",
            name: "graph-update",
            tensors: [
              {
                id: "tensor_a",
                name: "A",
                position: { x: 120, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_a_left", name: "left", dimension: 2, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_a_right", name: "right", dimension: 3, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_b",
                name: "B",
                position: { x: 360, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_b_left", name: "left", dimension: 3, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_b_right", name: "right", dimension: 5, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
            ],
            groups: [],
            edges: [
              {
                id: "edge_ab",
                name: "bond",
                left: { tensor_id: "tensor_a", index_id: "tensor_a_right" },
                right: { tensor_id: "tensor_b", index_id: "tensor_b_left" },
                metadata: {},
              },
            ],
            notes: [],
            contraction_plan: null,
            metadata: {},
          };
        }

        const ctx = await buildContext();
        await registerHistory(ctx);
        await registerGraphRender(ctx);
        const cyHarness = createCyStub();
        ctx.state.cy = cyHarness.cy;
        ctx.state.spec = ctx.normalizeSpec(buildSimpleGraphSpec());
        ctx.bumpSpecRevision();

        ctx.renderGraph();
        cyHarness.resetStats();

        const tensor = ctx.findTensorById("tensor_a");
        tensor.name = "A*";
        tensor.metadata.color = "#ff5a5a";
        tensor.position.x += 48;
        ctx.bumpSpecRevision();
        ctx.renderGraph();

        const tensorSnapshot = cyHarness.getElementSnapshot("tensor_a");
        if (!tensorSnapshot) {
          throw new Error("Expected tensor_a to stay mounted after the visual edit.");
        }
        if (tensorSnapshot.data.label !== "A*" || tensorSnapshot.position.x !== 168) {
          throw new Error(`Expected tensor_a to update in place, received ${JSON.stringify(tensorSnapshot)}.`);
        }
        if (cyHarness.stats.removeAllCalls !== 0 || cyHarness.stats.addCalls !== 0) {
          throw new Error(
            `Expected visual edits to update existing elements in place, received removeAll=${cyHarness.stats.removeAllCalls}, add=${cyHarness.stats.addCalls}.`
          );
        }
        if (
          !cyHarness.stats.dataSetIds.includes("tensor_a") ||
          !cyHarness.stats.positionSetIds.includes("tensor_a")
        ) {
          throw new Error(
            `Expected tensor_a to receive in-place data and position updates, received data=${cyHarness.stats.dataSetIds}, position=${cyHarness.stats.positionSetIds}.`
          );
        }
      """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The in-place graph update regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_sync_cy_selection_updates_only_delta_between_old_and_new_selection(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "cy_selection_delta_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildSimpleGraphSpec() {
          return {
            id: "network_selection_delta",
            name: "selection-delta",
            tensors: [
              {
                id: "tensor_a",
                name: "A",
                position: { x: 120, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_a_left", name: "left", dimension: 2, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_a_right", name: "right", dimension: 3, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_b",
                name: "B",
                position: { x: 360, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_b_left", name: "left", dimension: 3, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_b_right", name: "right", dimension: 5, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
            ],
            groups: [],
            edges: [],
            notes: [],
            contraction_plan: null,
            metadata: {},
          };
        }

        const ctx = await buildContext();
        await registerHistory(ctx);
        await registerGraphRender(ctx);
        const cyHarness = createCyStub();
        ctx.state.cy = cyHarness.cy;
        ctx.state.spec = ctx.normalizeSpec(buildSimpleGraphSpec());
        ctx.bumpSpecRevision();
        ctx.renderGraph();

        ctx.state.selectionIds = ["tensor_a"];
        ctx.syncCySelection();
        cyHarness.resetStats();

        ctx.state.selectionIds = ["tensor_b"];
        ctx.syncCySelection();

        if (cyHarness.stats.bulkUnselectCalls !== 0) {
          throw new Error(
            `Expected Cytoscape selection sync to avoid clearing the whole selection, received bulkUnselect=${cyHarness.stats.bulkUnselectCalls}.`
          );
        }
        if (
          cyHarness.stats.unselectCalls.join(",") !== "tensor_a" ||
          cyHarness.stats.selectCalls.join(",") !== "tensor_b"
        ) {
          throw new Error(
            `Expected selection sync to update only the delta, received unselect=${cyHarness.stats.unselectCalls}, select=${cyHarness.stats.selectCalls}.`
          );
        }
      """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The Cytoscape selection delta regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_pending_interaction_classes_touch_only_previous_and_current_ids(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "pending_interaction_delta_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildSimpleGraphSpec() {
          return {
            id: "network_pending_delta",
            name: "pending-delta",
            tensors: [
              {
                id: "tensor_a",
                name: "A",
                position: { x: 120, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_a_left", name: "left", dimension: 2, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_a_right", name: "right", dimension: 3, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_b",
                name: "B",
                position: { x: 360, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_b_left", name: "left", dimension: 3, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_b_right", name: "right", dimension: 5, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
            ],
            groups: [],
            edges: [],
            notes: [],
            contraction_plan: null,
            metadata: {},
          };
        }

        const ctx = await buildContext();
        await registerHistory(ctx);
        await registerGraphRender(ctx);
        const cyHarness = createCyStub();
        ctx.state.cy = cyHarness.cy;
        ctx.state.spec = ctx.normalizeSpec(buildSimpleGraphSpec());
        ctx.bumpSpecRevision();
        ctx.renderGraph();

        ctx.state.pendingPlannerSelectionId = "tensor_a";
        ctx.state.pendingIndexId = "tensor_a_left";
        ctx.syncPendingInteractionClasses();
        cyHarness.resetStats();

        ctx.state.pendingPlannerSelectionId = "tensor_b";
        ctx.state.pendingIndexId = "tensor_b_left";
        ctx.syncPendingInteractionClasses();

        const toggledIds = [...cyHarness.stats.toggleClassIds].sort();
        const expectedIds = ["tensor_a", "tensor_a_left", "tensor_b", "tensor_b_left"];
        if (cyHarness.stats.nodeQueryCalls.length !== 0) {
          throw new Error(
            `Expected pending classes to avoid full node scans, received selectors=${cyHarness.stats.nodeQueryCalls}.`
          );
        }
        if (JSON.stringify(toggledIds) !== JSON.stringify(expectedIds)) {
          throw new Error(
            `Expected pending classes to touch only previous and current ids, received ${JSON.stringify(toggledIds)}.`
          );
        }
      """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The pending interaction delta regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_bring_tensor_to_front_does_not_reposition_unrelated_ports(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "bring_tensor_to_front_layer_only_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildSimpleGraphSpec() {
          return {
            id: "network_layer_only",
            name: "layer-only",
            tensors: [
              {
                id: "tensor_a",
                name: "A",
                position: { x: 120, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_a_left", name: "left", dimension: 2, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_a_right", name: "right", dimension: 3, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_b",
                name: "B",
                position: { x: 360, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_b_left", name: "left", dimension: 3, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_b_right", name: "right", dimension: 5, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
            ],
            groups: [],
            edges: [],
            notes: [],
            contraction_plan: null,
            metadata: {},
          };
        }

        const ctx = await buildContext();
        await registerHistory(ctx);
        await registerGraphRender(ctx);
        const cyHarness = createCyStub();
        ctx.state.cy = cyHarness.cy;
        ctx.state.spec = ctx.normalizeSpec(buildSimpleGraphSpec());
        ctx.bumpSpecRevision();
        ctx.renderGraph();

        cyHarness.resetStats();
        ctx.bringTensorToFront("tensor_b");

        if (cyHarness.stats.positionSetIds.length !== 0) {
          throw new Error(
            `Expected bringTensorToFront() to update only layer data, received position updates for ${cyHarness.stats.positionSetIds}.`
          );
        }
      """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The bring-to-front layer-only regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_sync_cy_selection_clears_stray_cytoscape_selection_by_delta(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "cy_selection_stray_delta_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildSimpleGraphSpec() {
          return {
            id: "network_selection_stray_delta",
            name: "selection-stray-delta",
            tensors: [
              {
                id: "tensor_a",
                name: "A",
                position: { x: 120, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_a_left", name: "left", dimension: 2, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_a_right", name: "right", dimension: 3, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_b",
                name: "B",
                position: { x: 360, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_b_left", name: "left", dimension: 3, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_b_right", name: "right", dimension: 5, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
            ],
            groups: [],
            edges: [],
            notes: [],
            contraction_plan: null,
            metadata: {},
          };
        }

        const ctx = await buildContext();
        await registerHistory(ctx);
        await registerGraphRender(ctx);
        const cyHarness = createCyStub();
        ctx.state.cy = cyHarness.cy;
        ctx.state.spec = ctx.normalizeSpec(buildSimpleGraphSpec());
        ctx.bumpSpecRevision();
        ctx.renderGraph();

        cyHarness.cy.getElementById("tensor_a_right").select();
        cyHarness.resetStats();
        ctx.state.selectionIds = ["tensor_b"];
        ctx.state.cySelectionSyncedIds = [];
        ctx.syncCySelection();

        if (cyHarness.stats.bulkUnselectCalls !== 0) {
          throw new Error(
            `Expected stray Cytoscape selection cleanup to avoid a bulk clear, received bulkUnselect=${cyHarness.stats.bulkUnselectCalls}.`
          );
        }
        if (
          cyHarness.stats.unselectCalls.join(",") !== "tensor_a_right" ||
          cyHarness.stats.selectCalls.join(",") !== "tensor_b"
        ) {
          throw new Error(
            `Expected selection sync to clear stray selections by delta, received unselect=${cyHarness.stats.unselectCalls}, select=${cyHarness.stats.selectCalls}.`
          );
        }
      """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The stray Cytoscape selection delta regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_contraction_scene_render_diffs_visible_elements_without_full_reset(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "contraction_scene_render_diff_regression.mjs",
        _build_runtime_prelude()
        + """
        function buildContractionSpec() {
          return {
            id: "network_scene_diff",
            name: "scene-diff",
            tensors: [
              {
                id: "tensor_a",
                name: "A",
                position: { x: 120, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_a_left", name: "left", dimension: 2, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_a_right", name: "right", dimension: 3, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_b",
                name: "B",
                position: { x: 360, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_b_left", name: "left", dimension: 3, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_b_right", name: "right", dimension: 5, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
              {
                id: "tensor_c",
                name: "C",
                position: { x: 600, y: 120 },
                size: { width: 140, height: 84 },
                indices: [
                  { id: "tensor_c_left", name: "left", dimension: 5, offset: { x: -38, y: 0 }, metadata: {} },
                  { id: "tensor_c_right", name: "right", dimension: 7, offset: { x: 38, y: 0 }, metadata: {} },
                ],
                metadata: {},
              },
            ],
            groups: [],
            edges: [
              {
                id: "edge_ab",
                name: "ab",
                left: { tensor_id: "tensor_a", index_id: "tensor_a_right" },
                right: { tensor_id: "tensor_b", index_id: "tensor_b_left" },
                metadata: {},
              },
              {
                id: "edge_bc",
                name: "bc",
                left: { tensor_id: "tensor_b", index_id: "tensor_b_right" },
                right: { tensor_id: "tensor_c", index_id: "tensor_c_left" },
                metadata: {},
              },
            ],
            notes: [],
            contraction_plan: {
              id: "plan_scene_diff",
              name: "Scene diff",
              steps: [
                { id: "step_1", left_operand_id: "tensor_a", right_operand_id: "tensor_b", metadata: {} },
                { id: "step_2", left_operand_id: "step_1", right_operand_id: "tensor_c", metadata: {} },
              ],
              view_snapshots: [],
              metadata: {},
            },
            metadata: {},
          };
        }

        const ctx = await buildContext();
        await registerHistory(ctx);
        await registerContractionScene(ctx);
        await registerGraphRender(ctx);
        const cyHarness = createCyStub();
        ctx.state.cy = cyHarness.cy;
        ctx.state.spec = ctx.normalizeSpec(buildContractionSpec());
        ctx.bumpSpecRevision();

        ctx.renderGraph();
        if (!cyHarness.getElementIds().includes("step_2")) {
          throw new Error(`Expected the latest contraction scene to expose the final operand, received ${cyHarness.getElementIds()}.`);
        }

        cyHarness.resetStats();
        ctx.state.plannerInspectionStepCount = 1;
        ctx.renderGraph();

        const inspectedIds = cyHarness.getElementIds();
        if (
          !inspectedIds.includes("step_1") ||
          !inspectedIds.includes("tensor_c") ||
          inspectedIds.includes("step_2")
        ) {
          throw new Error(`Expected the inspected scene to expose the intermediate operands, received ${inspectedIds}.`);
        }
        if (cyHarness.stats.removeAllCalls !== 0) {
          throw new Error(
            `Expected contraction-scene inspection renders to diff visible elements without a full reset, received removeAll=${cyHarness.stats.removeAllCalls}.`
          );
        }
        if (!cyHarness.stats.addCalls && !cyHarness.stats.removedIds.length) {
          throw new Error("Expected the inspected scene render to apply an actual visible diff.");
        }
      """,
    )
    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The contraction-scene render diff regression script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )

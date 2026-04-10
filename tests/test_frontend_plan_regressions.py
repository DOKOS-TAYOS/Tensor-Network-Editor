from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


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

    async function buildContext() {{
      const [constantsModule, stateModule, utilitiesModule] = await Promise.all([
        import(constantsModuleUrl),
        import(stateModuleUrl),
        import(utilitiesModuleUrl),
      ]);
      const {{ constants }} = constantsModule;
      const {{ createInitialState }} = stateModule;
      const {{ registerUtilities }} = utilitiesModule;

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
      return ctx;
    }}

    async function registerHistory(ctx) {{
      const {{ registerHistorySelection }} = await import(historyModuleUrl);
      registerHistorySelection(ctx);
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
        ctx.state.plannerDisclosureState.automaticFuture = true;
        ctx.state.plannerDisclosureState.automaticPast = true;
        ctx.state.plannerInspectionStepCount = 0;

        ctx.renderPlanner();

        const html = ctx.dom.plannerPanel.innerHTML;
        if (!html.includes("Manual vs auto full")) {
          throw new Error(`Expected Manual vs auto full summary, received: ${html}`);
        }
        if (html.includes("Viewing the scene before step 1")) {
          throw new Error(`The planner should not render the old inspection helper message, received: ${html}`);
        }
        if (!html.includes("Auto - Manual")) {
          throw new Error(`Expected comparison chips to explain the Auto - Manual delta, received: ${html}`);
        }
        if (!html.includes(">FLOP</span>") || !html.includes("<strong>-376</strong>")) {
          throw new Error(`Expected the FLOP comparison chip to render the raw delta, received: ${html}`);
        }
        if (!html.includes(">Memory</span>") || !html.includes("<strong>-752 bytes</strong>")) {
          throw new Error(`Expected the memory comparison chip to render the raw delta, received: ${html}`);
        }
        if (!html.includes("Manual subtrees vs auto past")) {
          throw new Error(`Expected Manual subtrees vs auto past summary, received: ${html}`);
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

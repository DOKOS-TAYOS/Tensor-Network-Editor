export function startEditor(ctx) {
  const state = ctx.state;
  const {
    TENSOR_WIDTH,
    TENSOR_HEIGHT,
    MIN_TENSOR_WIDTH,
    MIN_TENSOR_HEIGHT,
    INDEX_RADIUS,
    INDEX_PADDING,
    HISTORY_LIMIT,
    REDO_SHORTCUT_LABEL,
    DEFAULT_INDEX_SLOTS,
  } = ctx.constants;
  const {
    statusMessage,
    propertiesPanel,
    generatedCode,
    engineSelect,
    addNoteButton,
    connectButton,
    loadInput,
    undoButton,
    redoButton,
    exportPyButton,
    exportPngButton,
    exportSvgButton,
    templateSelect,
    templateGraphSizeInput,
    templateBondDimensionInput,
    templatePhysicalDimensionInput,
    insertTemplateButton,
    createGroupButton,
    helpButton,
    helpModal,
    helpBackdrop,
    helpCloseButton,
    canvasShell,
    groupLayer,
    resizeLayer,
    selectionBox,
    minimapCanvas,
  } = ctx.dom;
  const { apiGet, apiPost, window, document, cytoscape } = ctx;

  document.addEventListener("DOMContentLoaded", () => {
    attachToolbarHandlers();
    bootstrap().catch((error) => {
      ctx.setStatus(`Failed to load the editor: ${error.message}`, "error");
    });
  });

  async function bootstrap() {
    const payload = await apiGet("/api/bootstrap");
    state.spec = ctx.normalizeSpec(payload.spec.network);
    state.schemaVersion = payload.schema_version;
    state.availableTemplates = Array.isArray(payload.templates) ? [...payload.templates] : [];
    state.templateDefinitions = payload.template_definitions && typeof payload.template_definitions === "object"
      ? { ...payload.template_definitions }
      : {};
    state.templateParametersByTemplate = ctx.buildTemplateParameterState(
      state.availableTemplates,
      state.templateDefinitions
    );
    state.selectedEngine = payload.default_engine;
    ctx.reconcileTensorOrder();
    ctx.populateEngineOptions(payload.engines);
    ctx.populateTemplateOptions(state.availableTemplates);
    ctx.syncTemplateParameterControls();
    ctx.initGraph();
    ctx.clearHistory();
    ctx.render();
    if (typeof ctx.refreshContractionAnalysis === "function") {
      ctx.refreshContractionAnalysis();
    }
    ctx.setStatus(
      "Editor ready. Drag the canvas to move, use the wheel to zoom, and right drag to box-select.",
      "success"
    );
  }

  function attachToolbarHandlers() {
    document.getElementById("new-design-button").addEventListener("click", ctx.handleNewDesign);
    document.getElementById("add-tensor-button").addEventListener("click", ctx.addTensorAtCenter);
    addNoteButton.addEventListener("click", ctx.addNoteAtCenter);
    document.getElementById("connect-button").addEventListener("click", ctx.toggleConnectMode);
    document.getElementById("delete-button").addEventListener("click", ctx.deleteSelection);
    document.getElementById("save-button").addEventListener("click", ctx.saveDesign);
    document.getElementById("load-button").addEventListener("click", () => loadInput.click());
    document.getElementById("generate-button").addEventListener("click", ctx.generateCode);
    document.getElementById("done-button").addEventListener("click", ctx.completeEditor);
    document.getElementById("cancel-button").addEventListener("click", ctx.cancelEditor);
    document.getElementById("copy-code-button").addEventListener("click", ctx.copyGeneratedCode);
    undoButton.addEventListener("click", ctx.performUndo);
    redoButton.addEventListener("click", ctx.performRedo);
    exportPyButton.addEventListener("click", ctx.downloadPythonExport);
    exportPngButton.addEventListener("click", ctx.downloadPngExport);
    exportSvgButton.addEventListener("click", ctx.downloadSvgExport);
    templateSelect.addEventListener("change", ctx.handleTemplateSelectionChange);
    templateGraphSizeInput.addEventListener("change", ctx.handleTemplateParameterInput);
    templateBondDimensionInput.addEventListener("change", ctx.handleTemplateParameterInput);
    templatePhysicalDimensionInput.addEventListener("change", ctx.handleTemplateParameterInput);
    insertTemplateButton.addEventListener("click", ctx.insertTemplate);
    createGroupButton.addEventListener("click", ctx.createGroupFromSelection);
    helpButton.addEventListener("click", () => ctx.toggleHelpModal(true));
    helpBackdrop.addEventListener("click", () => ctx.toggleHelpModal(false));
    helpCloseButton.addEventListener("click", () => ctx.toggleHelpModal(false));
    engineSelect.addEventListener("change", (event) => {
      state.selectedEngine = event.target.value;
    });
    loadInput.addEventListener("change", ctx.loadDesignFromFile);
    window.addEventListener("keydown", ctx.handleKeydown);
    window.addEventListener("beforeunload", ctx.sendCancelBeacon);
    window.addEventListener("pagehide", ctx.sendCancelBeacon);
    window.addEventListener("resize", ctx.handleWindowResize);
    window.addEventListener("mousemove", ctx.handleGlobalMouseMove);
    window.addEventListener("mouseup", ctx.handleGlobalMouseUp);
    canvasShell.addEventListener("contextmenu", ctx.handleCanvasContextMenu);
    canvasShell.addEventListener("mousedown", ctx.handleCanvasMouseDown, true);
    minimapCanvas.addEventListener("mousedown", ctx.handleMinimapMouseDown);
  }

}

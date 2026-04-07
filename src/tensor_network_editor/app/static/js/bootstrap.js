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
    workspace,
    statusMessage,
    propertiesPanel,
    generatedCode,
    engineSelect,
    collectionFormatSelect,
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
  let shortcutTooltip = null;
  let activeShortcutButton = null;

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
    state.availableCollectionFormats = Array.isArray(payload.collection_formats)
      ? [...payload.collection_formats]
      : ["list"];
    state.templateDefinitions = payload.template_definitions && typeof payload.template_definitions === "object"
      ? { ...payload.template_definitions }
      : {};
    state.templateParametersByTemplate = ctx.buildTemplateParameterState(
      state.availableTemplates,
      state.templateDefinitions
    );
    state.selectedEngine = payload.default_engine;
    state.selectedCollectionFormat = payload.default_collection_format || "list";
    ctx.reconcileTensorOrder();
    ctx.populateEngineOptions(payload.engines);
    ctx.populateCollectionFormatOptions(state.availableCollectionFormats);
    ctx.populateTemplateOptions(state.availableTemplates);
    ctx.syncTemplateParameterControls();
    ctx.initGraph();
    ctx.clearHistory();
    ctx.render();
    if (typeof ctx.refreshContractionAnalysis === "function") {
      ctx.refreshContractionAnalysis();
    }
    ctx.setStatus(
      "Editor ready. Drag the canvas to move, use Ctrl+wheel to zoom, use the wheel to pan, and right drag to box-select.",
      "success"
    );
  }

  function applyShortcutHint(buttonId, label, shortcut) {
    const button = document.getElementById(buttonId);
    if (!button) {
      return;
    }
    button.dataset.shortcut = shortcut;
    button.dataset.shortcutLabel = label;
    button.setAttribute("aria-label", `${label} (${shortcut})`);
    button.removeAttribute("title");
  }

  function ensureShortcutTooltip() {
    if (shortcutTooltip) {
      return shortcutTooltip;
    }
    shortcutTooltip = document.createElement("div");
    shortcutTooltip.className = "shortcut-tooltip is-hidden";
    shortcutTooltip.setAttribute("aria-hidden", "true");
    document.body.appendChild(shortcutTooltip);
    return shortcutTooltip;
  }

  function formatShortcutTooltipText(button) {
    const label = button.dataset.shortcutLabel || button.textContent.trim();
    const shortcut = button.dataset.shortcut || "";
    return label ? `${label} (${shortcut})` : shortcut;
  }

  function positionShortcutTooltip(button) {
    const tooltip = ensureShortcutTooltip();
    const rect = button.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    const margin = 8;
    let left = rect.right - tooltipRect.width;
    let top = rect.bottom + margin;

    if (top + tooltipRect.height > window.innerHeight - margin) {
      top = rect.top - tooltipRect.height - margin;
    }
    left = Math.min(
      Math.max(margin, left),
      Math.max(margin, window.innerWidth - tooltipRect.width - margin)
    );
    top = Math.min(
      Math.max(margin, top),
      Math.max(margin, window.innerHeight - tooltipRect.height - margin)
    );

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }

  function showShortcutTooltip(button) {
    if (!button || !button.dataset.shortcut || button.disabled) {
      return;
    }
    const tooltip = ensureShortcutTooltip();
    tooltip.textContent = formatShortcutTooltipText(button);
    tooltip.classList.remove("is-hidden");
    activeShortcutButton = button;
    positionShortcutTooltip(button);
  }

  function hideShortcutTooltip(button = null) {
    if (button && activeShortcutButton && button !== activeShortcutButton) {
      return;
    }
    if (!shortcutTooltip) {
      return;
    }
    shortcutTooltip.classList.add("is-hidden");
    activeShortcutButton = null;
  }

  function attachShortcutTooltipHandlers() {
    document.addEventListener("mouseover", (event) => {
      const button =
        event.target instanceof Element
          ? event.target.closest("button[data-shortcut]")
          : null;
      if (!button) {
        return;
      }
      showShortcutTooltip(button);
    });
    document.addEventListener("mouseout", (event) => {
      const button =
        event.target instanceof Element
          ? event.target.closest("button[data-shortcut]")
          : null;
      if (!button) {
        return;
      }
      const relatedButton =
        event.relatedTarget instanceof Element
          ? event.relatedTarget.closest("button[data-shortcut]")
          : null;
      if (relatedButton === button) {
        return;
      }
      hideShortcutTooltip(button);
    });
    document.addEventListener("focusin", (event) => {
      const button =
        event.target instanceof Element
          ? event.target.closest("button[data-shortcut]")
          : null;
      if (!button) {
        return;
      }
      showShortcutTooltip(button);
    });
    document.addEventListener("focusout", (event) => {
      const button =
        event.target instanceof Element
          ? event.target.closest("button[data-shortcut]")
          : null;
      if (!button) {
        return;
      }
      hideShortcutTooltip(button);
    });
    window.addEventListener("resize", () => {
      if (activeShortcutButton) {
        positionShortcutTooltip(activeShortcutButton);
      }
    });
    window.addEventListener(
      "scroll",
      () => {
        if (activeShortcutButton) {
          positionShortcutTooltip(activeShortcutButton);
        }
      },
      true
    );
  }

  function attachToolbarHandlers() {
    applyShortcutHint("add-tensor-button", "Add tensor", "N");
    applyShortcutHint("insert-template-button", "Insert Template", "T");
    applyShortcutHint("create-group-button", "Group", "G");
    applyShortcutHint("add-note-button", "Add note", "P");
    applyShortcutHint("connect-button", "Connect", "C");
    applyShortcutHint("delete-button", "Delete", "Delete");
    applyShortcutHint("save-button", "Save", "Ctrl/Cmd+S");
    applyShortcutHint("load-button", "Load", "Ctrl/Cmd+L");
    applyShortcutHint("generate-button", "Generate code", "Shift+G");
    applyShortcutHint("undo-button", "Undo", "Ctrl/Cmd+Z");
    applyShortcutHint("redo-button", "Redo", REDO_SHORTCUT_LABEL);
    applyShortcutHint("help-button", "Help", "?");
    attachShortcutTooltipHandlers();
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
    collectionFormatSelect.addEventListener("change", (event) => {
      state.selectedCollectionFormat = event.target.value;
    });
    loadInput.addEventListener("change", ctx.loadDesignFromFile);
    window.addEventListener("keydown", ctx.handleKeydown);
    window.addEventListener("beforeunload", ctx.sendCancelBeacon);
    window.addEventListener("pagehide", ctx.sendCancelBeacon);
    window.addEventListener("resize", ctx.handleWindowResize);
    window.addEventListener("mousemove", ctx.handleGlobalMouseMove);
    window.addEventListener("mouseup", ctx.handleGlobalMouseUp);
    canvasShell.addEventListener("contextmenu", ctx.handleCanvasContextMenu);
    canvasShell.addEventListener("wheel", ctx.handleCanvasWheel, { passive: false });
    canvasShell.addEventListener("mousedown", ctx.handleCanvasMouseDown, true);
    minimapCanvas.addEventListener("mousedown", ctx.handleMinimapMouseDown);
  }

}

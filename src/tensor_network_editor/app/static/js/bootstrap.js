import { createEditorBootstrapFlow } from "./shell/editorBootstrapFlow.js";
import { createEditorShellBindings } from "./shell/editorShellBindings.js";
import { createShortcutTooltip } from "./shell/shortcutTooltip.js";

function resolveContextAction(ctx, actionName, fallback = () => {}) {
  const candidate = ctx[actionName];
  return typeof candidate === "function" ? candidate.bind(ctx) : fallback;
}

function createShellActions(ctx) {
  return {
    normalizeSpec: (spec) => ctx.normalizeSpec(spec),
    applyTemplateCatalogPayload: (payload) => ctx.applyTemplateCatalogPayload(payload),
    reconcileTensorOrder: () => ctx.reconcileTensorOrder(),
    populateEngineOptions: (engines) => ctx.populateEngineOptions(engines),
    enforceLinearPeriodicEngineSupport: resolveContextAction(
      ctx,
      "enforceLinearPeriodicEngineSupport"
    ),
    populateCollectionFormatOptions: (formats) => ctx.populateCollectionFormatOptions(formats),
    initGraph: () => ctx.initGraph(),
    clearHistory: () => ctx.clearHistory(),
    render: () => ctx.render(),
    refreshContractionAnalysis: resolveContextAction(ctx, "refreshContractionAnalysis"),
    setStatus: (message, level) => ctx.setStatus(message, level),
    handleNewDesign: () => ctx.handleNewDesign(),
    addTensorAtCenter: () => ctx.addTensorAtCenter(),
    addNoteAtCenter: () => ctx.addNoteAtCenter(),
    toggleConnectMode: () => ctx.toggleConnectMode(),
    deleteSelection: () => ctx.deleteSelection(),
    saveDesign: () => ctx.saveDesign(),
    generateCode: () => ctx.generateCode(),
    completeEditor: () => ctx.completeEditor(),
    cancelEditor: () => ctx.cancelEditor(),
    copyGeneratedCode: () => ctx.copyGeneratedCode(),
    performUndo: () => ctx.performUndo(),
    performRedo: () => ctx.performRedo(),
    downloadSelectedExport: () => ctx.downloadSelectedExport(),
    downloadExportAs: (format) => ctx.downloadExportAs(format),
    openToolbarMenu: (menuName) => ctx.openToolbarMenu(menuName),
    toggleToolbarMenu: (menuName) => ctx.toggleToolbarMenu(menuName),
    closeTransientToolbarUi: () => ctx.closeTransientToolbarUi(),
    toggleTemplateSettingsPopover: () => ctx.toggleTemplateSettingsPopover(),
    toggleReflowLayoutPopover: () => ctx.toggleReflowLayoutPopover(),
    updateToolbarState: () => ctx.updateToolbarState(),
    toggleLinearPeriodicMode: () => ctx.toggleLinearPeriodicMode(),
    setLinearPeriodicMode: (enabled) => ctx.setLinearPeriodicMode(enabled),
    switchLinearPeriodicCell: (direction) => ctx.switchLinearPeriodicCell(direction),
    toggleBenchmarkMode: () => ctx.toggleBenchmarkMode(),
    setBenchmarkMode: (enabled) => ctx.setBenchmarkMode(enabled),
    switchBenchmarkPosition: (direction) => ctx.switchBenchmarkPosition(direction),
    renameActiveBenchmarkScheme: (name) => ctx.renameActiveBenchmarkScheme(name),
    openBenchmarkCompareModal: () => ctx.openBenchmarkCompareModal(),
    closeBenchmarkCompareModal: () => ctx.closeBenchmarkCompareModal(),
    handleTemplateSelectionChange: (event) => ctx.handleTemplateSelectionChange(event),
    handleTemplateParameterInput: (event) => ctx.handleTemplateParameterInput(event),
    insertTemplate: () => ctx.insertTemplate(),
    openSubnetworkPicker: () => ctx.openSubnetworkPicker(),
    saveSelectionAsSessionTemplate: () => ctx.saveSelectionAsSessionTemplate(),
    openSessionTemplatePicker: () => ctx.openSessionTemplatePicker(),
    exportSelectedTemplateSpec: () => ctx.exportSelectedTemplateSpec(),
    toggleTemplateManager: (forceOpen) => ctx.toggleTemplateManager(forceOpen),
    saveTemplateManagerChanges: () => ctx.saveTemplateManagerChanges(),
    discardTemplateManagerChanges: () => ctx.discardTemplateManagerChanges(),
    renameSelectedTemplate: () => ctx.renameSelectedTemplate(),
    deleteSelectedTemplate: () => ctx.deleteSelectedTemplate(),
    applyReflowLayoutAction: (layoutAction) => ctx.applyReflowLayoutAction(layoutAction),
    applyReflowIndicesAction: (layoutAction) =>
      ctx.applyReflowIndicesAction(layoutAction),
    reflowLastImportedTensors: () => ctx.reflowLastImportedTensors(),
    createGroupFromSelection: () => ctx.createGroupFromSelection(),
    toggleHelpModal: (isOpen, section) => ctx.toggleHelpModal(isOpen, section),
    openHelpSection: (section) => ctx.openHelpSection(section),
    renderPlanner: resolveContextAction(ctx, "renderPlanner"),
    formatEngineLabel: (engine) => ctx.formatEngineLabel(engine),
    loadDesignFromFile: (event) => ctx.loadDesignFromFile(event),
    loadSubnetworkFromFile: (event) => ctx.loadSubnetworkFromFile(event),
    loadSessionTemplatesFromFile: (event) => ctx.loadSessionTemplatesFromFile(event),
    handleKeydown: (event) => ctx.handleKeydown(event),
    sendCancelBeacon: (event) => ctx.sendCancelBeacon(event),
    handleWindowResize: (event) => ctx.handleWindowResize(event),
    handleGlobalMouseMove: (event) => ctx.handleGlobalMouseMove(event),
    handleGlobalMouseUp: (event) => ctx.handleGlobalMouseUp(event),
    handleCanvasContextMenu: (event) => ctx.handleCanvasContextMenu(event),
    handleCanvasWheel: (event) => ctx.handleCanvasWheel(event),
    handleCanvasMouseDown: (event) => ctx.handleCanvasMouseDown(event),
    handleMinimapMouseDown: (event) => ctx.handleMinimapMouseDown(event),
  };
}

export function startEditor(ctx) {
  const state = ctx.state;
  const store = ctx.store;
  const { window, document } = ctx;
  const sessionService = ctx.services.session;
  const actions = createShellActions(ctx);
  const shortcutTooltip = createShortcutTooltip({
    documentRef: document,
    windowRef: window,
  });
  const bootstrapFlow = createEditorBootstrapFlow({
    state,
    store,
    sessionService,
    actions,
  });
  const shellBindings = createEditorShellBindings({
    state,
    store,
    dom: ctx.dom,
    documentRef: document,
    windowRef: window,
    actions,
    shortcutTooltip,
    redoShortcutLabel: ctx.constants.REDO_SHORTCUT_LABEL,
  });

  document.addEventListener("DOMContentLoaded", () => {
    shellBindings.attachToolbarHandlers();
    bootstrapFlow.bootstrap().catch((error) => {
      actions.setStatus(`Failed to load the editor: ${error.message}`, "error");
    });
  });
}

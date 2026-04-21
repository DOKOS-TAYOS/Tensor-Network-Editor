function resolveContextAction(ctx, actionName, fallback = () => {}) {
  const candidate = ctx[actionName];
  return typeof candidate === "function" ? candidate.bind(ctx) : fallback;
}

export function createShellActions(ctx) {
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
    markContractionAnalysisDirty: resolveContextAction(
      ctx,
      "markContractionAnalysisDirty",
      () => {
        if (ctx.state) {
          ctx.state.contractionAnalysisDirty = true;
        }
      }
    ),
    refreshContractionAnalysis: resolveContextAction(ctx, "refreshContractionAnalysis"),
    setStatus: (message, level) => ctx.setStatus(message, level),
    isLinearPeriodicMode: () =>
      typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode(),
    isGridPeriodicMode: () =>
      typeof ctx.isGridPeriodicMode === "function" && ctx.isGridPeriodicMode(),
    isTreePeriodicMode: () =>
      typeof ctx.isTreePeriodicMode === "function" && ctx.isTreePeriodicMode(),
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
    toggleGeneratedCodeModal: (isOpen) => ctx.toggleGeneratedCodeModal(isOpen),
    performUndo: () => ctx.performUndo(),
    performRedo: () => ctx.performRedo(),
    downloadSelectedExport: () => ctx.downloadSelectedExport(),
    downloadExportAs: (format) => ctx.downloadExportAs(format),
    openToolbarMenu: (menuName) => ctx.openToolbarMenu(menuName),
    toggleToolbarMenu: (menuName) => ctx.toggleToolbarMenu(menuName),
    closeTransientToolbarUi: () => ctx.closeTransientToolbarUi(),
    toggleTemplateSettingsPopover: () => ctx.toggleTemplateSettingsPopover(),
    toggleReflowLayoutPopover: () => ctx.toggleReflowLayoutPopover(),
    toggleSubnetworkLibrary: (forceOpen) => ctx.toggleSubnetworkLibrary(forceOpen),
    updateToolbarState: () => ctx.updateToolbarState(),
    toggleLinearPeriodicMode: () => ctx.toggleLinearPeriodicMode(),
    setLinearPeriodicMode: (enabled) => ctx.setLinearPeriodicMode(enabled),
    switchLinearPeriodicCell: (direction) => ctx.switchLinearPeriodicCell(direction),
    toggleGridPeriodicMode: () => ctx.toggleGridPeriodicMode(),
    setGridPeriodicMode: (enabled) => ctx.setGridPeriodicMode(enabled),
    switchGridPeriodicCell: (direction) => ctx.switchGridPeriodicCell(direction),
    toggleTreePeriodicMode: () => ctx.toggleTreePeriodicMode(),
    setTreePeriodicMode: (enabled) => ctx.setTreePeriodicMode(enabled),
    switchTreePeriodicCell: (direction) => ctx.switchTreePeriodicCell(direction),
    toggleBenchmarkMode: () => ctx.toggleBenchmarkMode(),
    setBenchmarkMode: (enabled) => ctx.setBenchmarkMode(enabled),
    switchBenchmarkPosition: (direction) => ctx.switchBenchmarkPosition(direction),
    renameActiveBenchmarkScheme: (name) => ctx.renameActiveBenchmarkScheme(name),
    openBenchmarkCompareModal: () => ctx.openBenchmarkCompareModal(),
    closeBenchmarkCompareModal: () => ctx.closeBenchmarkCompareModal(),
    exportBenchmarkCompareAsCsv: resolveContextAction(
      ctx,
      "exportBenchmarkCompareAsCsv"
    ),
    exportBenchmarkCompareAsText: resolveContextAction(
      ctx,
      "exportBenchmarkCompareAsText"
    ),
    copyBenchmarkCompareAsLatex: resolveContextAction(
      ctx,
      "copyBenchmarkCompareAsLatex"
    ),
    handleTemplateSelectionChange: (event) => ctx.handleTemplateSelectionChange(event),
    handleTemplateParameterInput: (event) => ctx.handleTemplateParameterInput(event),
    insertTemplate: () => ctx.insertTemplate(),
    openSubnetworkPicker: () => ctx.openSubnetworkPicker(),
    saveSelectionToSubnetworkLibrary: () => ctx.saveSelectionToSubnetworkLibrary(),
    openSubnetworkLibrary: () => ctx.openSubnetworkLibrary(),
    insertSelectedSubnetworkFromLibrary: () =>
      ctx.insertSelectedSubnetworkFromLibrary(),
    updateSubnetworkLibrarySearch: (query) =>
      ctx.updateSubnetworkLibrarySearch(query),
    updateSubnetworkLibraryTagFilter: (tag) =>
      ctx.updateSubnetworkLibraryTagFilter(tag),
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

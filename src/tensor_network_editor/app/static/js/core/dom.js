export function getDomRefs(document) {
  return {
    workspace: document.getElementById("workspace"),
    statusMessage: document.getElementById("status-message"),
    propertiesPanel: document.getElementById("properties-panel"),
    generatedCode: document.getElementById("generated-code"),
    generatedCodeView: document.getElementById("generated-code-view"),
    generatedCodeModalView: document.getElementById("generated-code-modal-view"),
    engineSelectField: document.getElementById("engine-select-field"),
    engineSelect: document.getElementById("engine-select"),
    collectionFormatSelectField: document.getElementById(
      "collection-format-select-field"
    ),
    collectionFormatSelect: document.getElementById("collection-format-select"),
    fileMenuButton: document.getElementById("file-menu-button"),
    fileMenuPanel: document.getElementById("file-menu-panel"),
    themeMenuButton: document.getElementById("theme-menu-button"),
    themeMenuPanel: document.getElementById("theme-menu-panel"),
    modesMenuButton: document.getElementById("modes-menu-button"),
    modesMenuPanel: document.getElementById("modes-menu-panel"),
    templatesMenuButton: document.getElementById("templates-menu-button"),
    templatesMenuPanel: document.getElementById("templates-menu-panel"),
    helpMenuButton: document.getElementById("help-menu-button"),
    helpMenuPanel: document.getElementById("help-menu-panel"),
    newDesignButton: document.getElementById("new-design-button"),
    saveButton: document.getElementById("save-button"),
    loadDesignMenuItem: document.getElementById("load-design-menu-item"),
    exportFormatSelect: document.getElementById("export-format-select"),
    exportMenuItem: document.getElementById("export-menu-item"),
    exportSubmenuShell: document.getElementById("export-submenu-shell"),
    exportSubmenuPanel: document.getElementById("export-submenu-panel"),
    addNoteButton: document.getElementById("add-note-button"),
    connectButton: document.getElementById("connect-button"),
    loadInput: document.getElementById("load-input"),
    subnetworkLoadInput: document.getElementById("subnetwork-load-input"),
    templateLoadInput: document.getElementById("template-load-input"),
    undoButton: document.getElementById("undo-button"),
    redoButton: document.getElementById("redo-button"),
    exportPythonMenuItem: document.getElementById("export-python-menu-item"),
    exportPngMenuItem: document.getElementById("export-png-menu-item"),
    exportSvgMenuItem: document.getElementById("export-svg-menu-item"),
    exportPdfMenuItem: document.getElementById("export-pdf-menu-item"),
    exportTikzMenuItem: document.getElementById("export-tikz-menu-item"),
    exportDotMenuItem: document.getElementById("export-dot-menu-item"),
    exportShowTensorNamesMenuItem: document.getElementById(
      "export-show-tensor-names-menu-item"
    ),
    exportShowIndexNamesMenuItem: document.getElementById(
      "export-show-index-names-menu-item"
    ),
    exportShowBondNamesMenuItem: document.getElementById(
      "export-show-bond-names-menu-item"
    ),
    closeWithInfoMenuItem: document.getElementById("close-with-info-menu-item"),
    closeWithoutInfoMenuItem: document.getElementById(
      "close-without-info-menu-item"
    ),
    themeDarkMenuItem: document.getElementById("theme-dark-menu-item"),
    themeLightMenuItem: document.getElementById("theme-light-menu-item"),
    themeContrastMenuItem: document.getElementById("theme-contrast-menu-item"),
    themeColorblindMenuItem: document.getElementById(
      "theme-colorblind-menu-item"
    ),
    themeShinyMenuItem: document.getElementById("theme-shiny-menu-item"),
    singleModeMenuItem: document.getElementById("single-mode-menu-item"),
    linearPeriodicModeMenuItem: document.getElementById(
      "linear-periodic-mode-menu-item"
    ),
    gridPeriodicModeMenuItem: document.getElementById("grid-periodic-mode-menu-item"),
    treeModeMenuItem: document.getElementById("tree-mode-menu-item"),
    benchmarkModeMenuItem: document.getElementById("benchmark-mode-menu-item"),
    toolbarModeControls: document.querySelector(".toolbar-mode-controls"),
    linearPeriodicPreviousCellButton: document.getElementById(
      "linear-periodic-previous-cell-button"
    ),
    linearPeriodicCellLabel: document.getElementById("linear-periodic-cell-label"),
    gridPeriodicUpCellButton: document.getElementById("grid-periodic-up-cell-button"),
    gridPeriodicDownCellButton: document.getElementById(
      "grid-periodic-down-cell-button"
    ),
    linearPeriodicNextCellButton: document.getElementById(
      "linear-periodic-next-cell-button"
    ),
    benchmarkSchemeNameInput: document.getElementById("benchmark-scheme-name-input"),
    benchmarkCompareButton: document.getElementById("benchmark-compare-button"),
    copyCodeButton: document.getElementById("copy-code-button"),
    expandGeneratedCodeButton: document.getElementById("expand-generated-code-button"),
    generatedCodeModal: document.getElementById("generated-code-modal"),
    generatedCodeModalBackdrop: document.getElementById("generated-code-modal-backdrop"),
    generatedCodeModalCloseButton: document.getElementById(
      "generated-code-modal-close-button"
    ),
    templateSelectField: document.getElementById("template-select-field"),
    templateSelect: document.getElementById("template-select"),
    templateSettingsButton: document.getElementById("template-settings-button"),
    templateSettingsPopover: document.getElementById("template-settings-popover"),
    reflowLayoutPopover: document.getElementById("reflow-layout-popover"),
    templateParameterPanel: document.getElementById("template-parameter-panel"),
    insertTemplateButton: document.getElementById("insert-template-button"),
    saveSessionTemplateMenuItem: document.getElementById(
      "save-session-template-menu-item"
    ),
    saveSubnetworkLibraryMenuItem: document.getElementById(
      "save-subnetwork-library-menu-item"
    ),
    loadSessionTemplateMenuItem: document.getElementById(
      "load-session-template-menu-item"
    ),
    exportSessionTemplateMenuItem: document.getElementById(
      "export-session-template-menu-item"
    ),
    editSessionTemplateMenuItem: document.getElementById(
      "edit-session-template-menu-item"
    ),
    openSubnetworkLibraryMenuItem: document.getElementById(
      "open-subnetwork-library-menu-item"
    ),
    templateCatalogWarning: document.getElementById("template-catalog-warning"),
    subnetworkCatalogWarning: document.getElementById("subnetwork-catalog-warning"),
    reflowImportedButton: document.getElementById("reflow-imported-button"),
    reflowAlignLeftButton: document.getElementById("reflow-align-left-button"),
    reflowAlignRightButton: document.getElementById("reflow-align-right-button"),
    reflowAlignTopButton: document.getElementById("reflow-align-top-button"),
    reflowAlignMiddleButton: document.getElementById("reflow-align-middle-button"),
    reflowAlignBottomButton: document.getElementById("reflow-align-bottom-button"),
    reflowIndicesLeftButton: document.getElementById("reflow-indices-left-button"),
    reflowIndicesRightButton: document.getElementById("reflow-indices-right-button"),
    reflowIndicesTopButton: document.getElementById("reflow-indices-top-button"),
    reflowIndicesResetButton: document.getElementById("reflow-indices-reset-button"),
    reflowIndicesBottomButton: document.getElementById("reflow-indices-bottom-button"),
    reflowArrangeChainButton: document.getElementById("reflow-arrange-chain-button"),
    reflowArrangeTreeButton: document.getElementById("reflow-arrange-tree-button"),
    reflowArrangeGridButton: document.getElementById("reflow-arrange-grid-button"),
    reflowAutoLayoutButton: document.getElementById("reflow-auto-layout-button"),
    reflowDistributeHorizontalButton: document.getElementById(
      "reflow-distribute-horizontal-button"
    ),
    reflowDistributeVerticalButton: document.getElementById(
      "reflow-distribute-vertical-button"
    ),
    reflowSnapGridButton: document.getElementById("reflow-snap-grid-button"),
    createGroupButton: document.getElementById("create-group-button"),
    helpInfoMenuItem: document.getElementById("help-info-menu-item"),
    helpShortcutsMenuItem: document.getElementById("help-shortcuts-menu-item"),
    helpAboutMenuItem: document.getElementById("help-about-menu-item"),
    helpModal: document.getElementById("help-modal"),
    helpBackdrop: document.getElementById("help-backdrop"),
    helpCloseButton: document.getElementById("help-close-button"),
    helpSharedHeader: document.getElementById("help-shared-header"),
    helpTitle: document.getElementById("help-title"),
    helpNote: document.getElementById("help-note"),
    helpInfoSection: document.getElementById("help-info-section"),
    helpShortcutsSection: document.getElementById("help-shortcuts-section"),
    helpAboutSection: document.getElementById("help-about-section"),
    aboutRepositoryLink: document.getElementById("about-repository-link"),
    aboutVersion: document.getElementById("about-version"),
    aboutSchemaVersion: document.getElementById("about-schema-version"),
    aboutLicense: document.getElementById("about-license"),
    aboutAuthor: document.getElementById("about-author"),
    templateManagerModal: document.getElementById("template-manager-modal"),
    templateManagerBackdrop: document.getElementById("template-manager-backdrop"),
    templateManagerCloseButton: document.getElementById(
      "template-manager-close-button"
    ),
    templateManagerSaveButton: document.getElementById("template-manager-save-button"),
    templateManagerDiscardButton: document.getElementById(
      "template-manager-discard-button"
    ),
    templateManagerError: document.getElementById("template-manager-error"),
    templateManagerList: document.getElementById("template-manager-list"),
    subnetworkLibraryModal: document.getElementById("subnetwork-library-modal"),
    subnetworkLibraryBackdrop: document.getElementById(
      "subnetwork-library-backdrop"
    ),
    subnetworkLibraryCloseButton: document.getElementById(
      "subnetwork-library-close-button"
    ),
    subnetworkLibrarySearchInput: document.getElementById(
      "subnetwork-library-search-input"
    ),
    subnetworkLibraryTagFilter: document.getElementById(
      "subnetwork-library-tag-filter"
    ),
    subnetworkLibrarySelectAllInput: document.getElementById(
      "subnetwork-library-select-all-input"
    ),
    subnetworkLibrarySelectionSummary: document.getElementById(
      "subnetwork-library-selection-summary"
    ),
    subnetworkLibraryAddSelectedButton: document.getElementById(
      "subnetwork-library-add-selected-button"
    ),
    subnetworkLibraryWarning: document.getElementById(
      "subnetwork-library-warning"
    ),
    subnetworkLibraryList: document.getElementById("subnetwork-library-list"),
    benchmarkCompareModal: document.getElementById("benchmark-compare-modal"),
    benchmarkCompareBackdrop: document.getElementById(
      "benchmark-compare-backdrop"
    ),
    benchmarkCompareCloseButton: document.getElementById(
      "benchmark-compare-close-button"
    ),
    benchmarkCompareExportCsvButton: document.getElementById(
      "benchmark-compare-export-csv-button"
    ),
    benchmarkCompareExportTextButton: document.getElementById(
      "benchmark-compare-export-text-button"
    ),
    benchmarkCompareCopyLatexButton: document.getElementById(
      "benchmark-compare-copy-latex-button"
    ),
    benchmarkCompareTableBody: document.getElementById(
      "benchmark-compare-table-body"
    ),
    canvasShell: document.getElementById("canvas-shell"),
    groupLayer: document.getElementById("group-layer"),
    resizeLayer: document.getElementById("resize-layer"),
    notesLayer: document.getElementById("notes-layer"),
    selectionBox: document.getElementById("canvas-selection-box"),
    canvasTools: document.getElementById("canvas-tools"),
    canvasContextMenuRoot: document.getElementById("canvas-context-menu-root"),
    minimapShell: document.getElementById("minimap-shell"),
    minimapCanvas: document.getElementById("minimap"),
    sidebar: document.getElementById("sidebar"),
    sidebarPanel: document.getElementById("sidebar-panel"),
    sidebarResizeHandle: document.getElementById("sidebar-resize-handle"),
    sidebarToggleButton: document.getElementById("sidebar-toggle-button"),
    sidebarTabs: document.getElementById("sidebar-tabs"),
    sidebarTabSelection: document.getElementById("sidebar-tab-selection"),
    sidebarTabPlanner: document.getElementById("sidebar-tab-planner"),
    sidebarTabCode: document.getElementById("sidebar-tab-code"),
    sidebarPaneSelection: document.getElementById("sidebar-pane-selection"),
    sidebarPanePlanner: document.getElementById("sidebar-pane-planner"),
    sidebarPaneCode: document.getElementById("sidebar-pane-code"),
    plannerPanel: document.getElementById("planner-panel"),
    generateButton: document.getElementById("generate-button"),
    codeGenerationWarning: document.getElementById("code-generation-warning"),
  };
}

export function createEditorShellBindings({
  state,
  store,
  dom,
  documentRef,
  windowRef,
  actions,
  shortcutTooltip,
  redoShortcutLabel,
}) {
  const {
    addNoteButton,
    fileMenuButton,
    fileMenuPanel,
    themeMenuButton,
    themeMenuPanel,
    modesMenuButton,
    modesMenuPanel,
    templatesMenuButton,
    templatesMenuPanel,
    helpMenuButton,
    helpMenuPanel,
    newDesignButton,
    saveButton,
    loadDesignMenuItem,
    connectButton,
    loadInput,
    subnetworkLoadInput,
    templateLoadInput,
    undoButton,
    redoButton,
    exportMenuItem,
    exportSubmenuShell,
    exportSubmenuPanel,
    exportPythonMenuItem,
    exportPngMenuItem,
    exportSvgMenuItem,
    exportPdfMenuItem,
    exportTikzMenuItem,
    exportDotMenuItem,
    exportMermaidMenuItem,
    exportShowTensorNamesMenuItem,
    exportShowIndexNamesMenuItem,
    exportShowBondNamesMenuItem,
    closeWithInfoMenuItem,
    closeWithoutInfoMenuItem,
    themeDarkMenuItem,
    themeLightMenuItem,
    themeContrastMenuItem,
    themeColorblindMenuItem,
    themeShinyMenuItem,
    exportFormatSelect,
    singleModeMenuItem,
    linearPeriodicModeMenuItem,
    gridPeriodicModeMenuItem,
    treeModeMenuItem,
    benchmarkModeMenuItem,
    linearPeriodicPreviousCellButton,
    linearPeriodicNextCellButton,
    gridPeriodicUpCellButton,
    gridPeriodicDownCellButton,
    benchmarkSchemeNameInput,
    benchmarkCompareButton,
    copyCodeButton,
    expandGeneratedCodeButton,
    generatedCodeModalBackdrop,
    generatedCodeModalCloseButton,
    templateSelectField,
    templateSelect,
    engineSelectField,
    collectionFormatSelectField,
    templateSettingsButton,
    templateSettingsPopover,
    reflowLayoutPopover,
    templateParameterPanel,
    insertTemplateButton,
    saveSessionTemplateMenuItem,
    saveSubnetworkLibraryMenuItem,
    loadSessionTemplateMenuItem,
    exportSessionTemplateMenuItem,
    editSessionTemplateMenuItem,
    openSubnetworkLibraryMenuItem,
    reflowImportedButton,
    reflowAlignLeftButton,
    reflowAlignRightButton,
    reflowAlignTopButton,
    reflowAlignMiddleButton,
    reflowAlignBottomButton,
    reflowIndicesLeftButton,
    reflowIndicesRightButton,
    reflowIndicesTopButton,
    reflowIndicesResetButton,
    reflowIndicesBottomButton,
    reflowArrangeChainButton,
    reflowArrangeTreeButton,
    reflowArrangeGridButton,
    reflowAutoLayoutButton,
    reflowDistributeHorizontalButton,
    reflowDistributeVerticalButton,
    reflowSnapGridButton,
    createGroupButton,
    helpInfoMenuItem,
    helpShortcutsMenuItem,
    helpAboutMenuItem,
    helpBackdrop,
    helpCloseButton,
    templateManagerBackdrop,
    templateManagerCloseButton,
    templateManagerSaveButton,
    templateManagerDiscardButton,
    subnetworkLibraryBackdrop,
    subnetworkLibraryCloseButton,
    subnetworkLibrarySearchInput,
    subnetworkLibraryTagFilter,
    subnetworkLibrarySelectAllInput,
    subnetworkLibraryAddSelectedButton,
    benchmarkCompareBackdrop,
    benchmarkCompareCloseButton,
    benchmarkCompareExportCsvButton,
    benchmarkCompareExportTextButton,
    benchmarkCompareCopyLatexButton,
    canvasShell,
    minimapCanvas,
    engineSelect,
    collectionFormatSelect,
  } = dom;

  const toolbarMenus = [
    { name: "file", button: fileMenuButton, panel: fileMenuPanel },
    { name: "theme", button: themeMenuButton, panel: themeMenuPanel },
    { name: "modes", button: modesMenuButton, panel: modesMenuPanel },
    { name: "templates", button: templatesMenuButton, panel: templatesMenuPanel },
    { name: "help", button: helpMenuButton, panel: helpMenuPanel },
  ];

  function bindListener(target, eventName, handler, options) {
    if (!target || typeof target.addEventListener !== "function") {
      return;
    }
    target.addEventListener(eventName, handler, options);
  }

  function toggleAcademicExportLabel(labelKey) {
    if (!state.academicExportLabels) {
      state.academicExportLabels = { tensor: true, index: true, bond: true };
    }
    state.academicExportLabels[labelKey] = !state.academicExportLabels[labelKey];
    actions.updateToolbarState();
  }

  function targetWithinElement(target, element) {
    if (!target || !element) {
      return false;
    }
    if (target === element) {
      return true;
    }
    if (typeof element.contains === "function") {
      return element.contains(target);
    }
    if (
      typeof target.closest === "function"
      && typeof element.id === "string"
      && element.id
    ) {
      return Boolean(target.closest(`#${element.id}`));
    }
    return false;
  }

  function isWithinTransientToolbarUi(target) {
    return [
      ...toolbarMenus.flatMap((menu) => [menu.button, menu.panel]),
      exportSubmenuShell,
      exportSubmenuPanel,
      templateSettingsButton,
      templateSettingsPopover,
      reflowImportedButton,
      reflowLayoutPopover,
    ].some((element) => targetWithinElement(target, element));
  }

  function bindMenubarMenu(menuName, button, panel) {
    bindListener(button, "click", () => {
      actions.toggleToolbarMenu(menuName);
    });
    bindListener(button, "mouseenter", () => {
      if (state.openToolbarMenu && state.openToolbarMenu !== menuName) {
        actions.openToolbarMenu(menuName);
      }
    });
    bindListener(panel, "mouseenter", () => {
      if (state.openToolbarMenu && state.openToolbarMenu !== menuName) {
        actions.openToolbarMenu(menuName);
      }
    });
  }

  function readSelectChevronExpanded(fieldElement) {
    if (!fieldElement) {
      return false;
    }
    if (typeof fieldElement.getAttribute === "function") {
      return fieldElement.getAttribute("data-expanded") === "true";
    }
    return fieldElement.attributes?.["data-expanded"] === "true";
  }

  function setSelectChevronExpanded(fieldElement, isExpanded) {
    if (!fieldElement || typeof fieldElement.setAttribute !== "function") {
      return;
    }
    fieldElement.setAttribute("data-expanded", String(Boolean(isExpanded)));
  }

  function bindSelectChevronDisclosure(fieldElement, selectElement) {
    if (!fieldElement || !selectElement) {
      return;
    }
    setSelectChevronExpanded(fieldElement, false);
    bindListener(selectElement, "mousedown", () => {
      setSelectChevronExpanded(
        fieldElement,
        !readSelectChevronExpanded(fieldElement)
      );
    });
    bindListener(selectElement, "keydown", (event) => {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
        setSelectChevronExpanded(fieldElement, true);
      }
      if (["Escape", "Tab"].includes(event.key)) {
        setSelectChevronExpanded(fieldElement, false);
      }
    });
    bindListener(selectElement, "change", () => {
      setSelectChevronExpanded(fieldElement, false);
    });
    bindListener(selectElement, "blur", () => {
      setSelectChevronExpanded(fieldElement, false);
    });
  }

  function attachToolbarHandlers() {
    const bindReflowAction = (button, layoutAction) => {
      bindListener(button, "click", () => {
        actions.applyReflowLayoutAction(layoutAction);
        actions.closeTransientToolbarUi();
      });
    };
    const bindReflowIndicesAction = (button, layoutAction) => {
      bindListener(button, "click", () => {
        actions.applyReflowIndicesAction(layoutAction);
        actions.closeTransientToolbarUi();
      });
    };

    shortcutTooltip.applyShortcutHint(
      "add-tensor-button",
      "Add tensor",
      "N",
      "Place a new tensor at the center of the canvas."
    );
    shortcutTooltip.applyShortcutHint(
      "insert-template-button",
      "Insert template",
      "T",
      "Insert the selected template on the canvas."
    );
    shortcutTooltip.applyShortcutHint(
      "create-group-button",
      "Group",
      "G",
      "Wrap the selected tensors in a movable group."
    );
    shortcutTooltip.applyShortcutHint(
      "add-note-button",
      "Add note",
      "P",
      "Place a new note at the center of the canvas."
    );
    shortcutTooltip.applyShortcutHint(
      "connect-button",
      "Connect",
      "C",
      "Link two open indices that share the same dimension."
    );
    shortcutTooltip.applyShortcutHint(
      "delete-button",
      "Delete",
      "Delete",
      "Remove the current selection from the canvas."
    );
    shortcutTooltip.applyShortcutHint(
      "save-button",
      "Save tensor network",
      "Ctrl/Cmd+S",
      "Download the current design as JSON."
    );
    shortcutTooltip.applyShortcutHint(
      "load-design-menu-item",
      "Load tensor network",
      "Ctrl/Cmd+L",
      "Open a saved design from disk."
    );
    shortcutTooltip.applyShortcutHint(
      "generate-button",
      "Generate code",
      "Shift+G",
      "Build the current network with the selected engine."
    );
    shortcutTooltip.applyShortcutHint(
      "linear-periodic-mode-menu-item",
      "For unidimensional",
      "F",
      "Switch between single mode and the three-cell chain workflow."
    );
    shortcutTooltip.applyShortcutHint(
      "undo-button",
      "Undo",
      "Ctrl/Cmd+Z",
      "Revert the latest design change."
    );
    shortcutTooltip.applyShortcutHint(
      "redo-button",
      "Redo",
      redoShortcutLabel,
      "Restore the latest undone design change."
    );
    shortcutTooltip.applyShortcutHint(
      "help-info-menu-item",
      "Info",
      "?",
      "Open the editor guide."
    );
    if (typeof shortcutTooltip.applyTitleHint === "function") {
      [
        "new-design-button",
        "export-menu-item",
        "export-python-menu-item",
        "export-png-menu-item",
        "export-svg-menu-item",
        "export-tikz-menu-item",
        "export-dot-menu-item",
        "export-mermaid-menu-item",
        "single-mode-menu-item",
        "grid-periodic-mode-menu-item",
        "tree-mode-menu-item",
        "benchmark-mode-menu-item",
        "benchmark-compare-button",
        "save-session-template-menu-item",
        "save-subnetwork-library-menu-item",
        "load-session-template-menu-item",
        "export-session-template-menu-item",
        "edit-session-template-menu-item",
        "open-subnetwork-library-menu-item",
        "help-shortcuts-menu-item",
        "help-about-menu-item",
        "close-with-info-menu-item",
        "close-without-info-menu-item",
        "theme-dark-menu-item",
        "theme-light-menu-item",
        "theme-contrast-menu-item",
        "theme-colorblind-menu-item",
        "theme-shiny-menu-item",
        "linear-periodic-previous-cell-button",
        "grid-periodic-up-cell-button",
        "grid-periodic-down-cell-button",
        "linear-periodic-next-cell-button",
        "template-settings-button",
        "reflow-imported-button",
        "reflow-align-left-button",
        "reflow-align-right-button",
        "reflow-align-top-button",
        "reflow-align-middle-button",
        "reflow-align-bottom-button",
        "reflow-arrange-chain-button",
        "reflow-arrange-tree-button",
        "reflow-arrange-grid-button",
        "reflow-auto-layout-button",
        "reflow-snap-grid-button",
        "reflow-indices-left-button",
        "reflow-indices-right-button",
        "reflow-indices-top-button",
        "reflow-indices-reset-button",
        "reflow-indices-bottom-button",
        "copy-code-button",
        "expand-generated-code-button",
        "template-manager-save-button",
        "template-manager-discard-button",
      ].forEach((controlId) => {
        shortcutTooltip.applyTitleHint(controlId);
      });
      shortcutTooltip.applyTitleHint("template-select", { label: "Template" });
    }
    shortcutTooltip.applyShortcutHint(
      "collection-format-select-field",
      "Output type",
      "",
      "Choose how generated code returns the tensors: list keeps an ordered sequence, matrix arranges row and column structures when the template supports them, and dict returns named entries."
    );
    shortcutTooltip.applyShortcutHint(
      "reflow-imported-button",
      "Reflow",
      "R",
      "Open layout and index reflow tools."
    );
    shortcutTooltip.applyShortcutHint(
      "linear-periodic-previous-cell-button",
      "Previous cell",
      "Alt+ArrowLeft",
      "Move to the previous item in the current mode."
    );
    shortcutTooltip.applyShortcutHint(
      "linear-periodic-next-cell-button",
      "Next cell",
      "Alt+ArrowRight",
      "Move to the next item in the current mode."
    );
    shortcutTooltip.applyShortcutHint(
      "close-with-info-menu-item",
      "Close with info",
      "Ctrl/Cmd+Enter",
      "Close the editor and return the current design to Python."
    );
    shortcutTooltip.applyShortcutHint(
      "close-without-info-menu-item",
      "Close without info",
      "",
      "Close the editor without returning the current design to Python."
    );
    shortcutTooltip.applyShortcutHint(
      "sidebar-tab-selection",
      "Selection",
      "",
      "Inspect and edit the current selection, including metadata, layout, and extraction tools."
    );
    shortcutTooltip.applyShortcutHint(
      "sidebar-tab-planner",
      "Planner",
      "",
      "Review manual and automatic contraction paths, comparisons, and contraction steps."
    );
    shortcutTooltip.applyShortcutHint(
      "sidebar-tab-code",
      "Code",
      "",
      "Generate and inspect code for the current network with the selected engine."
    );
    shortcutTooltip.attachShortcutTooltipHandlers();

    toolbarMenus.forEach((menu) => {
      bindMenubarMenu(menu.name, menu.button, menu.panel);
    });

    bindListener(newDesignButton, "click", () => {
      actions.closeTransientToolbarUi();
      actions.handleNewDesign();
    });
    bindListener(documentRef.getElementById("add-tensor-button"), "click", actions.addTensorAtCenter);
    bindListener(addNoteButton, "click", actions.addNoteAtCenter);
    bindListener(connectButton, "click", actions.toggleConnectMode);
    bindListener(documentRef.getElementById("delete-button"), "click", actions.deleteSelection);
    bindListener(saveButton, "click", () => {
      actions.closeTransientToolbarUi();
      actions.saveDesign();
    });
    bindListener(loadDesignMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      if (loadInput && typeof loadInput.click === "function") {
        loadInput.click();
      }
    });
    bindListener(documentRef.getElementById("generate-button"), "click", actions.generateCode);
    bindListener(closeWithInfoMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.completeEditor();
    });
    bindListener(closeWithoutInfoMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.cancelEditor();
    });
    const bindThemeMenuItem = (menuItem, themeName) => {
      bindListener(menuItem, "click", () => {
        actions.closeTransientToolbarUi();
        actions.setEditorTheme(themeName);
      });
    };
    bindThemeMenuItem(themeDarkMenuItem, "dark");
    bindThemeMenuItem(themeLightMenuItem, "light");
    bindThemeMenuItem(themeContrastMenuItem, "contrast");
    bindThemeMenuItem(themeColorblindMenuItem, "colorblind");
    bindThemeMenuItem(themeShinyMenuItem, "shiny");
    bindListener(copyCodeButton, "click", actions.copyGeneratedCode);
    bindListener(expandGeneratedCodeButton, "click", () =>
      actions.toggleGeneratedCodeModal(true)
    );
    bindListener(generatedCodeModalBackdrop, "click", () =>
      actions.toggleGeneratedCodeModal(false)
    );
    bindListener(generatedCodeModalCloseButton, "click", () =>
      actions.toggleGeneratedCodeModal(false)
    );
    bindListener(undoButton, "click", actions.performUndo);
    bindListener(redoButton, "click", actions.performRedo);
    bindListener(exportPythonMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.downloadExportAs("py");
    });
    bindListener(exportPngMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.downloadExportAs("png");
    });
    bindListener(exportSvgMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.downloadExportAs("svg");
    });
    bindListener(exportPdfMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.downloadExportAs("pdf");
    });
    bindListener(exportTikzMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.downloadExportAs("tikz");
    });
    bindListener(exportDotMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.downloadExportAs("dot");
    });
    bindListener(exportMermaidMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.downloadExportAs("mermaid");
    });
    const openExportSubmenu = () => {
      if (typeof actions.openToolbarSubmenu === "function") {
        actions.openToolbarSubmenu("export");
      }
    };
    bindListener(exportSubmenuShell, "mouseenter", openExportSubmenu);
    bindListener(exportSubmenuShell, "focusin", openExportSubmenu);
    bindListener(exportMenuItem, "mouseenter", openExportSubmenu);
    bindListener(exportMenuItem, "focus", openExportSubmenu);
    bindListener(exportMenuItem, "click", openExportSubmenu);
    bindListener(exportSubmenuPanel, "mouseenter", openExportSubmenu);
    bindListener(exportSubmenuShell, "mouseleave", () => {
      if (typeof actions.closeToolbarSubmenu === "function") {
        actions.closeToolbarSubmenu("export");
      }
    });
    bindListener(exportShowTensorNamesMenuItem, "click", () => {
      toggleAcademicExportLabel("tensor");
    });
    bindListener(exportShowIndexNamesMenuItem, "click", () => {
      toggleAcademicExportLabel("index");
    });
    bindListener(exportShowBondNamesMenuItem, "click", () => {
      toggleAcademicExportLabel("bond");
    });
    bindListener(exportFormatSelect, "change", () => {
      actions.updateToolbarState();
    });
    bindListener(singleModeMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.setBenchmarkMode(false);
      actions.setLinearPeriodicMode(false);
      actions.setGridPeriodicMode(false);
      actions.setTreePeriodicMode(false);
      actions.updateToolbarState();
    });
    bindListener(linearPeriodicModeMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.setGridPeriodicMode(false);
      actions.setBenchmarkMode(false);
      actions.setLinearPeriodicMode(true);
      actions.updateToolbarState();
    });
    bindListener(gridPeriodicModeMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.setBenchmarkMode(false);
      actions.setGridPeriodicMode(true);
      actions.updateToolbarState();
    });
    bindListener(treeModeMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.setBenchmarkMode(false);
      actions.setTreePeriodicMode(true);
      actions.updateToolbarState();
    });
    bindListener(benchmarkModeMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.setLinearPeriodicMode(false);
      actions.setGridPeriodicMode(false);
      actions.setTreePeriodicMode(false);
      actions.setBenchmarkMode(true);
      actions.updateToolbarState();
    });
    bindListener(linearPeriodicPreviousCellButton, "click", () => {
      if (state.benchmarkSession && state.benchmarkSession.enabled) {
        actions.switchBenchmarkPosition(-1);
        return;
      }
      if (typeof actions.isGridPeriodicMode === "function" && actions.isGridPeriodicMode()) {
        actions.switchGridPeriodicCell("left");
        return;
      }
      actions.switchLinearPeriodicCell(-1);
    });
    bindListener(linearPeriodicNextCellButton, "click", () => {
      if (state.benchmarkSession && state.benchmarkSession.enabled) {
        actions.switchBenchmarkPosition(1);
        return;
      }
      if (typeof actions.isGridPeriodicMode === "function" && actions.isGridPeriodicMode()) {
        actions.switchGridPeriodicCell("right");
        return;
      }
      actions.switchLinearPeriodicCell(1);
    });
    bindListener(gridPeriodicUpCellButton, "click", () => {
      if (typeof actions.isTreePeriodicMode === "function" && actions.isTreePeriodicMode()) {
        actions.switchTreePeriodicCell("up");
        return;
      }
      actions.switchGridPeriodicCell("up");
    });
    bindListener(gridPeriodicDownCellButton, "click", () => {
      if (typeof actions.isTreePeriodicMode === "function" && actions.isTreePeriodicMode()) {
        actions.switchTreePeriodicCell("down");
        return;
      }
      actions.switchGridPeriodicCell("down");
    });
    bindListener(benchmarkSchemeNameInput, "input", (event) => {
      actions.renameActiveBenchmarkScheme(event.target.value);
    });
    bindListener(benchmarkCompareButton, "click", () => {
      actions.openBenchmarkCompareModal();
    });
    bindSelectChevronDisclosure(templateSelectField, templateSelect);
    bindSelectChevronDisclosure(engineSelectField, engineSelect);
    bindSelectChevronDisclosure(
      collectionFormatSelectField,
      collectionFormatSelect
    );
    bindListener(templateSelect, "change", (event) => {
      setSelectChevronExpanded(templateSelectField, false);
      actions.handleTemplateSelectionChange(event);
    });
    bindListener(
      templateParameterPanel,
      "mousedown",
      actions.handleTemplateParameterPanelDisclosure
    );
    bindListener(
      templateParameterPanel,
      "keydown",
      actions.handleTemplateParameterPanelKeydown
    );
    bindListener(
      templateParameterPanel,
      "focusout",
      actions.handleTemplateParameterPanelFocusOut
    );
    bindListener(
      templateParameterPanel,
      "change",
      actions.handleTemplateParameterInput
    );
    bindListener(templateSettingsButton, "click", () => {
      actions.toggleTemplateSettingsPopover();
    });
    bindListener(insertTemplateButton, "click", actions.insertTemplate);
    bindListener(saveSessionTemplateMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.saveSelectionAsSessionTemplate();
    });
    bindListener(saveSubnetworkLibraryMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.saveSelectionToSubnetworkLibrary();
    });
    bindListener(loadSessionTemplateMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.openSessionTemplatePicker();
    });
    bindListener(exportSessionTemplateMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.exportSelectedSubnetwork();
    });
    bindListener(editSessionTemplateMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.toggleTemplateManager(true);
    });
    bindListener(openSubnetworkLibraryMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.openSubnetworkLibrary();
    });
    bindListener(reflowImportedButton, "click", () => {
      actions.toggleReflowLayoutPopover();
    });
    bindReflowAction(reflowAlignLeftButton, "left");
    bindReflowAction(reflowAlignRightButton, "right");
    bindReflowAction(reflowAlignTopButton, "top");
    bindReflowAction(reflowAlignMiddleButton, "middle");
    bindReflowAction(reflowAlignBottomButton, "bottom");
    bindReflowIndicesAction(reflowIndicesLeftButton, "left");
    bindReflowIndicesAction(reflowIndicesRightButton, "right");
    bindReflowIndicesAction(reflowIndicesTopButton, "top");
    bindReflowIndicesAction(reflowIndicesResetButton, "reset");
    bindReflowIndicesAction(reflowIndicesBottomButton, "bottom");
    bindReflowAction(reflowArrangeChainButton, "chain");
    bindReflowAction(reflowArrangeTreeButton, "tree");
    bindReflowAction(reflowArrangeGridButton, "grid");
    bindReflowAction(reflowAutoLayoutButton, "auto");
    bindReflowAction(reflowDistributeHorizontalButton, "horizontal");
    bindReflowAction(reflowDistributeVerticalButton, "vertical");
    bindReflowAction(reflowSnapGridButton, "snap");
    bindListener(createGroupButton, "click", actions.createGroupFromSelection);
    bindListener(helpInfoMenuItem, "click", () => {
      actions.openHelpSection("info");
    });
    bindListener(helpShortcutsMenuItem, "click", () => {
      actions.openHelpSection("shortcuts");
    });
    bindListener(helpAboutMenuItem, "click", () => {
      actions.openHelpSection("about");
    });
    bindListener(helpBackdrop, "click", () => actions.toggleHelpModal(false));
    bindListener(helpCloseButton, "click", () => actions.toggleHelpModal(false));
    bindListener(templateManagerBackdrop, "click", () =>
      actions.toggleTemplateManager(false)
    );
    bindListener(templateManagerCloseButton, "click", () =>
      actions.toggleTemplateManager(false)
    );
    bindListener(templateManagerSaveButton, "click", () => {
      if (typeof actions.saveTemplateManagerChanges === "function") {
        actions.saveTemplateManagerChanges();
      }
    });
    bindListener(templateManagerDiscardButton, "click", () => {
      if (typeof actions.discardTemplateManagerChanges === "function") {
        actions.discardTemplateManagerChanges();
      }
    });
    bindListener(subnetworkLibraryBackdrop, "click", () =>
      actions.toggleSubnetworkLibrary(false)
    );
    bindListener(subnetworkLibraryCloseButton, "click", () =>
      actions.toggleSubnetworkLibrary(false)
    );
    bindListener(subnetworkLibrarySearchInput, "input", (event) => {
      actions.updateSubnetworkLibrarySearch(event.target.value);
    });
    bindListener(subnetworkLibraryTagFilter, "change", (event) => {
      actions.updateSubnetworkLibraryTagFilter(event.target.value);
    });
    bindListener(subnetworkLibrarySelectAllInput, "change", (event) => {
      actions.toggleSelectAllVisibleSubnetworks(Boolean(event.target.checked));
    });
    bindListener(subnetworkLibraryAddSelectedButton, "click", () => {
      actions.addSelectedSubnetworksToSessionTemplates();
    });
    bindListener(benchmarkCompareBackdrop, "click", () =>
      actions.closeBenchmarkCompareModal()
    );
    bindListener(benchmarkCompareCloseButton, "click", () =>
      actions.closeBenchmarkCompareModal()
    );
    bindListener(benchmarkCompareExportCsvButton, "click", () =>
      actions.exportBenchmarkCompareAsCsv()
    );
    bindListener(benchmarkCompareExportTextButton, "click", () =>
      actions.exportBenchmarkCompareAsText()
    );
    bindListener(benchmarkCompareCopyLatexButton, "click", () =>
      actions.copyBenchmarkCompareAsLatex()
    );
    bindListener(engineSelect, "change", (event) => {
      setSelectChevronExpanded(engineSelectField, false);
      store.setSelectedEngine(event.target.value);
      actions.enforceLinearPeriodicEngineSupport();
      actions.renderPlanner();
      actions.updateToolbarState();
      actions.setStatus(
        `Engine set to ${actions.formatEngineLabel(state.selectedEngine)}.`,
        "success"
      );
      if (typeof actions.scheduleDraftAutosave === "function") {
        actions.scheduleDraftAutosave();
      }
    });
    bindListener(collectionFormatSelect, "change", (event) => {
      setSelectChevronExpanded(collectionFormatSelectField, false);
      store.setSelectedCollectionFormat(event.target.value);
      if (typeof actions.scheduleDraftAutosave === "function") {
        actions.scheduleDraftAutosave();
      }
    });
    bindListener(documentRef, "mousedown", (event) => {
      if (!isWithinTransientToolbarUi(event.target)) {
        actions.closeTransientToolbarUi();
      }
    });
    bindListener(loadInput, "change", actions.loadDesignFromFile);
    bindListener(subnetworkLoadInput, "change", actions.loadSubnetworkFromFile);
    bindListener(templateLoadInput, "change", actions.loadSessionTemplatesFromFile);
    bindListener(windowRef, "keydown", actions.handleKeydown);
    bindListener(windowRef, "resize", actions.handleWindowResize);
    bindListener(windowRef, "mousemove", actions.handleGlobalMouseMove);
    bindListener(windowRef, "mouseup", actions.handleGlobalMouseUp);
    bindListener(canvasShell, "contextmenu", actions.handleCanvasContextMenu);
    bindListener(canvasShell, "wheel", actions.handleCanvasWheel, { passive: false });
    bindListener(canvasShell, "mousedown", actions.handleCanvasMouseDown, true);
    bindListener(minimapCanvas, "mousedown", actions.handleMinimapMouseDown);
  }

  return {
    attachToolbarHandlers,
  };
}

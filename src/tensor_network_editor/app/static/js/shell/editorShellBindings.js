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
    exportPythonMenuItem,
    exportPngMenuItem,
    exportSvgMenuItem,
    exportFormatSelect,
    singleModeMenuItem,
    linearPeriodicModeMenuItem,
    linearPeriodicPreviousCellButton,
    linearPeriodicNextCellButton,
    templateSelectField,
    templateSelect,
    templateSettingsButton,
    templateSettingsPopover,
    templateGraphSizeInput,
    templateBondDimensionInput,
    templatePhysicalDimensionInput,
    insertTemplateButton,
    saveSessionTemplateMenuItem,
    loadSessionTemplateMenuItem,
    exportSessionTemplateMenuItem,
    editSessionTemplateMenuItem,
    reflowImportedButton,
    createGroupButton,
    helpInfoMenuItem,
    helpShortcutsMenuItem,
    helpAboutMenuItem,
    helpBackdrop,
    helpCloseButton,
    templateManagerBackdrop,
    templateManagerCloseButton,
    canvasShell,
    minimapCanvas,
    engineSelect,
    collectionFormatSelect,
  } = dom;

  const toolbarMenus = [
    { name: "file", button: fileMenuButton, panel: fileMenuPanel },
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
      templateSettingsButton,
      templateSettingsPopover,
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

  function readTemplateSelectExpanded() {
    if (!templateSelectField) {
      return false;
    }
    if (typeof templateSelectField.getAttribute === "function") {
      return templateSelectField.getAttribute("data-expanded") === "true";
    }
    return templateSelectField.attributes?.["data-expanded"] === "true";
  }

  function setTemplateSelectExpanded(isExpanded) {
    if (!templateSelectField || typeof templateSelectField.setAttribute !== "function") {
      return;
    }
    templateSelectField.setAttribute("data-expanded", String(Boolean(isExpanded)));
  }

  function attachToolbarHandlers() {
    shortcutTooltip.applyShortcutHint("add-tensor-button", "Add tensor", "N");
    shortcutTooltip.applyShortcutHint("insert-template-button", "Insert template", "T");
    shortcutTooltip.applyShortcutHint("create-group-button", "Group", "G");
    shortcutTooltip.applyShortcutHint("add-note-button", "Add note", "P");
    shortcutTooltip.applyShortcutHint("connect-button", "Connect", "C");
    shortcutTooltip.applyShortcutHint("delete-button", "Delete", "Delete");
    shortcutTooltip.applyShortcutHint("save-button", "Save tensor network", "Ctrl/Cmd+S");
    shortcutTooltip.applyShortcutHint(
      "load-design-menu-item",
      "Load tensor network",
      "Ctrl/Cmd+L"
    );
    shortcutTooltip.applyShortcutHint("generate-button", "Generate code", "Shift+G");
    shortcutTooltip.applyShortcutHint(
      "linear-periodic-mode-menu-item",
      "For unidimensional",
      "F"
    );
    shortcutTooltip.applyShortcutHint("undo-button", "Undo", "Ctrl/Cmd+Z");
    shortcutTooltip.applyShortcutHint("redo-button", "Redo", redoShortcutLabel);
    shortcutTooltip.applyShortcutHint("help-info-menu-item", "Info", "?");
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
    bindListener(documentRef.getElementById("done-button"), "click", actions.completeEditor);
    bindListener(documentRef.getElementById("cancel-button"), "click", actions.cancelEditor);
    bindListener(documentRef.getElementById("copy-code-button"), "click", actions.copyGeneratedCode);
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
    bindListener(exportFormatSelect, "change", () => {
      actions.updateToolbarState();
    });
    bindListener(singleModeMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.setLinearPeriodicMode(false);
      actions.updateToolbarState();
    });
    bindListener(linearPeriodicModeMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.setLinearPeriodicMode(true);
      actions.updateToolbarState();
    });
    bindListener(linearPeriodicPreviousCellButton, "click", () => {
      actions.switchLinearPeriodicCell(-1);
    });
    bindListener(linearPeriodicNextCellButton, "click", () => {
      actions.switchLinearPeriodicCell(1);
    });
    setTemplateSelectExpanded(false);
    bindListener(templateSelect, "mousedown", () => {
      setTemplateSelectExpanded(!readTemplateSelectExpanded());
    });
    bindListener(templateSelect, "keydown", (event) => {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
        setTemplateSelectExpanded(true);
      }
      if (["Escape", "Tab"].includes(event.key)) {
        setTemplateSelectExpanded(false);
      }
    });
    bindListener(templateSelect, "blur", () => {
      setTemplateSelectExpanded(false);
    });
    bindListener(templateSelect, "change", (event) => {
      setTemplateSelectExpanded(false);
      actions.handleTemplateSelectionChange(event);
    });
    bindListener(templateGraphSizeInput, "change", actions.handleTemplateParameterInput);
    bindListener(templateBondDimensionInput, "change", actions.handleTemplateParameterInput);
    bindListener(
      templatePhysicalDimensionInput,
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
    bindListener(loadSessionTemplateMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.openSessionTemplatePicker();
    });
    bindListener(exportSessionTemplateMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.exportSelectedTemplateSpec();
    });
    bindListener(editSessionTemplateMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.toggleTemplateManager(true);
    });
    bindListener(reflowImportedButton, "click", actions.reflowLastImportedTensors);
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
    bindListener(engineSelect, "change", (event) => {
      store.setSelectedEngine(event.target.value);
      actions.enforceLinearPeriodicEngineSupport();
      actions.renderPlanner();
      actions.updateToolbarState();
      actions.setStatus(
        `Engine set to ${actions.formatEngineLabel(state.selectedEngine)}.`,
        "success"
      );
    });
    bindListener(collectionFormatSelect, "change", (event) => {
      store.setSelectedCollectionFormat(event.target.value);
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
    bindListener(windowRef, "beforeunload", actions.sendCancelBeacon);
    bindListener(windowRef, "pagehide", actions.sendCancelBeacon);
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

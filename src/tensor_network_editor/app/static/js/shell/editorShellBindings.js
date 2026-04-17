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
    loadButton,
    loadMenuPanel,
    loadDesignMenuItem,
    loadSubnetworkMenuItem,
    connectButton,
    loadInput,
    subnetworkLoadInput,
    undoButton,
    redoButton,
    exportButton,
    exportMenuPanel,
    exportPythonMenuItem,
    exportPngMenuItem,
    exportSvgMenuItem,
    exportFormatSelect,
    toggleLinearPeriodicButton,
    linearPeriodicPreviousCellButton,
    linearPeriodicNextCellButton,
    templateSelect,
    templateSettingsButton,
    templateSettingsPopover,
    templateGraphSizeInput,
    templateBondDimensionInput,
    templatePhysicalDimensionInput,
    insertTemplateButton,
    renameTemplateButton,
    deleteTemplateButton,
    reflowImportedButton,
    createGroupButton,
    helpButton,
    helpBackdrop,
    helpCloseButton,
    canvasShell,
    minimapCanvas,
    engineSelect,
    collectionFormatSelect,
  } = dom;

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
      loadButton,
      loadMenuPanel,
      exportButton,
      exportMenuPanel,
      templateSettingsButton,
      templateSettingsPopover,
    ].some((element) => targetWithinElement(target, element));
  }

  function attachToolbarHandlers() {
    shortcutTooltip.applyShortcutHint("add-tensor-button", "Add tensor", "N");
    shortcutTooltip.applyShortcutHint("insert-template-button", "Insert template", "T");
    shortcutTooltip.applyShortcutHint("create-group-button", "Group", "G");
    shortcutTooltip.applyShortcutHint("add-note-button", "Add note", "P");
    shortcutTooltip.applyShortcutHint("connect-button", "Connect", "C");
    shortcutTooltip.applyShortcutHint("delete-button", "Delete", "Delete");
    shortcutTooltip.applyShortcutHint("save-button", "Save", "Ctrl/Cmd+S");
    shortcutTooltip.applyShortcutHint("load-button", "Load", "Ctrl/Cmd+L");
    shortcutTooltip.applyShortcutHint("generate-button", "Generate code", "Shift+G");
    shortcutTooltip.applyShortcutHint("toggle-linear-periodic-button", "For mode", "F");
    shortcutTooltip.applyShortcutHint("undo-button", "Undo", "Ctrl/Cmd+Z");
    shortcutTooltip.applyShortcutHint("redo-button", "Redo", redoShortcutLabel);
    shortcutTooltip.applyShortcutHint("help-button", "Help", "?");
    shortcutTooltip.attachShortcutTooltipHandlers();

    bindListener(documentRef.getElementById("new-design-button"), "click", actions.handleNewDesign);
    bindListener(documentRef.getElementById("add-tensor-button"), "click", actions.addTensorAtCenter);
    bindListener(addNoteButton, "click", actions.addNoteAtCenter);
    bindListener(connectButton, "click", actions.toggleConnectMode);
    bindListener(documentRef.getElementById("delete-button"), "click", actions.deleteSelection);
    bindListener(documentRef.getElementById("save-button"), "click", actions.saveDesign);
    bindListener(loadButton, "click", () => {
      actions.toggleToolbarMenu("load");
    });
    bindListener(loadDesignMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      loadInput.click();
    });
    bindListener(loadSubnetworkMenuItem, "click", () => {
      actions.closeTransientToolbarUi();
      actions.openSubnetworkPicker();
    });
    bindListener(documentRef.getElementById("generate-button"), "click", actions.generateCode);
    bindListener(documentRef.getElementById("done-button"), "click", actions.completeEditor);
    bindListener(documentRef.getElementById("cancel-button"), "click", actions.cancelEditor);
    bindListener(documentRef.getElementById("copy-code-button"), "click", actions.copyGeneratedCode);
    bindListener(undoButton, "click", actions.performUndo);
    bindListener(redoButton, "click", actions.performRedo);
    bindListener(exportButton, "click", () => {
      actions.toggleToolbarMenu("export");
    });
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
    bindListener(toggleLinearPeriodicButton, "click", actions.toggleLinearPeriodicMode);
    bindListener(linearPeriodicPreviousCellButton, "click", () => {
      actions.switchLinearPeriodicCell(-1);
    });
    bindListener(linearPeriodicNextCellButton, "click", () => {
      actions.switchLinearPeriodicCell(1);
    });
    bindListener(templateSelect, "change", actions.handleTemplateSelectionChange);
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
    bindListener(renameTemplateButton, "click", actions.renameSelectedTemplate);
    bindListener(deleteTemplateButton, "click", actions.deleteSelectedTemplate);
    bindListener(reflowImportedButton, "click", actions.reflowLastImportedTensors);
    bindListener(createGroupButton, "click", actions.createGroupFromSelection);
    bindListener(helpButton, "click", () => actions.toggleHelpModal(true));
    bindListener(helpBackdrop, "click", () => actions.toggleHelpModal(false));
    bindListener(helpCloseButton, "click", () => actions.toggleHelpModal(false));
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

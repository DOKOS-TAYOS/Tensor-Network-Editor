export function createUtilityUiBindings({ ctx, state, dom, runtime }) {
  const {
    statusMessage,
    loadButton,
    loadMenuPanel,
    exportFormatSelect,
    generatedCode,
    generatedCodeView,
    codeGenerationWarning,
    undoButton,
    redoButton,
    exportButton,
    exportMenuPanel,
    toggleLinearPeriodicButton,
    linearPeriodicPreviousCellButton,
    linearPeriodicCellLabel,
    linearPeriodicNextCellButton,
    templateSelect,
    templateSettingsButton,
    templateSettingsPopover,
    insertTemplateButton,
    renameTemplateButton,
    deleteTemplateButton,
    templateCatalogWarning,
    reflowImportedButton,
    createGroupButton,
    generateButton,
  } = dom;

  const LINEAR_PERIODIC_CELL_LABELS = {
    initial: "Initial cell",
    periodic: "Periodic cell",
    final: "Final cell",
  };

  function toggleElementClass(element, className, isActive) {
    if (
      !element ||
      !element.classList ||
      typeof element.classList.toggle !== "function"
    ) {
      return;
    }
    element.classList.toggle(className, isActive);
  }

  function setExpandedState(button, isExpanded) {
    if (!button || typeof button.setAttribute !== "function") {
      return;
    }
    button.setAttribute("aria-expanded", String(isExpanded));
  }

  function syncToolbarTransientUi() {
    const openToolbarMenu =
      typeof state.openToolbarMenu === "string" ? state.openToolbarMenu : null;
    const isLoadMenuOpen =
      openToolbarMenu === "load" && loadButton && !loadButton.disabled;
    const isExportMenuOpen =
      openToolbarMenu === "export" && exportButton && !exportButton.disabled;
    const isTemplateSettingsOpen =
      Boolean(state.isTemplateSettingsOpen)
      && templateSettingsButton
      && !templateSettingsButton.disabled;

    if (loadMenuPanel) {
      loadMenuPanel.hidden = !isLoadMenuOpen;
    }
    setExpandedState(loadButton, isLoadMenuOpen);
    toggleElementClass(loadButton, "is-active", isLoadMenuOpen);

    if (exportMenuPanel) {
      exportMenuPanel.hidden = !isExportMenuOpen;
    }
    setExpandedState(exportButton, isExportMenuOpen);
    toggleElementClass(exportButton, "is-active", isExportMenuOpen);

    if (templateSettingsPopover) {
      templateSettingsPopover.hidden = !isTemplateSettingsOpen;
    }
    setExpandedState(templateSettingsButton, isTemplateSettingsOpen);
    toggleElementClass(templateSettingsButton, "is-active", isTemplateSettingsOpen);
  }

  function closeTransientToolbarUi() {
    const hadOpenUi = Boolean(state.openToolbarMenu || state.isTemplateSettingsOpen);
    if (!hadOpenUi) {
      return false;
    }
    state.openToolbarMenu = null;
    state.isTemplateSettingsOpen = false;
    syncToolbarTransientUi();
    return true;
  }

  function toggleToolbarMenu(menuName) {
    if (menuName !== "load" && menuName !== "export") {
      return state.openToolbarMenu;
    }
    state.openToolbarMenu = state.openToolbarMenu === menuName ? null : menuName;
    state.isTemplateSettingsOpen = false;
    syncToolbarTransientUi();
    return state.openToolbarMenu;
  }

  function toggleTemplateSettingsPopover() {
    if (!templateSettingsButton || templateSettingsButton.disabled) {
      return state.isTemplateSettingsOpen;
    }
    state.isTemplateSettingsOpen = !state.isTemplateSettingsOpen;
    state.openToolbarMenu = null;
    syncToolbarTransientUi();
    return state.isTemplateSettingsOpen;
  }

  function renderGeneratedCodePreview(code = state.generatedCode) {
    const renderedCode = typeof code === "string" ? code : "";
    if (generatedCode) {
      generatedCode.value = renderedCode;
    }
    if (!generatedCodeView) {
      return;
    }
    generatedCodeView.textContent = renderedCode;
    const prism =
      ctx.window && typeof ctx.window === "object" ? ctx.window.Prism : null;
    if (prism && typeof prism.highlightElement === "function") {
      prism.highlightElement(generatedCodeView);
    }
  }

  function syncCodeGenerationWarning() {
    if (!codeGenerationWarning) {
      return;
    }
    const warningMessage =
      typeof ctx.getTensorKrowchManualPlanIssueMessage === "function"
        ? ctx.getTensorKrowchManualPlanIssueMessage()
        : "";
    codeGenerationWarning.textContent = warningMessage;
    codeGenerationWarning.title = warningMessage;
    codeGenerationWarning.hidden = !warningMessage;
  }

  function syncTemplateCatalogWarning() {
    if (!templateCatalogWarning) {
      return;
    }
    const warningMessages = Array.isArray(state.templateCatalogWarnings)
      ? state.templateCatalogWarnings.filter(
          (warningMessage) => typeof warningMessage === "string" && warningMessage
        )
      : [];
    if (!warningMessages.length) {
      templateCatalogWarning.textContent = "";
      templateCatalogWarning.title = "";
      templateCatalogWarning.hidden = true;
      return;
    }
    const extraWarningCount = warningMessages.length - 1;
    templateCatalogWarning.textContent =
      extraWarningCount > 0
        ? `${warningMessages[0]} (+${extraWarningCount} more)`
        : warningMessages[0];
    templateCatalogWarning.title = warningMessages.join("\n");
    templateCatalogWarning.hidden = false;
  }

  function updateToolbarState() {
    const linearPeriodicMode = runtime.isLinearPeriodicMode();
    const activeLinearPeriodicCell = runtime.getActiveLinearPeriodicCellName();
    const selectedTensorIds =
      typeof ctx.getSelectedIdsByKind === "function"
        ? ctx.getSelectedIdsByKind("tensor")
        : [];
    const activeImportedTensorIds = Array.isArray(state.lastImportedTensorIds)
      ? state.lastImportedTensorIds.filter((tensorId) => Boolean(ctx.findTensorById(tensorId)))
      : [];
    runtime.enforceLinearPeriodicEngineSupport();
    const selectedExportFormat = exportFormatSelect ? exportFormatSelect.value : "py";
    const exportNeedsEngine = selectedExportFormat === "py";
    syncCodeGenerationWarning();
    syncTemplateCatalogWarning();

    undoButton.disabled = state.undoStack.length === 0;
    redoButton.disabled = state.redoStack.length === 0;
    if (exportButton) {
      exportButton.disabled = !state.spec || (exportNeedsEngine && !state.selectedEngine);
    }
    if (generateButton) {
      generateButton.disabled = !state.spec || !state.selectedEngine;
    }
    insertTemplateButton.disabled = !templateSelect.value;
    const selectedTemplateDefinition =
      typeof ctx.getTemplateDefinition === "function"
        ? ctx.getTemplateDefinition(templateSelect.value)
        : null;
    const selectedTemplateSource =
      selectedTemplateDefinition && typeof selectedTemplateDefinition.source === "string"
        ? selectedTemplateDefinition.source
        : "global";
    if (renameTemplateButton) {
      renameTemplateButton.disabled =
        !templateSelect.value || selectedTemplateSource !== "project";
      renameTemplateButton.title =
        !templateSelect.value
          ? "Choose a template first."
          : selectedTemplateSource !== "project"
            ? "Available only for project-local templates."
            : "Rename the selected project template.";
    }
    if (deleteTemplateButton) {
      deleteTemplateButton.disabled =
        !templateSelect.value || selectedTemplateSource !== "project";
      deleteTemplateButton.title =
        !templateSelect.value
          ? "Choose a template first."
          : selectedTemplateSource !== "project"
            ? "Available only for project-local templates."
            : "Delete the selected project template.";
    }
    if (reflowImportedButton) {
      reflowImportedButton.disabled = activeImportedTensorIds.length < 2;
      reflowImportedButton.title =
        activeImportedTensorIds.length < 2
          ? "Insert a template or subnetwork first."
          : "Reflow the last imported tensors.";
    }
    if (templateSettingsButton) {
      templateSettingsButton.disabled = !templateSelect.value;
      templateSettingsButton.title = !templateSelect.value
        ? "Choose a template first."
        : "Edit template parameters.";
    }
    if ((!templateSelect.value || linearPeriodicMode) && state.isTemplateSettingsOpen) {
      state.isTemplateSettingsOpen = false;
    }
    createGroupButton.disabled = selectedTensorIds.length < 2;
    if (toggleLinearPeriodicButton) {
      toggleLinearPeriodicButton.classList.toggle("is-active", linearPeriodicMode);
    }
    if (linearPeriodicCellLabel) {
      linearPeriodicCellLabel.textContent = linearPeriodicMode
        ? LINEAR_PERIODIC_CELL_LABELS[activeLinearPeriodicCell] || "For mode"
        : "Normal";
    }
    if (linearPeriodicPreviousCellButton) {
      linearPeriodicPreviousCellButton.disabled =
        !linearPeriodicMode || activeLinearPeriodicCell === "initial";
    }
    if (linearPeriodicNextCellButton) {
      linearPeriodicNextCellButton.disabled =
        !linearPeriodicMode || activeLinearPeriodicCell === "final";
    }
    syncToolbarTransientUi();
  }

  function formatIssues(issues) {
    if (!issues || !issues.length) {
      return "The design is not valid yet.";
    }
    return issues
      .slice(0, 3)
      .map((issue) => issue.message)
      .join(" ");
  }

  function setStatus(message, kind = "info") {
    if (!statusMessage) {
      return;
    }
    statusMessage.textContent = message;
    statusMessage.classList.remove("status-error", "status-success");
    if (kind === "error") {
      statusMessage.classList.add("status-error");
    }
    if (kind === "success") {
      statusMessage.classList.add("status-success");
    }
  }

  return {
    syncToolbarTransientUi,
    closeTransientToolbarUi,
    toggleToolbarMenu,
    toggleTemplateSettingsPopover,
    renderGeneratedCodePreview,
    updateToolbarState,
    syncCodeGenerationWarning,
    syncTemplateCatalogWarning,
    formatIssues,
    setStatus,
  };
}

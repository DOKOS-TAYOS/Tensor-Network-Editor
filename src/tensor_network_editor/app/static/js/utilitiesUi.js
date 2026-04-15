export function createUtilityUiBindings({ ctx, state, dom, runtime }) {
  const {
    statusMessage,
    exportFormatSelect,
    generatedCode,
    generatedCodeView,
    codeGenerationWarning,
    undoButton,
    redoButton,
    exportButton,
    toggleLinearPeriodicButton,
    linearPeriodicPreviousCellButton,
    linearPeriodicCellLabel,
    linearPeriodicNextCellButton,
    templateSelect,
    insertTemplateButton,
    insertSubnetworkButton,
    createGroupButton,
    generateButton,
  } = dom;

  const LINEAR_PERIODIC_CELL_LABELS = {
    initial: "Initial cell",
    periodic: "Periodic cell",
    final: "Final cell",
  };

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

  function updateToolbarState() {
    const linearPeriodicMode = runtime.isLinearPeriodicMode();
    const activeLinearPeriodicCell = runtime.getActiveLinearPeriodicCellName();
    const selectedTensorIds =
      typeof ctx.getSelectedIdsByKind === "function"
        ? ctx.getSelectedIdsByKind("tensor")
        : [];
    runtime.enforceLinearPeriodicEngineSupport();
    const selectedExportFormat = exportFormatSelect ? exportFormatSelect.value : "py";
    const exportNeedsEngine = selectedExportFormat === "py";
    syncCodeGenerationWarning();

    undoButton.disabled = state.undoStack.length === 0;
    redoButton.disabled = state.redoStack.length === 0;
    if (exportButton) {
      exportButton.disabled = !state.spec || (exportNeedsEngine && !state.selectedEngine);
    }
    if (generateButton) {
      generateButton.disabled = !state.spec || !state.selectedEngine;
    }
    insertTemplateButton.disabled = !templateSelect.value;
    if (insertSubnetworkButton) {
      insertSubnetworkButton.disabled = linearPeriodicMode;
      insertSubnetworkButton.title = linearPeriodicMode
        ? "Available only in normal graph mode."
        : "Insert a saved subnetwork JSON fragment.";
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
    renderGeneratedCodePreview,
    updateToolbarState,
    syncCodeGenerationWarning,
    formatIssues,
    setStatus,
  };
}

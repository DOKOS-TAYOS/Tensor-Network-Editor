export function createUtilityUiBindings({ ctx, state, dom, runtime }) {
  const {
    statusMessage,
    exportFormatSelect,
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
    createGroupButton,
    generateButton,
  } = dom;

  const LINEAR_PERIODIC_CELL_LABELS = {
    initial: "Initial cell",
    periodic: "Periodic cell",
    final: "Final cell",
  };

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
    updateToolbarState,
    syncCodeGenerationWarning,
    formatIssues,
    setStatus,
  };
}

export function createHistorySnapshotSupport({
  state,
  historyLimit,
  buildHistorySnapshotSpec,
  deepClone,
  updateToolbarState,
  normalizeSpec,
  bumpSpecRevision,
  reconcileTensorOrder,
  enforceLinearPeriodicEngineSupport,
  clearGeneratedCodePreview: clearGeneratedCodePreviewImpl,
  pruneSelectionToExisting,
  render,
  refreshContractionAnalysis,
  setStatus,
}) {
  function clearHistory() {
    state.undoStack = [];
    state.redoStack = [];
    updateToolbarState();
  }

  function createHistorySnapshot() {
    const snapshotSpec = buildHistorySnapshotSpec
      ? buildHistorySnapshotSpec()
      : null;
    return {
      spec: snapshotSpec == null ? deepClone(state.spec) : snapshotSpec,
      tensorOrder: Array.isArray(state.tensorOrder) ? [...state.tensorOrder] : [],
    };
  }

  function clearGeneratedCodePreview() {
    return clearGeneratedCodePreviewImpl();
  }

  function buildDesignStatusMessage(baseMessage, previewCleared) {
    if (!previewCleared) {
      return baseMessage;
    }
    return `${baseMessage} Generated code preview cleared; generate again to refresh it.`;
  }

  function commitHistorySnapshot(previousSnapshot) {
    state.undoStack.push(previousSnapshot);
    if (state.undoStack.length > historyLimit) {
      state.undoStack.shift();
    }
    state.redoStack = [];
    state.lastMutationClearedCode = clearGeneratedCodePreview();
    updateToolbarState();
    return true;
  }

  function restoreHistorySnapshot(snapshot) {
    state.spec = normalizeSpec(snapshot.spec);
    state.tensorOrder = Array.isArray(snapshot.tensorOrder)
      ? [...snapshot.tensorOrder]
      : [];
    if (typeof bumpSpecRevision === "function") {
      bumpSpecRevision();
    }
    reconcileTensorOrder();
    if (typeof enforceLinearPeriodicEngineSupport === "function") {
      enforceLinearPeriodicEngineSupport();
    }
    state.pendingIndexId = null;
    state.pendingPlannerOperandId = null;
    state.pendingPlannerSelectionId = null;
    state.plannerInspectionStepCount = null;
    state.plannerPreviewMode = null;
    state.plannerFutureBadgeDisclosure = {};
    state.activeNoteResize = null;
    state.activeSidebarTab = "selection";
    state.pendingPropertiesIndexFocusId = null;
    state.autoExpandedTensorIndex = null;
    state.tensorIndexDisclosureState = {};
    clearGeneratedCodePreview();
    pruneSelectionToExisting();
    render();
    if (typeof refreshContractionAnalysis === "function") {
      refreshContractionAnalysis();
    }
    updateToolbarState();
  }

  function performUndo() {
    if (!state.undoStack.length) {
      setStatus("There is nothing to undo.");
      return;
    }
    state.redoStack.push(createHistorySnapshot());
    restoreHistorySnapshot(state.undoStack.pop());
    setStatus("Undo applied.", "success");
  }

  function performRedo() {
    if (!state.redoStack.length) {
      setStatus("There is nothing to redo.");
      return;
    }
    state.undoStack.push(createHistorySnapshot());
    restoreHistorySnapshot(state.redoStack.pop());
    setStatus("Redo applied.", "success");
  }

  return {
    clearHistory,
    createHistorySnapshot,
    clearGeneratedCodePreview,
    buildDesignStatusMessage,
    commitHistorySnapshot,
    restoreHistorySnapshot,
    performUndo,
    performRedo,
  };
}

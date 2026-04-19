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
  function createEmptyBenchmarkCompareState() {
    return {
      open: false,
      loading: false,
      errorMessage: "",
      tableModel: null,
      rows: [],
      activeRequestId: 0,
    };
  }

  function createEmptyBenchmarkSession() {
    return {
      enabled: false,
      activePosition: 0,
      originalPlan: null,
      schemes: [],
      compareModal: createEmptyBenchmarkCompareState(),
    };
  }

  function restoreBenchmarkSession(snapshotBenchmarkSession) {
    const nextBenchmarkSession =
      snapshotBenchmarkSession && typeof snapshotBenchmarkSession === "object"
        ? deepClone(snapshotBenchmarkSession)
        : createEmptyBenchmarkSession();
    const compareModal =
      nextBenchmarkSession.compareModal &&
      typeof nextBenchmarkSession.compareModal === "object"
        ? nextBenchmarkSession.compareModal
        : createEmptyBenchmarkCompareState();
    compareModal.rows = Array.isArray(compareModal.rows)
      ? compareModal.rows
      : compareModal.tableModel && Array.isArray(compareModal.tableModel.rows)
        ? compareModal.tableModel.rows
        : [];
    nextBenchmarkSession.compareModal = compareModal;
    nextBenchmarkSession.schemes = Array.isArray(nextBenchmarkSession.schemes)
      ? nextBenchmarkSession.schemes
      : [];
    nextBenchmarkSession.activePosition = Number.isInteger(
      nextBenchmarkSession.activePosition
    )
      ? nextBenchmarkSession.activePosition
      : 0;
    nextBenchmarkSession.enabled = Boolean(nextBenchmarkSession.enabled);
    if (!nextBenchmarkSession.enabled) {
      nextBenchmarkSession.activePosition = 0;
    } else if (
      nextBenchmarkSession.activePosition > nextBenchmarkSession.schemes.length
    ) {
      nextBenchmarkSession.activePosition = nextBenchmarkSession.schemes.length;
    }
    state.benchmarkSession = nextBenchmarkSession;
    if (!nextBenchmarkSession.enabled) {
      return;
    }
    if (nextBenchmarkSession.activePosition <= 0) {
      state.spec.contraction_plan = null;
      return;
    }
    const activeScheme =
      nextBenchmarkSession.schemes[nextBenchmarkSession.activePosition - 1] || null;
    if (!activeScheme) {
      nextBenchmarkSession.activePosition = 0;
      state.spec.contraction_plan = null;
      return;
    }
    state.spec.contraction_plan = deepClone(activeScheme);
    nextBenchmarkSession.schemes[nextBenchmarkSession.activePosition - 1] =
      state.spec.contraction_plan;
  }

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
      benchmarkSession: deepClone(state.benchmarkSession || createEmptyBenchmarkSession()),
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
    restoreBenchmarkSession(snapshot.benchmarkSession);
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

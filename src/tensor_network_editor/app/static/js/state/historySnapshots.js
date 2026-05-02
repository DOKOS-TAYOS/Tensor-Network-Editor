import {
  createEmptyBenchmarkCompareState,
  createEmptyBenchmarkSession,
} from "./benchmarkState.js";

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
  function createHistorySnapshotBenchmarkPlan(plan) {
    if (!plan || typeof plan !== "object") {
      return {
        id: "",
        name: "",
        steps: [],
        view_snapshots: [],
        metadata: {},
      };
    }
    return {
      id: typeof plan.id === "string" ? plan.id : "",
      name: typeof plan.name === "string" ? plan.name : "",
      steps: Array.isArray(plan.steps) ? deepClone(plan.steps) : [],
      view_snapshots: [],
      metadata:
        plan.metadata && typeof plan.metadata === "object"
          ? deepClone(plan.metadata)
          : {},
    };
  }

  function createHistorySnapshotBenchmarkSession(benchmarkSession) {
    const sourceSession =
      benchmarkSession && typeof benchmarkSession === "object"
        ? benchmarkSession
        : createEmptyBenchmarkSession();
    const nextBenchmarkSession = {
      enabled: Boolean(sourceSession.enabled),
      activePosition: Number.isInteger(sourceSession.activePosition)
        ? sourceSession.activePosition
        : 0,
      originalPlan: sourceSession.originalPlan
        ? createHistorySnapshotBenchmarkPlan(sourceSession.originalPlan)
        : null,
      schemes: Array.isArray(sourceSession.schemes)
        ? sourceSession.schemes.map((scheme) =>
            createHistorySnapshotBenchmarkPlan(scheme)
          )
        : [],
      compareModal: createEmptyBenchmarkCompareState(),
    };
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
    return nextBenchmarkSession;
  }

  function restoreBenchmarkSession(
    snapshotBenchmarkSession,
    restoredContractionPlan = null
  ) {
    const nextBenchmarkSession = createHistorySnapshotBenchmarkSession(
      snapshotBenchmarkSession
    );
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
    const exactActiveScheme =
      restoredContractionPlan &&
      typeof restoredContractionPlan === "object" &&
      (!activeScheme.id || restoredContractionPlan.id === activeScheme.id)
        ? restoredContractionPlan
        : null;
    state.spec.contraction_plan = exactActiveScheme || deepClone(activeScheme);
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
      benchmarkSession: createHistorySnapshotBenchmarkSession(
        state.benchmarkSession
      ),
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
    if (typeof bumpSpecRevision === "function") {
      bumpSpecRevision();
    }
    state.lastMutationClearedCode = clearGeneratedCodePreview();
    updateToolbarState();
    return true;
  }

  function restoreHistorySnapshot(snapshot) {
    state.spec = normalizeSpec(snapshot.spec);
    state.tensorOrder = Array.isArray(snapshot.tensorOrder)
      ? [...snapshot.tensorOrder]
      : [];
    restoreBenchmarkSession(
      snapshot.benchmarkSession,
      state.spec && typeof state.spec === "object"
        ? state.spec.contraction_plan || null
        : null
    );
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
    state.plannerPreviewBadgeDisclosure = {};
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

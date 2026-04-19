export function createDesignMutationPipeline({
  state,
  isForMode,
  captureEditableFocus,
  restoreEditableFocus,
  resetDerivedStateCaches,
  syncCurrentGraphIntoLinearPeriodicChain,
  syncLinearPeriodicBoundaryTensors,
  syncCurrentGraphIntoGridPeriodicGrid,
  syncGridPeriodicBoundaryTensors,
  repairContractionPlan,
  reconcileTensorOrder,
  bumpSpecRevision,
  createHistorySnapshot,
  commitHistorySnapshot,
  buildDesignStatusMessage,
  pruneSelectionToExisting,
  updatePendingPropertiesIndexFocus,
  syncSelectedElementState,
  renderMutationState,
  markContractionAnalysisDirty,
  shouldRefreshContractionAnalysisImmediately,
  refreshContractionAnalysis,
  setStatus,
}) {
  function normalizeInvalidations(overrides = {}) {
    return {
      graph: true,
      lookups: true,
      analysis: true,
      properties: true,
      toolbar: true,
      overlays: true,
      planner: true,
      sidebarTabs: true,
      minimap: true,
      code: true,
      ...overrides,
    };
  }

  function applyDesignChange(mutator, options = {}) {
    const beforeSnapshot = createHistorySnapshot();
    const invalidate = normalizeInvalidations(options.invalidate);
    const preservedFocus = captureEditableFocus ? captureEditableFocus() : null;
    const previousSelectionIds = [...state.selectionIds];
    mutator();
    const shouldRefreshLookups = invalidate.lookups || isForMode();
    if (shouldRefreshLookups) {
      state.lookupRevision = -1;
      if (typeof resetDerivedStateCaches === "function") {
        resetDerivedStateCaches();
      }
    }
    state.plannerPreviewMode = null;
    state.plannerFutureBadgeDisclosure = {};
    if (typeof syncCurrentGraphIntoGridPeriodicGrid === "function") {
      syncCurrentGraphIntoGridPeriodicGrid();
    }
    if (typeof syncCurrentGraphIntoLinearPeriodicChain === "function") {
      syncCurrentGraphIntoLinearPeriodicChain();
    } else if (typeof syncGridPeriodicBoundaryTensors === "function") {
      syncGridPeriodicBoundaryTensors();
    } else if (typeof syncLinearPeriodicBoundaryTensors === "function") {
      syncLinearPeriodicBoundaryTensors();
    }
    if (typeof repairContractionPlan === "function") {
      repairContractionPlan();
    }
    reconcileTensorOrder();
    if (shouldRefreshLookups && typeof bumpSpecRevision === "function") {
      bumpSpecRevision();
    }
    commitHistorySnapshot(beforeSnapshot);

    if (Array.isArray(options.selectionIds)) {
      state.selectionIds = [...options.selectionIds];
      state.primarySelectionId = options.primaryId ||
        (options.selectionIds.length
          ? options.selectionIds[options.selectionIds.length - 1]
          : null);
    }

    pruneSelectionToExisting();
    updatePendingPropertiesIndexFocus(previousSelectionIds, state.selectionIds);
    syncSelectedElementState();
    renderMutationState(invalidate);
    if (typeof options.afterRender === "function") {
      options.afterRender();
    }
    if (typeof restoreEditableFocus === "function") {
      restoreEditableFocus(preservedFocus);
    }
    if (invalidate.analysis) {
      if (typeof markContractionAnalysisDirty === "function") {
        markContractionAnalysisDirty();
      } else {
        state.contractionAnalysisDirty = true;
      }
      const shouldRefreshImmediately =
        options.refreshAnalysisImmediately === true ||
        (typeof shouldRefreshContractionAnalysisImmediately === "function" &&
          shouldRefreshContractionAnalysisImmediately());
      if (
        shouldRefreshImmediately &&
        typeof refreshContractionAnalysis === "function"
      ) {
        refreshContractionAnalysis();
      }
    }

    if (options.statusMessage) {
      setStatus(
        buildDesignStatusMessage(
          options.statusMessage,
          state.lastMutationClearedCode
        ),
        options.statusKind || "success"
      );
    } else if (state.lastMutationClearedCode) {
      setStatus(
        "Design updated. Generated code preview cleared; generate again to refresh it.",
        "success"
      );
    }
    state.lastMutationClearedCode = false;
    return true;
  }

  return {
    normalizeInvalidations,
    applyDesignChange,
  };
}

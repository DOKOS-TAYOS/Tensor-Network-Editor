import { createDesignMutationPipeline } from "../actions/designMutationPipeline.js";
import { createHistorySnapshotSupport } from "../state/historySnapshots.js";
import { createSelectionEntrySupport } from "../state/selectionEntries.js";

export function registerHistorySelection(ctx) {
  const state = ctx.state;
  const { HISTORY_LIMIT } = ctx.constants;
  const { generatedCode } = ctx.dom;

  function callOptionalContext(name, ...args) {
    if (typeof ctx[name] === "function") {
      return ctx[name](...args);
    }
    return undefined;
  }

  function clearGeneratedCodePreview() {
    const hadGeneratedCode = Boolean(state.generatedCode && state.generatedCode.trim());
    state.generatedCode = "";
    if (typeof ctx.renderGeneratedCodePreview === "function") {
      ctx.renderGeneratedCodePreview("");
    } else {
      generatedCode.value = "";
    }
    return hadGeneratedCode;
  }

  function syncCySelection() {
    if (!state.cy) {
      return;
    }
    const actualSelectedElements = state.cy.$(":selected");
    const previousSelectionIds = [];
    if (
      actualSelectedElements &&
      typeof actualSelectedElements.forEach === "function"
    ) {
      actualSelectedElements.forEach((element) => {
        const elementId = typeof element.id === "function" ? element.id() : null;
        if (elementId && !previousSelectionIds.includes(elementId)) {
          previousSelectionIds.push(elementId);
        }
      });
    } else if (Array.isArray(state.cySelectionSyncedIds)) {
      previousSelectionIds.push(...state.cySelectionSyncedIds);
    }
    const nextSelectionIds = Array.isArray(state.selectionIds) ? state.selectionIds : [];
    const previousSelectionIdSet = new Set(previousSelectionIds);
    const nextSelectionIdSet = new Set(nextSelectionIds);
    state.cy.batch(() => {
      previousSelectionIds.forEach((selectionId) => {
        if (nextSelectionIdSet.has(selectionId)) {
          return;
        }
        const element = state.cy.getElementById(selectionId);
        if (element && element.length) {
          element.unselect();
        }
      });
      nextSelectionIds.forEach((selectionId) => {
        if (previousSelectionIdSet.has(selectionId)) {
          return;
        }
        const element = state.cy.getElementById(selectionId);
        if (element && element.length) {
          element.select();
        }
      });
    });
    state.cySelectionSyncedIds = nextSelectionIds.filter((selectionId) => {
      const element = state.cy.getElementById(selectionId);
      return Boolean(element && element.length);
    });
    ctx.renderOverlayDecorations();
  }

  function renderSelectionUi() {
    syncCySelection();
    ctx.render({
      graph: false,
      code: false,
    });
  }

  const selectionSupport = createSelectionEntrySupport({
    state,
    findGroupById: (groupId) => ctx.findGroupById(groupId),
    findTensorById: (tensorId) => ctx.findTensorById(tensorId),
    findVisibleTensorById: (tensorId) =>
      typeof ctx.findVisibleTensorById === "function"
        ? ctx.findVisibleTensorById(tensorId)
        : null,
    findIndexOwner: (indexId) => ctx.findIndexOwner(indexId),
    findEdgeById: (edgeId) => ctx.findEdgeById(edgeId),
    findHyperedgeById: (hyperedgeId) =>
      typeof ctx.findHyperedgeById === "function" ? ctx.findHyperedgeById(hyperedgeId) : null,
    findNoteById: (noteId) =>
      typeof ctx.findNoteById === "function" ? ctx.findNoteById(noteId) : null,
    getVisibleTensors: () =>
      typeof ctx.getVisibleTensors === "function"
        ? ctx.getVisibleTensors()
        : state.spec.tensors,
    isContractionSceneVisible: () =>
      typeof ctx.isContractionSceneVisible === "function" &&
      ctx.isContractionSceneVisible(),
    isInspectingPastStage: () =>
      typeof ctx.isInspectingPastStage === "function" && ctx.isInspectingPastStage(),
    isPlannerOperandAvailable: (operandId) =>
      typeof ctx.isPlannerOperandAvailable === "function" &&
      ctx.isPlannerOperandAvailable(operandId),
    renderSelectionUi,
  });

  const historySupport = createHistorySnapshotSupport({
    state,
    historyLimit: HISTORY_LIMIT,
    buildHistorySnapshotSpec: () => callOptionalContext("buildHistorySnapshotSpec"),
    deepClone: (value) => ctx.deepClone(value),
    updateToolbarState: () => ctx.updateToolbarState(),
    normalizeSpec: (spec) => ctx.normalizeSpec(spec),
    bumpSpecRevision: () => callOptionalContext("bumpSpecRevision"),
    reconcileTensorOrder: () => ctx.reconcileTensorOrder(),
    enforceLinearPeriodicEngineSupport: () =>
      callOptionalContext("enforceLinearPeriodicEngineSupport"),
    clearGeneratedCodePreview,
    pruneSelectionToExisting: () => selectionSupport.pruneSelectionToExisting(),
    render: () => ctx.render(),
    refreshContractionAnalysis: () =>
      callOptionalContext("refreshContractionAnalysis"),
    scheduleDraftAutosave: () => callOptionalContext("scheduleDraftAutosave"),
    setStatus: (message, level) => ctx.setStatus(message, level),
  });

  const mutationPipeline = createDesignMutationPipeline({
    state,
    isForMode: () =>
      (typeof ctx.isForMode === "function" && ctx.isForMode()) ||
      (typeof ctx.isLinearPeriodicMode === "function" && ctx.isLinearPeriodicMode()) ||
      (typeof ctx.isGridPeriodicMode === "function" && ctx.isGridPeriodicMode()) ||
      (typeof ctx.isTreePeriodicMode === "function" && ctx.isTreePeriodicMode()),
    captureEditableFocus: () => callOptionalContext("captureEditableFocus"),
    restoreEditableFocus: (focusState) =>
      callOptionalContext("restoreEditableFocus", focusState),
    resetDerivedStateCaches: () => callOptionalContext("resetDerivedStateCaches"),
    syncCurrentGraphIntoLinearPeriodicChain: () =>
      callOptionalContext("syncCurrentGraphIntoLinearPeriodicChain"),
    syncLinearPeriodicBoundaryTensors: () =>
      callOptionalContext("syncLinearPeriodicBoundaryTensors"),
    syncCurrentGraphIntoGridPeriodicGrid: () =>
      callOptionalContext("syncCurrentGraphIntoGridPeriodicGrid"),
    syncGridPeriodicBoundaryTensors: () =>
      callOptionalContext("syncGridPeriodicBoundaryTensors"),
    syncCurrentGraphIntoTreePeriodicTree: () =>
      callOptionalContext("syncCurrentGraphIntoTreePeriodicTree"),
    repairContractionPlan: () => callOptionalContext("repairContractionPlan"),
    reconcileTensorOrder: () => ctx.reconcileTensorOrder(),
    bumpSpecRevision: () => callOptionalContext("bumpSpecRevision"),
    createHistorySnapshot: () => historySupport.createHistorySnapshot(),
    commitHistorySnapshot: (snapshot) =>
      historySupport.commitHistorySnapshot(snapshot),
    buildDesignStatusMessage: (message, previewCleared) =>
      historySupport.buildDesignStatusMessage(message, previewCleared),
    pruneSelectionToExisting: () => selectionSupport.pruneSelectionToExisting(),
    updatePendingPropertiesIndexFocus: (previousIds, nextIds) =>
      selectionSupport.updatePendingPropertiesIndexFocus(previousIds, nextIds),
    syncSelectedElementState: () => selectionSupport.syncSelectedElementState(),
    renderMutationState: (invalidate) =>
      ctx.render({
        graph: invalidate.graph,
        properties: invalidate.properties,
        code: invalidate.code,
        toolbar: invalidate.toolbar,
        overlays: invalidate.overlays,
        planner: invalidate.planner,
        sidebarTabs: invalidate.sidebarTabs,
        minimap: invalidate.minimap,
        syncSelection: true,
      }),
    markContractionAnalysisDirty: () => {
      state.contractionAnalysisDirty = true;
    },
    shouldRefreshContractionAnalysisImmediately: () =>
      state.activeSidebarTab === "planner" ||
      state.plannerMode ||
      Boolean(state.plannerPreviewMode) ||
      (typeof ctx.isInspectingPastStage === "function" &&
        ctx.isInspectingPastStage()) ||
      (typeof ctx.isContractionSceneVisible === "function" &&
        ctx.isContractionSceneVisible()),
    refreshContractionAnalysis: () =>
      callOptionalContext("refreshContractionAnalysis"),
    setStatus: (message, level) => ctx.setStatus(message, level),
  });

  Object.assign(ctx, historySupport, mutationPipeline, selectionSupport, {
    syncCySelection,
  });
}

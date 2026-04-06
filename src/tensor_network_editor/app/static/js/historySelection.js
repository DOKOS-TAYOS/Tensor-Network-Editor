export function registerHistorySelection(ctx) {
  const state = ctx.state;
  const {
    TENSOR_WIDTH,
    TENSOR_HEIGHT,
    MIN_TENSOR_WIDTH,
    MIN_TENSOR_HEIGHT,
    INDEX_RADIUS,
    INDEX_PADDING,
    HISTORY_LIMIT,
    REDO_SHORTCUT_LABEL,
    DEFAULT_INDEX_SLOTS,
  } = ctx.constants;
  const {
    statusMessage,
    propertiesPanel,
    generatedCode,
    engineSelect,
    connectButton,
    loadInput,
    undoButton,
    redoButton,
    exportPyButton,
    exportPngButton,
    exportSvgButton,
    templateSelect,
    insertTemplateButton,
    createGroupButton,
    helpButton,
    helpModal,
    helpBackdrop,
    helpCloseButton,
    canvasShell,
    groupLayer,
    resizeLayer,
    selectionBox,
    minimapCanvas,
  } = ctx.dom;
  const { apiGet, apiPost, window, document, cytoscape } = ctx;

  function clearHistory() {
    state.undoStack = [];
    state.redoStack = [];
    ctx.updateToolbarState();
  }

  function createHistorySnapshot() {
    return {
      spec: ctx.deepClone(state.spec),
      tensorOrder: Array.isArray(state.tensorOrder) ? [...state.tensorOrder] : [],
    };
  }

  function snapshotsEqual(leftSnapshot, rightSnapshot) {
    return JSON.stringify(leftSnapshot) === JSON.stringify(rightSnapshot);
  }

  function commitHistorySnapshot(previousSnapshot) {
    const nextSnapshot = createHistorySnapshot();
    if (snapshotsEqual(previousSnapshot, nextSnapshot)) {
      return false;
    }
    state.undoStack.push(previousSnapshot);
    if (state.undoStack.length > HISTORY_LIMIT) {
      state.undoStack.shift();
    }
    state.redoStack = [];
    state.lastMutationClearedCode = clearGeneratedCodePreview();
    ctx.updateToolbarState();
    return true;
  }

  function restoreHistorySnapshot(snapshot) {
    state.spec = ctx.normalizeSpec(snapshot.spec);
    state.tensorOrder = Array.isArray(snapshot.tensorOrder) ? [...snapshot.tensorOrder] : [];
    ctx.reconcileTensorOrder();
    state.pendingIndexId = null;
    state.pendingPlannerOperandId = null;
    state.pendingPlannerSelectionId = null;
    state.plannerPreviewMode = null;
    state.plannerManualOrderByTensorId = {};
    state.plannerPreviewOrderByTensorId = {};
    state.activeNoteResize = null;
    state.activeSidebarTab = "selection";
    state.pendingPropertiesIndexFocusId = null;
    state.autoExpandedTensorIndex = null;
    state.tensorIndexDisclosureState = {};
    clearGeneratedCodePreview();
    pruneSelectionToExisting();
    ctx.render();
    if (typeof ctx.refreshContractionAnalysis === "function") {
      ctx.refreshContractionAnalysis();
    }
    ctx.updateToolbarState();
  }

  function performUndo() {
    if (!state.undoStack.length) {
      ctx.setStatus("There is nothing to undo.");
      return;
    }
    state.redoStack.push(createHistorySnapshot());
    restoreHistorySnapshot(state.undoStack.pop());
    ctx.setStatus("Undo applied.", "success");
  }

  function performRedo() {
    if (!state.redoStack.length) {
      ctx.setStatus("There is nothing to redo.");
      return;
    }
    state.undoStack.push(createHistorySnapshot());
    restoreHistorySnapshot(state.redoStack.pop());
    ctx.setStatus("Redo applied.", "success");
  }

  function clearGeneratedCodePreview() {
    const hadGeneratedCode = Boolean(state.generatedCode && state.generatedCode.trim());
    state.generatedCode = "";
    generatedCode.value = "";
    return hadGeneratedCode;
  }

  function buildDesignStatusMessage(baseMessage, previewCleared) {
    if (!previewCleared) {
      return baseMessage;
    }
    return `${baseMessage} Generated code preview cleared; generate again to refresh it.`;
  }

  function applyDesignChange(mutator, options = {}) {
    const beforeSnapshot = createHistorySnapshot();
    const preservedFocus =
      typeof ctx.captureEditableFocus === "function"
        ? ctx.captureEditableFocus()
        : null;
    const previousSelectionIds = [...state.selectionIds];
    mutator();
    state.plannerPreviewMode = null;
    state.plannerPreviewOrderByTensorId = {};
    if (typeof ctx.repairContractionPlan === "function") {
      ctx.repairContractionPlan();
    }
    ctx.reconcileTensorOrder();
    const changed = commitHistorySnapshot(beforeSnapshot);
    if (!changed) {
      ctx.render();
      return false;
    }

    if (Array.isArray(options.selectionIds)) {
      state.selectionIds = [...options.selectionIds];
      state.primarySelectionId =
        options.primaryId || (options.selectionIds.length ? options.selectionIds[options.selectionIds.length - 1] : null);
    }

    pruneSelectionToExisting();
    updatePendingPropertiesIndexFocus(previousSelectionIds, state.selectionIds);
    syncSelectedElementState();
    ctx.render();
    if (typeof options.afterRender === "function") {
      options.afterRender();
    }
    if (typeof ctx.restoreEditableFocus === "function") {
      ctx.restoreEditableFocus(preservedFocus);
    }
    if (!options.skipContractionAnalysisRefresh && typeof ctx.refreshContractionAnalysis === "function") {
      ctx.refreshContractionAnalysis();
    }

    if (options.statusMessage) {
      ctx.setStatus(buildDesignStatusMessage(options.statusMessage, state.lastMutationClearedCode), options.statusKind || "success");
    } else if (state.lastMutationClearedCode) {
      ctx.setStatus("Design updated. Generated code preview cleared; generate again to refresh it.", "success");
    }
    state.lastMutationClearedCode = false;
    return true;
  }

  function resolveSelectionKind(selectionId) {
    const entry = getSelectionEntry(selectionId);
    return entry ? entry.kind : null;
  }

  function getSelectionEntry(selectionId) {
    const group = ctx.findGroupById(selectionId);
    if (group) {
      return { kind: "group", id: group.id, group };
    }
    const tensor = ctx.findTensorById(selectionId);
    if (tensor) {
      return { kind: "tensor", id: tensor.id, tensor };
    }
    const locatedIndex = ctx.findIndexOwner(selectionId);
    if (locatedIndex) {
      return { kind: "index", id: selectionId, located: locatedIndex };
    }
    const edge = ctx.findEdgeById(selectionId);
    if (edge) {
      return { kind: "edge", id: edge.id, edge };
    }
    const note = typeof ctx.findNoteById === "function" ? ctx.findNoteById(selectionId) : null;
    if (note) {
      return { kind: "note", id: note.id, note };
    }
    return null;
  }

  function getSelectedEntries() {
    return state.selectionIds.map((selectionId) => getSelectionEntry(selectionId)).filter(Boolean);
  }

  function getSelectedIdsByKind(kind) {
    return getSelectedEntries()
      .filter((entry) => entry.kind === kind)
      .map((entry) => entry.id);
  }

  function syncSelectedElementState() {
    if (state.selectionIds.length === 1) {
      const selectionId = state.selectionIds[0];
      const kind = resolveSelectionKind(selectionId);
      state.selectedElement = kind ? { kind, id: selectionId } : null;
      return;
    }
    state.selectedElement = null;
  }

  function updatePendingPropertiesIndexFocus(previousSelectionIds, nextSelectionIds) {
    releaseAutoExpandedTensorIndex(
      nextSelectionIds.length === 1 ? nextSelectionIds[0] : null
    );
    const previousPropertiesTensorId = getPropertiesTensorId(previousSelectionIds);
    const nextPropertiesTensorId = getPropertiesTensorId(nextSelectionIds);
    if (previousPropertiesTensorId !== nextPropertiesTensorId) {
      state.tensorIndexDisclosureState = {};
    }

    const previousSingleSelectionId =
      previousSelectionIds.length === 1 ? previousSelectionIds[0] : null;
    const nextSingleSelectionId =
      nextSelectionIds.length === 1 ? nextSelectionIds[0] : null;

    if (previousSingleSelectionId === nextSingleSelectionId) {
      return;
    }

    const nextEntry = nextSingleSelectionId
      ? getSelectionEntry(nextSingleSelectionId)
      : null;
    state.pendingPropertiesIndexFocusId =
      nextEntry && nextEntry.kind === "index" ? nextEntry.id : null;
  }

  function releaseAutoExpandedTensorIndex(nextSingleSelectionId) {
    const autoExpanded = state.autoExpandedTensorIndex;
    if (!autoExpanded || nextSingleSelectionId === autoExpanded.indexId) {
      return;
    }
    if (!autoExpanded.wasOpen) {
      const disclosureState =
        state.tensorIndexDisclosureState[autoExpanded.tensorId];
      if (disclosureState) {
        delete disclosureState[autoExpanded.indexId];
        if (!Object.keys(disclosureState).length) {
          delete state.tensorIndexDisclosureState[autoExpanded.tensorId];
        }
      }
    }
    state.autoExpandedTensorIndex = null;
  }

  function getPropertiesTensorId(selectionIds) {
    if (!Array.isArray(selectionIds) || selectionIds.length !== 1) {
      return null;
    }
    const entry = getSelectionEntry(selectionIds[0]);
    if (!entry) {
      return null;
    }
    if (entry.kind === "tensor") {
      return entry.id;
    }
    if (entry.kind === "index") {
      return entry.located.tensor.id;
    }
    return null;
  }

  function syncCySelection() {
    if (!state.cy) {
      return;
    }
    state.cy.batch(() => {
      state.cy.$(":selected").unselect();
      state.selectionIds.forEach((selectionId) => {
        const element = state.cy.getElementById(selectionId);
        if (element && element.length) {
          element.select();
        }
      });
    });
    ctx.renderOverlayDecorations();
  }

  function pruneSelectionToExisting() {
    state.selectionIds = state.selectionIds.filter((selectionId) => Boolean(resolveSelectionKind(selectionId)));
    if (!state.selectionIds.includes(state.primarySelectionId)) {
      state.primarySelectionId = state.selectionIds.length ? state.selectionIds[state.selectionIds.length - 1] : null;
    }
    if (state.pendingIndexId && resolveSelectionKind(state.pendingIndexId) !== "index") {
      state.pendingIndexId = null;
    }
    if (
      state.pendingPlannerOperandId &&
      typeof ctx.isPlannerOperandAvailable === "function" &&
      !ctx.isPlannerOperandAvailable(state.pendingPlannerOperandId)
    ) {
      state.pendingPlannerOperandId = null;
      state.pendingPlannerSelectionId = null;
    }
  }

  function setSelection(selectionIds, options = {}) {
    const previousSelectionIds = [...state.selectionIds];
    const uniqueIds = [];
    selectionIds.forEach((selectionId) => {
      if (resolveSelectionKind(selectionId) && !uniqueIds.includes(selectionId)) {
        uniqueIds.push(selectionId);
      }
    });
    state.selectionIds = uniqueIds;
    state.primarySelectionId =
      uniqueIds.includes(options.primaryId) ? options.primaryId : uniqueIds.length ? uniqueIds[uniqueIds.length - 1] : null;
    updatePendingPropertiesIndexFocus(previousSelectionIds, uniqueIds);
    syncSelectedElementState();
    syncCySelection();
    ctx.renderProperties();
    ctx.renderMinimap();
    ctx.updateToolbarState();
  }

  function selectElement(kind, id, options = {}) {
    if (options.additive) {
      if (state.selectionIds.includes(id)) {
        setSelection(
          state.selectionIds.filter((selectionId) => selectionId !== id),
          {
            primaryId:
              state.primarySelectionId === id && state.selectionIds.length > 1
                ? state.selectionIds[state.selectionIds.length - 2]
                : state.primarySelectionId,
          }
        );
        return;
      }
      setSelection([...state.selectionIds, id], { primaryId: id });
      return;
    }
    setSelection([id], { primaryId: id });
  }

  function setSelectedElement(kind, id) {
    setSelection([id], { primaryId: id });
  }

  function clearSelection(options = {}) {
    const previousSelectionIds = [...state.selectionIds];
    state.selectionIds = [];
    state.primarySelectionId = null;
    state.selectedElement = null;
    if (!options.preservePendingIndex) {
      state.pendingIndexId = null;
    }
    updatePendingPropertiesIndexFocus(previousSelectionIds, []);
    syncCySelection();
    ctx.renderProperties();
    ctx.renderMinimap();
    ctx.updateToolbarState();
  }

  function selectAllTensors() {
    const tensorIds = state.spec.tensors.map((tensor) => tensor.id);
    setSelection(tensorIds, { primaryId: tensorIds.length ? tensorIds[tensorIds.length - 1] : null });
  }

  Object.assign(ctx, {
    clearHistory,
    createHistorySnapshot,
    snapshotsEqual,
    commitHistorySnapshot,
    restoreHistorySnapshot,
    performUndo,
    performRedo,
    clearGeneratedCodePreview,
    buildDesignStatusMessage,
    applyDesignChange,
    resolveSelectionKind,
    getSelectionEntry,
    getSelectedEntries,
    getSelectedIdsByKind,
    getPropertiesTensorId,
    releaseAutoExpandedTensorIndex,
    syncSelectedElementState,
    syncCySelection,
    pruneSelectionToExisting,
    setSelection,
    selectElement,
    setSelectedElement,
    clearSelection,
    selectAllTensors
  });
}

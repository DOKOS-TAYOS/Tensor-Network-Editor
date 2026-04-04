import { HISTORY_LIMIT, refs, state } from "./core.js";
import {
  createSnapshotSignature,
  deepClone,
  findEdgeById,
  findIndexOwner,
  findTensorById,
  normalizeSpec,
  reconcileTensorOrder,
  setStatus,
  updateToolbarState,
} from "./utilities.js";

const { generatedCode } = refs;

export function createHistorySelection(deps) {
  const { render, renderMinimap, renderProperties } = deps;

  function clearHistory() {
    state.undoStack = [];
    state.redoStack = [];
    updateToolbarState();
  }

  function createHistorySnapshot() {
    return {
      spec: deepClone(state.spec),
      tensorOrder: Array.isArray(state.tensorOrder) ? [...state.tensorOrder] : [],
      signature: createSnapshotSignature(state.spec, state.tensorOrder),
    };
  }

  function snapshotsEqual(leftSnapshot, rightSnapshot) {
    return leftSnapshot.signature === rightSnapshot.signature;
  }

  function commitHistorySnapshot(previousSnapshot) {
    const nextSignature = createSnapshotSignature(state.spec, state.tensorOrder);
    if (previousSnapshot.signature === nextSignature) {
      return false;
    }
    state.undoStack.push(previousSnapshot);
    if (state.undoStack.length > HISTORY_LIMIT) {
      state.undoStack.shift();
    }
    state.redoStack = [];
    state.lastMutationClearedCode = clearGeneratedCodePreview();
    updateToolbarState();
    return true;
  }

  function restoreHistorySnapshot(snapshot) {
    state.spec = normalizeSpec(snapshot.spec);
    state.tensorOrder = Array.isArray(snapshot.tensorOrder) ? [...snapshot.tensorOrder] : [];
    reconcileTensorOrder();
    state.pendingIndexId = null;
    clearGeneratedCodePreview();
    pruneSelectionToExisting();
    render();
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
    mutator();
    reconcileTensorOrder();
    const changed = commitHistorySnapshot(beforeSnapshot);
    if (!changed) {
      render();
      return false;
    }

    if (Array.isArray(options.selectionIds)) {
      state.selectionIds = [...options.selectionIds];
      state.primarySelectionId =
        options.primaryId || (options.selectionIds.length ? options.selectionIds[options.selectionIds.length - 1] : null);
    }

    pruneSelectionToExisting();
    syncSelectedElementState();
    render();
    if (typeof options.afterRender === "function") {
      options.afterRender();
    }

    if (options.statusMessage) {
      setStatus(buildDesignStatusMessage(options.statusMessage, state.lastMutationClearedCode), options.statusKind || "success");
    } else if (state.lastMutationClearedCode) {
      setStatus("Design updated. Generated code preview cleared; generate again to refresh it.", "success");
    }
    state.lastMutationClearedCode = false;
    return true;
  }

  function resolveSelectionKind(selectionId) {
    const entry = getSelectionEntry(selectionId);
    return entry ? entry.kind : null;
  }

  function getSelectionEntry(selectionId) {
    const tensor = findTensorById(selectionId);
    if (tensor) {
      return { kind: "tensor", id: tensor.id, tensor };
    }
    const locatedIndex = findIndexOwner(selectionId);
    if (locatedIndex) {
      return { kind: "index", id: selectionId, located: locatedIndex };
    }
    const edge = findEdgeById(selectionId);
    if (edge) {
      return { kind: "edge", id: edge.id, edge };
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
  }

  function pruneSelectionToExisting() {
    state.selectionIds = state.selectionIds.filter((selectionId) => Boolean(resolveSelectionKind(selectionId)));
    if (!state.selectionIds.includes(state.primarySelectionId)) {
      state.primarySelectionId = state.selectionIds.length ? state.selectionIds[state.selectionIds.length - 1] : null;
    }
    if (state.pendingIndexId && resolveSelectionKind(state.pendingIndexId) !== "index") {
      state.pendingIndexId = null;
    }
  }

  function setSelection(selectionIds, options = {}) {
    const uniqueIds = [];
    selectionIds.forEach((selectionId) => {
      if (resolveSelectionKind(selectionId) && !uniqueIds.includes(selectionId)) {
        uniqueIds.push(selectionId);
      }
    });
    state.selectionIds = uniqueIds;
    state.primarySelectionId =
      uniqueIds.includes(options.primaryId) ? options.primaryId : uniqueIds.length ? uniqueIds[uniqueIds.length - 1] : null;
    syncSelectedElementState();
    syncCySelection();
    renderProperties();
    renderMinimap();
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
    state.selectionIds = [];
    state.primarySelectionId = null;
    state.selectedElement = null;
    if (!options.preservePendingIndex) {
      state.pendingIndexId = null;
    }
    syncCySelection();
    renderProperties();
    renderMinimap();
  }

  function selectAllTensors() {
    const tensorIds = state.spec.tensors.map((tensor) => tensor.id);
    setSelection(tensorIds, { primaryId: tensorIds.length ? tensorIds[tensorIds.length - 1] : null });
  }

  return {
    applyDesignChange,
    clearGeneratedCodePreview,
    clearHistory,
    clearSelection,
    commitHistorySnapshot,
    createHistorySnapshot,
    getSelectedEntries,
    getSelectedIdsByKind,
    getSelectionEntry,
    performRedo,
    performUndo,
    pruneSelectionToExisting,
    resolveSelectionKind,
    selectAllTensors,
    selectElement,
    setSelectedElement,
    setSelection,
    snapshotsEqual,
    syncCySelection,
    syncSelectedElementState,
  };
}

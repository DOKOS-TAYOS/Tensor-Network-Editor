export function createSelectionEntrySupport({
  state,
  findGroupById,
  findTensorById,
  findVisibleTensorById,
  findIndexOwner,
  findEdgeById,
  findHyperedgeById = () => null,
  findNoteById,
  getVisibleTensors,
  isContractionSceneVisible,
  isInspectingPastStage,
  isPlannerOperandAvailable,
  renderSelectionUi,
}) {
  function getSelectionEntry(selectionId) {
    const inContractionScene = isContractionSceneVisible();
    const inspectingPastStage = isInspectingPastStage();
    const group = findGroupById(selectionId);
    if (group) {
      return { kind: "group", id: group.id, group };
    }
    const visibleTensor = findVisibleTensorById(selectionId);
    if (visibleTensor && inContractionScene) {
      return {
        kind: visibleTensor.isDerived ? "contraction-tensor" : "tensor",
        id: visibleTensor.id,
        tensor: visibleTensor,
        isBaseTensor: !visibleTensor.isDerived,
      };
    }
    const tensor = findTensorById(selectionId);
    if (tensor && inContractionScene) {
      return null;
    }
    if (tensor) {
      return {
        kind: "tensor",
        id: tensor.id,
        tensor,
        isBaseTensor: true,
      };
    }
    if (visibleTensor) {
      return {
        kind: "contraction-tensor",
        id: visibleTensor.id,
        tensor: visibleTensor,
        isBaseTensor: false,
      };
    }
    const locatedIndex = findIndexOwner(selectionId);
    if (locatedIndex) {
      if (inContractionScene && inspectingPastStage) {
        return null;
      }
      if (inContractionScene) {
        return { kind: "contraction-index", id: selectionId, located: locatedIndex };
      }
      return { kind: "index", id: selectionId, located: locatedIndex };
    }
    const edge = findEdgeById(selectionId);
    if (edge) {
      if (inContractionScene && inspectingPastStage) {
        return null;
      }
      return { kind: "edge", id: selectionId, edge };
    }
    const hyperedge = findHyperedgeById(selectionId);
    if (hyperedge) {
      return { kind: "hyperedge", id: selectionId, hyperedge };
    }
    const note = findNoteById(selectionId);
    if (note) {
      return { kind: "note", id: note.id, note };
    }
    return null;
  }

  function resolveSelectionKind(selectionId) {
    const entry = getSelectionEntry(selectionId);
    return entry ? entry.kind : null;
  }

  function getSelectedEntries() {
    return state.selectionIds
      .map((selectionId) => getSelectionEntry(selectionId))
      .filter(Boolean);
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

  function releaseAutoExpandedTensorIndex(nextSingleSelectionId) {
    const autoExpanded = state.autoExpandedTensorIndex;
    if (!autoExpanded || nextSingleSelectionId === autoExpanded.indexId) {
      return;
    }
    if (!autoExpanded.wasOpen) {
      const disclosureState = state.tensorIndexDisclosureState[autoExpanded.tensorId];
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

  function pruneSelectionToExisting() {
    state.selectionIds = state.selectionIds.filter((selectionId) =>
      Boolean(resolveSelectionKind(selectionId))
    );
    if (!state.selectionIds.includes(state.primarySelectionId)) {
      state.primarySelectionId = state.selectionIds.length
        ? state.selectionIds[state.selectionIds.length - 1]
        : null;
    }
    const pendingIndexKind = state.pendingIndexId
      ? resolveSelectionKind(state.pendingIndexId)
      : null;
    if (
      state.pendingIndexId &&
      pendingIndexKind !== "index" &&
      pendingIndexKind !== "contraction-index"
    ) {
      state.pendingIndexId = null;
    }
    if (
      state.pendingPlannerOperandId &&
      !isPlannerOperandAvailable(state.pendingPlannerOperandId)
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
    state.primarySelectionId = uniqueIds.includes(options.primaryId)
      ? options.primaryId
      : uniqueIds.length
        ? uniqueIds[uniqueIds.length - 1]
        : null;
    updatePendingPropertiesIndexFocus(previousSelectionIds, uniqueIds);
    syncSelectedElementState();
    renderSelectionUi();
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
    renderSelectionUi();
  }

  function selectAllTensors() {
    const tensorIds = getVisibleTensors().map((tensor) => tensor.id);
    setSelection(tensorIds, {
      primaryId: tensorIds.length ? tensorIds[tensorIds.length - 1] : null,
    });
  }

  return {
    getSelectionEntry,
    resolveSelectionKind,
    getSelectedEntries,
    getSelectedIdsByKind,
    syncSelectedElementState,
    releaseAutoExpandedTensorIndex,
    getPropertiesTensorId,
    updatePendingPropertiesIndexFocus,
    pruneSelectionToExisting,
    setSelection,
    selectElement,
    setSelectedElement,
    clearSelection,
    selectAllTensors,
  };
}

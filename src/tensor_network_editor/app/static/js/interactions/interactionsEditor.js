function noop() {}

function resolveContextAction(ctx, name) {
  return (...args) => {
    if (typeof ctx[name] === "function") {
      return ctx[name](...args);
    }
    return undefined;
  };
}

function resolveContextValue(ctx, name, fallback) {
  return typeof ctx[name] === "function" ? ctx[name].bind(ctx) : fallback;
}

export function createInteractionEditorBindings({
  ctx,
  state,
  runtime,
  editorActions = {},
}) {
  const { window } = ctx;
  const resolvedEditorActions = {
    bumpSpecRevision:
      editorActions.bumpSpecRevision || resolveContextAction(ctx, "bumpSpecRevision"),
    enforceLinearPeriodicEngineSupport:
      editorActions.enforceLinearPeriodicEngineSupport ||
      resolveContextAction(ctx, "enforceLinearPeriodicEngineSupport"),
    refreshContractionAnalysis:
      editorActions.refreshContractionAnalysis ||
      resolveContextAction(ctx, "refreshContractionAnalysis"),
    findVisibleTensorById:
      editorActions.findVisibleTensorById ||
      resolveContextValue(ctx, "findVisibleTensorById", (tensorId) =>
        ctx.findTensorById(tensorId)
      ),
    canEditCurrentContractionStage:
      editorActions.canEditCurrentContractionStage ||
      resolveContextValue(ctx, "canEditCurrentContractionStage", () => false),
    updateCurrentStageOperandLayout:
      editorActions.updateCurrentStageOperandLayout ||
      resolveContextAction(ctx, "updateCurrentStageOperandLayout"),
    isInspectingPastStage:
      editorActions.isInspectingPastStage ||
      resolveContextValue(ctx, "isInspectingPastStage", () => false),
    resolveConnectableIndexOwner:
      editorActions.resolveConnectableIndexOwner ||
      resolveContextValue(ctx, "resolveConnectableIndexOwner", (indexId) =>
        ctx.findIndexOwner(indexId)
      ),
    findConnectionByIndexId:
      editorActions.findConnectionByIndexId ||
      resolveContextValue(ctx, "findConnectionByIndexId", (indexId) =>
        ctx.findEdgeByIndexId(indexId)
      ),
    toggleSidebarCollapsed:
      editorActions.toggleSidebarCollapsed ||
      resolveContextAction(ctx, "toggleSidebarCollapsed"),
    setActiveSidebarTab:
      editorActions.setActiveSidebarTab ||
      resolveContextAction(ctx, "setActiveSidebarTab"),
    syncPendingInteractionClasses:
      editorActions.syncPendingInteractionClasses ||
      resolveContextAction(ctx, "syncPendingInteractionClasses"),
    findVisibleEdgeSelectionIdByBaseEdgeId:
      editorActions.findVisibleEdgeSelectionIdByBaseEdgeId ||
      resolveContextValue(ctx, "findVisibleEdgeSelectionIdByBaseEdgeId", (edgeId) =>
        edgeId
      ),
    removeNote: editorActions.removeNote || resolveContextAction(ctx, "removeNote"),
  };
  const {
    bumpSpecRevision = noop,
    enforceLinearPeriodicEngineSupport = noop,
    refreshContractionAnalysis = noop,
    findVisibleTensorById = (tensorId) => ctx.findTensorById(tensorId),
    canEditCurrentContractionStage = () => false,
    updateCurrentStageOperandLayout = noop,
    isInspectingPastStage = () => false,
    resolveConnectableIndexOwner = (indexId) => ctx.findIndexOwner(indexId),
    findConnectionByIndexId = (indexId) => ctx.findEdgeByIndexId(indexId),
    toggleSidebarCollapsed = noop,
    setActiveSidebarTab = noop,
    syncPendingInteractionClasses = noop,
    findVisibleEdgeSelectionIdByBaseEdgeId = (edgeId) => edgeId,
    removeNote = noop,
  } = resolvedEditorActions;

  function isForBoundaryTensor(tensor) {
    return (
      (typeof ctx.isForBoundaryTensor === "function" && ctx.isForBoundaryTensor(tensor)) ||
      ctx.isLinearPeriodicBoundaryTensor(tensor)
    );
  }

  function handleNewDesign() {
    if (
      !window.confirm(
        "Start a new design? Unsaved changes in this browser tab will be lost."
      )
    ) {
      return;
    }

    resetDesignState(
      {
        id: ctx.makeId("network"),
        name: "Untitled Network",
        tensors: [],
        groups: [],
        edges: [],
        hyperedges: [],
        notes: [],
        contraction_plan: null,
        linear_periodic_chain: null,
        metadata: {},
      },
      "Started a new empty design. History cleared."
    );
  }

  function resetDesignState(spec, message, schemaVersion = state.schemaVersion) {
    state.spec = ctx.normalizeSpec(spec);
    bumpSpecRevision();
    state.schemaVersion = schemaVersion;
    state.generatedCode = "";
    state.lastImportedTensorIds = [];
    state.activeSidebarTab = "selection";
    state.selectionIds = [];
    state.primarySelectionId = null;
    state.selectedElement = null;
    state.pendingIndexId = null;
    state.pendingPropertiesIndexFocusId = null;
    state.tensorIndexDisclosureState = {};
    state.autoExpandedTensorIndex = null;
    state.pendingPlannerOperandId = null;
    state.pendingPlannerSelectionId = null;
    state.plannerInspectionStepCount = null;
    state.connectMode = false;
    state.plannerMode = false;
    state.hasFitCanvas = false;
    state.activeResize = null;
    state.activeGroupDrag = null;
    state.noteDragState = null;
    state.activeNoteResize = null;
    state.contractionAnalysis = null;
    state.benchmarkSession = {
      enabled: false,
      activePosition: 0,
      originalPlan: null,
      schemes: [],
      compareModal: {
        open: false,
        loading: false,
        errorMessage: "",
        tableModel: null,
        rows: [],
        activeRequestId: 0,
      },
    };
    state.templateCatalogWarnings = Array.isArray(state.templateCatalogWarnings)
      ? [...state.templateCatalogWarnings]
      : [];
    state.plannerPreviewMode = null;
    state.plannerFutureBadgeDisclosure = {};
    state.plannerPreviewBadgeDisclosure = {};
    enforceLinearPeriodicEngineSupport();
    ctx.reconcileTensorOrder();
    ctx.clearHistory();
    ctx.render();
    refreshContractionAnalysis();
    ctx.setStatus(message, "success");
  }

  function viewportCenterPosition() {
    if (!state.cy) {
      return { x: 240, y: 200 };
    }
    const zoom = state.cy.zoom();
    const pan = state.cy.pan();
    return {
      x: Math.round((state.cy.width() / 2 - pan.x) / zoom),
      y: Math.round((state.cy.height() / 2 - pan.y) / zoom),
    };
  }

  function isTensorPositionOccupied(candidate) {
    return state.spec.tensors.some((tensor) => {
      return (
        Math.abs(tensor.position.x - candidate.x) <
          Math.max(170, ctx.tensorWidth(tensor) * 0.8) &&
        Math.abs(tensor.position.y - candidate.y) <
          Math.max(120, ctx.tensorHeight(tensor) * 0.8)
      );
    });
  }

  function suggestTensorPosition(center) {
    const offsets = [
      { x: 0, y: 0 },
      { x: 220, y: 0 },
      { x: -220, y: 0 },
      { x: 0, y: 170 },
      { x: 0, y: -170 },
      { x: 220, y: 170 },
      { x: 220, y: -170 },
      { x: -220, y: 170 },
      { x: -220, y: -170 },
      { x: 440, y: 0 },
      { x: -440, y: 0 },
      { x: 0, y: 340 },
      { x: 0, y: -340 },
    ];

    for (const offset of offsets) {
      const candidate = { x: center.x + offset.x, y: center.y + offset.y };
      if (!isTensorPositionOccupied(candidate)) {
        return candidate;
      }
    }

    return {
      x: center.x + state.spec.tensors.length * 36,
      y: center.y + state.spec.tensors.length * 28,
    };
  }

  function addTensorAtCenter() {
    const center = viewportCenterPosition();
    const suggestedPosition = suggestTensorPosition(center);
    const tensor = ctx.createTensor(suggestedPosition.x, suggestedPosition.y);
    ctx.applyDesignChange(
      () => {
        state.spec.tensors.push(tensor);
        ctx.bringTensorToFront(tensor.id);
      },
      {
        selectionIds: [tensor.id],
        primaryId: tensor.id,
        statusMessage: `Added tensor ${tensor.name}.`,
      }
    );
  }

  function centerTensor(tensorId) {
    const tensor = findVisibleTensorById(tensorId);
    if (!tensor) {
      return;
    }
    const center = viewportCenterPosition();
    if (canEditCurrentContractionStage()) {
      updateCurrentStageOperandLayout(tensor.id, { position: center });
      tensor.position.x = center.x;
      tensor.position.y = center.y;
      return;
    }
    const baseTensor = ctx.findTensorById(tensorId);
    if (!baseTensor) {
      return;
    }
    baseTensor.position.x = center.x;
    baseTensor.position.y = center.y;
  }

  function toggleConnectMode() {
    if (!state.connectMode && isInspectingPastStage()) {
      ctx.setStatus(
        "Return to the latest contraction step before editing ports.",
        "error"
      );
      return;
    }
    state.connectMode = !state.connectMode;
    state.pendingIndexId = null;
    ctx.render();
    ctx.setStatus(
      state.connectMode
        ? "Connect mode active. Click two open indices with the same dimension."
        : "Connect mode disabled."
    );
  }

  function handleConnectClick(indexId) {
    if (findConnectionByIndexId(indexId)) {
      ctx.setStatus(
        "This index is already connected. Delete the connection first.",
        "error"
      );
      return;
    }
    const located = resolveConnectableIndexOwner(indexId);
    if (!located) {
      ctx.setStatus(
        "This port is not available for new connections in the current view.",
        "error"
      );
      return;
    }
    if (findConnectionByIndexId(located.index.id)) {
      ctx.setStatus(
        "This index is already connected. Delete the connection first.",
        "error"
      );
      return;
    }

    if (!state.pendingIndexId) {
      state.pendingIndexId = indexId;
      toggleSidebarCollapsed(false);
      setActiveSidebarTab("selection");
      ctx.setSelectedElement("index", indexId);
      syncPendingInteractionClasses();
      ctx.renderOverlayDecorations();
      ctx.setStatus(
        "First index selected. Click another compatible open index to connect."
      );
      return;
    }

    if (state.pendingIndexId === indexId) {
      state.pendingIndexId = null;
      syncPendingInteractionClasses();
      ctx.renderOverlayDecorations();
      ctx.setStatus("Connection cancelled.");
      return;
    }

    const left = resolveConnectableIndexOwner(state.pendingIndexId);
    if (!left) {
      state.pendingIndexId = null;
      syncPendingInteractionClasses();
      ctx.renderOverlayDecorations();
      return;
    }
    if (left.index.dimension !== located.index.dimension) {
      ctx.setStatus("Connected indices must have the same dimension.", "error");
      return;
    }
    if (
      isForBoundaryTensor(left.tensor) &&
      isForBoundaryTensor(located.tensor)
    ) {
      ctx.setStatus(
        "Virtual boundary tensors can only connect to real tensors inside the current cell.",
        "error"
      );
      return;
    }

    const newEdgeId = ctx.makeId("edge");
    state.pendingIndexId = null;
    ctx.applyDesignChange(
      () => {
        state.spec.edges.push({
          id: newEdgeId,
          name: ctx.nextName("bond", state.spec.edges.map((edge) => edge.name)),
          left: { tensor_id: left.tensor.id, index_id: left.index.id },
          right: { tensor_id: located.tensor.id, index_id: located.index.id },
          metadata: {},
        });
      },
      {
        selectionIds: [findVisibleEdgeSelectionIdByBaseEdgeId(newEdgeId)].filter(
          Boolean
        ),
        primaryId: findVisibleEdgeSelectionIdByBaseEdgeId(newEdgeId),
        statusMessage: "Connection created.",
      }
    );
  }

  function removeSelectedElements() {
    const selectedEntries = ctx.getSelectedEntries();
    const contractionTensorSourceIds = [
      ...new Set(
        selectedEntries
          .filter((entry) => entry.kind === "contraction-tensor")
          .flatMap((entry) =>
            Array.isArray(entry.tensor && entry.tensor.sourceTensorIds)
              ? entry.tensor.sourceTensorIds
              : []
          )
      ),
    ];
    const selectedTensorIds = new Set([
      ...ctx.getSelectedIdsByKind("tensor"),
      ...contractionTensorSourceIds,
    ]);
    const selectedIndexIds = new Set(ctx.getSelectedIdsByKind("index"));
    const selectedEdgeIds = new Set(ctx.getSelectedIdsByKind("edge"));
    const selectedHyperedgeIds = new Set(ctx.getSelectedIdsByKind("hyperedge"));
    const selectedGroupIds = new Set(ctx.getSelectedIdsByKind("group"));
    const selectedNoteIds = new Set(ctx.getSelectedIdsByKind("note"));

    selectedTensorIds.forEach((tensorId) => {
      const tensor = ctx.findTensorById(tensorId);
      if (!isForBoundaryTensor(tensor)) {
        ctx.removeTensor(tensorId);
      }
    });

    selectedIndexIds.forEach((indexId) => {
      const located = ctx.findIndexOwner(indexId);
      if (
        located &&
        !selectedTensorIds.has(located.tensor.id) &&
        !isForBoundaryTensor(located.tensor)
      ) {
        ctx.removeIndex(located.tensor.id, indexId);
      }
    });

    selectedEdgeIds.forEach((edgeId) => {
      if (ctx.findEdgeById(edgeId)) {
        ctx.removeEdge(edgeId);
      }
    });

    selectedHyperedgeIds.forEach((hyperedgeId) => {
      if (typeof ctx.findHyperedgeById === "function" && ctx.findHyperedgeById(hyperedgeId)) {
        ctx.removeHyperedge(hyperedgeId);
      }
    });

    if (selectedGroupIds.size) {
      state.spec.groups = state.spec.groups.filter(
        (group) => !selectedGroupIds.has(group.id)
      );
    }

    selectedNoteIds.forEach((noteId) => {
      removeNote(noteId);
    });
  }

  function deleteSelection() {
    if (!state.selectionIds.length) {
      ctx.setStatus("Nothing is selected to delete.");
      return;
    }
    const selectedEntries = ctx.getSelectedEntries();
    const hasMutableSelection = selectedEntries.some(
      (entry) =>
        (entry.kind === "tensor" &&
          !isForBoundaryTensor(entry.tensor)) ||
        (entry.kind === "index" &&
          !isForBoundaryTensor(entry.located.tensor)) ||
        entry.kind === "contraction-tensor" ||
        entry.kind === "edge" ||
        entry.kind === "hyperedge" ||
        entry.kind === "group" ||
        entry.kind === "note"
    );
    if (!hasMutableSelection) {
      ctx.setStatus("Contracted result tensors are view-only in this scene.", "error");
      return;
    }
    ctx.applyDesignChange(
      () => {
        removeSelectedElements();
      },
      {
        selectionIds: [],
        statusMessage: "Selection deleted.",
      }
    );
  }

  function getKeyboardNudgeDistance(options = {}) {
    const baseDistance =
      Number.isFinite(ctx.constants?.GRID_SNAP_SIZE) && ctx.constants.GRID_SNAP_SIZE > 0
        ? ctx.constants.GRID_SNAP_SIZE
        : 20;
    return options.fast ? baseDistance * 3 : baseDistance;
  }

  function getKeyboardNudgeDelta(direction, options = {}) {
    const distance = getKeyboardNudgeDistance(options);
    if (direction === "left") {
      return { x: -distance, y: 0 };
    }
    if (direction === "right") {
      return { x: distance, y: 0 };
    }
    if (direction === "up") {
      return { x: 0, y: -distance };
    }
    if (direction === "down") {
      return { x: 0, y: distance };
    }
    return null;
  }

  function nudgeSelectedElements(direction, options = {}) {
    if (!Array.isArray(state.selectionIds) || !state.selectionIds.length) {
      return false;
    }
    const delta = getKeyboardNudgeDelta(direction, options);
    if (!delta) {
      return false;
    }
    const selectedEntries =
      typeof ctx.getSelectedEntries === "function" ? ctx.getSelectedEntries() : [];
    if (!selectedEntries.length) {
      return false;
    }
    const anchorSelectionId =
      state.primarySelectionId || state.selectionIds[state.selectionIds.length - 1] || null;
    const dragSelection =
      anchorSelectionId && typeof ctx.buildCanvasSelectionDragState === "function"
        ? ctx.buildCanvasSelectionDragState(anchorSelectionId)
        : null;
    const draggedTensorIds = Array.isArray(dragSelection?.tensorIds)
      ? dragSelection.tensorIds
      : [];
    const draggedNoteIds = Array.isArray(dragSelection?.noteIds)
      ? dragSelection.noteIds
      : [];
    const movableIndexEntries = selectedEntries.filter(
      (entry) =>
        entry.kind === "index" &&
        entry.located &&
        !draggedTensorIds.includes(entry.located.tensor.id)
    );
    const hasDragMove = draggedTensorIds.length > 0 || draggedNoteIds.length > 0;
    if (!hasDragMove && !movableIndexEntries.length) {
      return false;
    }

    const snapshot =
      dragSelection?.snapshot ||
      (typeof ctx.createHistorySnapshot === "function"
        ? ctx.createHistorySnapshot()
        : null);
    let changed = false;

    if (hasDragMove && typeof ctx.applyCanvasSelectionDragDelta === "function") {
      ctx.applyCanvasSelectionDragDelta(dragSelection, delta.x, delta.y);
      changed =
        draggedTensorIds.some((tensorId) => {
          const tensor =
            typeof ctx.findVisibleTensorById === "function"
              ? ctx.findVisibleTensorById(tensorId)
              : ctx.findTensorById(tensorId);
          const startPosition = dragSelection.tensorStartPositions[tensorId];
          return (
            tensor &&
            startPosition &&
            (tensor.position.x !== startPosition.x || tensor.position.y !== startPosition.y)
          );
        }) ||
        draggedNoteIds.some((noteId) => {
          const note = typeof ctx.findNoteById === "function" ? ctx.findNoteById(noteId) : null;
          const startPosition = dragSelection.noteStartPositions[noteId];
          return (
            note &&
            startPosition &&
            (note.position.x !== startPosition.x || note.position.y !== startPosition.y)
          );
        });
    }

    const moveIndices = () => {
      movableIndexEntries.forEach((entry) => {
        const located =
          typeof ctx.findIndexOwner === "function" ? ctx.findIndexOwner(entry.id) : null;
        if (!located) {
          return;
        }
        const nextOffset =
          typeof ctx.clampIndexOffset === "function"
            ? ctx.clampIndexOffset(
                {
                  x: located.index.offset.x + delta.x,
                  y: located.index.offset.y + delta.y,
                },
                located.tensor
              )
            : {
                x: located.index.offset.x + delta.x,
                y: located.index.offset.y + delta.y,
              };
        if (
          nextOffset.x === located.index.offset.x &&
          nextOffset.y === located.index.offset.y
        ) {
          return;
        }
        located.index.offset = nextOffset;
        changed = true;
        if (typeof ctx.syncSingleIndexNodePosition === "function") {
          ctx.syncSingleIndexNodePosition(located.tensor, located.index);
        }
      });
    };

    if (movableIndexEntries.length) {
      if (typeof ctx.runWithIndexSync === "function") {
        ctx.runWithIndexSync(moveIndices);
      } else {
        moveIndices();
      }
    }

    if (!changed || !snapshot || typeof ctx.commitHistorySnapshot !== "function") {
      return changed;
    }
    ctx.commitHistorySnapshot(snapshot);
    if (typeof ctx.renderOverlayDecorations === "function") {
      ctx.renderOverlayDecorations();
    }
    if (typeof ctx.renderMinimap === "function") {
      ctx.renderMinimap();
    }
    if (typeof ctx.renderProperties === "function") {
      ctx.renderProperties();
    }
    if (typeof ctx.updateToolbarState === "function") {
      ctx.updateToolbarState();
    }
    return true;
  }

  return {
    handleNewDesign,
    resetDesignState,
    addTensorAtCenter,
    viewportCenterPosition,
    suggestTensorPosition,
    isTensorPositionOccupied,
    centerTensor,
    toggleConnectMode,
    handleConnectClick,
    deleteSelection,
    removeSelectedElements,
    nudgeSelectedElements,
  };
}

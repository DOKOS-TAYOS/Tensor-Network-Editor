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
    toggleSidebarCollapsed = noop,
    setActiveSidebarTab = noop,
    syncPendingInteractionClasses = noop,
    findVisibleEdgeSelectionIdByBaseEdgeId = (edgeId) => edgeId,
    removeNote = noop,
  } = resolvedEditorActions;

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
    state.templateCatalogWarnings = Array.isArray(state.templateCatalogWarnings)
      ? [...state.templateCatalogWarnings]
      : [];
    state.plannerPreviewMode = null;
    state.plannerFutureBadgeDisclosure = {};
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
    if (ctx.findEdgeByIndexId(indexId)) {
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
    if (ctx.findEdgeByIndexId(located.index.id)) {
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
      ctx.isLinearPeriodicBoundaryTensor(left.tensor) &&
      ctx.isLinearPeriodicBoundaryTensor(located.tensor)
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
    const selectedGroupIds = new Set(ctx.getSelectedIdsByKind("group"));
    const selectedNoteIds = new Set(ctx.getSelectedIdsByKind("note"));

    selectedTensorIds.forEach((tensorId) => {
      const tensor = ctx.findTensorById(tensorId);
      if (!ctx.isLinearPeriodicBoundaryTensor(tensor)) {
        ctx.removeTensor(tensorId);
      }
    });

    selectedIndexIds.forEach((indexId) => {
      const located = ctx.findIndexOwner(indexId);
      if (
        located &&
        !selectedTensorIds.has(located.tensor.id) &&
        !ctx.isLinearPeriodicBoundaryTensor(located.tensor)
      ) {
        ctx.removeIndex(located.tensor.id, indexId);
      }
    });

    selectedEdgeIds.forEach((edgeId) => {
      if (ctx.findEdgeById(edgeId)) {
        ctx.removeEdge(edgeId);
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
          !ctx.isLinearPeriodicBoundaryTensor(entry.tensor)) ||
        (entry.kind === "index" &&
          !ctx.isLinearPeriodicBoundaryTensor(entry.located.tensor)) ||
        entry.kind === "contraction-tensor" ||
        entry.kind === "edge" ||
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
  };
}

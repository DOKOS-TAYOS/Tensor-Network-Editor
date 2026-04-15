export function createInteractionEditorBindings({ ctx, state, runtime }) {
  const { window } = ctx;

  function handleNewDesign() {
    if (!window.confirm("Start a new design? Unsaved changes in this browser tab will be lost.")) {
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
    if (typeof ctx.bumpSpecRevision === "function") {
      ctx.bumpSpecRevision();
    }
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
    if (typeof ctx.enforceLinearPeriodicEngineSupport === "function") {
      ctx.enforceLinearPeriodicEngineSupport();
    }
    ctx.reconcileTensorOrder();
    ctx.clearHistory();
    ctx.render();
    if (typeof ctx.refreshContractionAnalysis === "function") {
      ctx.refreshContractionAnalysis();
    }
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
        Math.abs(tensor.position.x - candidate.x) < Math.max(170, ctx.tensorWidth(tensor) * 0.8) &&
        Math.abs(tensor.position.y - candidate.y) < Math.max(120, ctx.tensorHeight(tensor) * 0.8)
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
    const tensor =
      typeof ctx.findVisibleTensorById === "function"
        ? ctx.findVisibleTensorById(tensorId)
        : ctx.findTensorById(tensorId);
    if (!tensor) {
      return;
    }
    const center = viewportCenterPosition();
    if (
      typeof ctx.canEditCurrentContractionStage === "function" &&
      ctx.canEditCurrentContractionStage() &&
      typeof ctx.updateCurrentStageOperandLayout === "function"
    ) {
      ctx.updateCurrentStageOperandLayout(tensor.id, { position: center });
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
    if (
      !state.connectMode &&
      typeof ctx.isInspectingPastStage === "function" &&
      ctx.isInspectingPastStage()
    ) {
      ctx.setStatus("Return to the latest contraction step before editing ports.", "error");
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
      ctx.setStatus("This index is already connected. Delete the connection first.", "error");
      return;
    }
    const located =
      typeof ctx.resolveConnectableIndexOwner === "function"
        ? ctx.resolveConnectableIndexOwner(indexId)
        : ctx.findIndexOwner(indexId);
    if (!located) {
      ctx.setStatus(
        "This port is not available for new connections in the current view.",
        "error"
      );
      return;
    }
    if (ctx.findEdgeByIndexId(located.index.id)) {
      ctx.setStatus("This index is already connected. Delete the connection first.", "error");
      return;
    }

    if (!state.pendingIndexId) {
      state.pendingIndexId = indexId;
      if (typeof ctx.toggleSidebarCollapsed === "function") {
        ctx.toggleSidebarCollapsed(false);
      }
      if (typeof ctx.setActiveSidebarTab === "function") {
        ctx.setActiveSidebarTab("selection");
      }
      ctx.setSelectedElement("index", indexId);
      if (typeof ctx.syncPendingInteractionClasses === "function") {
        ctx.syncPendingInteractionClasses();
      }
      ctx.renderOverlayDecorations();
      ctx.setStatus("First index selected. Click another compatible open index to connect.");
      return;
    }

    if (state.pendingIndexId === indexId) {
      state.pendingIndexId = null;
      if (typeof ctx.syncPendingInteractionClasses === "function") {
        ctx.syncPendingInteractionClasses();
      }
      ctx.renderOverlayDecorations();
      ctx.setStatus("Connection cancelled.");
      return;
    }

    const left =
      typeof ctx.resolveConnectableIndexOwner === "function"
        ? ctx.resolveConnectableIndexOwner(state.pendingIndexId)
        : ctx.findIndexOwner(state.pendingIndexId);
    if (!left) {
      state.pendingIndexId = null;
      if (typeof ctx.syncPendingInteractionClasses === "function") {
        ctx.syncPendingInteractionClasses();
      }
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
        selectionIds: [
          typeof ctx.findVisibleEdgeSelectionIdByBaseEdgeId === "function"
            ? ctx.findVisibleEdgeSelectionIdByBaseEdgeId(newEdgeId)
            : newEdgeId,
        ].filter(Boolean),
        primaryId:
          typeof ctx.findVisibleEdgeSelectionIdByBaseEdgeId === "function"
            ? ctx.findVisibleEdgeSelectionIdByBaseEdgeId(newEdgeId)
            : newEdgeId,
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
      state.spec.groups = state.spec.groups.filter((group) => !selectedGroupIds.has(group.id));
    }

    selectedNoteIds.forEach((noteId) => {
      if (typeof ctx.removeNote === "function") {
        ctx.removeNote(noteId);
      }
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
        (entry.kind === "tensor" && !ctx.isLinearPeriodicBoundaryTensor(entry.tensor)) ||
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

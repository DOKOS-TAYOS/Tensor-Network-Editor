export function registerInteractions(ctx) {
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
    workspace,
    statusMessage,
    propertiesPanel,
    generatedCode,
    engineSelect,
    collectionFormatSelect,
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

  function handleCanvasContextMenu(event) {
    event.preventDefault();
  }

  function handleCanvasWheel(event) {
    if (!state.cy || state.isHelpOpen) {
      return;
    }
    if (event.ctrlKey || event.metaKey) {
      event.preventDefault();
      event.stopPropagation();
      const container = state.cy.container();
      const rect = container.getBoundingClientRect();
      const renderedPosition = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
      const zoomFactor = Math.exp(-event.deltaY * 0.0025);
      const nextZoom = ctx.clamp(
        state.cy.zoom() * zoomFactor,
        state.cy.minZoom(),
        state.cy.maxZoom()
      );
      state.cy.zoom({
        level: nextZoom,
        renderedPosition,
      });
      ctx.renderOverlayDecorations();
      ctx.renderMinimap();
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    state.cy.panBy({
      x: -event.deltaX,
      y: -event.deltaY,
    });
    ctx.renderOverlayDecorations();
    ctx.renderMinimap();
  }

  function handleCanvasMouseDown(event) {
    if (state.isHelpOpen) {
      return;
    }
    if (
      event.target.closest(".resize-handle") ||
      event.target.closest(".group-overlay") ||
      event.target.closest(".canvas-note")
    ) {
      return;
    }
    if (event.button === 2) {
      event.preventDefault();
      event.stopPropagation();
      startBoxSelection(event);
    }
  }

  function handleGlobalMouseMove(event) {
    if (state.boxSelection) {
      updateBoxSelection(event);
      return;
    }
    if (state.activeResize) {
      ctx.updateActiveResize(event);
      return;
    }
    if (state.activeGroupDrag) {
      ctx.updateActiveGroupDrag(event);
      return;
    }
    if (state.noteDragState) {
      ctx.updateActiveNoteDrag(event);
      return;
    }
    if (state.activeNoteResize) {
      ctx.updateActiveNoteResize(event);
      return;
    }
    if (state.minimapDrag) {
      ctx.updateViewportFromMinimapClientPoint(event.clientX, event.clientY);
    }
  }

  function handleGlobalMouseUp(event) {
    if (state.boxSelection && event.button === 2) {
      finishBoxSelection(false);
      return;
    }
    if (state.activeResize && event.button === 0) {
      ctx.finishActiveResize();
      return;
    }
    if (state.activeGroupDrag && event.button === 0) {
      ctx.finishActiveGroupDrag();
      return;
    }
    if (state.noteDragState && event.button === 0) {
      ctx.finishActiveNoteDrag();
      return;
    }
    if (state.activeNoteResize && event.button === 0) {
      ctx.finishActiveNoteResize();
      return;
    }
    if (state.minimapDrag && event.button === 0) {
      state.minimapDrag = null;
      minimapCanvas.classList.remove("is-dragging");
    }
  }

  function startBoxSelection(event) {
    const point = ctx.clientPointToCanvasPoint(event.clientX, event.clientY);
    state.boxSelection = {
      start: point,
      current: point,
      additive: Boolean(event.shiftKey),
    };
    updateSelectionBoxElement();
  }

  function updateBoxSelection(event) {
    state.boxSelection.current = ctx.clientPointToCanvasPoint(event.clientX, event.clientY);
    updateSelectionBoxElement();
  }

  function finishBoxSelection(cancelled) {
    if (!state.boxSelection) {
      return;
    }
    const boxSelectionState = state.boxSelection;
    state.boxSelection = null;
    selectionBox.classList.add("is-hidden");
    if (cancelled || !state.cy) {
      return;
    }
    const box = ctx.normalizedBox(boxSelectionState.start, boxSelectionState.current);
    const hitIds = state.cy
      .elements("node, edge")
      .toArray()
      .filter((element) => element.data("kind") !== "index-label")
      .filter((element) => ctx.boxesIntersect(box, element.renderedBoundingBox()))
      .map((element) => element.id());
    const hitNoteIds = collectBoxSelectedNoteIds(box);
    const selectionIds = [...hitIds, ...hitNoteIds];
    if (boxSelectionState.additive) {
      ctx.setSelection([...state.selectionIds, ...selectionIds], {
        primaryId: selectionIds.length
          ? selectionIds[selectionIds.length - 1]
          : state.primarySelectionId,
      });
      return;
    }
    ctx.setSelection(selectionIds, {
      primaryId: selectionIds.length ? selectionIds[selectionIds.length - 1] : null,
    });
  }

  function collectBoxSelectedNoteIds(box) {
    if (
      !state.spec ||
      !Array.isArray(state.spec.notes) ||
      typeof ctx.noteCanvasBounds !== "function"
    ) {
      return [];
    }
    return state.spec.notes
      .filter((note) => ctx.boxesIntersect(box, ctx.noteCanvasBounds(note)))
      .map((note) => note.id);
  }

  function updateSelectionBoxElement() {
    if (!state.boxSelection) {
      selectionBox.classList.add("is-hidden");
      return;
    }
    const box = ctx.normalizedBox(state.boxSelection.start, state.boxSelection.current);
    selectionBox.classList.remove("is-hidden");
    selectionBox.style.left = `${box.left}px`;
    selectionBox.style.top = `${box.top}px`;
    selectionBox.style.width = `${Math.max(1, box.width)}px`;
    selectionBox.style.height = `${Math.max(1, box.height)}px`;
  }

  function openSidebarTab(tabName) {
    if (typeof ctx.toggleSidebarCollapsed === "function") {
      ctx.toggleSidebarCollapsed(false);
    }
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab(tabName);
    }
  }

  function setSelectedEngine(engineName) {
    if (!engineSelect) {
      return;
    }
    const hasEngine = Array.from(engineSelect.options).some(
      (option) => option.value === engineName
    );
    if (!hasEngine) {
      ctx.setStatus(`The ${engineName} engine is not available in this session.`, "error");
      return;
    }
    state.selectedEngine = engineName;
    engineSelect.value = engineName;
    ctx.setStatus(`Engine set to ${ctx.formatEngineLabel(engineName)}.`);
  }

  function toggleAutomaticPreview(mode) {
    openSidebarTab("planner");
    if (typeof ctx.startAutomaticPreview === "function") {
      ctx.startAutomaticPreview(mode);
    }
  }

  function acceptAutomaticShortcut(mode) {
    openSidebarTab("planner");
    if (typeof ctx.acceptAutomaticPlan === "function") {
      ctx.acceptAutomaticPlan(mode);
    }
  }

  function toggleSidebarVisibility() {
    if (typeof ctx.toggleSidebarCollapsed !== "function") {
      return;
    }
    ctx.toggleSidebarCollapsed();
    ctx.setStatus(
      state.sidebarCollapsed ? "Sidebar collapsed." : "Sidebar expanded."
    );
  }

  function toggleMinimapVisibility() {
    if (typeof ctx.toggleMinimapVisibility !== "function") {
      return;
    }
    ctx.toggleMinimapVisibility();
    ctx.setStatus(
      state.minimapHidden ? "Minimap hidden." : "Minimap shown."
    );
  }

  function handleKeydown(event) {
    const activeElement = document.activeElement;
    const inTextInput =
      ctx.isTextInput(event.target) || ctx.isTextInput(activeElement);
    const lowerKey = event.key.toLowerCase();

    if (event.key === "Escape") {
      event.preventDefault();
      if (state.isHelpOpen) {
        toggleHelpModal(false);
        return;
      }
      if (state.boxSelection) {
        finishBoxSelection(true);
        return;
      }
      if (state.connectMode) {
        state.pendingIndexId = null;
        state.connectMode = false;
        if (typeof ctx.syncPendingInteractionClasses === "function") {
          ctx.syncPendingInteractionClasses();
        }
        ctx.render();
        ctx.setStatus("Connect mode cancelled.");
        return;
      }
      if (state.pendingPlannerOperandId) {
        state.pendingPlannerOperandId = null;
        state.pendingPlannerSelectionId = null;
        if (typeof ctx.syncPendingInteractionClasses === "function") {
          ctx.syncPendingInteractionClasses();
        }
        if (typeof ctx.renderPlanner === "function") {
          ctx.renderPlanner();
        }
        ctx.renderOverlayDecorations();
        ctx.setStatus("Manual planner operand selection cleared.");
        return;
      }
      if (state.plannerPreviewMode) {
        if (typeof ctx.clearAutomaticPreview === "function") {
          ctx.clearAutomaticPreview();
        } else {
          state.plannerPreviewMode = null;
        }
        ctx.render();
        ctx.setStatus("Automatic preview cleared.");
        return;
      }
      if (
        Number.isInteger(state.plannerInspectionStepCount) &&
        typeof ctx.clearPastInspection === "function"
      ) {
        ctx.clearPastInspection();
        ctx.render();
        ctx.setStatus("Returned to the latest contracted view.");
        return;
      }
      ctx.clearSelection();
      return;
    }

    if (inTextInput) {
      return;
    }

    const hasModifier = event.ctrlKey || event.metaKey;
    if (hasModifier && lowerKey === "z") {
      event.preventDefault();
      if (event.shiftKey) {
        ctx.performRedo();
      } else {
        ctx.performUndo();
      }
      return;
    }
    if (hasModifier && event.shiftKey && lowerKey === "a") {
      event.preventDefault();
      acceptAutomaticShortcut("automaticPast");
      return;
    }
    if (hasModifier && lowerKey === "a") {
      event.preventDefault();
      acceptAutomaticShortcut("automaticFuture");
      return;
    }
    if (hasModifier && lowerKey === "s") {
      event.preventDefault();
      saveDesign();
      return;
    }
    if (hasModifier && lowerKey === "l") {
      event.preventDefault();
      loadInput.click();
      return;
    }
    if (hasModifier && lowerKey === "y") {
      event.preventDefault();
      setSelectedEngine("einsum_numpy");
      return;
    }
    if (hasModifier && lowerKey === "p") {
      event.preventDefault();
      setSelectedEngine("einsum_torch");
      return;
    }
    if (hasModifier && lowerKey === "k") {
      event.preventDefault();
      setSelectedEngine("tensorkrowch");
      return;
    }
    if (hasModifier && lowerKey === "q") {
      event.preventDefault();
      setSelectedEngine("quimb");
      return;
    }
    if (hasModifier && lowerKey === "t") {
      event.preventDefault();
      setSelectedEngine("tensornetwork");
      return;
    }
    if (hasModifier && lowerKey === "c") {
      event.preventDefault();
      if (typeof ctx.copySelectedSubgraphToClipboard === "function") {
        ctx.copySelectedSubgraphToClipboard();
      }
      return;
    }
    if (hasModifier && lowerKey === "v") {
      event.preventDefault();
      if (typeof ctx.pasteClipboardToCanvas === "function") {
        ctx.pasteClipboardToCanvas();
      }
      return;
    }
    if (event.shiftKey && lowerKey === "r") {
      event.preventDefault();
      if (typeof ctx.trimContractionPlan === "function" && state.spec.contraction_plan) {
        ctx.trimContractionPlan(0);
      } else {
        ctx.setStatus("There is no contraction path to reset.");
      }
      return;
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      deleteSelection();
      return;
    }
    if (event.shiftKey && lowerKey === "a") {
      event.preventDefault();
      toggleAutomaticPreview("automaticPast");
      return;
    }
    if (event.shiftKey && lowerKey === "m") {
      event.preventDefault();
      toggleMinimapVisibility();
      return;
    }
    if (lowerKey === "a") {
      event.preventDefault();
      toggleAutomaticPreview("automaticFuture");
      return;
    }
    if (lowerKey === "m") {
      event.preventDefault();
      openSidebarTab("planner");
      if (typeof ctx.togglePlannerMode === "function") {
        ctx.togglePlannerMode();
      }
      return;
    }
    if (event.shiftKey && lowerKey === "g") {
      event.preventDefault();
      generateCode();
      return;
    }
    if (lowerKey === "s") {
      event.preventDefault();
      toggleSidebarVisibility();
      return;
    }
    if (lowerKey === "g") {
      event.preventDefault();
      if (typeof ctx.createGroupFromSelection === "function") {
        ctx.createGroupFromSelection();
      }
      return;
    }
    if (lowerKey === "t") {
      event.preventDefault();
      insertTemplate();
      return;
    }
    if (lowerKey === "p") {
      event.preventDefault();
      if (typeof ctx.addNoteAtCenter === "function") {
        ctx.addNoteAtCenter();
      }
      return;
    }
    if (lowerKey === "f") {
      event.preventDefault();
      if (typeof ctx.toggleLinearPeriodicMode === "function") {
        ctx.toggleLinearPeriodicMode();
      }
      return;
    }
    if (lowerKey === "n") {
      event.preventDefault();
      addTensorAtCenter();
      return;
    }
    if (lowerKey === "c") {
      event.preventDefault();
      toggleConnectMode();
      return;
    }
    if (event.key === "?") {
      event.preventDefault();
      toggleHelpModal(true);
    }
  }

  function toggleHelpModal(forceOpen) {
    state.isHelpOpen = typeof forceOpen === "boolean" ? forceOpen : !state.isHelpOpen;
    helpModal.classList.toggle("is-hidden", !state.isHelpOpen);
    if (state.isHelpOpen) {
      helpCloseButton.focus();
    }
  }

  function sendCancelBeacon() {
    if (state.editorFinished || !navigator.sendBeacon) {
      return;
    }
    const payload = new Blob([JSON.stringify({})], { type: "application/json" });
    navigator.sendBeacon("/api/cancel", payload);
  }

  function handleWindowResize() {
    if (state.cy) {
      state.cy.resize();
    }
    ctx.renderOverlayDecorations();
    ctx.renderMinimap();
  }

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

  function isTensorPositionOccupied(candidate) {
    return state.spec.tensors.some((tensor) => {
      return (
        Math.abs(tensor.position.x - candidate.x) < Math.max(170, ctx.tensorWidth(tensor) * 0.8) &&
        Math.abs(tensor.position.y - candidate.y) < Math.max(120, ctx.tensorHeight(tensor) * 0.8)
      );
    });
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

  function removeSelectedElements() {
    const selectedTensorIds = new Set(ctx.getSelectedIdsByKind("tensor"));
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

  async function generateCode() {
    if (typeof ctx.toggleSidebarCollapsed === "function") {
      ctx.toggleSidebarCollapsed(false);
    }
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("code");
    }
    try {
      const payload = await apiPost("/api/generate", {
        engine: state.selectedEngine,
        collection_format: state.selectedCollectionFormat,
        spec: ctx.serializeCurrentSpec({ persistViewSnapshots: false }),
      });
      if (!payload.ok) {
        ctx.setStatus(payload.message || ctx.formatIssues(payload.issues), "error");
        return;
      }
      state.generatedCode = ctx.stripImportLines(payload.code);
      generatedCode.value = state.generatedCode;
      ctx.setStatus(`Generated ${payload.engine} code.`, "success");
    } catch (error) {
      ctx.setStatus(`Code generation failed: ${error.message}`, "error");
    }
  }

  async function completeEditor() {
    try {
      const payload = await apiPost("/api/complete", {
        engine: state.selectedEngine,
        collection_format: state.selectedCollectionFormat,
        spec: ctx.serializeCurrentSpec({ persistViewSnapshots: true }),
      });
      if (!payload.ok) {
        ctx.setStatus(payload.message || ctx.formatIssues(payload.issues), "error");
        return;
      }
      state.editorFinished = true;
      ctx.setStatus("Returning the design to Python. You can close this tab.", "success");
      window.setTimeout(() => {
        window.close();
      }, 150);
    } catch (error) {
      ctx.setStatus(`Could not finish the editor session: ${error.message}`, "error");
    }
  }

  async function cancelEditor() {
    try {
      state.editorFinished = true;
      await apiPost("/api/cancel", {});
      ctx.setStatus("Editor cancelled. You can close this tab.", "success");
      window.setTimeout(() => {
        window.close();
      }, 150);
    } catch (error) {
      ctx.setStatus(`Could not cancel the editor session: ${error.message}`, "error");
    }
  }

  function saveDesign() {
    const blob = new Blob([JSON.stringify(ctx.serializeCurrentSpec({ persistViewSnapshots: true }), null, 2)], {
      type: "application/json",
    });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `${ctx.sanitizeFilename(state.spec.name || "tensor-network")}.json`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
    ctx.setStatus("Design downloaded as JSON.");
  }

  function loadDesignFromFile(event) {
    const file = event.target.files[0];
    if (!file) {
      return;
    }

    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const fileText = typeof reader.result === "string" ? reader.result : "";
        const isPythonSource = file.name.toLowerCase().endsWith(".py");
        const response = isPythonSource
          ? await apiPost("/api/validate", { python_code: fileText })
          : await apiPost("/api/validate", { spec: JSON.parse(fileText) });
        if (!response.ok) {
          ctx.setStatus(ctx.formatIssues(response.issues), "error");
          return;
        }
        resetDesignState(
          response.spec.network,
          `Loaded design from ${file.name}. History cleared.`,
          response.spec.schema_version
        );
      } catch (error) {
        ctx.setStatus(`Could not load ${file.name}: ${error.message}`, "error");
      } finally {
        loadInput.value = "";
      }
    };
    reader.readAsText(file, "utf-8");
  }

  async function copyGeneratedCode() {
    const codeToCopy = ctx.stripImportLines(generatedCode.value);
    if (!codeToCopy.trim()) {
      ctx.setStatus("There is no generated code to copy yet.");
      return;
    }
    try {
      await navigator.clipboard.writeText(codeToCopy);
      ctx.setStatus("Generated code copied to the clipboard without import lines.", "success");
    } catch (error) {
      ctx.setStatus(`Could not copy the generated code: ${error.message}`, "error");
    }
  }

  async function downloadPythonExport() {
    if (typeof ctx.toggleSidebarCollapsed === "function") {
      ctx.toggleSidebarCollapsed(false);
    }
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("code");
    }
    try {
      const payload = await apiPost("/api/generate", {
        engine: state.selectedEngine,
        collection_format: state.selectedCollectionFormat,
        spec: ctx.serializeCurrentSpec({ persistViewSnapshots: false }),
      });
      if (!payload.ok) {
        ctx.setStatus(payload.message || ctx.formatIssues(payload.issues), "error");
        return;
      }
      state.generatedCode = ctx.stripImportLines(payload.code);
      generatedCode.value = state.generatedCode;
      ctx.downloadBlob(
        `${ctx.sanitizeFilename(state.spec.name || "tensor-network")}-${ctx.sanitizeFilename(state.selectedEngine || "engine")}.py`,
        new Blob([payload.code], { type: "text/x-python;charset=utf-8" })
      );
      ctx.setStatus(`Exported ${payload.engine} Python code.`, "success");
    } catch (error) {
      ctx.setStatus(`Could not export Python code: ${error.message}`, "error");
    }
  }

  async function insertTemplate() {
    const templateName = templateSelect.value;
    if (!templateName) {
      ctx.setStatus("Choose a template first.");
      return;
    }
    const parameters = ctx.persistTemplateParametersFromControls();
    try {
      const payload = await apiPost("/api/template", {
        template: templateName,
        parameters,
      });
      const importedSpec = ctx.uniquifyImportedSpec(payload.spec.network, ctx.makeId("template"));
      const translatedSpec = ctx.translateImportedSpec(importedSpec, suggestTensorPosition(viewportCenterPosition()));
      ctx.applyDesignChange(
        () => {
          state.spec.tensors.push(...translatedSpec.tensors);
          state.spec.edges.push(...translatedSpec.edges);
          state.spec.groups.push(...translatedSpec.groups);
        },
        {
          selectionIds: translatedSpec.tensors.map((tensor) => tensor.id),
          primaryId: translatedSpec.tensors.length
            ? translatedSpec.tensors[translatedSpec.tensors.length - 1].id
            : null,
          statusMessage: `Inserted ${translatedSpec.name}.`,
        }
      );
    } catch (error) {
      ctx.setStatus(`Could not insert the template: ${error.message}`, "error");
    }
  }

  Object.assign(ctx, {
    handleCanvasContextMenu,
    handleCanvasWheel,
    handleCanvasMouseDown,
    handleGlobalMouseMove,
    handleGlobalMouseUp,
    startBoxSelection,
    updateBoxSelection,
    finishBoxSelection,
    updateSelectionBoxElement,
    handleKeydown,
    toggleHelpModal,
    sendCancelBeacon,
    handleWindowResize,
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
    generateCode,
    completeEditor,
    cancelEditor,
    saveDesign,
    loadDesignFromFile,
    copyGeneratedCode,
    downloadPythonExport,
    insertTemplate
  });
}

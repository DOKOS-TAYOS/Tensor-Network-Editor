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

  function handleCanvasContextMenu(event) {
    event.preventDefault();
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
    if (boxSelectionState.additive) {
      ctx.setSelection([...state.selectionIds, ...hitIds], {
        primaryId: hitIds.length ? hitIds[hitIds.length - 1] : state.primarySelectionId,
      });
      return;
    }
    ctx.setSelection(hitIds, { primaryId: hitIds.length ? hitIds[hitIds.length - 1] : null });
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

  function handleKeydown(event) {
    const activeElement = document.activeElement;
    const inTextInput = ctx.isTextInput(activeElement);

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
        state.plannerPreviewMode = null;
        state.plannerPreviewOrderByTensorId = {};
        ctx.render();
        ctx.setStatus("Automatic preview cleared.");
        return;
      }
      ctx.clearSelection();
      return;
    }

    if (inTextInput) {
      return;
    }

    const hasModifier = event.ctrlKey || event.metaKey;
    if (hasModifier && event.key.toLowerCase() === "z") {
      event.preventDefault();
      if (event.shiftKey) {
        ctx.performRedo();
      } else {
        ctx.performUndo();
      }
      return;
    }
    if (hasModifier && event.key.toLowerCase() === "y") {
      event.preventDefault();
      ctx.performRedo();
      return;
    }
    if (hasModifier && event.key.toLowerCase() === "a") {
      event.preventDefault();
      ctx.selectAllTensors();
      return;
    }
    if (hasModifier && event.key.toLowerCase() === "c") {
      event.preventDefault();
      if (typeof ctx.copySelectedSubgraphToClipboard === "function") {
        ctx.copySelectedSubgraphToClipboard();
      }
      return;
    }
    if (hasModifier && event.key.toLowerCase() === "v") {
      event.preventDefault();
      if (typeof ctx.pasteClipboardToCanvas === "function") {
        ctx.pasteClipboardToCanvas();
      }
      return;
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      deleteSelection();
      return;
    }
    if (event.key.toLowerCase() === "n") {
      event.preventDefault();
      addTensorAtCenter();
      return;
    }
    if (event.key.toLowerCase() === "c") {
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
        metadata: {},
      },
      "Started a new empty design. History cleared."
    );
  }

  function resetDesignState(spec, message, schemaVersion = state.schemaVersion) {
    state.spec = ctx.normalizeSpec(spec);
    state.schemaVersion = schemaVersion;
    state.generatedCode = "";
    state.activeSidebarTab = "selection";
    state.selectionIds = [];
    state.primarySelectionId = null;
    state.selectedElement = null;
    state.pendingIndexId = null;
    state.pendingPlannerOperandId = null;
    state.pendingPlannerSelectionId = null;
    state.connectMode = false;
    state.plannerMode = false;
    state.hasFitCanvas = false;
    state.activeResize = null;
    state.activeGroupDrag = null;
    state.noteDragState = null;
    state.activeNoteResize = null;
    state.contractionAnalysis = null;
    state.plannerPreviewMode = null;
    state.plannerManualOrderByTensorId = {};
    state.plannerPreviewOrderByTensorId = {};
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
    const tensor = ctx.findTensorById(tensorId);
    if (!tensor) {
      return;
    }
    const center = viewportCenterPosition();
    tensor.position.x = center.x;
    tensor.position.y = center.y;
  }

  function toggleConnectMode() {
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
    const located = ctx.findIndexOwner(indexId);
    if (!located) {
      return;
    }
    if (ctx.findEdgeByIndexId(indexId)) {
      ctx.setStatus("This index is already connected. Delete the connection first.", "error");
      return;
    }

    if (!state.pendingIndexId) {
      state.pendingIndexId = indexId;
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

    const left = ctx.findIndexOwner(state.pendingIndexId);
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
        selectionIds: [newEdgeId],
        primaryId: newEdgeId,
        statusMessage: "Connection created.",
      }
    );
  }

  function deleteSelection() {
    if (!state.selectionIds.length) {
      ctx.setStatus("Nothing is selected to delete.");
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
      ctx.removeTensor(tensorId);
    });

    selectedIndexIds.forEach((indexId) => {
      const located = ctx.findIndexOwner(indexId);
      if (located && !selectedTensorIds.has(located.tensor.id)) {
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
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("code");
    }
    try {
      const payload = await apiPost("/api/generate", {
        engine: state.selectedEngine,
        spec: ctx.serializeCurrentSpec(),
      });
      if (!payload.ok) {
        ctx.setStatus(ctx.formatIssues(payload.issues), "error");
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
        spec: ctx.serializeCurrentSpec(),
      });
      if (!payload.ok) {
        ctx.setStatus(ctx.formatIssues(payload.issues), "error");
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
    const blob = new Blob([JSON.stringify(ctx.serializeCurrentSpec(), null, 2)], {
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
        const payload = JSON.parse(reader.result);
        const response = await apiPost("/api/validate", { spec: payload });
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
    if (typeof ctx.setActiveSidebarTab === "function") {
      ctx.setActiveSidebarTab("code");
    }
    try {
      const payload = await apiPost("/api/generate", {
        engine: state.selectedEngine,
        spec: ctx.serializeCurrentSpec(),
      });
      if (!payload.ok) {
        ctx.setStatus(ctx.formatIssues(payload.issues), "error");
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
    try {
      const payload = await apiPost("/api/template", { template: templateName });
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

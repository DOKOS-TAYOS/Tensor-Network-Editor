(function () {
  "use strict";

  const TENSOR_WIDTH = 180;
  const TENSOR_HEIGHT = 108;
  const INDEX_RADIUS = 15;
  const INDEX_PADDING = 8;
  const HISTORY_LIMIT = 100;
  const REDO_SHORTCUT_LABEL = "Ctrl+Shift+Z";
  const DEFAULT_INDEX_SLOTS = [
    { x: -58, y: -20 },
    { x: 58, y: -20 },
    { x: -58, y: 20 },
    { x: 58, y: 20 },
    { x: 0, y: -28 },
    { x: 0, y: 28 },
    { x: -24, y: -34 },
    { x: 24, y: 34 },
  ];

  const state = {
    spec: null,
    selectedEngine: null,
    selectedElement: null,
    selectionIds: [],
    primarySelectionId: null,
    generatedCode: "",
    connectMode: false,
    pendingIndexId: null,
    cy: null,
    hasFitCanvas: false,
    editorFinished: false,
    tensorOrder: [],
    syncingIndexPositions: false,
    syncingTensorPositions: false,
    undoStack: [],
    redoStack: [],
    lastMutationClearedCode: false,
    activeTensorDrag: null,
    activeIndexDrag: null,
    boxSelection: null,
    isHelpOpen: false,
    minimapDrag: null,
    minimapTransform: null,
  };

  const statusMessage = document.getElementById("status-message");
  const propertiesPanel = document.getElementById("properties-panel");
  const generatedCode = document.getElementById("generated-code");
  const engineSelect = document.getElementById("engine-select");
  const connectButton = document.getElementById("connect-button");
  const loadInput = document.getElementById("load-input");
  const undoButton = document.getElementById("undo-button");
  const redoButton = document.getElementById("redo-button");
  const exportPngButton = document.getElementById("export-png-button");
  const exportSvgButton = document.getElementById("export-svg-button");
  const helpButton = document.getElementById("help-button");
  const helpModal = document.getElementById("help-modal");
  const helpBackdrop = document.getElementById("help-backdrop");
  const helpCloseButton = document.getElementById("help-close-button");
  const canvasShell = document.getElementById("canvas-shell");
  const selectionBox = document.getElementById("canvas-selection-box");
  const minimapCanvas = document.getElementById("minimap");

  document.addEventListener("DOMContentLoaded", () => {
    attachToolbarHandlers();
    bootstrap().catch((error) => {
      setStatus(`Failed to load the editor: ${error.message}`, "error");
    });
  });

  async function bootstrap() {
    const payload = await apiGet("/api/bootstrap");
    state.spec = normalizeSpec(payload.spec.network);
    state.selectedEngine = payload.default_engine;
    reconcileTensorOrder();
    populateEngineOptions(payload.engines);
    initGraph();
    clearHistory();
    render();
    setStatus(
      "Editor ready. Drag the canvas to move, use the wheel to zoom, and right drag to box-select.",
      "success"
    );
  }

  function attachToolbarHandlers() {
    document.getElementById("new-design-button").addEventListener("click", handleNewDesign);
    document.getElementById("add-tensor-button").addEventListener("click", addTensorAtCenter);
    document.getElementById("connect-button").addEventListener("click", toggleConnectMode);
    document.getElementById("delete-button").addEventListener("click", deleteSelection);
    document.getElementById("save-button").addEventListener("click", saveDesign);
    document.getElementById("load-button").addEventListener("click", () => loadInput.click());
    document.getElementById("generate-button").addEventListener("click", generateCode);
    document.getElementById("done-button").addEventListener("click", completeEditor);
    document.getElementById("cancel-button").addEventListener("click", cancelEditor);
    document.getElementById("copy-code-button").addEventListener("click", copyGeneratedCode);
    undoButton.addEventListener("click", performUndo);
    redoButton.addEventListener("click", performRedo);
    exportPngButton.addEventListener("click", downloadPngExport);
    exportSvgButton.addEventListener("click", downloadSvgExport);
    helpButton.addEventListener("click", () => toggleHelpModal(true));
    helpBackdrop.addEventListener("click", () => toggleHelpModal(false));
    helpCloseButton.addEventListener("click", () => toggleHelpModal(false));
    engineSelect.addEventListener("change", (event) => {
      state.selectedEngine = event.target.value;
    });
    loadInput.addEventListener("change", loadDesignFromFile);
    window.addEventListener("keydown", handleKeydown);
    window.addEventListener("beforeunload", sendCancelBeacon);
    window.addEventListener("pagehide", sendCancelBeacon);
    window.addEventListener("resize", handleWindowResize);
    window.addEventListener("mousemove", handleGlobalMouseMove);
    window.addEventListener("mouseup", handleGlobalMouseUp);
    canvasShell.addEventListener("contextmenu", handleCanvasContextMenu);
    canvasShell.addEventListener("mousedown", handleCanvasMouseDown, true);
    minimapCanvas.addEventListener("mousedown", handleMinimapMouseDown);
  }

  // SECTION: history-selection

  function clearHistory() {
    state.undoStack = [];
    state.redoStack = [];
    updateToolbarState();
  }

  function createHistorySnapshot() {
    return {
      spec: deepClone(state.spec),
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

  // SECTION: graph-render

  function initGraph() {
    state.cy = cytoscape({
      container: document.getElementById("canvas"),
      layout: { name: "preset" },
      minZoom: 0.3,
      maxZoom: 2.5,
      selectionType: "additive",
      wheelSensitivity: 0.18,
      userPanningEnabled: true,
      userZoomingEnabled: true,
      boxSelectionEnabled: false,
      style: [
        {
          selector: "node, edge",
          style: {
            "z-index-compare": "manual",
          },
        },
        {
          selector: "node[kind = 'tensor']",
          style: {
            shape: "round-rectangle",
            width: TENSOR_WIDTH,
            height: TENSOR_HEIGHT,
            "background-color": "data(backgroundColor)",
            "border-width": 2,
            "border-color": "data(borderColor)",
            color: "data(textColor)",
            label: "data(label)",
            "font-size": 18,
            "font-family": "Georgia",
            "text-valign": "top",
            "text-halign": "center",
            "text-margin-y": 20,
            "padding-top": 42,
            "padding-bottom": 18,
            "padding-left": 24,
            "padding-right": 24,
            "z-index": "data(zIndex)",
          },
        },
        {
          selector: "node[kind = 'index']",
          style: {
            width: INDEX_RADIUS * 2,
            height: INDEX_RADIUS * 2,
            label: "data(orderLabel)",
            "font-size": 12,
            "font-weight": 700,
            color: "data(textColor)",
            "text-valign": "center",
            "text-halign": "center",
            "border-width": 2,
            "border-color": "data(borderColor)",
            "background-color": "data(backgroundColor)",
            "overlay-opacity": 0,
            "z-index": "data(zIndex)",
          },
        },
        {
          selector: "node[kind = 'index-label']",
          style: {
            width: 1,
            height: 1,
            shape: "round-rectangle",
            label: "data(label)",
            color: "data(textColor)",
            "background-opacity": 0,
            "border-opacity": 0,
            "overlay-opacity": 0,
            "font-size": 10,
            "text-wrap": "wrap",
            "text-max-width": 90,
            "text-valign": "top",
            "text-halign": "center",
            "z-index": "data(zIndex)",
            events: "no",
          },
        },
        {
          selector: "node.index-open",
          style: {
            "background-color": "data(backgroundColor)",
            "border-color": "data(borderColor)",
            color: "data(textColor)",
          },
        },
        {
          selector: "node.index-connected",
          style: {
            "background-color": "data(backgroundColor)",
            "border-color": "data(borderColor)",
            color: "data(textColor)",
          },
        },
        {
          selector: "edge",
          style: {
            width: 3,
            "line-color": "data(lineColor)",
            "curve-style": "bezier",
            label: "data(label)",
            "font-size": 11,
            color: "data(textColor)",
            "text-background-color": "#101720",
            "text-background-opacity": 0.92,
            "text-background-padding": 4,
            "text-rotation": "autorotate",
            "target-arrow-shape": "none",
            "source-arrow-shape": "none",
            "z-index": "data(zIndex)",
          },
        },
        {
          selector: ":selected",
          style: {
            "overlay-color": "#61a8ff",
            "overlay-opacity": 0.14,
            "border-color": "#8bc2ff",
            "line-color": "#8bc2ff",
          },
        },
      ],
    });

    state.cy.on("tap", "node, edge", (event) => {
      if (state.boxSelection) {
        return;
      }
      const element = event.target;
      const kind = element.data("kind");
      if (kind === "index-label") {
        return;
      }
      if (state.connectMode && isIndexNode(element)) {
        handleConnectClick(element.id());
        return;
      }
      if (kind === "tensor") {
        bringTensorToFront(element.id());
      } else if (kind === "index") {
        const located = findIndexOwner(element.id());
        if (located) {
          bringTensorToFront(located.tensor.id);
        }
      }
      selectElement(kind, element.id(), { additive: Boolean(event.originalEvent && event.originalEvent.shiftKey) });
    });

    state.cy.on("tap", (event) => {
      if (event.target === state.cy && !state.boxSelection) {
        clearSelection({ preservePendingIndex: true });
      }
    });

    state.cy.on("grab", "node[kind = 'tensor']", (event) => {
      const tensorId = event.target.id();
      bringTensorToFront(tensorId);
      if (!state.selectionIds.includes(tensorId)) {
        setSelection([tensorId], { primaryId: tensorId });
      }
      state.activeTensorDrag = createTensorDragState(tensorId);
    });

    state.cy.on("position", "node[kind = 'tensor']", (event) => {
      if (state.syncingTensorPositions) {
        return;
      }
      const tensor = findTensorById(event.target.id());
      if (!tensor) {
        return;
      }
      tensor.position.x = Math.round(event.target.position("x"));
      tensor.position.y = Math.round(event.target.position("y"));
      syncIndexNodePositions(tensor);
      if (state.activeTensorDrag && state.activeTensorDrag.anchorId === tensor.id) {
        moveCompanionTensorsDuringDrag();
      }
    });

    state.cy.on("dragfree", "node[kind = 'tensor']", (event) => {
      const tensor = findTensorById(event.target.id());
      if (tensor) {
        syncIndexNodePositions(tensor);
      }
      finishTensorDrag(event.target.id());
      renderProperties();
      renderMinimap();
    });

    state.cy.on("grab", "node[kind = 'index']", (event) => {
      const located = findIndexOwner(event.target.id());
      if (located) {
        bringTensorToFront(located.tensor.id);
      }
      state.activeIndexDrag = {
        indexId: event.target.id(),
        snapshot: createHistorySnapshot(),
      };
    });

    state.cy.on("position", "node[kind = 'index']", (event) => {
      if (state.syncingIndexPositions) {
        return;
      }
      const located = findIndexOwner(event.target.id());
      if (!located) {
        return;
      }
      located.index.offset = clampIndexOffset({
        x: event.target.position("x") - located.tensor.position.x,
        y: event.target.position("y") - located.tensor.position.y,
      });
      const absolutePosition = indexAbsolutePosition(located.tensor, located.index);
      syncIndexLabelNodePosition(located.index, absolutePosition);
      if (
        Math.abs(absolutePosition.x - event.target.position("x")) > 0.5 ||
        Math.abs(absolutePosition.y - event.target.position("y")) > 0.5
      ) {
        runWithIndexSync(() => {
          event.target.position(absolutePosition);
        });
      }
    });

    state.cy.on("dragfree", "node[kind = 'index']", (event) => {
      const located = findIndexOwner(event.target.id());
      if (located) {
        located.index.offset = clampIndexOffset(located.index.offset);
        syncSingleIndexNodePosition(located.tensor, located.index);
      }
      finishIndexDrag(event.target.id());
      renderProperties();
      renderMinimap();
    });

    state.cy.on("pan zoom resize", () => {
      renderMinimap();
    });
  }

  function render() {
    renderGraph();
    renderProperties();
    generatedCode.value = state.generatedCode;
    connectButton.classList.toggle("is-active", state.connectMode);
    helpModal.classList.toggle("is-hidden", !state.isHelpOpen);
    updateToolbarState();
    renderMinimap();
  }

  function renderGraph() {
    if (!state.cy || !state.spec) {
      return;
    }
    reconcileTensorOrder();
    const elements = buildGraphElements();
    state.cy.batch(() => {
      state.cy.elements().remove();
      state.cy.add(elements);
    });
    applyTensorLayerData();
    if (!state.hasFitCanvas) {
      if (state.spec.tensors.length) {
        state.cy.fit(undefined, 40);
      } else {
        state.cy.center();
      }
      state.hasFitCanvas = true;
    }
    syncCySelection();
  }

  function buildGraphElements() {
    const tensorElements = [];
    const edgeElements = [];
    const indexElements = [];
    const indexLabelElements = [];
    const connectedIndexIds = new Set();

    state.spec.edges.forEach((edge) => {
      connectedIndexIds.add(edge.left.index_id);
      connectedIndexIds.add(edge.right.index_id);
    });

    state.spec.tensors.forEach((tensor) => {
      ensureTensorIndexOffsets(tensor);
      const tensorRank = tensorLayerRank(tensor.id);
      const tensorColor = getMetadataColor(tensor.metadata, "#18212c");
      tensorElements.push({
        group: "nodes",
        data: {
          id: tensor.id,
          label: tensor.name,
          kind: "tensor",
          backgroundColor: tensorColor,
          borderColor: shiftColor(tensorColor, 26),
          textColor: readableTextColor(tensorColor),
          zIndex: 100 + tensorRank * 20,
        },
        position: { x: tensor.position.x, y: tensor.position.y },
      });

      tensor.indices.forEach((index, indexPosition) => {
        const indexColor = getIndexColor(index, connectedIndexIds.has(index.id));
        const indexPositionAbsolute = indexAbsolutePosition(tensor, index);
        indexElements.push({
          group: "nodes",
          data: {
            id: index.id,
            kind: "index",
            tensor_id: tensor.id,
            orderLabel: String(indexPosition + 1),
            backgroundColor: indexColor,
            borderColor: shiftColor(indexColor, 34),
            textColor: readableTextColor(indexColor),
            zIndex: 300 + tensorRank * 20 + indexPosition,
          },
          classes: connectedIndexIds.has(index.id) ? "index-connected" : "index-open",
          position: indexPositionAbsolute,
          grabbable: true,
          selectable: true,
        });

        indexLabelElements.push({
          group: "nodes",
          data: {
            id: indexLabelNodeId(index.id),
            kind: "index-label",
            label: `${index.name} · ${index.dimension}`,
            textColor: shiftColor(indexColor, 64),
            zIndex: 310 + tensorRank * 20 + indexPosition,
          },
          position: indexLabelPosition(indexPositionAbsolute),
          grabbable: false,
          selectable: false,
        });
      });
    });

    state.spec.edges.forEach((edge) => {
      const edgeColor = getMetadataColor(edge.metadata, "#8da1c3");
      edgeElements.push({
        group: "edges",
        data: {
          id: edge.id,
          source: edge.left.index_id,
          target: edge.right.index_id,
          label: edge.name,
          kind: "edge",
          lineColor: edgeColor,
          textColor: shiftColor(edgeColor, 72),
          zIndex: 220,
        },
      });
    });

    return [...tensorElements, ...edgeElements, ...indexElements, ...indexLabelElements];
  }

  function createTensorDragState(anchorId) {
    const selectedTensorIds = getSelectedIdsByKind("tensor");
    const dragIds = selectedTensorIds.includes(anchorId) ? selectedTensorIds : [anchorId];
    const startPositions = {};
    dragIds.forEach((tensorId) => {
      const tensor = findTensorById(tensorId);
      if (tensor) {
        startPositions[tensorId] = { x: tensor.position.x, y: tensor.position.y };
      }
    });
    return {
      anchorId,
      snapshot: createHistorySnapshot(),
      dragIds,
      startPositions,
    };
  }

  function moveCompanionTensorsDuringDrag() {
    if (!state.activeTensorDrag || !state.cy) {
      return;
    }
    const anchor = findTensorById(state.activeTensorDrag.anchorId);
    const anchorStartPosition = state.activeTensorDrag.startPositions[state.activeTensorDrag.anchorId];
    if (!anchor || !anchorStartPosition) {
      return;
    }
    const deltaX = anchor.position.x - anchorStartPosition.x;
    const deltaY = anchor.position.y - anchorStartPosition.y;
    state.syncingTensorPositions = true;
    try {
      state.activeTensorDrag.dragIds.forEach((tensorId) => {
        if (tensorId === anchor.id) {
          return;
        }
        const tensor = findTensorById(tensorId);
        const startPosition = state.activeTensorDrag.startPositions[tensorId];
        if (!tensor || !startPosition) {
          return;
        }
        tensor.position.x = Math.round(startPosition.x + deltaX);
        tensor.position.y = Math.round(startPosition.y + deltaY);
        const tensorElement = state.cy.getElementById(tensor.id);
        if (tensorElement && tensorElement.length) {
          tensorElement.position(tensor.position);
        }
        syncIndexNodePositions(tensor);
      });
    } finally {
      state.syncingTensorPositions = false;
    }
  }

  function finishTensorDrag(anchorId) {
    if (!state.activeTensorDrag || state.activeTensorDrag.anchorId !== anchorId) {
      return;
    }
    commitHistorySnapshot(state.activeTensorDrag.snapshot);
    state.activeTensorDrag = null;
    updateToolbarState();
  }

  function finishIndexDrag(indexId) {
    if (!state.activeIndexDrag || state.activeIndexDrag.indexId !== indexId) {
      return;
    }
    commitHistorySnapshot(state.activeIndexDrag.snapshot);
    state.activeIndexDrag = null;
    updateToolbarState();
  }

  // SECTION: properties

  function renderProperties() {
    pruneSelectionToExisting();
    if (!state.selectionIds.length) {
      renderNetworkProperties();
      return;
    }
    if (state.selectionIds.length > 1) {
      renderMultiSelectionProperties();
      return;
    }
    const singleSelection = getSelectionEntry(state.selectionIds[0]);
    if (!singleSelection) {
      renderNetworkProperties();
      return;
    }
    if (singleSelection.kind === "tensor") {
      renderTensorProperties(singleSelection.id);
      return;
    }
    if (singleSelection.kind === "index") {
      renderIndexProperties(singleSelection.id);
      return;
    }
    if (singleSelection.kind === "edge") {
      renderEdgeProperties(singleSelection.id);
      return;
    }
    renderNetworkProperties();
  }

  function renderNetworkProperties() {
    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="network-name-input">Design name</label>
        <input id="network-name-input" value="${escapeHtml(state.spec.name)}" />
      </div>
      <div class="properties-chip">
        <span>Tensors</span>
        <strong>${state.spec.tensors.length}</strong>
      </div>
      <div class="properties-chip">
        <span>Connections</span>
        <strong>${state.spec.edges.length}</strong>
      </div>
    `;

    const networkNameInput = document.getElementById("network-name-input");
    networkNameInput.addEventListener("change", () => {
      const proposedName = networkNameInput.value.trim() || "Untitled Network";
      if (proposedName === state.spec.name) {
        return;
      }
      applyDesignChange(
        () => {
          state.spec.name = proposedName;
        },
        {
          statusMessage: "Updated design name.",
        }
      );
    });
  }

  function renderMultiSelectionProperties() {
    const selectedEntries = getSelectedEntries();
    const tensorCount = selectedEntries.filter((entry) => entry.kind === "tensor").length;
    const indexCount = selectedEntries.filter((entry) => entry.kind === "index").length;
    const edgeCount = selectedEntries.filter((entry) => entry.kind === "edge").length;
    const tensorsOnly = tensorCount > 0 && tensorCount === selectedEntries.length;
    const batchColor = getBatchColorValue(selectedEntries);

    propertiesPanel.innerHTML = `
      <div class="properties-summary">
        <div class="properties-chip">
          <span>Selected</span>
          <strong>${selectedEntries.length}</strong>
        </div>
        <div class="properties-chip-wrap">
          <div class="properties-chip">
            <span>Tensors</span>
            <strong>${tensorCount}</strong>
          </div>
          <div class="properties-chip">
            <span>Indices</span>
            <strong>${indexCount}</strong>
          </div>
          <div class="properties-chip">
            <span>Connections</span>
            <strong>${edgeCount}</strong>
          </div>
        </div>
      </div>
      <div class="button-row">
        <label class="control-inline-color" for="multi-color-input">
          <input id="multi-color-input" type="color" title="Choose tint" aria-label="Choose tint" value="${escapeHtml(batchColor)}" />
        </label>
        <button id="apply-multi-color-button" type="button">Apply Color</button>
        ${tensorsOnly ? '<button id="add-index-to-selection-button" type="button">Add Index to Tensors</button>' : ""}
        <button id="delete-selection-button" type="button" class="danger">Delete Selected</button>
      </div>
      <p class="property-meta">
        Drag any selected tensor to move the selected tensor group together.
      </p>
    `;

    document.getElementById("apply-multi-color-button").addEventListener("click", () => {
      const colorValue = document.getElementById("multi-color-input").value;
      applyDesignChange(
        () => {
          applyColorToSelection(colorValue);
        },
        {
          preserveSelection: true,
          statusMessage: "Updated the selection color.",
        }
      );
    });

    const addIndexButton = document.getElementById("add-index-to-selection-button");
    if (addIndexButton) {
      addIndexButton.addEventListener("click", () => {
        applyDesignChange(
          () => {
            getSelectedIdsByKind("tensor").forEach((tensorId) => {
              const tensor = findTensorById(tensorId);
              if (tensor) {
                tensor.indices.push(createIndex(tensor, tensor.indices.length));
              }
            });
          },
          {
            preserveSelection: true,
            statusMessage: "Added one index to each selected tensor.",
          }
        );
      });
    }

    document.getElementById("delete-selection-button").addEventListener("click", deleteSelection);
  }

  function renderTensorProperties(tensorId) {
    const tensor = findTensorById(tensorId);
    if (!tensor) {
      clearSelection();
      return;
    }

    const indexList = tensor.indices
      .map(
        (index, indexPosition) => `
          <button type="button" class="properties-chip index-select-button" data-index-id="${index.id}">
            <span>${indexPosition + 1}. ${escapeHtml(index.name)}</span>
            <strong>${index.dimension}</strong>
          </button>
        `
      )
      .join("");

    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="tensor-name-input">Tensor name</label>
        <input id="tensor-name-input" value="${escapeHtml(tensor.name)}" />
      </div>
      <div class="button-row">
        <button id="add-index-button" type="button">Add Index</button>
        <button id="center-tensor-button" type="button">Center in View</button>
        <label class="control-inline-color" for="tensor-color-input">
          <input id="tensor-color-input" type="color" title="Choose tint" aria-label="Choose tint" value="${escapeHtml(getMetadataColor(tensor.metadata, "#18212c"))}" />
        </label>
      </div>
      <div class="button-row">
        <button id="delete-tensor-button" type="button" class="danger">Delete Tensor</button>
        <button id="apply-tensor-button" type="button" class="apply-button is-hidden">Apply Changes</button>
      </div>
      <div class="properties-list">${indexList || "<p class='property-meta'>This tensor has no indices yet.</p>"}</div>
    `;

    const tensorNameInput = document.getElementById("tensor-name-input");
    const tensorColorInput = document.getElementById("tensor-color-input");
    installDirtyApply({
      buttonElement: document.getElementById("apply-tensor-button"),
      inputElements: [tensorNameInput, tensorColorInput],
      isDirty: () =>
        tensorNameInput.value !== tensor.name ||
        tensorColorInput.value !== getMetadataColor(tensor.metadata, "#18212c"),
      onApply: () => {
        applyDesignChange(
          () => {
            tensor.name = tensorNameInput.value.trim() || tensor.name;
            tensor.metadata.color = tensorColorInput.value;
          },
          {
            selectionIds: [tensor.id],
            primaryId: tensor.id,
            statusMessage: `Updated tensor ${tensorNameInput.value.trim() || tensor.name}.`,
          }
        );
      },
    });

    document.getElementById("add-index-button").addEventListener("click", () => {
      applyDesignChange(
        () => {
          tensor.indices.push(createIndex(tensor, tensor.indices.length));
        },
        {
          selectionIds: [tensor.id],
          primaryId: tensor.id,
          statusMessage: `Added one index to ${tensor.name}.`,
        }
      );
    });
    document.getElementById("center-tensor-button").addEventListener("click", () => {
      applyDesignChange(
        () => {
          centerTensor(tensor.id);
        },
        {
          selectionIds: [tensor.id],
          primaryId: tensor.id,
          statusMessage: `Centered tensor ${tensor.name} in the current view.`,
        }
      );
    });
    document.getElementById("delete-tensor-button").addEventListener("click", () => {
      applyDesignChange(
        () => {
          removeTensor(tensor.id);
        },
        {
          selectionIds: [],
          statusMessage: `Deleted tensor ${tensor.name}.`,
        }
      );
    });
    document.querySelectorAll(".index-select-button").forEach((button) => {
      button.addEventListener("click", () => {
        setSelection([button.dataset.indexId], { primaryId: button.dataset.indexId });
      });
    });
  }

  function renderIndexProperties(indexId) {
    const located = findIndexOwner(indexId);
    if (!located) {
      clearSelection();
      return;
    }

    const { tensor, index, indexPosition } = located;
    propertiesPanel.innerHTML = `
      <div class="field-row">
        <div class="field-group">
          <label for="index-name-input">Index name</label>
          <input id="index-name-input" value="${escapeHtml(index.name)}" />
        </div>
        <div class="field-group compact-number-field">
          <label for="index-dimension-input">Dimension</label>
          <input id="index-dimension-input" type="number" min="1" step="1" value="${index.dimension}" />
        </div>
      </div>
      <div class="button-row">
        <button id="move-index-up-button" type="button">Move Earlier</button>
        <button id="move-index-down-button" type="button">Move Later</button>
        <label class="control-inline-color" for="index-color-input">
          <input id="index-color-input" type="color" title="Choose tint" aria-label="Choose tint" value="${escapeHtml(getMetadataColor(index.metadata, getIndexColor(index, Boolean(findEdgeByIndexId(indexId)))))}" />
        </label>
        <button id="delete-index-button" type="button" class="danger">Delete Index</button>
        <button id="apply-index-button" type="button" class="apply-button is-hidden">Apply Changes</button>
      </div>
    `;

    const indexNameInput = document.getElementById("index-name-input");
    const indexDimensionInput = document.getElementById("index-dimension-input");
    const indexColorInput = document.getElementById("index-color-input");
    installDirtyApply({
      buttonElement: document.getElementById("apply-index-button"),
      inputElements: [indexNameInput, indexDimensionInput, indexColorInput],
      isDirty: () =>
        indexNameInput.value !== index.name ||
        indexDimensionInput.value !== String(index.dimension) ||
        indexColorInput.value !== getMetadataColor(index.metadata, getIndexColor(index, Boolean(findEdgeByIndexId(indexId)))),
      onApply: () => {
        const proposedName = indexNameInput.value.trim();
        const parsed = Number.parseInt(indexDimensionInput.value, 10);
        if (!proposedName) {
          setStatus("Index name cannot be empty.", "error");
          return;
        }
        if (!Number.isFinite(parsed) || parsed <= 0) {
          setStatus("Index dimension must be a positive integer.", "error");
          return;
        }
        if (tensorIndexNameExists(tensor, proposedName, index.id)) {
          setStatus(`Tensor ${tensor.name} already has an index named ${proposedName}.`, "error");
          return;
        }
        applyDesignChange(
          () => {
            index.name = proposedName;
            index.dimension = parsed;
            index.metadata.color = indexColorInput.value;
          },
          {
            selectionIds: [index.id],
            primaryId: index.id,
            statusMessage: `Updated index ${proposedName}.`,
          }
        );
      },
    });

    document.getElementById("move-index-up-button").addEventListener("click", () => {
      applyDesignChange(
        () => {
          moveIndex(tensor.id, indexPosition, -1);
        },
        {
          selectionIds: [index.id],
          primaryId: index.id,
          statusMessage: `Moved index ${index.name}.`,
        }
      );
    });
    document.getElementById("move-index-down-button").addEventListener("click", () => {
      applyDesignChange(
        () => {
          moveIndex(tensor.id, indexPosition, 1);
        },
        {
          selectionIds: [index.id],
          primaryId: index.id,
          statusMessage: `Moved index ${index.name}.`,
        }
      );
    });
    document.getElementById("delete-index-button").addEventListener("click", () => {
      applyDesignChange(
        () => {
          removeIndex(tensor.id, index.id);
        },
        {
          selectionIds: [],
          statusMessage: `Deleted index ${index.name}.`,
        }
      );
    });
  }

  function renderEdgeProperties(edgeId) {
    const edge = findEdgeById(edgeId);
    if (!edge) {
      clearSelection();
      return;
    }
    propertiesPanel.innerHTML = `
      <div class="field-group">
        <label for="edge-name-input">Edge name</label>
        <input id="edge-name-input" value="${escapeHtml(edge.name)}" />
      </div>
      <div class="button-row">
        <label class="control-inline-color" for="edge-color-input">
          <input id="edge-color-input" type="color" title="Choose tint" aria-label="Choose tint" value="${escapeHtml(getMetadataColor(edge.metadata, "#8da1c3"))}" />
        </label>
        <button id="delete-edge-button" type="button" class="danger">Delete Connection</button>
        <button id="apply-edge-button" type="button" class="apply-button is-hidden">Apply Changes</button>
      </div>
    `;

    const edgeNameInput = document.getElementById("edge-name-input");
    const edgeColorInput = document.getElementById("edge-color-input");
    installDirtyApply({
      buttonElement: document.getElementById("apply-edge-button"),
      inputElements: [edgeNameInput, edgeColorInput],
      isDirty: () =>
        edgeNameInput.value !== edge.name ||
        edgeColorInput.value !== getMetadataColor(edge.metadata, "#8da1c3"),
      onApply: () => {
        applyDesignChange(
          () => {
            edge.name = edgeNameInput.value.trim() || edge.name;
            edge.metadata.color = edgeColorInput.value;
          },
          {
            selectionIds: [edge.id],
            primaryId: edge.id,
            statusMessage: `Updated connection ${edgeNameInput.value.trim() || edge.name}.`,
          }
        );
      },
    });

    document.getElementById("delete-edge-button").addEventListener("click", () => {
      applyDesignChange(
        () => {
          removeEdge(edge.id);
        },
        {
          selectionIds: [],
          statusMessage: `Deleted connection ${edge.name}.`,
        }
      );
    });
  }

  function installDirtyApply({ buttonElement, inputElements, isDirty, onApply }) {
    const refreshVisibility = () => {
      buttonElement.classList.toggle("is-hidden", !isDirty());
    };
    inputElements.forEach((element) => {
      element.addEventListener("input", refreshVisibility);
    });
    buttonElement.addEventListener("click", onApply);
    refreshVisibility();
  }

  // SECTION: interaction-actions

  function handleCanvasContextMenu(event) {
    event.preventDefault();
  }

  function handleCanvasMouseDown(event) {
    if (state.isHelpOpen) {
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
    if (state.minimapDrag) {
      updateViewportFromMinimapClientPoint(event.clientX, event.clientY);
    }
  }

  function handleGlobalMouseUp(event) {
    if (state.boxSelection && event.button === 2) {
      finishBoxSelection(false);
      return;
    }
    if (state.minimapDrag && event.button === 0) {
      state.minimapDrag = null;
      minimapCanvas.classList.remove("is-dragging");
    }
  }

  function startBoxSelection(event) {
    const point = clientPointToCanvasPoint(event.clientX, event.clientY);
    state.boxSelection = {
      start: point,
      current: point,
      additive: Boolean(event.shiftKey),
    };
    updateSelectionBoxElement();
  }

  function updateBoxSelection(event) {
    state.boxSelection.current = clientPointToCanvasPoint(event.clientX, event.clientY);
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
    const box = normalizedBox(boxSelectionState.start, boxSelectionState.current);
    const hitIds = state.cy
      .elements("node, edge")
      .toArray()
      .filter((element) => element.data("kind") !== "index-label")
      .filter((element) => boxesIntersect(box, element.renderedBoundingBox()))
      .map((element) => element.id());
    if (boxSelectionState.additive) {
      setSelection([...state.selectionIds, ...hitIds], {
        primaryId: hitIds.length ? hitIds[hitIds.length - 1] : state.primarySelectionId,
      });
      return;
    }
    setSelection(hitIds, { primaryId: hitIds.length ? hitIds[hitIds.length - 1] : null });
  }

  function updateSelectionBoxElement() {
    if (!state.boxSelection) {
      selectionBox.classList.add("is-hidden");
      return;
    }
    const box = normalizedBox(state.boxSelection.start, state.boxSelection.current);
    selectionBox.classList.remove("is-hidden");
    selectionBox.style.left = `${box.left}px`;
    selectionBox.style.top = `${box.top}px`;
    selectionBox.style.width = `${Math.max(1, box.width)}px`;
    selectionBox.style.height = `${Math.max(1, box.height)}px`;
  }

  function handleKeydown(event) {
    const activeElement = document.activeElement;
    const inTextInput = isTextInput(activeElement);

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
        render();
        setStatus("Connect mode cancelled.");
        return;
      }
      clearSelection();
      return;
    }

    if (inTextInput) {
      return;
    }

    const hasModifier = event.ctrlKey || event.metaKey;
    if (hasModifier && event.key.toLowerCase() === "z") {
      event.preventDefault();
      if (event.shiftKey) {
        performRedo();
      } else {
        performUndo();
      }
      return;
    }
    if (hasModifier && event.key.toLowerCase() === "y") {
      event.preventDefault();
      performRedo();
      return;
    }
    if (hasModifier && event.key.toLowerCase() === "a") {
      event.preventDefault();
      selectAllTensors();
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
    renderMinimap();
  }

  function handleNewDesign() {
    if (!window.confirm("Start a new design? Unsaved changes in this browser tab will be lost.")) {
      return;
    }

    resetDesignState(
      {
        id: makeId("network"),
        name: "Untitled Network",
        tensors: [],
        edges: [],
        metadata: {},
      },
      "Started a new empty design. History cleared."
    );
  }

  function resetDesignState(spec, message) {
    state.spec = normalizeSpec(spec);
    state.generatedCode = "";
    state.selectionIds = [];
    state.primarySelectionId = null;
    state.selectedElement = null;
    state.pendingIndexId = null;
    state.connectMode = false;
    state.hasFitCanvas = false;
    reconcileTensorOrder();
    clearHistory();
    render();
    setStatus(message, "success");
  }

  function addTensorAtCenter() {
    const center = viewportCenterPosition();
    const suggestedPosition = suggestTensorPosition(center);
    const tensor = createTensor(suggestedPosition.x, suggestedPosition.y);
    applyDesignChange(
      () => {
        state.spec.tensors.push(tensor);
        bringTensorToFront(tensor.id);
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
        Math.abs(tensor.position.x - candidate.x) < 170 &&
        Math.abs(tensor.position.y - candidate.y) < 120
      );
    });
  }

  function centerTensor(tensorId) {
    const tensor = findTensorById(tensorId);
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
    render();
    setStatus(
      state.connectMode
        ? "Connect mode active. Click two open indices with the same dimension."
        : "Connect mode disabled."
    );
  }

  function handleConnectClick(indexId) {
    const located = findIndexOwner(indexId);
    if (!located) {
      return;
    }
    if (findEdgeByIndexId(indexId)) {
      setStatus("This index is already connected. Delete the connection first.", "error");
      return;
    }

    if (!state.pendingIndexId) {
      state.pendingIndexId = indexId;
      setSelectedElement("index", indexId);
      setStatus("First index selected. Click another compatible open index to connect.");
      return;
    }

    if (state.pendingIndexId === indexId) {
      state.pendingIndexId = null;
      setStatus("Connection cancelled.");
      return;
    }

    const left = findIndexOwner(state.pendingIndexId);
    if (!left) {
      state.pendingIndexId = null;
      return;
    }
    if (left.index.dimension !== located.index.dimension) {
      setStatus("Connected indices must have the same dimension.", "error");
      return;
    }

    const newEdgeId = makeId("edge");
    state.pendingIndexId = null;
    applyDesignChange(
      () => {
        state.spec.edges.push({
          id: newEdgeId,
          name: nextName("bond", state.spec.edges.map((edge) => edge.name)),
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
      setStatus("Nothing is selected to delete.");
      return;
    }
    applyDesignChange(
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
    const selectedTensorIds = new Set(getSelectedIdsByKind("tensor"));
    const selectedIndexIds = new Set(getSelectedIdsByKind("index"));
    const selectedEdgeIds = new Set(getSelectedIdsByKind("edge"));

    selectedTensorIds.forEach((tensorId) => {
      removeTensor(tensorId);
    });

    selectedIndexIds.forEach((indexId) => {
      const located = findIndexOwner(indexId);
      if (located && !selectedTensorIds.has(located.tensor.id)) {
        removeIndex(located.tensor.id, indexId);
      }
    });

    selectedEdgeIds.forEach((edgeId) => {
      if (findEdgeById(edgeId)) {
        removeEdge(edgeId);
      }
    });
  }

  async function generateCode() {
    try {
      const payload = await apiPost("/api/generate", {
        engine: state.selectedEngine,
        spec: serializeCurrentSpec(),
      });
      if (!payload.ok) {
        setStatus(formatIssues(payload.issues), "error");
        return;
      }
      state.generatedCode = stripImportLines(payload.code);
      generatedCode.value = state.generatedCode;
      setStatus(`Generated ${payload.engine} code.`, "success");
    } catch (error) {
      setStatus(`Code generation failed: ${error.message}`, "error");
    }
  }

  async function completeEditor() {
    try {
      const payload = await apiPost("/api/complete", {
        engine: state.selectedEngine,
        spec: serializeCurrentSpec(),
      });
      if (!payload.ok) {
        setStatus(formatIssues(payload.issues), "error");
        return;
      }
      state.editorFinished = true;
      setStatus("Returning the design to Python. You can close this tab.", "success");
      window.setTimeout(() => {
        window.close();
      }, 150);
    } catch (error) {
      setStatus(`Could not finish the editor session: ${error.message}`, "error");
    }
  }

  async function cancelEditor() {
    try {
      state.editorFinished = true;
      await apiPost("/api/cancel", {});
      setStatus("Editor cancelled. You can close this tab.", "success");
      window.setTimeout(() => {
        window.close();
      }, 150);
    } catch (error) {
      setStatus(`Could not cancel the editor session: ${error.message}`, "error");
    }
  }

  function saveDesign() {
    const blob = new Blob([JSON.stringify(serializeCurrentSpec(), null, 2)], {
      type: "application/json",
    });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `${sanitizeFilename(state.spec.name || "tensor-network")}.json`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
    setStatus("Design downloaded as JSON.");
  }

  function loadDesignFromFile(event) {
    const file = event.target.files[0];
    if (!file) {
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const payload = JSON.parse(reader.result);
        resetDesignState(payload.network ? payload.network : payload, `Loaded design from ${file.name}. History cleared.`);
      } catch (error) {
        setStatus(`Could not load ${file.name}: ${error.message}`, "error");
      } finally {
        loadInput.value = "";
      }
    };
    reader.readAsText(file, "utf-8");
  }

  async function copyGeneratedCode() {
    const codeToCopy = stripImportLines(generatedCode.value);
    if (!codeToCopy.trim()) {
      setStatus("There is no generated code to copy yet.");
      return;
    }
    try {
      await navigator.clipboard.writeText(codeToCopy);
      setStatus("Generated code copied to the clipboard without import lines.", "success");
    } catch (error) {
      setStatus(`Could not copy the generated code: ${error.message}`, "error");
    }
  }

  // SECTION: export-minimap

  function handleMinimapMouseDown(event) {
    event.preventDefault();
    event.stopPropagation();
    if (!state.minimapTransform) {
      return;
    }
    state.minimapDrag = { active: true };
    minimapCanvas.classList.add("is-dragging");
    updateViewportFromMinimapClientPoint(event.clientX, event.clientY);
  }

  function updateViewportFromMinimapClientPoint(clientX, clientY) {
    if (!state.minimapTransform || !state.cy) {
      return;
    }
    const rect = minimapCanvas.getBoundingClientRect();
    const localX = clamp(clientX - rect.left, 0, rect.width);
    const localY = clamp(clientY - rect.top, 0, rect.height);
    const transform = state.minimapTransform;
    const modelPoint = {
      x: (localX - transform.offsetX) / transform.scale + transform.bounds.x1,
      y: (localY - transform.offsetY) / transform.scale + transform.bounds.y1,
    };
    centerViewportAt(modelPoint);
  }

  function centerViewportAt(point) {
    if (!state.cy) {
      return;
    }
    const zoom = state.cy.zoom();
    state.cy.pan({
      x: state.cy.width() / 2 - point.x * zoom,
      y: state.cy.height() / 2 - point.y * zoom,
    });
    renderMinimap();
  }

  function renderMinimap() {
    const context = minimapCanvas.getContext("2d");
    if (!context) {
      return;
    }
    const canvasWidth = minimapCanvas.width;
    const canvasHeight = minimapCanvas.height;
    context.clearRect(0, 0, canvasWidth, canvasHeight);
    context.fillStyle = "#0d121b";
    context.fillRect(0, 0, canvasWidth, canvasHeight);

    if (!state.spec || !state.spec.tensors.length) {
      context.fillStyle = "#95a3b8";
      context.font = '12px "Segoe UI", "Helvetica Neue", sans-serif';
      context.textAlign = "center";
      context.fillText("Minimap will appear here.", canvasWidth / 2, canvasHeight / 2);
      state.minimapTransform = null;
      return;
    }

    const worldBounds = computeDesignBounds(48);
    const innerWidth = canvasWidth - 16;
    const innerHeight = canvasHeight - 16;
    const scale = Math.min(innerWidth / Math.max(1, worldBounds.x2 - worldBounds.x1), innerHeight / Math.max(1, worldBounds.y2 - worldBounds.y1));
    const drawnWidth = (worldBounds.x2 - worldBounds.x1) * scale;
    const drawnHeight = (worldBounds.y2 - worldBounds.y1) * scale;
    const offsetX = (canvasWidth - drawnWidth) / 2;
    const offsetY = (canvasHeight - drawnHeight) / 2;

    state.minimapTransform = {
      bounds: worldBounds,
      scale,
      offsetX,
      offsetY,
    };

    context.save();
    context.translate(offsetX, offsetY);
    context.scale(scale, scale);

    state.spec.edges.forEach((edge) => {
      const left = findIndexOwner(edge.left.index_id);
      const right = findIndexOwner(edge.right.index_id);
      if (!left || !right) {
        return;
      }
      const source = indexAbsolutePosition(left.tensor, left.index);
      const target = indexAbsolutePosition(right.tensor, right.index);
      const curve = buildQuadraticCurve(source, target);
      context.beginPath();
      context.strokeStyle = state.selectionIds.includes(edge.id) ? "#8bc2ff" : getMetadataColor(edge.metadata, "#8da1c3");
      context.lineWidth = 3 / scale;
      context.moveTo(source.x - worldBounds.x1, source.y - worldBounds.y1);
      context.quadraticCurveTo(
        curve.control.x - worldBounds.x1,
        curve.control.y - worldBounds.y1,
        target.x - worldBounds.x1,
        target.y - worldBounds.y1
      );
      context.stroke();
    });

    state.spec.tensors.forEach((tensor) => {
      const tensorColor = getMetadataColor(tensor.metadata, "#18212c");
      const left = tensor.position.x - TENSOR_WIDTH / 2 - worldBounds.x1;
      const top = tensor.position.y - TENSOR_HEIGHT / 2 - worldBounds.y1;
      drawRoundRectPath(context, left, top, TENSOR_WIDTH, TENSOR_HEIGHT, 22);
      context.fillStyle = tensorColor;
      context.fill();
      context.lineWidth = (state.selectionIds.includes(tensor.id) ? 3 : 2) / scale;
      context.strokeStyle = state.selectionIds.includes(tensor.id) ? "#8bc2ff" : shiftColor(tensorColor, 26);
      context.stroke();

      tensor.indices.forEach((index) => {
        const absolutePosition = indexAbsolutePosition(tensor, index);
        const indexColor = getIndexColor(index, Boolean(findEdgeByIndexId(index.id)));
        context.beginPath();
        context.fillStyle = indexColor;
        context.strokeStyle = state.selectionIds.includes(index.id) ? "#8bc2ff" : shiftColor(indexColor, 34);
        context.lineWidth = (state.selectionIds.includes(index.id) ? 3 : 1.5) / scale;
        context.arc(absolutePosition.x - worldBounds.x1, absolutePosition.y - worldBounds.y1, INDEX_RADIUS, 0, Math.PI * 2);
        context.fill();
        context.stroke();
      });
    });

    context.restore();

    if (state.cy) {
      const extent = state.cy.extent();
      const topLeft = worldToMinimapPoint({ x: extent.x1, y: extent.y1 });
      const bottomRight = worldToMinimapPoint({ x: extent.x2, y: extent.y2 });
      context.fillStyle = "rgba(97, 168, 255, 0.12)";
      context.strokeStyle = "#8bc2ff";
      context.lineWidth = 2;
      context.fillRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
      context.strokeRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
    }
  }

  function worldToMinimapPoint(point) {
    return {
      x: state.minimapTransform.offsetX + (point.x - state.minimapTransform.bounds.x1) * state.minimapTransform.scale,
      y: state.minimapTransform.offsetY + (point.y - state.minimapTransform.bounds.y1) * state.minimapTransform.scale,
    };
  }

  function downloadPngExport() {
    if (!state.cy || !state.spec) {
      return;
    }
    try {
      const pngDataUrl = withSelectionSuppressed(() =>
        state.cy.png({
          full: true,
          scale: 2,
          bg: "#0b0f14",
        })
      );
      downloadDataUrl(`${sanitizeFilename(state.spec.name || "tensor-network")}.png`, pngDataUrl);
      setStatus("Exported a PNG image of the current design.", "success");
    } catch (error) {
      setStatus(`Could not export PNG: ${error.message}`, "error");
    }
  }

  function downloadSvgExport() {
    if (!state.spec) {
      return;
    }
    try {
      const svgText = buildSvgExport();
      const blob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
      downloadBlob(`${sanitizeFilename(state.spec.name || "tensor-network")}.svg`, blob);
      setStatus("Exported an SVG image of the current design.", "success");
    } catch (error) {
      setStatus(`Could not export SVG: ${error.message}`, "error");
    }
  }

  function buildSvgExport() {
    const bounds = computeDesignBounds(56);
    const width = Math.max(240, Math.ceil(bounds.x2 - bounds.x1));
    const height = Math.max(180, Math.ceil(bounds.y2 - bounds.y1));
    const lines = [];

    lines.push('<?xml version="1.0" encoding="UTF-8"?>');
    lines.push(
      `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="${bounds.x1} ${bounds.y1} ${width} ${height}">`
    );
    lines.push(`<rect x="${bounds.x1}" y="${bounds.y1}" width="${width}" height="${height}" fill="#0b0f14" />`);

    state.spec.edges.forEach((edge) => {
      const left = findIndexOwner(edge.left.index_id);
      const right = findIndexOwner(edge.right.index_id);
      if (!left || !right) {
        return;
      }
      const source = indexAbsolutePosition(left.tensor, left.index);
      const target = indexAbsolutePosition(right.tensor, right.index);
      const curve = buildQuadraticCurve(source, target);
      const edgeColor = getMetadataColor(edge.metadata, "#8da1c3");
      const labelPosition = quadraticPointAt(source, curve.control, target, 0.5);
      lines.push(
        `<path d="M ${source.x} ${source.y} Q ${curve.control.x} ${curve.control.y} ${target.x} ${target.y}" fill="none" stroke="${edgeColor}" stroke-width="3" />`
      );
      lines.push(
        `<text x="${labelPosition.x}" y="${labelPosition.y - 10}" fill="${shiftColor(edgeColor, 72)}" font-size="11" font-family="Segoe UI, Helvetica Neue, sans-serif" text-anchor="middle">${escapeSvgText(edge.name)}</text>`
      );
    });

    state.spec.tensors.forEach((tensor) => {
      const tensorColor = getMetadataColor(tensor.metadata, "#18212c");
      const borderColor = shiftColor(tensorColor, 26);
      lines.push(
        `<rect x="${tensor.position.x - TENSOR_WIDTH / 2}" y="${tensor.position.y - TENSOR_HEIGHT / 2}" width="${TENSOR_WIDTH}" height="${TENSOR_HEIGHT}" rx="22" ry="22" fill="${tensorColor}" stroke="${borderColor}" stroke-width="2" />`
      );
      lines.push(
        `<text x="${tensor.position.x}" y="${tensor.position.y - TENSOR_HEIGHT / 2 + 26}" fill="${readableTextColor(tensorColor)}" font-size="18" font-family="Georgia, Times New Roman, serif" text-anchor="middle">${escapeSvgText(tensor.name)}</text>`
      );

      tensor.indices.forEach((index, indexPosition) => {
        const absolutePosition = indexAbsolutePosition(tensor, index);
        const indexColor = getIndexColor(index, Boolean(findEdgeByIndexId(index.id)));
        lines.push(
          `<circle cx="${absolutePosition.x}" cy="${absolutePosition.y}" r="${INDEX_RADIUS}" fill="${indexColor}" stroke="${shiftColor(indexColor, 34)}" stroke-width="2" />`
        );
        lines.push(
          `<text x="${absolutePosition.x}" y="${absolutePosition.y + 4}" fill="${readableTextColor(indexColor)}" font-size="12" font-family="Segoe UI, Helvetica Neue, sans-serif" font-weight="700" text-anchor="middle">${indexPosition + 1}</text>`
        );
        lines.push(
          `<text x="${absolutePosition.x}" y="${absolutePosition.y + 28}" fill="${shiftColor(indexColor, 64)}" font-size="10" font-family="Segoe UI, Helvetica Neue, sans-serif" text-anchor="middle">${escapeSvgText(`${index.name} · ${index.dimension}`)}</text>`
        );
      });
    });

    lines.push("</svg>");
    return lines.join("\n");
  }

  function withSelectionSuppressed(action) {
    if (!state.cy) {
      return action();
    }
    const selectionIds = [...state.selectionIds];
    state.cy.$(":selected").unselect();
    try {
      return action();
    } finally {
      state.selectionIds = selectionIds;
      syncSelectedElementState();
      syncCySelection();
    }
  }

  // SECTION: utilities

  function populateEngineOptions(engines) {
    engineSelect.innerHTML = "";
    engines.forEach((engineName) => {
      const option = document.createElement("option");
      option.value = engineName;
      option.textContent = engineName;
      if (engineName === state.selectedEngine) {
        option.selected = true;
      }
      engineSelect.appendChild(option);
    });
  }

  function serializeCurrentSpec() {
    return {
      schema_version: 1,
      network: state.spec,
    };
  }

  function stripImportLines(code) {
    const keptLines = code
      .split(/\r?\n/)
      .filter((line) => !/^\s*(import|from)\s+/.test(line));
    while (keptLines.length && keptLines[0].trim() === "") {
      keptLines.shift();
    }
    while (keptLines.length && keptLines[keptLines.length - 1].trim() === "") {
      keptLines.pop();
    }
    return keptLines.join("\n");
  }

  function moveIndex(tensorId, indexPosition, direction) {
    const tensor = findTensorById(tensorId);
    if (!tensor) {
      return;
    }
    const targetPosition = indexPosition + direction;
    if (targetPosition < 0 || targetPosition >= tensor.indices.length) {
      return;
    }
    const [index] = tensor.indices.splice(indexPosition, 1);
    tensor.indices.splice(targetPosition, 0, index);
  }

  function removeTensor(tensorId) {
    const tensor = findTensorById(tensorId);
    if (!tensor) {
      return;
    }
    const tensorIndexIds = new Set(tensor.indices.map((index) => index.id));
    state.spec.edges = state.spec.edges.filter(
      (edge) => !tensorIndexIds.has(edge.left.index_id) && !tensorIndexIds.has(edge.right.index_id)
    );
    state.spec.tensors = state.spec.tensors.filter((candidate) => candidate.id !== tensorId);
    state.tensorOrder = state.tensorOrder.filter((candidateId) => candidateId !== tensorId);
  }

  function removeIndex(tensorId, indexId) {
    const tensor = findTensorById(tensorId);
    if (!tensor) {
      return;
    }
    state.spec.edges = state.spec.edges.filter(
      (edge) => edge.left.index_id !== indexId && edge.right.index_id !== indexId
    );
    tensor.indices = tensor.indices.filter((index) => index.id !== indexId);
  }

  function removeEdge(edgeId) {
    state.spec.edges = state.spec.edges.filter((edge) => edge.id !== edgeId);
  }

  function findTensorById(tensorId) {
    return state.spec.tensors.find((tensor) => tensor.id === tensorId) || null;
  }

  function findEdgeById(edgeId) {
    return state.spec.edges.find((edge) => edge.id === edgeId) || null;
  }

  function findIndexOwner(indexId) {
    for (const tensor of state.spec.tensors) {
      const indexPosition = tensor.indices.findIndex((index) => index.id === indexId);
      if (indexPosition >= 0) {
        return { tensor, index: tensor.indices[indexPosition], indexPosition };
      }
    }
    return null;
  }

  function findEdgeByIndexId(indexId) {
    return (
      state.spec.edges.find(
        (edge) => edge.left.index_id === indexId || edge.right.index_id === indexId
      ) || null
    );
  }

  function createTensor(x, y) {
    const tensor = {
      id: makeId("tensor"),
      name: nextName("T", state.spec.tensors.map((tensor) => tensor.name)),
      position: { x, y },
      indices: [],
      metadata: {},
    };
    tensor.indices.push(createIndex(tensor, 0));
    tensor.indices.push(createIndex(tensor, 1));
    return tensor;
  }

  function createIndex(tensor, indexPosition) {
    return {
      id: makeId("index"),
      name: nextName("i", tensor.indices.map((index) => index.name)),
      dimension: 2,
      offset: defaultIndexOffsetForOrder(indexPosition),
      metadata: {},
    };
  }

  function normalizeSpec(spec) {
    const normalized = deepClone(spec || {});
    normalized.metadata = isObject(normalized.metadata) ? normalized.metadata : {};
    normalized.tensors = Array.isArray(normalized.tensors) ? normalized.tensors : [];
    normalized.edges = Array.isArray(normalized.edges) ? normalized.edges : [];

    normalized.tensors.forEach((tensor) => {
      tensor.metadata = isObject(tensor.metadata) ? tensor.metadata : {};
      tensor.position = {
        x: asFiniteNumber(tensor.position && tensor.position.x, 120),
        y: asFiniteNumber(tensor.position && tensor.position.y, 120),
      };
      tensor.indices = Array.isArray(tensor.indices) ? tensor.indices : [];
      tensor.indices.forEach((index, indexPosition) => {
        index.metadata = isObject(index.metadata) ? index.metadata : {};
        index.dimension = Math.max(1, Math.round(asFiniteNumber(index.dimension, 2)));
        index.offset = {
          x: asFiniteNumber(index.offset && index.offset.x, 0),
          y: asFiniteNumber(index.offset && index.offset.y, 0),
        };
        if (!index.name) {
          index.name = nextName("i", tensor.indices.slice(0, indexPosition).map((candidate) => candidate.name));
        }
      });
      ensureTensorIndexOffsets(tensor);
    });

    return normalized;
  }

  function ensureTensorIndexOffsets(tensor) {
    const needsAutoLayout =
      tensor.indices.length > 0 &&
      tensor.indices.every((index) => Math.abs(index.offset.x) < 0.001 && Math.abs(index.offset.y) < 0.001);

    tensor.indices.forEach((index, indexPosition) => {
      if (needsAutoLayout) {
        index.offset = defaultIndexOffsetForOrder(indexPosition);
      } else {
        index.offset = clampIndexOffset(index.offset);
      }
    });
  }

  function defaultIndexOffsetForOrder(indexPosition) {
    const slot = DEFAULT_INDEX_SLOTS[indexPosition];
    if (slot) {
      return clampIndexOffset(slot);
    }
    return clampIndexOffset({
      x: indexPosition % 2 === 0 ? -58 : 58,
      y: -30 + Math.floor(indexPosition / 2) * 18,
    });
  }

  function clampIndexOffset(offset) {
    return {
      x: clamp(
        asFiniteNumber(offset.x, 0),
        -TENSOR_WIDTH / 2 + INDEX_RADIUS + INDEX_PADDING,
        TENSOR_WIDTH / 2 - INDEX_RADIUS - INDEX_PADDING
      ),
      y: clamp(
        asFiniteNumber(offset.y, 0),
        -TENSOR_HEIGHT / 2 + INDEX_RADIUS + INDEX_PADDING,
        TENSOR_HEIGHT / 2 - INDEX_RADIUS - INDEX_PADDING
      ),
    };
  }

  function indexAbsolutePosition(tensor, index) {
    const offset = clampIndexOffset(index.offset);
    index.offset = offset;
    return {
      x: tensor.position.x + offset.x,
      y: tensor.position.y + offset.y,
    };
  }

  function syncIndexNodePositions(tensor) {
    runWithIndexSync(() => {
      tensor.indices.forEach((index) => {
        syncSingleIndexNodePosition(tensor, index);
      });
    });
  }

  function syncSingleIndexNodePosition(tensor, index) {
    if (!state.cy) {
      return;
    }
    const indexElement = state.cy.getElementById(index.id);
    const absolutePosition = indexAbsolutePosition(tensor, index);
    if (indexElement && indexElement.length) {
      indexElement.position(absolutePosition);
    }
    syncIndexLabelNodePosition(index, absolutePosition);
    if (!indexElement || !indexElement.length) {
      return;
    }
  }

  function syncIndexLabelNodePosition(index, absolutePosition) {
    if (!state.cy) {
      return;
    }
    const labelElement = state.cy.getElementById(indexLabelNodeId(index.id));
    if (!labelElement || !labelElement.length) {
      return;
    }
    labelElement.position(indexLabelPosition(absolutePosition));
    labelElement.data("label", `${index.name} · ${index.dimension}`);
    labelElement.data(
      "textColor",
      shiftColor(getIndexColor(index, Boolean(findEdgeByIndexId(index.id))), 64)
    );
  }

  function runWithIndexSync(action) {
    state.syncingIndexPositions = true;
    try {
      action();
    } finally {
      state.syncingIndexPositions = false;
    }
  }

  function reconcileTensorOrder() {
    const tensorIds = state.spec ? state.spec.tensors.map((tensor) => tensor.id) : [];
    const nextOrder = (Array.isArray(state.tensorOrder) ? state.tensorOrder : []).filter((tensorId) =>
      tensorIds.includes(tensorId)
    );
    tensorIds.forEach((tensorId) => {
      if (!nextOrder.includes(tensorId)) {
        nextOrder.push(tensorId);
      }
    });
    state.tensorOrder = nextOrder;
  }

  function tensorLayerRank(tensorId) {
    const index = state.tensorOrder.indexOf(tensorId);
    return index >= 0 ? index : 0;
  }

  function bringTensorToFront(tensorId) {
    if (!tensorId) {
      return;
    }
    reconcileTensorOrder();
    state.tensorOrder = state.tensorOrder.filter((id) => id !== tensorId);
    state.tensorOrder.push(tensorId);
    applyTensorLayerData();
  }

  function applyTensorLayerData() {
    if (!state.cy) {
      return;
    }
    reconcileTensorOrder();
    state.tensorOrder.forEach((tensorId, tensorRank) => {
      const tensorElement = state.cy.getElementById(tensorId);
      if (tensorElement && tensorElement.length) {
        const tensor = findTensorById(tensorId);
        const tensorColor = tensor ? getMetadataColor(tensor.metadata, "#18212c") : "#18212c";
        tensorElement.data("zIndex", 100 + tensorRank * 20);
        tensorElement.data("backgroundColor", tensorColor);
        tensorElement.data("borderColor", shiftColor(tensorColor, 26));
        tensorElement.data("textColor", readableTextColor(tensorColor));
      }
      const tensor = findTensorById(tensorId);
      if (!tensor) {
        return;
      }
      tensor.indices.forEach((index, indexPosition) => {
        const absolutePosition = indexAbsolutePosition(tensor, index);
        const indexElement = state.cy.getElementById(index.id);
        if (indexElement && indexElement.length) {
          const indexColor = getIndexColor(index, Boolean(findEdgeByIndexId(index.id)));
          indexElement.data("zIndex", 300 + tensorRank * 20 + indexPosition);
          indexElement.data("orderLabel", String(indexPosition + 1));
          indexElement.data("backgroundColor", indexColor);
          indexElement.data("borderColor", shiftColor(indexColor, 34));
          indexElement.data("textColor", readableTextColor(indexColor));
          indexElement.position(absolutePosition);
        }
        syncIndexLabelNodePosition(index, absolutePosition);
        const labelElement = state.cy.getElementById(indexLabelNodeId(index.id));
        if (labelElement && labelElement.length) {
          labelElement.data("zIndex", 310 + tensorRank * 20 + indexPosition);
        }
      });
    });
    state.cy.edges().forEach((edgeElement) => {
      const edge = findEdgeById(edgeElement.id());
      const edgeColor = edge ? getMetadataColor(edge.metadata, "#8da1c3") : "#8da1c3";
      edgeElement.data("zIndex", 220);
      edgeElement.data("lineColor", edgeColor);
      edgeElement.data("textColor", shiftColor(edgeColor, 72));
    });
  }

  function applyColorToSelection(colorValue) {
    getSelectedEntries().forEach((entry) => {
      if (entry.kind === "tensor") {
        entry.tensor.metadata.color = colorValue;
      } else if (entry.kind === "index") {
        entry.located.index.metadata.color = colorValue;
      } else if (entry.kind === "edge") {
        entry.edge.metadata.color = colorValue;
      }
    });
  }

  function getBatchColorValue(selectedEntries) {
    if (!selectedEntries.length) {
      return "#61a8ff";
    }
    return getEntryColor(selectedEntries[0]);
  }

  function getEntryColor(entry) {
    if (entry.kind === "tensor") {
      return getMetadataColor(entry.tensor.metadata, "#18212c");
    }
    if (entry.kind === "index") {
      return getMetadataColor(
        entry.located.index.metadata,
        getIndexColor(entry.located.index, Boolean(findEdgeByIndexId(entry.id)))
      );
    }
    return getMetadataColor(entry.edge.metadata, "#8da1c3");
  }

  function buildQuadraticCurve(source, target) {
    const midpoint = {
      x: (source.x + target.x) / 2,
      y: (source.y + target.y) / 2,
    };
    const deltaX = target.x - source.x;
    const deltaY = target.y - source.y;
    const distance = Math.max(1, Math.sqrt(deltaX * deltaX + deltaY * deltaY));
    const normal = { x: -deltaY / distance, y: deltaX / distance };
    const bend = clamp(distance * 0.18, 18, 60);
    return {
      control: {
        x: midpoint.x + normal.x * bend,
        y: midpoint.y + normal.y * bend,
      },
    };
  }

  function quadraticPointAt(source, control, target, t) {
    const inverse = 1 - t;
    return {
      x: inverse * inverse * source.x + 2 * inverse * t * control.x + t * t * target.x,
      y: inverse * inverse * source.y + 2 * inverse * t * control.y + t * t * target.y,
    };
  }

  function drawRoundRectPath(context, x, y, width, height, radius) {
    const effectiveRadius = Math.min(radius, width / 2, height / 2);
    context.beginPath();
    context.moveTo(x + effectiveRadius, y);
    context.lineTo(x + width - effectiveRadius, y);
    context.quadraticCurveTo(x + width, y, x + width, y + effectiveRadius);
    context.lineTo(x + width, y + height - effectiveRadius);
    context.quadraticCurveTo(x + width, y + height, x + width - effectiveRadius, y + height);
    context.lineTo(x + effectiveRadius, y + height);
    context.quadraticCurveTo(x, y + height, x, y + height - effectiveRadius);
    context.lineTo(x, y + effectiveRadius);
    context.quadraticCurveTo(x, y, x + effectiveRadius, y);
    context.closePath();
  }

  function computeDesignBounds(padding) {
    const bounds = {
      x1: Number.POSITIVE_INFINITY,
      y1: Number.POSITIVE_INFINITY,
      x2: Number.NEGATIVE_INFINITY,
      y2: Number.NEGATIVE_INFINITY,
    };

    state.spec.tensors.forEach((tensor) => {
      expandBounds(bounds, tensor.position.x - TENSOR_WIDTH / 2, tensor.position.y - TENSOR_HEIGHT / 2);
      expandBounds(bounds, tensor.position.x + TENSOR_WIDTH / 2, tensor.position.y + TENSOR_HEIGHT / 2);
      tensor.indices.forEach((index) => {
        const absolutePosition = indexAbsolutePosition(tensor, index);
        expandBounds(bounds, absolutePosition.x - INDEX_RADIUS, absolutePosition.y - INDEX_RADIUS);
        expandBounds(bounds, absolutePosition.x + INDEX_RADIUS, absolutePosition.y + INDEX_RADIUS);
        expandBounds(bounds, absolutePosition.x + 50, absolutePosition.y + 42);
      });
    });

    state.spec.edges.forEach((edge) => {
      const left = findIndexOwner(edge.left.index_id);
      const right = findIndexOwner(edge.right.index_id);
      if (!left || !right) {
        return;
      }
      const source = indexAbsolutePosition(left.tensor, left.index);
      const target = indexAbsolutePosition(right.tensor, right.index);
      const curve = buildQuadraticCurve(source, target);
      expandBounds(bounds, source.x, source.y);
      expandBounds(bounds, target.x, target.y);
      expandBounds(bounds, curve.control.x, curve.control.y);
    });

    if (!Number.isFinite(bounds.x1)) {
      if (state.cy) {
        const extent = state.cy.extent();
        return {
          x1: extent.x1 - padding,
          y1: extent.y1 - padding,
          x2: extent.x2 + padding,
          y2: extent.y2 + padding,
        };
      }
      return {
        x1: -padding,
        y1: -padding,
        x2: 240 + padding,
        y2: 200 + padding,
      };
    }

    return {
      x1: bounds.x1 - padding,
      y1: bounds.y1 - padding,
      x2: bounds.x2 + padding,
      y2: bounds.y2 + padding,
    };
  }

  function expandBounds(bounds, x, y) {
    bounds.x1 = Math.min(bounds.x1, x);
    bounds.y1 = Math.min(bounds.y1, y);
    bounds.x2 = Math.max(bounds.x2, x);
    bounds.y2 = Math.max(bounds.y2, y);
  }

  function downloadDataUrl(filename, dataUrl) {
    const anchor = document.createElement("a");
    anchor.href = dataUrl;
    anchor.download = filename;
    anchor.click();
  }

  function downloadBlob(filename, blob) {
    const anchor = document.createElement("a");
    const objectUrl = URL.createObjectURL(blob);
    anchor.href = objectUrl;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(objectUrl);
  }

  async function apiGet(path) {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(await response.text());
    }
    return response.json();
  }

  async function apiPost(path, payload) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.message || "Request failed.");
    }
    return body;
  }

  function updateToolbarState() {
    undoButton.disabled = state.undoStack.length === 0;
    redoButton.disabled = state.redoStack.length === 0;
  }

  function formatIssues(issues) {
    if (!issues || !issues.length) {
      return "The design is not valid yet.";
    }
    return issues
      .slice(0, 3)
      .map((issue) => issue.message)
      .join(" ");
  }

  function setStatus(message, kind = "info") {
    if (!statusMessage) {
      return;
    }
    statusMessage.textContent = message;
    statusMessage.classList.remove("status-error", "status-success");
    if (kind === "error") {
      statusMessage.classList.add("status-error");
    }
    if (kind === "success") {
      statusMessage.classList.add("status-success");
    }
  }

  function sanitizeFilename(value) {
    const sanitized = value.toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
    return sanitized.replace(/^-+|-+$/g, "") || "tensor-network";
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function escapeSvgText(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&apos;");
  }

  function isIndexNode(element) {
    return element.isNode() && element.data("kind") === "index";
  }

  function isTextInput(element) {
    return Boolean(element) && ["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName);
  }

  function deepClone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function clientPointToCanvasPoint(clientX, clientY) {
    const rect = canvasShell.getBoundingClientRect();
    return {
      x: clamp(clientX - rect.left, 0, rect.width),
      y: clamp(clientY - rect.top, 0, rect.height),
    };
  }

  function normalizedBox(startPoint, currentPoint) {
    const left = Math.min(startPoint.x, currentPoint.x);
    const top = Math.min(startPoint.y, currentPoint.y);
    const width = Math.abs(currentPoint.x - startPoint.x);
    const height = Math.abs(currentPoint.y - startPoint.y);
    return { left, top, width, height };
  }

  function boxesIntersect(leftBox, rightBox) {
    return !(
      leftBox.left + leftBox.width < rightBox.x1 ||
      leftBox.left > rightBox.x2 ||
      leftBox.top + leftBox.height < rightBox.y1 ||
      leftBox.top > rightBox.y2
    );
  }

  function indexLabelNodeId(indexId) {
    return `${indexId}__label`;
  }

  function indexLabelPosition(indexPositionAbsolute) {
    return {
      x: indexPositionAbsolute.x,
      y: indexPositionAbsolute.y + 28,
    };
  }

  function getIndexColor(index, isConnected) {
    return getMetadataColor(index.metadata, isConnected ? "#61a8ff" : "#e0b566");
  }

  function getMetadataColor(metadata, fallback) {
    const candidate = metadata && typeof metadata.color === "string" ? metadata.color.trim() : "";
    return /^#[0-9a-fA-F]{6}$/.test(candidate) ? candidate.toLowerCase() : fallback;
  }

  function shiftColor(hexColor, amount) {
    const { red, green, blue } = parseHexColor(hexColor);
    return formatColorHex({
      red: clamp(Math.round(red + amount), 0, 255),
      green: clamp(Math.round(green + amount), 0, 255),
      blue: clamp(Math.round(blue + amount), 0, 255),
    });
  }

  function readableTextColor(hexColor) {
    const { red, green, blue } = parseHexColor(hexColor);
    const luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255;
    return luminance > 0.62 ? "#091018" : "#f5f9ff";
  }

  function parseHexColor(hexColor) {
    const normalized = getMetadataColor({ color: hexColor }, "#000000");
    return {
      red: Number.parseInt(normalized.slice(1, 3), 16),
      green: Number.parseInt(normalized.slice(3, 5), 16),
      blue: Number.parseInt(normalized.slice(5, 7), 16),
    };
  }

  function formatColorHex({ red, green, blue }) {
    return `#${[red, green, blue]
      .map((component) => component.toString(16).padStart(2, "0"))
      .join("")}`;
  }

  function isObject(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  }

  function asFiniteNumber(value, fallback) {
    const numericValue = Number(value);
    return Number.isFinite(numericValue) ? numericValue : fallback;
  }

  function makeId(prefix) {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return `${prefix}_${window.crypto.randomUUID().replace(/-/g, "").slice(0, 10)}`;
    }
    return `${prefix}_${Math.random().toString(16).slice(2, 12)}`;
  }

  function nextName(prefix, existingNames) {
    let counter = 1;
    while (existingNames.includes(`${prefix}${counter}`)) {
      counter += 1;
    }
    return `${prefix}${counter}`;
  }

  function tensorIndexNameExists(tensor, candidateName, excludedIndexId = null) {
    const normalizedCandidate = candidateName.trim();
    return tensor.indices.some(
      (index) =>
        index.id !== excludedIndexId &&
        typeof index.name === "string" &&
        index.name.trim() === normalizedCandidate
    );
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }
})();

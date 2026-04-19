import { GRAPH_THEME, UI_THEME } from "./theme.js";
import { createCytoscapeGraphAdapter } from "./views/cytoscapeGraphAdapter.js";
import { createGraphElementModelBuilder } from "./views/graphElementModel.js";

export function registerGraphRender(ctx) {
  const state = ctx.state;
  const TENSOR_BASE_Z_INDEX = 10;
  const EDGE_Z_INDEX = 100;
  const PORT_BASE_Z_INDEX = 200;
  const INDEX_LABEL_BASE_Z_INDEX = 230;
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
  const graphAdapter = createCytoscapeGraphAdapter({
    state,
    getCy: () => state.cy,
  });
  const graphElementModelBuilder = createGraphElementModelBuilder({
    state,
    buildContractionScene: () =>
      typeof ctx.buildContractionScene === "function" ? ctx.buildContractionScene() : null,
    ensureTensorIndexOffsets: (tensor) => ctx.ensureTensorIndexOffsets(tensor),
    findTensorById: (tensorId) => ctx.findTensorById(tensorId),
    getIndexColor: (index, isConnected) => ctx.getIndexColor(index, isConnected),
    getMetadataColor: (metadata, fallbackColor) =>
      ctx.getMetadataColor(metadata, fallbackColor),
    getMetadataFilterEntityState: (entityKind, entityId, metadataFilterHighlight) =>
      ctx.getMetadataFilterEntityState(entityKind, entityId, metadataFilterHighlight),
    getMetadataFilterHighlight: () =>
      typeof ctx.getMetadataFilterHighlight === "function"
        ? ctx.getMetadataFilterHighlight()
        : null,
    indexAbsolutePosition: (tensor, index) => ctx.indexAbsolutePosition(tensor, index),
    indexLabelNodeId: (indexId) => ctx.indexLabelNodeId(indexId),
    indexLabelPosition: (position) => ctx.indexLabelPosition(position),
    isInspectingPastStage: () =>
      typeof ctx.isInspectingPastStage === "function" && ctx.isInspectingPastStage(),
    readableTextColor: (color) => ctx.readableTextColor(color),
    shiftColor: (color, amount) => ctx.shiftColor(color, amount),
    tensorHeight: (tensor) => ctx.tensorHeight(tensor),
    tensorLayerRank: (tensorId) => ctx.tensorLayerRank(tensorId),
    tensorWidth: (tensor) => ctx.tensorWidth(tensor),
    zIndexes: {
      edge: EDGE_Z_INDEX,
      indexLabel: INDEX_LABEL_BASE_Z_INDEX,
      port: PORT_BASE_Z_INDEX,
      tensor: TENSOR_BASE_Z_INDEX,
    },
  });

  function initGraph() {
    state.cy = cytoscape({
      container: document.getElementById("canvas"),
      layout: { name: "preset" },
      minZoom: 0.3,
      maxZoom: 2.5,
      selectionType: "additive",
      wheelSensitivity: 0.18,
      userPanningEnabled: true,
      userZoomingEnabled: false,
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
            width: "data(width)",
            height: "data(height)",
            "background-color": "data(backgroundColor)",
            "border-width": 2,
            "border-color": "data(borderColor)",
            color: "data(textColor)",
            label: "data(label)",
            "font-size": 18,
            "font-family": UI_THEME.fontFamily,
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
          selector: "node.planner-pending-tensor",
          style: {
            "border-color": GRAPH_THEME.pendingTensor,
            "border-width": 4,
            "overlay-color": GRAPH_THEME.pendingTensor,
            "overlay-opacity": 0.1,
          },
        },
        {
          selector: "node.planner-pending-index",
          style: {
            "border-color": GRAPH_THEME.pendingIndex,
            "border-width": 4,
            "overlay-color": GRAPH_THEME.pendingIndex,
            "overlay-opacity": 0.18,
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
            "text-background-color": GRAPH_THEME.selectionTextBackground,
            "text-background-opacity": 0.92,
            "text-background-padding": 4,
            "text-rotation": "autorotate",
            "target-arrow-shape": "none",
            "source-arrow-shape": "none",
            "z-index": "data(zIndex)",
          },
        },
        {
          selector: ".metadata-filter-context",
          style: {
            opacity: 0.62,
            "text-opacity": 0.82,
          },
        },
        {
          selector: ".metadata-filter-dim",
          style: {
            opacity: 0.22,
            "text-opacity": 0.38,
          },
        },
        {
          selector: "node[kind = 'tensor']:selected",
          style: {
            "border-color": GRAPH_THEME.selection,
            "border-width": 4,
            "overlay-opacity": 0,
          },
        },
        {
          selector: "node[kind = 'index']:selected",
          style: {
            "border-color": GRAPH_THEME.selection,
            "overlay-opacity": 0,
          },
        },
        {
          selector: "edge:selected",
          style: {
            "line-color": GRAPH_THEME.selection,
          },
        },
      ],
    });

    state.cy.on("tap", "node, edge", (event) => {
      if (
        event.originalEvent &&
        Number.isFinite(event.originalEvent.button) &&
        event.originalEvent.button === 2
      ) {
        return;
      }
      if (typeof ctx.closeCanvasContextMenu === "function") {
        ctx.closeCanvasContextMenu();
      }
      if (state.boxSelection) {
        return;
      }
      const element = event.target;
      const kind = element.data("kind");
      if (kind === "index-label") {
        return;
      }
      if (
        kind === "edge" &&
        typeof ctx.isInspectingPastStage === "function" &&
        ctx.isInspectingPastStage()
      ) {
        return;
      }
      if (state.plannerMode && kind === "tensor") {
        ctx.handlePlannerOperandClick(element.id());
        return;
      }
      if (state.connectMode && ctx.isIndexNode(element)) {
        ctx.handleConnectClick(element.id());
        return;
      }
      if (kind === "tensor") {
        ctx.bringTensorToFront(element.id());
      } else if (kind === "index") {
        const located = ctx.findIndexOwner(element.id());
        if (located) {
          ctx.bringTensorToFront(located.tensor.id);
        }
      }
      if (typeof ctx.toggleSidebarCollapsed === "function") {
        ctx.toggleSidebarCollapsed(false);
      }
      if (typeof ctx.setActiveSidebarTab === "function") {
        ctx.setActiveSidebarTab("selection");
      }
      ctx.selectElement(kind, element.id(), { additive: Boolean(event.originalEvent && event.originalEvent.shiftKey) });
    });

    state.cy.on("cxttap", "node, edge", (event) => {
      const element = event.target;
      const kind = element.data("kind");
      if (kind !== "tensor" && kind !== "index" && kind !== "edge") {
        return;
      }
      if (typeof ctx.cancelPendingBoxSelection === "function") {
        ctx.cancelPendingBoxSelection();
      }
      if (event.originalEvent) {
        event.originalEvent.preventDefault();
        event.originalEvent.stopPropagation();
      }
      if (typeof ctx.openCanvasContextMenu === "function") {
        const selectedTensorIds =
          kind === "tensor" && typeof ctx.getSelectedIdsByKind === "function"
            ? ctx.getSelectedIdsByKind("tensor")
            : [];
        const menuKind =
          kind === "tensor" &&
          Array.isArray(state.selectionIds) &&
          selectedTensorIds.length >= 2 &&
          selectedTensorIds.includes(element.id())
            ? "selection"
            : kind;
        ctx.openCanvasContextMenu({
          kind: menuKind,
          id: element.id(),
          clientX:
            event.originalEvent && Number.isFinite(event.originalEvent.clientX)
              ? event.originalEvent.clientX
              : 0,
          clientY:
            event.originalEvent && Number.isFinite(event.originalEvent.clientY)
              ? event.originalEvent.clientY
              : 0,
        });
      }
    });

    state.cy.on("tap", (event) => {
      if (
        event.originalEvent &&
        Number.isFinite(event.originalEvent.button) &&
        event.originalEvent.button === 2
      ) {
        return;
      }
      if (event.target === state.cy && !state.boxSelection) {
        if (typeof ctx.closeCanvasContextMenu === "function") {
          ctx.closeCanvasContextMenu();
        }
        ctx.clearSelection({ preservePendingIndex: true });
      }
    });

    state.cy.on("grab", "node[kind = 'tensor']", (event) => {
      const tensorId = event.target.id();
      ctx.bringTensorToFront(tensorId);
      if (!state.selectionIds.includes(tensorId)) {
        ctx.setSelection([tensorId], { primaryId: tensorId });
      }
      state.activeTensorDrag = createTensorDragState(tensorId);
    });

    state.cy.on("position", "node[kind = 'tensor']", (event) => {
      if (state.syncingTensorPositions) {
        return;
      }
      const tensor = typeof ctx.findVisibleTensorById === "function"
        ? ctx.findVisibleTensorById(event.target.id())
        : ctx.findTensorById(event.target.id());
      if (!tensor) {
        return;
      }
      const candidatePosition = {
        x: event.target.position("x"),
        y: event.target.position("y"),
      };
      const nextPosition = {
        x: Math.round(candidatePosition.x),
        y: Math.round(candidatePosition.y),
      };
      if (
        typeof ctx.canEditCurrentContractionStage === "function" &&
        ctx.canEditCurrentContractionStage() &&
        typeof ctx.updateCurrentStageOperandLayout === "function"
      ) {
        ctx.updateCurrentStageOperandLayout(tensor.id, { position: nextPosition });
      } else {
        tensor.position.x = nextPosition.x;
        tensor.position.y = nextPosition.y;
      }
      if (
        Math.abs(candidatePosition.x - nextPosition.x) > 0.5 ||
        Math.abs(candidatePosition.y - nextPosition.y) > 0.5
      ) {
        ctx.runWithTensorSync(() => {
          event.target.position(nextPosition);
        });
      }
      ctx.syncIndexNodePositions(tensor);
      if (state.activeTensorDrag && state.activeTensorDrag.anchorId === tensor.id) {
        moveCompanionTensorsDuringDrag();
      }
      ctx.renderOverlayDecorations();
    });

    state.cy.on("dragfree", "node[kind = 'tensor']", (event) => {
      const tensor = typeof ctx.findVisibleTensorById === "function"
        ? ctx.findVisibleTensorById(event.target.id())
        : ctx.findTensorById(event.target.id());
      if (tensor) {
        ctx.syncIndexNodePositions(tensor);
      }
      finishTensorDrag(event.target.id());
      ctx.renderProperties();
      ctx.renderMinimap();
    });

    state.cy.on("grab", "node[kind = 'index']", (event) => {
      const located = ctx.findIndexOwner(event.target.id());
      if (located) {
        ctx.bringTensorToFront(located.tensor.id);
      }
      state.activeIndexDrag = {
        indexId: event.target.id(),
        startOffset: located
          ? {
            x: located.index.offset.x,
            y: located.index.offset.y,
          }
          : null,
        snapshot: ctx.createHistorySnapshot(),
      };
    });

    state.cy.on("position", "node[kind = 'index']", (event) => {
      if (state.syncingIndexPositions) {
        return;
      }
      const located = ctx.findIndexOwner(event.target.id());
      if (!located) {
        return;
      }
      located.index.offset = ctx.clampIndexOffset({
        x: event.target.position("x") - located.tensor.position.x,
        y: event.target.position("y") - located.tensor.position.y,
      }, located.tensor);
      const absolutePosition = ctx.indexAbsolutePosition(located.tensor, located.index);
      ctx.syncIndexLabelNodePosition(located.index, absolutePosition);
      if (
        Math.abs(absolutePosition.x - event.target.position("x")) > 0.5 ||
        Math.abs(absolutePosition.y - event.target.position("y")) > 0.5
      ) {
        ctx.runWithIndexSync(() => {
          event.target.position(absolutePosition);
        });
      }
    });

    state.cy.on("dragfree", "node[kind = 'index']", (event) => {
      const located = ctx.findIndexOwner(event.target.id());
      if (located) {
        located.index.offset = ctx.clampIndexOffset(located.index.offset, located.tensor);
        ctx.syncSingleIndexNodePosition(located.tensor, located.index);
      }
      finishIndexDrag(event.target.id());
      ctx.renderProperties();
      ctx.renderMinimap();
    });

    state.cy.on("pan zoom resize", () => {
      ctx.renderOverlayDecorations();
      ctx.renderMinimap();
    });
  }

  function render(options = {}) {
    const resolvedOptions = {
      graph: true,
      properties: true,
      code: true,
      toolbar: true,
      overlays: true,
      planner: true,
      sidebarTabs: true,
      minimap: true,
      syncSelection: false,
      ...options,
    };
    if (resolvedOptions.graph) {
      renderGraph();
    } else if (resolvedOptions.syncSelection) {
      ctx.syncCySelection();
    }
    if (resolvedOptions.properties) {
      if (typeof ctx.renderMetadataFilters === "function") {
        ctx.renderMetadataFilters();
      }
      ctx.renderProperties();
    }
    if (resolvedOptions.code) {
      if (typeof ctx.renderGeneratedCodePreview === "function") {
        ctx.renderGeneratedCodePreview(state.generatedCode);
      }
    }
    connectButton.classList.toggle("is-active", state.connectMode);
    helpModal.classList.toggle("is-hidden", !state.isHelpOpen);
    if (resolvedOptions.toolbar) {
      ctx.updateToolbarState();
    }
    if (resolvedOptions.overlays) {
      ctx.renderOverlayDecorations();
    }
    if (resolvedOptions.planner && typeof ctx.renderPlanner === "function") {
      ctx.renderPlanner();
    }
    if (resolvedOptions.sidebarTabs && typeof ctx.renderSidebarTabs === "function") {
      ctx.renderSidebarTabs();
    }
    if (resolvedOptions.minimap) {
      ctx.renderMinimap();
    }
  }

  function buildLegacyGraphElementModel(contractionScene = null) {
    const descriptorsById = {};
    const orderedIds = [];
    const connectedIndexIds = new Set();
    const resolvedContractionScene =
      contractionScene ||
      (typeof ctx.buildContractionScene === "function" ? ctx.buildContractionScene() : null);
    const visibleTensors = resolvedContractionScene
      ? resolvedContractionScene.tensors
      : state.spec.tensors;
    const visibleEdges = resolvedContractionScene
      ? resolvedContractionScene.edges
      : state.spec.edges;
    const readOnlyScene = Boolean(
      resolvedContractionScene &&
        typeof ctx.isInspectingPastStage === "function" &&
        ctx.isInspectingPastStage()
    );
    const indexNodesInteractive = !readOnlyScene;
    const metadataFilterHighlight =
      typeof ctx.getMetadataFilterHighlight === "function"
        ? ctx.getMetadataFilterHighlight()
        : null;

    function appendDescriptor(descriptor) {
      orderedIds.push(descriptor.data.id);
      descriptorsById[descriptor.data.id] = descriptor;
    }

    function joinClasses(...classNames) {
      return classNames.filter(Boolean).join(" ");
    }

    function getMetadataFilterClass(entityKind, entityId) {
      if (
        !metadataFilterHighlight ||
        typeof ctx.getMetadataFilterEntityState !== "function"
      ) {
        return "";
      }
      const entityState = ctx.getMetadataFilterEntityState(
        entityKind,
        entityId,
        metadataFilterHighlight
      );
      if (entityState === "context") {
        return "metadata-filter-context";
      }
      if (entityState === "dim") {
        return "metadata-filter-dim";
      }
      return "";
    }

    visibleEdges.forEach((edge) => {
      connectedIndexIds.add(edge.leftIndexId || edge.left.index_id);
      connectedIndexIds.add(edge.rightIndexId || edge.right.index_id);
    });

    visibleTensors.forEach((tensor) => {
      if (!resolvedContractionScene) {
        ctx.ensureTensorIndexOffsets(tensor);
      }
      const tensorRank = ctx.tensorLayerRank(tensor.id);
      const anchorTensor =
        tensor.isDerived &&
        Array.isArray(tensor.sourceTensorIds) &&
        tensor.sourceTensorIds.length
          ? ctx.findTensorById(tensor.sourceTensorIds[0])
          : ctx.findTensorById(tensor.id);
      const tensorColor = ctx.getMetadataColor(
        anchorTensor ? anchorTensor.metadata : null,
        GRAPH_THEME.tensorFallback
      );
      appendDescriptor({
        group: "nodes",
        data: {
          id: tensor.id,
          label: tensor.name,
          kind: "tensor",
          width: ctx.tensorWidth(tensor),
          height: ctx.tensorHeight(tensor),
          resultCount: Number(tensor.resultCount || 0),
          backgroundColor: tensorColor,
          borderColor: ctx.shiftColor(tensorColor, 26),
          textColor: ctx.readableTextColor(tensorColor),
          zIndex: TENSOR_BASE_Z_INDEX + tensorRank,
        },
        classes: joinClasses(
          state.pendingPlannerSelectionId === tensor.id
            ? "planner-pending-tensor"
            : "",
          getMetadataFilterClass("tensor", tensor.id)
        ),
        position: { x: tensor.position.x, y: tensor.position.y },
        grabbable: !readOnlyScene,
        selectable: true,
      });

      tensor.indices.forEach((index, indexPosition) => {
        const indexColor = ctx.getIndexColor(index, connectedIndexIds.has(index.id));
        const indexPositionAbsolute = resolvedContractionScene
          ? {
              x: tensor.position.x + index.offset.x,
              y: tensor.position.y + index.offset.y,
            }
          : ctx.indexAbsolutePosition(tensor, index);
        appendDescriptor({
          group: "nodes",
          data: {
            id: index.id,
            kind: "index",
            tensor_id: tensor.id,
            orderLabel: String(indexPosition + 1),
            backgroundColor: indexColor,
            borderColor: ctx.shiftColor(indexColor, 34),
            textColor: ctx.readableTextColor(indexColor),
            zIndex: PORT_BASE_Z_INDEX + tensorRank * 10 + indexPosition,
          },
          classes: [
            connectedIndexIds.has(index.id) ? "index-connected" : "index-open",
            state.pendingIndexId === index.id ? "planner-pending-index" : "",
            getMetadataFilterClass("index", index.id),
          ]
            .filter(Boolean)
            .join(" "),
          position: indexPositionAbsolute,
          grabbable: indexNodesInteractive,
          selectable: indexNodesInteractive,
        });

        appendDescriptor({
          group: "nodes",
          data: {
            id: ctx.indexLabelNodeId(index.id),
            kind: "index-label",
            label: `${index.name} · ${index.dimension}`,
            textColor: ctx.shiftColor(indexColor, 64),
            zIndex: INDEX_LABEL_BASE_Z_INDEX + tensorRank * 10 + indexPosition,
          },
          classes: getMetadataFilterClass("index", index.id),
          position: ctx.indexLabelPosition(indexPositionAbsolute),
          grabbable: false,
          selectable: false,
        });
      });
    });

    visibleEdges.forEach((edge) => {
      const edgeColor = ctx.getMetadataColor(edge.metadata, GRAPH_THEME.edge);
      appendDescriptor({
        group: "edges",
        data: {
          id: edge.id,
          source: edge.leftIndexId || edge.left.index_id,
          target: edge.rightIndexId || edge.right.index_id,
          label: edge.name || edge.label || "",
          kind: "edge",
          lineColor: edgeColor,
          textColor: ctx.shiftColor(edgeColor, 72),
          zIndex: EDGE_Z_INDEX,
        },
        classes: getMetadataFilterClass("edge", edge.id),
        position: null,
        grabbable: false,
        selectable: !readOnlyScene,
      });
    });

    return {
      descriptorsById,
      elements: orderedIds.map((elementId) => descriptorsById[elementId]),
      ephemeralSignature: [
        state.pendingPlannerSelectionId || "",
        state.pendingIndexId || "",
        readOnlyScene ? "readonly" : "editable",
      ].join("|"),
      orderedIds,
      visibleSignature: orderedIds.join("|"),
      visibleTensors,
    };
  }

  function renderGraph() {
    if (!state.cy || !state.spec) {
      return;
    }
    graphAdapter.ensureForCurrentCy();
    ctx.reconcileTensorOrder();
    const contractionScene =
      typeof ctx.buildContractionScene === "function" ? ctx.buildContractionScene() : null;
    const graphModel = graphElementModelBuilder(contractionScene);
    state.cy.batch(() => {
      graphAdapter.applyModel(graphModel);
    });
    syncPendingInteractionClasses();
    if (!state.hasFitCanvas) {
      if (graphModel.visibleTensors.length) {
        state.cy.fit(undefined, 40);
      } else {
        state.cy.center();
      }
      state.hasFitCanvas = true;
    }
    ctx.syncCySelection();
  }

  function syncPendingInteractionClasses() {
    if (!state.cy) {
      return;
    }
    const previousPlannerSelectionId = state.pendingInteractionRenderedPlannerSelectionId;
    const previousIndexId = state.pendingInteractionRenderedIndexId;
    const nextPlannerSelectionId = state.pendingPlannerSelectionId || null;
    const nextIndexId = state.pendingIndexId || null;
    if (
      previousPlannerSelectionId === nextPlannerSelectionId &&
      previousIndexId === nextIndexId
    ) {
      return;
    }
    state.cy.batch(() => {
      new Set([previousPlannerSelectionId, nextPlannerSelectionId]).forEach((tensorId) => {
        if (!tensorId) {
          return;
        }
        const tensorNode = state.cy.getElementById(tensorId);
        if (tensorNode && tensorNode.length) {
          tensorNode.toggleClass(
            "planner-pending-tensor",
            tensorId === nextPlannerSelectionId
          );
        }
      });
      new Set([previousIndexId, nextIndexId]).forEach((indexId) => {
        if (!indexId) {
          return;
        }
        const indexNode = state.cy.getElementById(indexId);
        if (indexNode && indexNode.length) {
          indexNode.toggleClass("planner-pending-index", indexId === nextIndexId);
        }
      });
    });
    state.pendingInteractionRenderedPlannerSelectionId = nextPlannerSelectionId;
    state.pendingInteractionRenderedIndexId = nextIndexId;
  }

  function buildGraphElements(contractionScene = null) {
    return graphElementModelBuilder(contractionScene).elements;
  }

  function createTensorDragState(anchorId) {
    const dragSelection = ctx.buildCanvasSelectionDragState(anchorId);
    return {
      anchorId,
      ...dragSelection,
    };
  }

  function moveCompanionTensorsDuringDrag() {
    if (!state.activeTensorDrag || !state.cy) {
      return;
    }
    const anchor = typeof ctx.findVisibleTensorById === "function"
      ? ctx.findVisibleTensorById(state.activeTensorDrag.anchorId)
      : ctx.findTensorById(state.activeTensorDrag.anchorId);
    const anchorStartPosition =
      state.activeTensorDrag.tensorStartPositions[state.activeTensorDrag.anchorId];
    if (!anchor || !anchorStartPosition) {
      return;
    }
    const deltaX = anchor.position.x - anchorStartPosition.x;
    const deltaY = anchor.position.y - anchorStartPosition.y;
    ctx.runWithTensorSync(() => {
      state.activeTensorDrag.tensorIds.forEach((tensorId) => {
        if (tensorId === anchor.id) {
          return;
        }
        const tensor = typeof ctx.findVisibleTensorById === "function"
          ? ctx.findVisibleTensorById(tensorId)
          : ctx.findTensorById(tensorId);
        const startPosition = state.activeTensorDrag.tensorStartPositions[tensorId];
        if (!tensor || !startPosition) {
          return;
        }
        const nextPosition = {
          x: Math.round(startPosition.x + deltaX),
          y: Math.round(startPosition.y + deltaY),
        };
        if (
          typeof ctx.canEditCurrentContractionStage === "function" &&
          ctx.canEditCurrentContractionStage() &&
          typeof ctx.updateCurrentStageOperandLayout === "function"
        ) {
          ctx.updateCurrentStageOperandLayout(tensor.id, { position: nextPosition });
          tensor.position = nextPosition;
        } else {
          tensor.position.x = nextPosition.x;
          tensor.position.y = nextPosition.y;
        }
        const tensorElement = state.cy.getElementById(tensor.id);
        if (tensorElement && tensorElement.length) {
          tensorElement.position(tensor.position);
        }
        ctx.syncIndexNodePositions(tensor);
      });
    });
    state.activeTensorDrag.noteIds.forEach((noteId) => {
      const note = ctx.findNoteById(noteId);
      const startPosition = state.activeTensorDrag.noteStartPositions[noteId];
      if (!note || !startPosition) {
        return;
      }
      note.position.x = Math.round(startPosition.x + deltaX);
      note.position.y = Math.round(startPosition.y + deltaY);
    });
  }

  function finishTensorDrag(anchorId) {
    if (!state.activeTensorDrag || state.activeTensorDrag.anchorId !== anchorId) {
      return;
    }
    const changed =
      state.activeTensorDrag.tensorIds.some((tensorId) => {
        const tensor = typeof ctx.findVisibleTensorById === "function"
          ? ctx.findVisibleTensorById(tensorId)
          : ctx.findTensorById(tensorId);
        const startPosition = state.activeTensorDrag.tensorStartPositions[tensorId];
        return (
          tensor &&
          startPosition &&
          (tensor.position.x !== startPosition.x || tensor.position.y !== startPosition.y)
        );
      }) ||
      state.activeTensorDrag.noteIds.some((noteId) => {
        const note = ctx.findNoteById(noteId);
        const startPosition = state.activeTensorDrag.noteStartPositions[noteId];
        return (
          note &&
          startPosition &&
          (note.position.x !== startPosition.x || note.position.y !== startPosition.y)
        );
      });
    if (changed) {
      ctx.commitHistorySnapshot(state.activeTensorDrag.snapshot);
    }
    state.activeTensorDrag = null;
    ctx.updateToolbarState();
  }

  function finishIndexDrag(indexId) {
    if (!state.activeIndexDrag || state.activeIndexDrag.indexId !== indexId) {
      return;
    }
    const located = ctx.findIndexOwner(indexId);
    const changed =
      located &&
      state.activeIndexDrag.startOffset &&
      (
        located.index.offset.x !== state.activeIndexDrag.startOffset.x ||
        located.index.offset.y !== state.activeIndexDrag.startOffset.y
      );
    if (changed) {
      ctx.commitHistorySnapshot(state.activeIndexDrag.snapshot);
    }
    state.activeIndexDrag = null;
    ctx.updateToolbarState();
  }

  Object.assign(ctx, {
    initGraph,
    render,
    renderGraph,
    syncPendingInteractionClasses,
    buildGraphElements,
    createTensorDragState,
    moveCompanionTensorsDuringDrag,
    finishTensorDrag,
    finishIndexDrag
  });
}

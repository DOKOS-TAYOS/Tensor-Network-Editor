import { GRAPH_THEME, UI_THEME } from "../core/theme.js";
import { createCytoscapeGraphAdapter } from "../views/cytoscapeGraphAdapter.js";
import { createGraphElementModelBuilder } from "../views/graphElementModel.js";
import { createGraphRenderDragSupport } from "./graphRenderDrag.js";
import { createGraphRenderLifecycle } from "./graphRenderLifecycle.js";
import { createGraphRenderTooltipSupport } from "./graphRenderTooltips.js";

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
    findIndexOwner: (indexId) => ctx.findIndexOwner(indexId),
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
    hyperedgeHubNodeId: (hyperedgeId) => ctx.hyperedgeHubNodeId(hyperedgeId),
    hyperedgeSpokeEdgeId: (hyperedgeId, endpointPosition) =>
      ctx.hyperedgeSpokeEdgeId(hyperedgeId, endpointPosition),
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
  const { hideBoundaryTensorTooltip, showBoundaryTensorTooltip } =
    createGraphRenderTooltipSupport({
      ctx,
      canvasShell,
    });
  const dragSupport = createGraphRenderDragSupport({
    ctx,
    state,
  });
  const lifecycle = createGraphRenderLifecycle({
    ctx,
    state,
    connectButton,
    helpModal,
    graphAdapter,
    graphElementModelBuilder,
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
          selector: "node[kind = 'hyperedge-hub']",
          style: {
            shape: "diamond",
            width: 20,
            height: 20,
            label: "data(label)",
            "font-size": 10,
            color: "data(textColor)",
            "text-valign": "top",
            "text-halign": "center",
            "text-margin-y": -16,
            "border-width": 2,
            "border-color": "data(borderColor)",
            "background-color": "data(backgroundColor)",
            "overlay-opacity": 0,
            "z-index": "data(zIndex)",
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
          selector: "edge[kind = 'hyperedge-spoke']",
          style: {
            width: 2,
            "line-style": "dashed",
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
          selector: "node[kind = 'hyperedge-hub']:selected",
          style: {
            "border-color": GRAPH_THEME.selection,
            "border-width": 4,
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
      hideBoundaryTensorTooltip();
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
      const baseHyperedgeId =
        typeof element.data === "function" ? element.data("baseHyperedgeId") : null;
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
      if (kind === "hyperedge-hub" || baseHyperedgeId) {
        if (typeof ctx.toggleSidebarCollapsed === "function") {
          ctx.toggleSidebarCollapsed(false);
        }
        if (typeof ctx.setActiveSidebarTab === "function") {
          ctx.setActiveSidebarTab("selection");
        }
        const additiveSelection =
          typeof ctx.isAdditiveSelectionModifier === "function" &&
          ctx.isAdditiveSelectionModifier(event.originalEvent);
        const selectionId =
          kind === "hyperedge-hub" ? element.id() : ctx.hyperedgeHubNodeId(baseHyperedgeId);
        ctx.selectElement("hyperedge", selectionId, {
          additive: additiveSelection,
        });
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
      const additiveSelection =
        typeof ctx.isAdditiveSelectionModifier === "function" &&
        ctx.isAdditiveSelectionModifier(event.originalEvent);
      if (
        kind === "tensor" &&
        additiveSelection &&
        state.activeTensorDrag &&
        state.activeTensorDrag.anchorId === element.id() &&
        state.activeTensorDrag.addedSelectionOnGrab
      ) {
        return;
      }
      if (
        kind === "index" &&
        additiveSelection &&
        state.activeIndexDrag &&
        state.activeIndexDrag.indexId === element.id() &&
        state.activeIndexDrag.addedSelectionOnGrab
      ) {
        return;
      }
      ctx.selectElement(kind, element.id(), {
        additive: additiveSelection,
      });
    });

    state.cy.on("cxttap", "node, edge", (event) => {
      hideBoundaryTensorTooltip();
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
      hideBoundaryTensorTooltip();
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

    state.cy.on("mouseover", "node[kind = 'tensor']", (event) => {
      showBoundaryTensorTooltip(event.target);
    });

    state.cy.on("mouseout", "node[kind = 'tensor']", () => {
      hideBoundaryTensorTooltip();
    });

    state.cy.on("grab", "node[kind = 'tensor']", (event) => {
      hideBoundaryTensorTooltip();
      const tensorId = event.target.id();
      const additiveSelection =
        typeof ctx.isAdditiveSelectionModifier === "function" &&
        ctx.isAdditiveSelectionModifier(event.originalEvent);
      const tensorWasSelected = state.selectionIds.includes(tensorId);
      ctx.bringTensorToFront(tensorId);
      if (additiveSelection && !tensorWasSelected) {
        ctx.setSelection([...state.selectionIds, tensorId], { primaryId: tensorId });
      } else if (!tensorWasSelected) {
        ctx.setSelection([tensorId], { primaryId: tensorId });
      }
      state.activeTensorDrag = {
        addedSelectionOnGrab: additiveSelection && !tensorWasSelected,
        ...createTensorDragState(tensorId),
      };
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

    function handleTensorRelease(event) {
      const tensor = typeof ctx.findVisibleTensorById === "function"
        ? ctx.findVisibleTensorById(event.target.id())
        : ctx.findTensorById(event.target.id());
      if (tensor) {
        ctx.syncIndexNodePositions(tensor);
      }
      finishTensorDrag(event.target.id());
      ctx.renderProperties();
      ctx.renderMinimap();
    }

    state.cy.on("dragfree", "node[kind = 'tensor']", handleTensorRelease);
    state.cy.on("free", "node[kind = 'tensor']", handleTensorRelease);

    state.cy.on("grab", "node[kind = 'index']", (event) => {
      hideBoundaryTensorTooltip();
      const indexId = event.target.id();
      const additiveSelection =
        typeof ctx.isAdditiveSelectionModifier === "function" &&
        ctx.isAdditiveSelectionModifier(event.originalEvent);
      const indexWasSelected = state.selectionIds.includes(indexId);
      const located = ctx.findIndexOwner(indexId);
      if (located) {
        ctx.bringTensorToFront(located.tensor.id);
      }
      if (additiveSelection && !indexWasSelected) {
        ctx.setSelection([...state.selectionIds, indexId], { primaryId: indexId });
      } else if (!indexWasSelected) {
        ctx.setSelection([indexId], { primaryId: indexId });
      }
      state.activeIndexDrag = {
        addedSelectionOnGrab: additiveSelection && !indexWasSelected,
        indexId,
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
      if (typeof ctx.syncHyperedgeHubNodePositions === "function") {
        ctx.syncHyperedgeHubNodePositions([located.index.id]);
      }
      if (
        Math.abs(absolutePosition.x - event.target.position("x")) > 0.5 ||
        Math.abs(absolutePosition.y - event.target.position("y")) > 0.5
      ) {
        ctx.runWithIndexSync(() => {
          event.target.position(absolutePosition);
        });
      }
    });

    function handleIndexRelease(event) {
      const located = ctx.findIndexOwner(event.target.id());
      if (located) {
        located.index.offset = ctx.clampIndexOffset(located.index.offset, located.tensor);
        ctx.syncSingleIndexNodePosition(located.tensor, located.index);
      }
      finishIndexDrag(event.target.id());
      ctx.renderProperties();
      ctx.renderMinimap();
    }

    state.cy.on("dragfree", "node[kind = 'index']", handleIndexRelease);
    state.cy.on("free", "node[kind = 'index']", handleIndexRelease);

    state.cy.on("pan zoom resize", () => {
      hideBoundaryTensorTooltip();
      ctx.renderOverlayDecorations();
      ctx.renderMinimap();
    });
  }

  const { render, renderGraph, syncPendingInteractionClasses, buildGraphElements } =
    lifecycle;
  const {
    createTensorDragState,
    moveCompanionTensorsDuringDrag,
    finishTensorDrag,
    finishIndexDrag,
  } = dragSupport;

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

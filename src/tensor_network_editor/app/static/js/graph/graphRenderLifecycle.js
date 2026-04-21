import { GRAPH_THEME } from "../core/theme.js";

export function createGraphRenderLifecycle({
  ctx,
  state,
  connectButton,
  helpModal,
  graphAdapter,
  graphElementModelBuilder,
}) {
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
    if (resolvedOptions.code && typeof ctx.renderGeneratedCodePreview === "function") {
      ctx.renderGeneratedCodePreview(state.generatedCode);
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
          zIndex: ctx.constants ? undefined : undefined,
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

      const tensorDescriptor = descriptorsById[tensor.id];
      tensorDescriptor.data.zIndex = 10 + tensorRank;

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
            zIndex: 200 + tensorRank * 10 + indexPosition,
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
            zIndex: 230 + tensorRank * 10 + indexPosition,
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
          zIndex: 100,
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

  return {
    buildGraphElements,
    buildLegacyGraphElementModel,
    render,
    renderGraph,
    syncPendingInteractionClasses,
  };
}

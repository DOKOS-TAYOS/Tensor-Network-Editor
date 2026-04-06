export function registerGraphRender(ctx) {
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
          selector: "node.planner-pending-tensor",
          style: {
            "border-color": "#ff8c87",
            "border-width": 4,
            "overlay-color": "#ff8c87",
            "overlay-opacity": 0.1,
          },
        },
        {
          selector: "node.planner-pending-index",
          style: {
            "border-color": "#61c7ff",
            "border-width": 4,
            "overlay-color": "#61c7ff",
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
      if (
        kind === "edge" &&
        typeof ctx.isContractionSceneVisible === "function" &&
        ctx.isContractionSceneVisible()
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

    state.cy.on("tap", (event) => {
      if (event.target === state.cy && !state.boxSelection) {
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

  function render() {
    renderGraph();
    ctx.renderProperties();
    generatedCode.value = state.generatedCode;
    connectButton.classList.toggle("is-active", state.connectMode);
    helpModal.classList.toggle("is-hidden", !state.isHelpOpen);
    ctx.updateToolbarState();
    ctx.renderOverlayDecorations();
    if (typeof ctx.renderPlanner === "function") {
      ctx.renderPlanner();
    }
    if (typeof ctx.renderSidebarTabs === "function") {
      ctx.renderSidebarTabs();
    }
    ctx.renderMinimap();
  }

  function renderGraph() {
    if (!state.cy || !state.spec) {
      return;
    }
    ctx.reconcileTensorOrder();
    const elements = buildGraphElements();
    const visibleTensors =
      typeof ctx.getVisibleTensors === "function" ? ctx.getVisibleTensors() : state.spec.tensors;
    state.cy.batch(() => {
      state.cy.elements().remove();
      state.cy.add(elements);
    });
    ctx.applyTensorLayerData();
    syncPendingInteractionClasses();
    if (!state.hasFitCanvas) {
      if (visibleTensors.length) {
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
    state.cy.nodes("node[kind = 'tensor']").forEach((node) => {
      node.toggleClass("planner-pending-tensor", node.id() === state.pendingPlannerSelectionId);
    });
    state.cy.nodes("node[kind = 'index']").forEach((node) => {
      node.toggleClass("planner-pending-index", node.id() === state.pendingIndexId);
    });
  }

  function buildGraphElements() {
    const tensorElements = [];
    const edgeElements = [];
    const indexElements = [];
    const indexLabelElements = [];
    const connectedIndexIds = new Set();
    const contractionScene =
      typeof ctx.buildContractionScene === "function" ? ctx.buildContractionScene() : null;
    const visibleTensors = contractionScene ? contractionScene.tensors : state.spec.tensors;
    const visibleEdges = contractionScene ? contractionScene.edges : state.spec.edges;
    const readOnlyScene = Boolean(
      contractionScene && typeof ctx.isInspectingPastStage === "function" && ctx.isInspectingPastStage()
    );
    const indexNodesInteractive = !contractionScene;

    visibleEdges.forEach((edge) => {
      connectedIndexIds.add(edge.leftIndexId || edge.left.index_id);
      connectedIndexIds.add(edge.rightIndexId || edge.right.index_id);
    });

    visibleTensors.forEach((tensor) => {
      if (!contractionScene) {
        ctx.ensureTensorIndexOffsets(tensor);
      }
      const tensorRank = ctx.tensorLayerRank(tensor.id);
      const anchorTensor = tensor.isDerived && Array.isArray(tensor.sourceTensorIds) && tensor.sourceTensorIds.length
        ? ctx.findTensorById(tensor.sourceTensorIds[0])
        : ctx.findTensorById(tensor.id);
      const tensorColor = ctx.getMetadataColor(anchorTensor ? anchorTensor.metadata : null, "#18212c");
      tensorElements.push({
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
          zIndex: 100 + tensorRank * 20,
        },
        classes: state.pendingPlannerSelectionId === tensor.id ? "planner-pending-tensor" : "",
        position: { x: tensor.position.x, y: tensor.position.y },
        grabbable: !readOnlyScene,
        selectable: true,
      });

      tensor.indices.forEach((index, indexPosition) => {
        const indexColor = ctx.getIndexColor(index, connectedIndexIds.has(index.id));
        const indexPositionAbsolute = contractionScene
          ? {
              x: tensor.position.x + index.offset.x,
              y: tensor.position.y + index.offset.y,
            }
          : ctx.indexAbsolutePosition(tensor, index);
        indexElements.push({
          group: "nodes",
          data: {
            id: index.id,
            kind: "index",
            tensor_id: tensor.id,
            orderLabel: String(indexPosition + 1),
            backgroundColor: indexColor,
            borderColor: ctx.shiftColor(indexColor, 34),
            textColor: ctx.readableTextColor(indexColor),
            zIndex: 300 + tensorRank * 20 + indexPosition,
          },
          classes: [
            connectedIndexIds.has(index.id) ? "index-connected" : "index-open",
            state.pendingIndexId === index.id ? "planner-pending-index" : "",
          ]
            .filter(Boolean)
            .join(" "),
          position: indexPositionAbsolute,
          grabbable: indexNodesInteractive,
          selectable: indexNodesInteractive,
        });

        indexLabelElements.push({
          group: "nodes",
          data: {
            id: ctx.indexLabelNodeId(index.id),
            kind: "index-label",
            label: `${index.name} · ${index.dimension}`,
            textColor: ctx.shiftColor(indexColor, 64),
            zIndex: 310 + tensorRank * 20 + indexPosition,
          },
          position: ctx.indexLabelPosition(indexPositionAbsolute),
          grabbable: false,
          selectable: false,
        });
      });
    });

    visibleEdges.forEach((edge) => {
      const edgeColor = ctx.getMetadataColor(edge.metadata, "#8da1c3");
      edgeElements.push({
        group: "edges",
        data: {
          id: edge.id,
          source: edge.leftIndexId || edge.left.index_id,
          target: edge.rightIndexId || edge.right.index_id,
          label: edge.name || edge.label || "",
          kind: "edge",
          lineColor: edgeColor,
          textColor: ctx.shiftColor(edgeColor, 72),
          zIndex: 220,
        },
        selectable: !contractionScene,
      });
    });

    return [...tensorElements, ...edgeElements, ...indexElements, ...indexLabelElements];
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
    ctx.commitHistorySnapshot(state.activeTensorDrag.snapshot);
    state.activeTensorDrag = null;
    ctx.updateToolbarState();
  }

  function finishIndexDrag(indexId) {
    if (!state.activeIndexDrag || state.activeIndexDrag.indexId !== indexId) {
      return;
    }
    ctx.commitHistorySnapshot(state.activeIndexDrag.snapshot);
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

export function registerOverlaysLayoutTemplates(ctx) {
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

  function renderOverlayDecorations() {
    renderGroupOverlays();
    renderResizeHandles();
    renderContractionBadges();
    if (typeof ctx.renderNotes === "function") {
      ctx.renderNotes();
    }
  }

  function renderGroupOverlays() {
    if (!groupLayer) {
      return;
    }
    groupLayer.innerHTML = "";
    const scene =
      typeof ctx.buildContractionScene === "function" ? ctx.buildContractionScene() : null;
    const hideGroupsForContractedScene = Boolean(
      scene &&
      (scene.tensors.some((tensor) => tensor.isDerived) ||
        scene.tensors.length !== state.spec.tensors.length)
    );
    if (hideGroupsForContractedScene) {
      return;
    }
    state.spec.groups.forEach((group) => {
      const rect = groupDisplayRect(group);
      if (!rect) {
        return;
      }
      const color = ctx.getMetadataColor(group.metadata, "#61a8ff");
      const overlay = document.createElement("div");
      overlay.className = "group-overlay";
      if (state.selectionIds.includes(group.id)) {
        overlay.classList.add("is-selected");
      }
      if (Boolean(group.metadata && group.metadata.collapsed)) {
        overlay.classList.add("is-collapsed");
      }
      overlay.dataset.groupId = group.id;
      overlay.style.left = `${rect.left}px`;
      overlay.style.top = `${rect.top}px`;
      overlay.style.width = `${rect.width}px`;
      overlay.style.height = `${rect.height}px`;
      overlay.style.borderColor = color;
      overlay.style.background = `${color}14`;

      const label = document.createElement("div");
      label.className = "group-label";
      label.textContent = `${group.name} (${group.tensor_ids.length})`;
      overlay.appendChild(label);

      overlay.addEventListener("mousedown", (event) => startGroupDrag(event, group.id));
      overlay.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        ctx.selectElement("group", group.id, {
          additive: Boolean(event.shiftKey),
        });
      });
      groupLayer.appendChild(overlay);
    });
  }

  function renderResizeHandles() {
    if (!resizeLayer) {
      return;
    }
    resizeLayer.innerHTML = "";
    if (state.selectionIds.length !== 1) {
      return;
    }
    const selectedEntry = ctx.getSelectionEntry(state.selectionIds[0]);
    if (
      !selectedEntry ||
      (selectedEntry.kind !== "tensor" && selectedEntry.kind !== "contraction-tensor")
    ) {
      return;
    }
    if (
      selectedEntry.kind === "contraction-tensor" &&
      !(
        typeof ctx.canEditCurrentContractionStage === "function" &&
        ctx.canEditCurrentContractionStage()
      )
    ) {
      return;
    }
    const tensor = selectedEntry.tensor;
    const rect = tensorScreenRect(tensor);
    [
      { corner: "nw", left: rect.left, top: rect.top },
      { corner: "ne", left: rect.left + rect.width, top: rect.top },
      { corner: "sw", left: rect.left, top: rect.top + rect.height },
      { corner: "se", left: rect.left + rect.width, top: rect.top + rect.height },
    ].forEach((handleSpec) => {
      const handle = document.createElement("div");
      handle.className = `resize-handle corner-${handleSpec.corner}`;
      handle.style.left = `${handleSpec.left - 7}px`;
      handle.style.top = `${handleSpec.top - 7}px`;
      handle.addEventListener("mousedown", (event) => startTensorResize(event, tensor.id, handleSpec.corner));
      resizeLayer.appendChild(handle);
    });
  }

  function renderContractionBadges() {
    if (!resizeLayer) {
      return;
    }
    const previewOrderByTensorId =
      state.plannerPreviewOrderByTensorId && typeof state.plannerPreviewOrderByTensorId === "object"
        ? state.plannerPreviewOrderByTensorId
        : {};
    const visibleTensors =
      typeof ctx.getVisibleTensors === "function" ? ctx.getVisibleTensors() : state.spec.tensors;
    visibleTensors.forEach((tensor) => {
      const previewOrders = previewOrderByTensorId[tensor.id];
      if (Array.isArray(previewOrders) && previewOrders.length) {
        resizeLayer.appendChild(
          createPreviewBadgeStack(tensor, previewOrders)
        );
      }
    });
    const scene =
      typeof ctx.buildContractionScene === "function" ? ctx.buildContractionScene() : null;
    if (!scene) {
      return;
    }
    scene.tensors.forEach((tensor) => {
      if (Number(tensor.resultCount || 0) > 1) {
        resizeLayer.appendChild(createResultCountBadge(tensor));
      }
      const futureOrders = scene.futureOrdersByOperandId[tensor.id];
      if (
        typeof ctx.isInspectingPastStage === "function" &&
        ctx.isInspectingPastStage() &&
        Array.isArray(futureOrders) &&
        futureOrders.length
      ) {
        resizeLayer.appendChild(createFutureStepBadgeStack(tensor, futureOrders));
      }
    });
  }

  function createResultCountBadge(tensor) {
    const rect = tensorScreenRect(tensor);
    const badge = document.createElement("div");
    badge.className = "planner-result-count-badge";
    badge.textContent = String(Number(tensor.resultCount || 0));
    badge.style.left = `${rect.left + 8}px`;
    badge.style.top = `${rect.top + rect.height - 28}px`;
    return badge;
  }

  function createPreviewBadgeStack(tensor, orders) {
    const rect = tensorScreenRect(tensor);
    const stack = document.createElement("div");
    stack.className = "planner-order-badge-stack is-preview";
    stack.style.left = `${rect.left + rect.width - 1}px`;
    stack.style.top = `${rect.top + 1}px`;
    orders.forEach((order) => {
      const badge = document.createElement("div");
      badge.className = "planner-preview-badge";
      badge.textContent = String(order);
      stack.appendChild(badge);
    });
    return stack;
  }

  function createFutureStepBadgeStack(tensor, orders) {
    const rect = tensorScreenRect(tensor);
    const stack = document.createElement("div");
    const isOpen = Boolean(state.plannerFutureBadgeDisclosure[tensor.id]);
    stack.className = `planner-order-badge-stack planner-future-badge-stack${
      isOpen ? " is-open" : ""
    }`;
    stack.style.left = `${rect.left + 1}px`;
    stack.style.top = `${rect.top + 1}px`;

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "planner-order-badge planner-future-badge-toggle";
    toggle.textContent = String(orders[0]);
    toggle.addEventListener("mousedown", (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
    toggle.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (typeof ctx.toggleFutureBadgeDisclosure === "function") {
        ctx.toggleFutureBadgeDisclosure(tensor.id);
      }
      renderOverlayDecorations();
    });
    stack.appendChild(toggle);

    if (isOpen) {
      orders.slice(1).forEach((order) => {
        const badge = document.createElement("div");
        badge.className = "planner-order-badge";
        badge.textContent = String(order);
        stack.appendChild(badge);
      });
    }

    return stack;
  }

  function createGroupFromSelection() {
    const selectedTensorIds = ctx.getSelectedIdsByKind("tensor");
    if (selectedTensorIds.length < 2) {
      ctx.setStatus("Select at least two tensors to create a group.", "error");
      return;
    }
    const alreadyGrouped = selectedTensorIds.find((tensorId) => ctx.findGroupsByTensorId(tensorId).length > 0);
    if (alreadyGrouped) {
      ctx.setStatus("A tensor can only belong to one group in this editor.", "error");
      return;
    }
    const groupId = ctx.makeId("group");
    ctx.applyDesignChange(
      () => {
        state.spec.groups.push({
          id: groupId,
          name: ctx.nextName("Group ", state.spec.groups.map((group) => group.name)),
          tensor_ids: [...selectedTensorIds],
          metadata: {},
        });
      },
      {
        selectionIds: [groupId],
        primaryId: groupId,
        statusMessage: "Created a new tensor group.",
      }
    );
  }

  function toggleGroupCollapse(groupId) {
    const group = ctx.findGroupById(groupId);
    if (!group) {
      return;
    }
    const nextCollapsed = !Boolean(group.metadata.collapsed);
    ctx.applyDesignChange(
      () => {
        group.metadata.collapsed = nextCollapsed;
      },
      {
        selectionIds: [group.id],
        primaryId: group.id,
        statusMessage: nextCollapsed
          ? `Collapsed ${group.name}.`
          : `Expanded ${group.name}.`,
      }
    );
  }

  function startGroupDrag(event, groupId) {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const group = ctx.findGroupById(groupId);
    if (!group) {
      return;
    }
    if (Boolean(event.shiftKey) && !state.selectionIds.includes(groupId)) {
      ctx.setSelection([...state.selectionIds, groupId], { primaryId: groupId });
    } else if (!state.selectionIds.includes(groupId)) {
      ctx.setSelection([groupId], { primaryId: groupId });
    }
    const dragSelection = ctx.buildCanvasSelectionDragState(groupId);
    state.activeGroupDrag = {
      groupId,
      startPoint: ctx.clientPointToWorldPoint(event.clientX, event.clientY),
      ...dragSelection,
    };
  }

  function updateActiveGroupDrag(event) {
    if (!state.activeGroupDrag || !state.cy) {
      return;
    }
    const group = ctx.findGroupById(state.activeGroupDrag.groupId);
    if (!group) {
      return;
    }
    const worldPoint = ctx.clientPointToWorldPoint(event.clientX, event.clientY);
    const deltaX = worldPoint.x - state.activeGroupDrag.startPoint.x;
    const deltaY = worldPoint.y - state.activeGroupDrag.startPoint.y;
    ctx.applyCanvasSelectionDragDelta(state.activeGroupDrag, deltaX, deltaY);
    renderOverlayDecorations();
    ctx.renderMinimap();
  }

  function finishActiveGroupDrag() {
    if (!state.activeGroupDrag) {
      return;
    }
    ctx.commitHistorySnapshot(state.activeGroupDrag.snapshot);
    state.activeGroupDrag = null;
    ctx.updateToolbarState();
    ctx.render();
  }

  function startTensorResize(event, tensorId, corner) {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const tensor =
      typeof ctx.findVisibleTensorById === "function"
        ? ctx.findVisibleTensorById(tensorId)
        : ctx.findTensorById(tensorId);
    if (!tensor) {
      return;
    }
    state.activeResize = {
      tensorId,
      corner,
      snapshot: ctx.createHistorySnapshot(),
      center: { x: tensor.position.x, y: tensor.position.y },
      startSize: { width: tensorWidth(tensor), height: tensorHeight(tensor) },
      startOffsets: Object.fromEntries(
        (Array.isArray(tensor.indices) ? tensor.indices : []).map((index) => [
          index.id,
          { x: index.offset.x, y: index.offset.y },
        ])
      ),
      usesSnapshot:
        typeof ctx.canEditCurrentContractionStage === "function" &&
        ctx.canEditCurrentContractionStage(),
    };
  }

  function updateActiveResize(event) {
    if (!state.activeResize || !state.cy) {
      return;
    }
    const tensor =
      typeof ctx.findVisibleTensorById === "function"
        ? ctx.findVisibleTensorById(state.activeResize.tensorId)
        : ctx.findTensorById(state.activeResize.tensorId);
    if (!tensor) {
      return;
    }
    const worldPoint = ctx.clientPointToWorldPoint(event.clientX, event.clientY);
    const width = Math.max(
      MIN_TENSOR_WIDTH,
      Math.abs(worldPoint.x - state.activeResize.center.x) * 2
    );
    const height = Math.max(
      MIN_TENSOR_HEIGHT,
      Math.abs(worldPoint.y - state.activeResize.center.y) * 2
    );
    const widthRatio = width / Math.max(1, state.activeResize.startSize.width);
    const heightRatio = height / Math.max(1, state.activeResize.startSize.height);
    if (
      state.activeResize.usesSnapshot &&
      typeof ctx.updateCurrentStageOperandLayout === "function"
    ) {
      ctx.updateCurrentStageOperandLayout(tensor.id, {
        size: { width, height },
      });
      tensor.size.width = Math.round(width);
      tensor.size.height = Math.round(height);
    } else {
      tensor.size.width = Math.round(width);
      tensor.size.height = Math.round(height);
      tensor.indices.forEach((index) => {
        const startOffset = state.activeResize.startOffsets[index.id] || { x: 0, y: 0 };
        index.offset = ctx.clampIndexOffset(
          {
            x: startOffset.x * widthRatio,
            y: startOffset.y * heightRatio,
          },
          tensor
        );
      });
    }
    syncTensorElementSize(tensor);
    ctx.syncIndexNodePositions(tensor);
    renderOverlayDecorations();
    ctx.renderMinimap();
  }

  function finishActiveResize() {
    if (!state.activeResize) {
      return;
    }
    ctx.commitHistorySnapshot(state.activeResize.snapshot);
    state.activeResize = null;
    ctx.updateToolbarState();
    ctx.render();
  }

  function tensorWidth(tensor) {
    return Math.max(MIN_TENSOR_WIDTH, ctx.asFiniteNumber(tensor.size && tensor.size.width, TENSOR_WIDTH));
  }

  function tensorHeight(tensor) {
    return Math.max(MIN_TENSOR_HEIGHT, ctx.asFiniteNumber(tensor.size && tensor.size.height, TENSOR_HEIGHT));
  }

  function tensorScreenRect(tensor) {
    const topLeft = ctx.worldToCanvasPoint({
      x: tensor.position.x - tensorWidth(tensor) / 2,
      y: tensor.position.y - tensorHeight(tensor) / 2,
    });
    return {
      left: topLeft.x,
      top: topLeft.y,
      width: tensorWidth(tensor) * state.cy.zoom(),
      height: tensorHeight(tensor) * state.cy.zoom(),
    };
  }

  function groupDisplayRect(group) {
    const bounds = groupWorldBounds(group);
    if (!bounds) {
      return null;
    }
    if (Boolean(group.metadata && group.metadata.collapsed)) {
      const anchor = ctx.worldToCanvasPoint({
        x: (bounds.x1 + bounds.x2) / 2,
        y: bounds.y1 - 30,
      });
      return {
        left: anchor.x - 80,
        top: anchor.y - 20,
        width: 160,
        height: 40,
      };
    }
    const topLeft = ctx.worldToCanvasPoint({ x: bounds.x1, y: bounds.y1 });
    const bottomRight = ctx.worldToCanvasPoint({ x: bounds.x2, y: bounds.y2 });
    return {
      left: topLeft.x,
      top: topLeft.y,
      width: Math.max(48, bottomRight.x - topLeft.x),
      height: Math.max(48, bottomRight.y - topLeft.y),
    };
  }

  function groupWorldBounds(group) {
    const tensors = group.tensor_ids.map((tensorId) => ctx.findTensorById(tensorId)).filter(Boolean);
    if (!tensors.length) {
      return null;
    }
    return tensors.reduce(
      (bounds, tensor) => ({
        x1: Math.min(bounds.x1, tensor.position.x - tensorWidth(tensor) / 2 - 24),
        y1: Math.min(bounds.y1, tensor.position.y - tensorHeight(tensor) / 2 - 24),
        x2: Math.max(bounds.x2, tensor.position.x + tensorWidth(tensor) / 2 + 24),
        y2: Math.max(bounds.y2, tensor.position.y + tensorHeight(tensor) / 2 + 24),
      }),
      {
        x1: Number.POSITIVE_INFINITY,
        y1: Number.POSITIVE_INFINITY,
        x2: Number.NEGATIVE_INFINITY,
        y2: Number.NEGATIVE_INFINITY,
      }
    );
  }

  function syncTensorElementSize(tensor) {
    if (!state.cy) {
      return;
    }
    const tensorElement = state.cy.getElementById(tensor.id);
    if (!tensorElement || !tensorElement.length) {
      return;
    }
    tensorElement.data("width", tensorWidth(tensor));
    tensorElement.data("height", tensorHeight(tensor));
  }

  function uniquifyImportedSpec(spec, prefix) {
    const cloned = ctx.deepClone(spec);
    const tensorIdMap = {};
    const indexIdMap = {};
    const groupIdMap = {};

    cloned.id = `${prefix}_${ctx.sanitizeFilename(cloned.name || "template")}`;
    cloned.tensors.forEach((tensor) => {
      const nextTensorId = ctx.makeId("tensor");
      tensorIdMap[tensor.id] = nextTensorId;
      tensor.id = nextTensorId;
      tensor.indices.forEach((index) => {
        const nextIndexId = ctx.makeId("index");
        indexIdMap[index.id] = nextIndexId;
        index.id = nextIndexId;
      });
    });
    cloned.edges.forEach((edge) => {
      edge.id = ctx.makeId("edge");
      edge.left.tensor_id = tensorIdMap[edge.left.tensor_id];
      edge.right.tensor_id = tensorIdMap[edge.right.tensor_id];
      edge.left.index_id = indexIdMap[edge.left.index_id];
      edge.right.index_id = indexIdMap[edge.right.index_id];
    });
    cloned.groups.forEach((group) => {
      const nextGroupId = ctx.makeId("group");
      groupIdMap[group.id] = nextGroupId;
      group.id = nextGroupId;
      group.tensor_ids = group.tensor_ids.map((tensorId) => tensorIdMap[tensorId]);
    });
    return ctx.normalizeSpec(cloned);
  }

  function translateImportedSpec(spec, targetCenter) {
    const translated = ctx.deepClone(spec);
    const bounds = computeSpecBounds(translated);
    const sourceCenter = {
      x: (bounds.x1 + bounds.x2) / 2,
      y: (bounds.y1 + bounds.y2) / 2,
    };
    const deltaX = targetCenter.x - sourceCenter.x;
    const deltaY = targetCenter.y - sourceCenter.y;
    translated.tensors.forEach((tensor) => {
      tensor.position.x += deltaX;
      tensor.position.y += deltaY;
    });
    return ctx.normalizeSpec(translated);
  }

  function computeSpecBounds(spec) {
    const bounds = {
      x1: Number.POSITIVE_INFINITY,
      y1: Number.POSITIVE_INFINITY,
      x2: Number.NEGATIVE_INFINITY,
      y2: Number.NEGATIVE_INFINITY,
    };
    spec.tensors.forEach((tensor) => {
      ctx.expandBounds(bounds, tensor.position.x - tensorWidth(tensor) / 2, tensor.position.y - tensorHeight(tensor) / 2);
      ctx.expandBounds(bounds, tensor.position.x + tensorWidth(tensor) / 2, tensor.position.y + tensorHeight(tensor) / 2);
    });
    return bounds;
  }

  Object.assign(ctx, {
    renderOverlayDecorations,
    renderGroupOverlays,
    renderResizeHandles,
    renderContractionBadges,
    createGroupFromSelection,
    toggleGroupCollapse,
    startGroupDrag,
    updateActiveGroupDrag,
    finishActiveGroupDrag,
    startTensorResize,
    updateActiveResize,
    finishActiveResize,
    tensorWidth,
    tensorHeight,
    tensorScreenRect,
    groupDisplayRect,
    groupWorldBounds,
    syncTensorElementSize,
    uniquifyImportedSpec,
    translateImportedSpec,
    computeSpecBounds
  });
}

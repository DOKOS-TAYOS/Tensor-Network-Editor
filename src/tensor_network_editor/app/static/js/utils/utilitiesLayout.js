export function createUtilityLayoutBindings({ ctx, state, constants }) {
  const GRID_SNAP_SIZE =
    Number.isFinite(constants && constants.GRID_SNAP_SIZE) &&
    constants.GRID_SNAP_SIZE > 0
      ? constants.GRID_SNAP_SIZE
      : 20;
  const LAYOUT_HORIZONTAL_GAP =
    Number.isFinite(constants && constants.LAYOUT_HORIZONTAL_GAP) &&
    constants.LAYOUT_HORIZONTAL_GAP >= 0
      ? constants.LAYOUT_HORIZONTAL_GAP
      : 80;
  const LAYOUT_VERTICAL_GAP =
    Number.isFinite(constants && constants.LAYOUT_VERTICAL_GAP) &&
    constants.LAYOUT_VERTICAL_GAP >= 0
      ? constants.LAYOUT_VERTICAL_GAP
      : 100;
  const LAYOUT_COMPONENT_GAP =
    Number.isFinite(constants && constants.LAYOUT_COMPONENT_GAP) &&
    constants.LAYOUT_COMPONENT_GAP >= 0
      ? constants.LAYOUT_COMPONENT_GAP
      : 140;
  const LAYOUT_NON_OVERLAP_GAP =
    Number.isFinite(constants && constants.LAYOUT_NON_OVERLAP_GAP) &&
    constants.LAYOUT_NON_OVERLAP_GAP >= 0
      ? constants.LAYOUT_NON_OVERLAP_GAP
      : 36;
  const INDEX_RADIUS =
    Number.isFinite(constants && constants.INDEX_RADIUS) &&
    constants.INDEX_RADIUS >= 0
      ? constants.INDEX_RADIUS
      : 15;
  const INDEX_PADDING =
    Number.isFinite(constants && constants.INDEX_PADDING) &&
    constants.INDEX_PADDING >= 0
      ? constants.INDEX_PADDING
      : 8;
  const INDEX_REFLOW_GAP = INDEX_RADIUS * 2 + Math.max(INDEX_PADDING, 8);

  function getSelectedLayoutTensorIds() {
    return typeof ctx.getSelectedIdsByKind === "function"
      ? ctx.getSelectedIdsByKind("tensor")
      : [];
  }

  function getSelectedLayoutTensors() {
    return getLayoutTensorsById(getSelectedLayoutTensorIds());
  }

  function getLayoutTensorsById(tensorIds) {
    return tensorIds
      .map((tensorId) => ctx.findTensorById(tensorId))
      .filter(Boolean);
  }

  function applyTensorLayoutChangeForIds(
    tensorIds,
    mutator,
    statusMessage,
    primaryId = state.primarySelectionId
  ) {
    if (
      !Array.isArray(tensorIds) ||
      tensorIds.length < 2 ||
      typeof ctx.applyDesignChange !== "function"
    ) {
      return false;
    }
    ctx.applyDesignChange(mutator, {
      invalidate: {
        lookups: false,
        analysis: false,
      },
      selectionIds: [...tensorIds],
      primaryId: tensorIds.includes(primaryId)
        ? primaryId
        : tensorIds[tensorIds.length - 1],
      statusMessage,
    });
    return true;
  }

  function applyTensorLayoutChange(mutator, statusMessage) {
    return applyTensorLayoutChangeForIds(
      getSelectedLayoutTensorIds(),
      mutator,
      statusMessage
    );
  }

  function applyIndexLayoutChangeForIds(
    tensorIds,
    mutator,
    statusMessage,
    primaryId = state.primarySelectionId
  ) {
    if (
      !Array.isArray(tensorIds) ||
      tensorIds.length < 1 ||
      typeof ctx.applyDesignChange !== "function"
    ) {
      return false;
    }
    ctx.applyDesignChange(mutator, {
      invalidate: {
        lookups: false,
        analysis: false,
      },
      selectionIds: [...tensorIds],
      primaryId: tensorIds.includes(primaryId)
        ? primaryId
        : tensorIds[tensorIds.length - 1],
      statusMessage,
    });
    return true;
  }

  function alignSelectedTensors(mode) {
    const tensors = getSelectedLayoutTensors();
    if (tensors.length < 2) {
      ctx.setStatus("Select at least two tensors to align.");
      return false;
    }

    const bounds = computeTensorBounds(tensors);
    const centerX = (bounds.left + bounds.right) / 2;
    const centerY = (bounds.top + bounds.bottom) / 2;
    const statusLabels = {
      left: "Aligned tensors to the left.",
      center: "Aligned tensor centers horizontally.",
      right: "Aligned tensors to the right.",
      top: "Aligned tensors to the top.",
      middle: "Aligned tensor centers vertically.",
      bottom: "Aligned tensors to the bottom.",
    };

    const targetPositions = buildAlignedTensorPositions(tensors, mode, {
      bounds,
      centerX,
      centerY,
    });
    if (!targetPositions) {
      ctx.setStatus(`Unknown alignment mode '${mode}'.`, "error");
      return false;
    }
    return applyTensorPositions(
      tensors.map((tensor) => tensor.id),
      targetPositions,
      statusLabels[mode] || "Aligned tensors."
    );
  }

  function distributeSelectedTensors(axis) {
    const tensors = getSelectedLayoutTensors();
    if (tensors.length < 3) {
      ctx.setStatus("Select at least three tensors to distribute.");
      return false;
    }

    const sortedTensors = [...tensors].sort((leftTensor, rightTensor) =>
      axis === "vertical"
        ? leftTensor.position.y - rightTensor.position.y
        : leftTensor.position.x - rightTensor.position.x
    );
    const firstTensor = sortedTensors[0];
    const lastTensor = sortedTensors[sortedTensors.length - 1];
    const start =
      axis === "vertical" ? firstTensor.position.y : firstTensor.position.x;
    const end = axis === "vertical" ? lastTensor.position.y : lastTensor.position.x;
    const step = (end - start) / (sortedTensors.length - 1);

    return applyTensorLayoutChange(
      () => {
        sortedTensors.forEach((tensor, index) => {
          if (axis === "vertical") {
            tensor.position.y = start + step * index;
          } else {
            tensor.position.x = start + step * index;
          }
        });
      },
      axis === "vertical"
        ? "Distributed tensors vertically."
        : "Distributed tensors horizontally."
    );
  }

  function snapSelectedTensorsToGrid() {
    const tensors = getSelectedLayoutTensors();
    if (tensors.length < 2) {
      ctx.setStatus("Select at least two tensors to snap.");
      return false;
    }

    return applyTensorLayoutChange(() => {
      tensors.forEach((tensor) => {
        tensor.position.x =
          Math.round(tensor.position.x / GRID_SNAP_SIZE) * GRID_SNAP_SIZE;
        tensor.position.y =
          Math.round(tensor.position.y / GRID_SNAP_SIZE) * GRID_SNAP_SIZE;
      });
    }, "Snapped tensors to the grid.");
  }

  function arrangeSelectedTensors(mode) {
    const tensorIds = getSelectedLayoutTensorIds();
    if (tensorIds.length < 2) {
      ctx.setStatus("Select at least two tensors to arrange.");
      return false;
    }
    const targetPositions = buildArrangedSelectionPositions(tensorIds, mode);
    if (!targetPositions) {
      ctx.setStatus(`Unknown layout mode '${mode}'.`, "error");
      return false;
    }
    return applyTensorPositions(
      tensorIds,
      targetPositions,
      {
        chain: "Arranged selection as a chain.",
        tree: "Arranged selection as a tree.",
        grid: "Arranged selection as a grid.",
      }[mode] || "Arranged the selected tensors."
    );
  }

  function applyReflowLayoutAction(layoutAction) {
    const action = typeof layoutAction === "string" ? layoutAction : "";
    if (
      action === "left" ||
      action === "right" ||
      action === "top" ||
      action === "middle" ||
      action === "bottom"
    ) {
      return alignSelectedTensors(action);
    }
    if (action === "chain" || action === "tree" || action === "grid") {
      return arrangeSelectedTensors(action);
    }
    if (action === "horizontal" || action === "vertical") {
      return distributeSelectedTensors(action);
    }
    if (action === "snap") {
      return snapSelectedTensorsToGrid();
    }
    if (action === "smart") {
      return reflowLastImportedTensors();
    }
    ctx.setStatus(`Unknown reflow action '${action}'.`, "error");
    return false;
  }

  function applyReflowIndicesAction(layoutAction) {
    const action = typeof layoutAction === "string" ? layoutAction : "";
    const tensorIds = getSelectedLayoutTensorIds();
    if (tensorIds.length < 1) {
      ctx.setStatus("Select at least one tensor to reflow indices.");
      return false;
    }
    const tensors = getLayoutTensorsById(tensorIds).filter(
      (tensor) => Array.isArray(tensor.indices) && tensor.indices.length > 0
    );
    if (!tensors.length) {
      ctx.setStatus("The selected tensors have no indices to reflow.");
      return false;
    }
    const targetOffsets = buildReflowIndexOffsets(tensors, action);
    if (!targetOffsets) {
      ctx.setStatus(`Unknown index reflow action '${action}'.`, "error");
      return false;
    }
    return applyIndexLayoutChangeForIds(
      tensorIds,
      () => {
        tensors.forEach((tensor) => {
          const tensorOffsets = targetOffsets[tensor.id];
          if (!Array.isArray(tensorOffsets)) {
            return;
          }
          tensor.indices.forEach((index, indexPosition) => {
            const targetOffset = tensorOffsets[indexPosition];
            if (!targetOffset) {
              return;
            }
            index.offset = targetOffset;
          });
        });
      },
      {
        left: "Moved selected tensor indices to the left.",
        right: "Moved selected tensor indices to the right.",
        top: "Moved selected tensor indices to the top.",
        bottom: "Moved selected tensor indices to the bottom.",
        reset: "Reset selected tensor indices.",
      }[action] || "Reflowed the selected tensor indices."
    );
  }

  function reflowLastImportedTensors() {
    const tensorIds = getSelectedLayoutTensorIds();
    if (tensorIds.length < 2) {
      ctx.setStatus("Select at least two tensors to reflow.");
      return false;
    }
    const targetPositions = buildImportedReflowPositions(tensorIds);
    return applyTensorPositions(
      tensorIds,
      targetPositions,
      "Reflowed the selected tensors."
    );
  }

  function applyTensorPositions(tensorIds, targetPositions, statusMessage) {
    if (!targetPositions) {
      return false;
    }
    return applyTensorLayoutChangeForIds(
      tensorIds,
      () => {
        tensorIds.forEach((tensorId) => {
          const tensor = ctx.findTensorById(tensorId);
          const targetPosition = targetPositions[tensorId];
          if (!tensor || !targetPosition) {
            return;
          }
          tensor.position.x = targetPosition.x;
          tensor.position.y = targetPosition.y;
        });
      },
      statusMessage
    );
  }

  function buildArrangedSelectionPositions(tensorIds, mode) {
    const tensors = getLayoutTensorsById(tensorIds);
    const targetCenter = computeTensorBounds(tensors);
    const adjacency = buildTensorAdjacency(tensorIds);
    const internalEdgeCount = getInternalEdgeCount(tensorIds, adjacency);
    if (mode === "grid" && internalEdgeCount === 0) {
      return centerLayoutPositions(
        buildGridLocalPositions(sortTensorIdsByPosition(tensorIds)),
        tensorIds,
        {
          x: (targetCenter.left + targetCenter.right) / 2,
          y: (targetCenter.top + targetCenter.bottom) / 2,
        }
      );
    }
    if (mode === "tree" && internalEdgeCount === 0) {
      return centerLayoutPositions(
        buildSyntheticTreeLocalPositions(
          tensorIds,
          tensorIds.includes(state.primarySelectionId) ? state.primarySelectionId : null
        ),
        tensorIds,
        {
          x: (targetCenter.left + targetCenter.right) / 2,
          y: (targetCenter.top + targetCenter.bottom) / 2,
        }
      );
    }
    const componentLayouts = buildConnectedComponents(tensorIds, adjacency).map(
      (componentIds) =>
        buildSelectionComponentLayout(
          componentIds,
          adjacency,
          mode,
          componentIds.includes(state.primarySelectionId)
            ? state.primarySelectionId
            : null
        )
    );
    if (!componentLayouts.every(Boolean)) {
      return null;
    }
    return centerPackedComponentLayouts(componentLayouts, tensorIds, {
      x: (targetCenter.left + targetCenter.right) / 2,
      y: (targetCenter.top + targetCenter.bottom) / 2,
    });
  }

  function buildImportedReflowPositions(tensorIds) {
    const adjacency = buildTensorAdjacency(tensorIds);
    const componentLayouts = buildConnectedComponents(tensorIds, adjacency).map(
      (componentIds) => {
        if (isPathComponent(componentIds, adjacency)) {
          return buildComponentLayoutFromLocalPositions(
            componentIds,
            buildChainLocalPositions(buildPathOrder(componentIds, adjacency))
          );
        }
        if (isTreeComponent(componentIds, adjacency)) {
          return buildTreeComponentLayout(componentIds, adjacency);
        }
        return buildComponentLayoutFromLocalPositions(
          componentIds,
          buildGridLocalPositions(
            buildComponentTraversalOrder(componentIds, adjacency)
          )
        );
      }
    );
    const importedTensors = getLayoutTensorsById(tensorIds);
    const importedBounds = computeTensorBounds(importedTensors);
    return centerPackedComponentLayouts(componentLayouts, tensorIds, {
      x: (importedBounds.left + importedBounds.right) / 2,
      y: (importedBounds.top + importedBounds.bottom) / 2,
    });
  }

  function buildAlignedTensorPositions(
    tensors,
    mode,
    { bounds, centerX, centerY }
  ) {
    if (mode === "left" || mode === "center" || mode === "right") {
      const packedYPositions = buildAlignedNonOverlappingAxisPositions(tensors, "y");
      return Object.fromEntries(
        tensors.map((tensor) => [
          tensor.id,
          {
            x:
              mode === "left"
                ? bounds.left + ctx.tensorWidth(tensor) / 2
                : mode === "right"
                  ? bounds.right - ctx.tensorWidth(tensor) / 2
                  : centerX,
            y: packedYPositions[tensor.id],
          },
        ])
      );
    }
    if (mode === "top" || mode === "middle" || mode === "bottom") {
      const packedXPositions = buildAlignedNonOverlappingAxisPositions(tensors, "x");
      return Object.fromEntries(
        tensors.map((tensor) => [
          tensor.id,
          {
            x: packedXPositions[tensor.id],
            y:
              mode === "top"
                ? bounds.top + ctx.tensorHeight(tensor) / 2
                : mode === "bottom"
                  ? bounds.bottom - ctx.tensorHeight(tensor) / 2
                  : centerY,
          },
        ])
      );
    }
    return null;
  }

  function buildReflowIndexOffsets(tensors, mode) {
    if (
      mode !== "left" &&
      mode !== "right" &&
      mode !== "top" &&
      mode !== "bottom" &&
      mode !== "reset"
    ) {
      return null;
    }
    return Object.fromEntries(
      tensors.map((tensor) => [tensor.id, buildTensorIndexOffsets(tensor, mode)])
    );
  }

  function buildTensorIndexOffsets(tensor, mode) {
    if (mode === "reset") {
      return tensor.indices.map((index, indexPosition) =>
        ctx.defaultIndexOffsetForOrder(indexPosition, tensor)
      );
    }

    const leftOffset = -ctx.tensorWidth(tensor) / 2 + INDEX_RADIUS + INDEX_PADDING;
    const rightOffset = ctx.tensorWidth(tensor) / 2 - INDEX_RADIUS - INDEX_PADDING;
    const topOffset = -ctx.tensorHeight(tensor) / 2 + INDEX_RADIUS + INDEX_PADDING;
    const bottomOffset = ctx.tensorHeight(tensor) / 2 - INDEX_RADIUS - INDEX_PADDING;

    if (mode === "left" || mode === "right") {
      return buildPackedBoundaryIndexOffsets(
        tensor,
        tensor.indices.length,
        topOffset,
        bottomOffset,
        (offsetAlongEdge, bandIndex) => ({
          x:
            (mode === "left" ? leftOffset : rightOffset) +
            (mode === "left" ? 1 : -1) * bandIndex * INDEX_REFLOW_GAP,
          y: offsetAlongEdge,
        })
      );
    }

    return buildPackedBoundaryIndexOffsets(
      tensor,
      tensor.indices.length,
      leftOffset,
      rightOffset,
      (offsetAlongEdge, bandIndex) => ({
        x: offsetAlongEdge,
        y:
          (mode === "top" ? topOffset : bottomOffset) +
          (mode === "top" ? 1 : -1) * bandIndex * INDEX_REFLOW_GAP,
      })
    );
  }

  function buildPackedBoundaryIndexOffsets(
    tensor,
    count,
    start,
    end,
    offsetBuilder
  ) {
    return buildIndexBands(count, start, end).flatMap((bandOffsets, bandIndex) =>
      bandOffsets.map((offsetAlongEdge) =>
        ctx.clampIndexOffset(offsetBuilder(offsetAlongEdge, bandIndex), tensor)
      )
    );
  }

  function buildIndexBands(count, start, end) {
    if (count <= 0) {
      return [];
    }
    const span = Math.abs(end - start);
    const maxPerBand = Math.max(1, Math.floor(span / INDEX_REFLOW_GAP) + 1);
    const bands = [];
    for (let index = 0; index < count; index += maxPerBand) {
      const bandCount = Math.min(maxPerBand, count - index);
      bands.push(buildDistributedIndexAxisOffsets(bandCount, start, end));
    }
    return bands;
  }

  function buildDistributedIndexAxisOffsets(count, start, end) {
    if (count <= 1) {
      return [(start + end) / 2];
    }
    const step = (end - start) / (count - 1);
    return Array.from({ length: count }, (_, index) => start + step * index);
  }

  function buildAlignedNonOverlappingAxisPositions(tensors, axis) {
    const orderedTensors = [...tensors].sort((leftTensor, rightTensor) => {
      const leftPrimary = axis === "y" ? leftTensor.position.y : leftTensor.position.x;
      const rightPrimary =
        axis === "y" ? rightTensor.position.y : rightTensor.position.x;
      if (leftPrimary !== rightPrimary) {
        return leftPrimary - rightPrimary;
      }
      const leftSecondary =
        axis === "y" ? leftTensor.position.x : leftTensor.position.y;
      const rightSecondary =
        axis === "y" ? rightTensor.position.x : rightTensor.position.y;
      return leftSecondary - rightSecondary;
    });
    if (!needsAlignmentPacking(orderedTensors, axis)) {
      return Object.fromEntries(
        orderedTensors.map((tensor) => [
          tensor.id,
          axis === "y" ? tensor.position.y : tensor.position.x,
        ])
      );
    }

    const packedAxisPositions = {};
    let nextCenter = 0;
    let previousHalfSize = 0;
    orderedTensors.forEach((tensor, index) => {
      const halfSize = getTensorHalfSize(tensor, axis);
      if (index === 0) {
        nextCenter = halfSize;
      } else {
        nextCenter += previousHalfSize + LAYOUT_NON_OVERLAP_GAP + halfSize;
      }
      packedAxisPositions[tensor.id] = nextCenter;
      previousHalfSize = halfSize;
    });

    const packedBounds = computeAxisBounds(orderedTensors, packedAxisPositions, axis);
    const originalBounds = computeTensorBounds(orderedTensors);
    const targetAxisCenter =
      axis === "y"
        ? (originalBounds.top + originalBounds.bottom) / 2
        : (originalBounds.left + originalBounds.right) / 2;
    const packedAxisCenter = (packedBounds.start + packedBounds.end) / 2;
    const delta = targetAxisCenter - packedAxisCenter;
    return Object.fromEntries(
      Object.entries(packedAxisPositions).map(([tensorId, position]) => [
        tensorId,
        position + delta,
      ])
    );
  }

  function needsAlignmentPacking(orderedTensors, axis) {
    for (let index = 1; index < orderedTensors.length; index += 1) {
      const previousTensor = orderedTensors[index - 1];
      const currentTensor = orderedTensors[index];
      const previousCenter =
        axis === "y" ? previousTensor.position.y : previousTensor.position.x;
      const currentCenter =
        axis === "y" ? currentTensor.position.y : currentTensor.position.x;
      const minimumCenterDistance =
        getTensorHalfSize(previousTensor, axis) +
        LAYOUT_NON_OVERLAP_GAP +
        getTensorHalfSize(currentTensor, axis);
      if (currentCenter - previousCenter < minimumCenterDistance) {
        return true;
      }
    }
    return false;
  }

  function getTensorHalfSize(tensor, axis) {
    return (
      (axis === "y" ? ctx.tensorHeight(tensor) : ctx.tensorWidth(tensor)) / 2
    );
  }

  function computeAxisBounds(tensors, axisPositions, axis) {
    return {
      start: Math.min(
        ...tensors.map((tensor) => axisPositions[tensor.id] - getTensorHalfSize(tensor, axis))
      ),
      end: Math.max(
        ...tensors.map((tensor) => axisPositions[tensor.id] + getTensorHalfSize(tensor, axis))
      ),
    };
  }

  function buildTreeComponentLayout(componentIds, adjacency, preferredRootId = null) {
    const { rootId, childrenById } = buildSpanningTree(
      componentIds,
      adjacency,
      preferredRootId
    );
    return buildTreeSubtreeLayout(rootId, childrenById);
  }

  function buildComponentLayoutFromLocalPositions(componentIds, localPositions) {
    const bounds = computePositionBounds(componentIds, localPositions);
    return {
      ids: [...componentIds],
      positions: localPositions,
      width: bounds.right - bounds.left,
      height: bounds.bottom - bounds.top,
      bounds,
    };
  }

  function centerPackedComponentLayouts(componentLayouts, tensorIds, targetCenter) {
    const packedPositions = {};
    let left = 0;
    componentLayouts.forEach((layout, componentIndex) => {
      const deltaX = left - layout.bounds.left;
      const deltaY = -layout.bounds.top;
      layout.ids.forEach((tensorId) => {
        const position = layout.positions[tensorId];
        packedPositions[tensorId] = {
          x: position.x + deltaX,
          y: position.y + deltaY,
        };
      });
      left += layout.width;
      if (componentIndex < componentLayouts.length - 1) {
        left += LAYOUT_COMPONENT_GAP;
      }
    });
    return centerLayoutPositions(packedPositions, tensorIds, targetCenter);
  }

  function centerLayoutPositions(positions, tensorIds, targetCenter) {
    const bounds = computePositionBounds(tensorIds, positions);
    const deltaX = targetCenter.x - (bounds.left + bounds.right) / 2;
    const deltaY = targetCenter.y - (bounds.top + bounds.bottom) / 2;
    return Object.fromEntries(
      Object.entries(positions).map(([tensorId, position]) => [
        tensorId,
        {
          x: position.x + deltaX,
          y: position.y + deltaY,
        },
      ])
    );
  }

  function buildHorizontalRowPositions(orderedIds) {
    const positions = {};
    let nextCenterX = 0;
    let previousHalfWidth = 0;
    orderedIds.forEach((tensorId, index) => {
      const tensor = ctx.findTensorById(tensorId);
      const halfWidth = ctx.tensorWidth(tensor) / 2;
      if (index === 0) {
        nextCenterX = halfWidth;
      } else {
        nextCenterX += previousHalfWidth + LAYOUT_HORIZONTAL_GAP + halfWidth;
      }
      positions[tensorId] = {
        x: nextCenterX,
        y: 0,
      };
      previousHalfWidth = halfWidth;
    });
    const bounds = computePositionBounds(orderedIds, positions);
    return {
      positions,
      width: bounds.right - bounds.left,
    };
  }

  function buildChainLocalPositions(orderedIds) {
    return buildHorizontalRowPositions(orderedIds).positions;
  }

  function buildGridLocalPositions(orderedIds) {
    const columnCount = Math.max(1, Math.ceil(Math.sqrt(orderedIds.length)));
    const rows = [];
    for (let index = 0; index < orderedIds.length; index += columnCount) {
      rows.push(orderedIds.slice(index, index + columnCount));
    }
    const rowHeights = rows.map((rowIds) =>
      Math.max(...rowIds.map((tensorId) => ctx.tensorHeight(ctx.findTensorById(tensorId))))
    );
    const columnWidths = Array.from({ length: columnCount }, (_, columnIndex) =>
      Math.max(
        0,
        ...rows
          .map((rowIds) => rowIds[columnIndex])
          .filter(Boolean)
          .map((tensorId) => ctx.tensorWidth(ctx.findTensorById(tensorId)))
      )
    );
    const positions = {};
    let top = 0;
    rows.forEach((rowIds, rowIndex) => {
      let left = 0;
      rowIds.forEach((tensorId, columnIndex) => {
        positions[tensorId] = {
          x: left + columnWidths[columnIndex] / 2,
          y: top + rowHeights[rowIndex] / 2,
        };
        left += columnWidths[columnIndex] + LAYOUT_HORIZONTAL_GAP;
      });
      top += rowHeights[rowIndex] + LAYOUT_VERTICAL_GAP;
    });
    return positions;
  }

  function buildSyntheticTreeLocalPositions(tensorIds, preferredRootId = null) {
    const sortedIds = sortTensorIdsByPosition(tensorIds);
    const rootId =
      preferredRootId && tensorIds.includes(preferredRootId)
        ? preferredRootId
        : sortedIds[0];
    const remainingIds = sortedIds.filter((tensorId) => tensorId !== rootId);
    const childrenById = new Map(tensorIds.map((tensorId) => [tensorId, []]));
    const parentQueue = [rootId];
    let parentIndex = 0;
    remainingIds.forEach((tensorId) => {
      while (
        parentIndex < parentQueue.length &&
        (childrenById.get(parentQueue[parentIndex]) || []).length >= 2
      ) {
        parentIndex += 1;
      }
      const parentId = parentQueue[parentIndex] || rootId;
      childrenById.get(parentId).push(tensorId);
      parentQueue.push(tensorId);
    });
    return buildTreeSubtreeLayout(rootId, childrenById).positions;
  }

  function buildBreadthFirstLevels(rootId, adjacency) {
    const levels = [];
    const visited = new Set([rootId]);
    const queue = [{ tensorId: rootId, depth: 0 }];
    while (queue.length) {
      const current = queue.shift();
      if (!levels[current.depth]) {
        levels[current.depth] = [];
      }
      levels[current.depth].push(current.tensorId);
      (adjacency.get(current.tensorId) || []).forEach((neighborId) => {
        if (visited.has(neighborId)) {
          return;
        }
        visited.add(neighborId);
        queue.push({ tensorId: neighborId, depth: current.depth + 1 });
      });
    }
    return levels;
  }

  function buildConnectedComponents(tensorIds, adjacency) {
    const visited = new Set();
    const components = [];
    sortTensorIdsByPosition(tensorIds).forEach((tensorId) => {
      if (visited.has(tensorId)) {
        return;
      }
      const queue = [tensorId];
      const componentIds = [];
      visited.add(tensorId);
      while (queue.length) {
        const currentId = queue.shift();
        componentIds.push(currentId);
        (adjacency.get(currentId) || []).forEach((neighborId) => {
          if (visited.has(neighborId)) {
            return;
          }
          visited.add(neighborId);
          queue.push(neighborId);
        });
      }
      components.push(componentIds);
    });
    return components;
  }

  function buildTensorAdjacency(tensorIds) {
    const tensorIdSet = new Set(tensorIds);
    const adjacency = new Map(
      tensorIds.map((tensorId) => [tensorId, []])
    );
    (Array.isArray(state.spec && state.spec.edges) ? state.spec.edges : []).forEach(
      (edge) => {
        const leftTensorId = edge && edge.left ? edge.left.tensor_id : null;
        const rightTensorId = edge && edge.right ? edge.right.tensor_id : null;
        if (!tensorIdSet.has(leftTensorId) || !tensorIdSet.has(rightTensorId)) {
          return;
        }
        adjacency.get(leftTensorId).push(rightTensorId);
        adjacency.get(rightTensorId).push(leftTensorId);
      }
    );
    return adjacency;
  }

  function buildPathOrder(tensorIds, adjacency) {
    const endpoints = sortTensorIdsByPosition(
      tensorIds.filter(
        (tensorId) => (adjacency.get(tensorId) || []).length <= 1
      )
    );
    if (!endpoints.length) {
      return sortTensorIdsByPosition(tensorIds);
    }
    const orderedIds = [];
    let previousId = null;
    let currentId = endpoints[0];
    while (currentId) {
      orderedIds.push(currentId);
      const nextId = (adjacency.get(currentId) || []).find(
        (neighborId) => neighborId !== previousId
      );
      previousId = currentId;
      currentId = nextId || null;
    }
    return orderedIds;
  }

  function isPathComponent(tensorIds, adjacency) {
    return (
      tensorIds.length > 1 &&
      getInternalEdgeCount(tensorIds, adjacency) === tensorIds.length - 1 &&
      tensorIds.every((tensorId) => (adjacency.get(tensorId) || []).length <= 2)
    );
  }

  function isTreeComponent(tensorIds, adjacency) {
    return (
      tensorIds.length > 1 &&
      getInternalEdgeCount(tensorIds, adjacency) === tensorIds.length - 1
    );
  }

  function buildSelectionComponentLayout(
    componentIds,
    adjacency,
    mode,
    preferredRootId = null
  ) {
    if (mode === "tree") {
      return buildTreeComponentLayout(componentIds, adjacency, preferredRootId);
    }
    const orderedIds = buildComponentTraversalOrder(
      componentIds,
      adjacency,
      preferredRootId
    );
    if (mode === "chain") {
      return buildComponentLayoutFromLocalPositions(
        componentIds,
        buildChainLocalPositions(orderedIds)
      );
    }
    if (mode === "grid") {
      return buildComponentLayoutFromLocalPositions(
        componentIds,
        buildGridLocalPositions(orderedIds)
      );
    }
    return null;
  }

  function buildComponentTraversalOrder(
    componentIds,
    adjacency,
    preferredRootId = null
  ) {
    if (componentIds.length <= 1) {
      return [...componentIds];
    }
    if (isPathComponent(componentIds, adjacency)) {
      return buildPathOrder(componentIds, adjacency);
    }
    const { rootId, childrenById } = buildSpanningTree(
      componentIds,
      adjacency,
      preferredRootId
    );
    const orderedIds = [];
    function visitSubtree(tensorId) {
      orderedIds.push(tensorId);
      (childrenById.get(tensorId) || []).forEach((childId) => {
        visitSubtree(childId);
      });
    }
    visitSubtree(rootId);
    return orderedIds;
  }

  function buildSpanningTree(componentIds, adjacency, preferredRootId = null) {
    const rootId = resolveComponentRootId(componentIds, adjacency, preferredRootId);
    const childrenById = new Map(componentIds.map((tensorId) => [tensorId, []]));
    const visited = new Set([rootId]);
    const queue = [rootId];
    while (queue.length) {
      const currentId = queue.shift();
      sortTensorIdsByPosition(adjacency.get(currentId) || []).forEach((neighborId) => {
        if (!childrenById.has(neighborId) || visited.has(neighborId)) {
          return;
        }
        visited.add(neighborId);
        childrenById.get(currentId).push(neighborId);
        queue.push(neighborId);
      });
    }
    return {
      rootId,
      childrenById,
    };
  }

  function resolveComponentRootId(componentIds, adjacency, preferredRootId = null) {
    if (preferredRootId && componentIds.includes(preferredRootId)) {
      return preferredRootId;
    }
    const endpointIds = componentIds.filter(
      (tensorId) => (adjacency.get(tensorId) || []).length <= 1
    );
    if (endpointIds.length) {
      return sortTensorIdsByPosition(endpointIds)[0];
    }
    return sortTensorIdsByPosition(componentIds)[0];
  }

  function buildTreeSubtreeLayout(rootId, childrenById) {
    const rootTensor = ctx.findTensorById(rootId);
    const rootWidth = ctx.tensorWidth(rootTensor);
    const rootHeight = ctx.tensorHeight(rootTensor);
    const childIds = childrenById.get(rootId) || [];
    if (!childIds.length) {
      return buildComponentLayoutFromLocalPositions([rootId], {
        [rootId]: {
          x: rootWidth / 2,
          y: rootHeight / 2,
        },
      });
    }

    const childLayouts = childIds.map((childId) =>
      buildTreeSubtreeLayout(childId, childrenById)
    );
    const packedChildPositions = {};
    let childLeft = 0;
    childLayouts.forEach((layout, index) => {
      const deltaX = childLeft - layout.bounds.left;
      const deltaY = rootHeight + LAYOUT_VERTICAL_GAP - layout.bounds.top;
      layout.ids.forEach((tensorId) => {
        const position = layout.positions[tensorId];
        packedChildPositions[tensorId] = {
          x: position.x + deltaX,
          y: position.y + deltaY,
        };
      });
      childLeft += layout.width;
      if (index < childLayouts.length - 1) {
        childLeft += LAYOUT_HORIZONTAL_GAP;
      }
    });

    const childBounds = computePositionBounds(
      Object.keys(packedChildPositions),
      packedChildPositions
    );
    const childBandWidth = childBounds.right - childBounds.left;
    const totalWidth = Math.max(rootWidth, childBandWidth);
    const childOffsetX = (totalWidth - childBandWidth) / 2 - childBounds.left;
    const positions = {
      [rootId]: {
        x: totalWidth / 2,
        y: rootHeight / 2,
      },
    };
    Object.entries(packedChildPositions).forEach(([tensorId, position]) => {
      positions[tensorId] = {
        x: position.x + childOffsetX,
        y: position.y,
      };
    });

    return buildComponentLayoutFromLocalPositions(
      [rootId, ...childLayouts.flatMap((layout) => layout.ids)],
      positions
    );
  }

  function getInternalEdgeCount(tensorIds, adjacency) {
    return (
      tensorIds.reduce(
        (edgeCount, tensorId) => edgeCount + (adjacency.get(tensorId) || []).length,
        0
      ) / 2
    );
  }

  function sortTensorIdsByPosition(tensorIds) {
    return [...tensorIds].sort((leftId, rightId) => {
      const leftTensor = ctx.findTensorById(leftId);
      const rightTensor = ctx.findTensorById(rightId);
      if (!leftTensor || !rightTensor) {
        return 0;
      }
      if (leftTensor.position.y !== rightTensor.position.y) {
        return leftTensor.position.y - rightTensor.position.y;
      }
      return leftTensor.position.x - rightTensor.position.x;
    });
  }

  function computeTensorBounds(tensors) {
    return {
      left: Math.min(
        ...tensors.map((tensor) => tensor.position.x - ctx.tensorWidth(tensor) / 2)
      ),
      right: Math.max(
        ...tensors.map((tensor) => tensor.position.x + ctx.tensorWidth(tensor) / 2)
      ),
      top: Math.min(
        ...tensors.map((tensor) => tensor.position.y - ctx.tensorHeight(tensor) / 2)
      ),
      bottom: Math.max(
        ...tensors.map((tensor) => tensor.position.y + ctx.tensorHeight(tensor) / 2)
      ),
    };
  }

  function computePositionBounds(tensorIds, positions) {
    return {
      left: Math.min(
        ...tensorIds.map((tensorId) => {
          const tensor = ctx.findTensorById(tensorId);
          const position = positions[tensorId];
          return position.x - ctx.tensorWidth(tensor) / 2;
        })
      ),
      right: Math.max(
        ...tensorIds.map((tensorId) => {
          const tensor = ctx.findTensorById(tensorId);
          const position = positions[tensorId];
          return position.x + ctx.tensorWidth(tensor) / 2;
        })
      ),
      top: Math.min(
        ...tensorIds.map((tensorId) => {
          const tensor = ctx.findTensorById(tensorId);
          const position = positions[tensorId];
          return position.y - ctx.tensorHeight(tensor) / 2;
        })
      ),
      bottom: Math.max(
        ...tensorIds.map((tensorId) => {
          const tensor = ctx.findTensorById(tensorId);
          const position = positions[tensorId];
          return position.y + ctx.tensorHeight(tensor) / 2;
        })
      ),
    };
  }

  return {
    GRID_SNAP_SIZE,
    alignSelectedTensors,
    applyReflowIndicesAction,
    applyReflowLayoutAction,
    arrangeSelectedTensors,
    distributeSelectedTensors,
    reflowLastImportedTensors,
    snapSelectedTensorsToGrid,
  };
}

export function createUtilityLayoutAlgorithmSupport({
  ctx,
  state,
  constants,
  selection,
}) {
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

  function buildArrangedSelectionPositions(tensorIds, mode) {
    const tensors = selection.getLayoutTensorsById(tensorIds);
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
    return buildAutoLayoutPositions(tensorIds);
  }

  function buildAutoLayoutPositions(tensorIds, preferredRootId = null) {
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
          return buildTreeComponentLayout(
            componentIds,
            adjacency,
            componentIds.includes(preferredRootId) ? preferredRootId : null
          );
        }
        return buildLayeredComponentLayout(
          componentIds,
          adjacency,
          componentIds.includes(preferredRootId) ? preferredRootId : null
        );
      }
    );
    const importedTensors = selection.getLayoutTensorsById(tensorIds);
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

  function buildLayeredComponentLayout(
    componentIds,
    adjacency,
    preferredRootId = null
  ) {
    const rootId = resolveLayeredRootId(componentIds, adjacency, preferredRootId);
    const { depthById, idsByDepth } = buildBreadthFirstLevels(
      componentIds,
      adjacency,
      rootId
    );
    const layeredIds = [...idsByDepth.keys()]
      .sort((leftDepth, rightDepth) => leftDepth - rightDepth)
      .map((depth) =>
        sortLayerTensorIds(idsByDepth.get(depth) || [], adjacency, depthById)
      )
      .filter((layerIds) => layerIds.length > 0);
    return buildComponentLayoutFromLocalPositions(
      componentIds,
      buildLayeredLocalPositions(layeredIds)
    );
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

  function buildLayerRowPositions(orderedIds) {
    const positions = {};
    const rowHeight = Math.max(
      0,
      ...orderedIds.map((tensorId) =>
        ctx.tensorHeight(ctx.findTensorById(tensorId))
      )
    );
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
        y: rowHeight / 2,
      };
      previousHalfWidth = halfWidth;
    });
    const bounds = computePositionBounds(orderedIds, positions);
    return {
      positions,
      rowHeight,
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

  function buildLayeredLocalPositions(layeredIds) {
    const layerRows = layeredIds.map((layerIds) => ({
      ids: [...layerIds],
      ...buildLayerRowPositions(layerIds),
    }));
    const maxLayerWidth = Math.max(
      0,
      ...layerRows.map((layerRow) => layerRow.width)
    );
    const positions = {};
    let top = 0;
    layerRows.forEach((layerRow, rowIndex) => {
      const offsetX = (maxLayerWidth - layerRow.width) / 2;
      layerRow.ids.forEach((tensorId) => {
        const position = layerRow.positions[tensorId];
        positions[tensorId] = {
          x: position.x + offsetX,
          y: position.y + top,
        };
      });
      top += layerRow.rowHeight;
      if (rowIndex < layerRows.length - 1) {
        top += LAYOUT_VERTICAL_GAP;
      }
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

  function buildBreadthFirstLevels(componentIds, adjacency, rootId) {
    const componentIdSet = new Set(componentIds);
    const depthById = new Map([[rootId, 0]]);
    const idsByDepth = new Map([[0, [rootId]]]);
    const queue = [rootId];
    while (queue.length) {
      const currentId = queue.shift();
      const currentDepth = depthById.get(currentId) || 0;
      sortTensorIdsByPosition(adjacency.get(currentId) || []).forEach((neighborId) => {
        if (!componentIdSet.has(neighborId) || depthById.has(neighborId)) {
          return;
        }
        const nextDepth = currentDepth + 1;
        depthById.set(neighborId, nextDepth);
        if (!idsByDepth.has(nextDepth)) {
          idsByDepth.set(nextDepth, []);
        }
        idsByDepth.get(nextDepth).push(neighborId);
        queue.push(neighborId);
      });
    }
    return {
      depthById,
      idsByDepth,
    };
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

  function sortLayerTensorIds(layerIds, adjacency, depthById) {
    return [...layerIds].sort((leftId, rightId) => {
      const leftAnchor = computeLayerAnchorX(leftId, adjacency, depthById);
      const rightAnchor = computeLayerAnchorX(rightId, adjacency, depthById);
      if (leftAnchor !== rightAnchor) {
        return leftAnchor - rightAnchor;
      }
      return compareTensorIdsByPosition(leftId, rightId);
    });
  }

  function computeLayerAnchorX(tensorId, adjacency, depthById) {
    const tensorDepth = depthById.get(tensorId) || 0;
    const parentXs = (adjacency.get(tensorId) || [])
      .filter((neighborId) => (depthById.get(neighborId) || 0) < tensorDepth)
      .map((neighborId) => {
        const neighborTensor = ctx.findTensorById(neighborId);
        return neighborTensor ? neighborTensor.position.x : 0;
      });
    if (parentXs.length) {
      return (
        parentXs.reduce(
          (sum, currentValue) => sum + currentValue,
          0
        ) / parentXs.length
      );
    }
    const tensor = ctx.findTensorById(tensorId);
    return tensor ? tensor.position.x : 0;
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
      childrenById,
      rootId,
    };
  }

  function resolveLayeredRootId(componentIds, adjacency, preferredRootId = null) {
    if (preferredRootId && componentIds.includes(preferredRootId)) {
      return preferredRootId;
    }
    return [...componentIds].sort((leftId, rightId) => {
      const leftDegree = (adjacency.get(leftId) || []).length;
      const rightDegree = (adjacency.get(rightId) || []).length;
      if (leftDegree !== rightDegree) {
        return rightDegree - leftDegree;
      }
      return compareTensorIdsByPosition(leftId, rightId);
    })[0];
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

  function compareTensorIdsByPosition(leftId, rightId) {
    const leftTensor = ctx.findTensorById(leftId);
    const rightTensor = ctx.findTensorById(rightId);
    if (!leftTensor || !rightTensor) {
      return 0;
    }
    if (leftTensor.position.y !== rightTensor.position.y) {
      return leftTensor.position.y - rightTensor.position.y;
    }
    if (leftTensor.position.x !== rightTensor.position.x) {
      return leftTensor.position.x - rightTensor.position.x;
    }
    return String(leftId).localeCompare(String(rightId));
  }

  function sortTensorIdsByPosition(tensorIds) {
    return [...tensorIds].sort(compareTensorIdsByPosition);
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
    buildAlignedTensorPositions,
    buildArrangedSelectionPositions,
    buildAutoLayoutPositions,
    buildImportedReflowPositions,
    computeTensorBounds,
  };
}

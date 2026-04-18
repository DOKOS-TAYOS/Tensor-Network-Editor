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

    return applyTensorLayoutChange(() => {
      tensors.forEach((tensor) => {
        if (mode === "left") {
          tensor.position.x = bounds.left + ctx.tensorWidth(tensor) / 2;
        } else if (mode === "center") {
          tensor.position.x = centerX;
        } else if (mode === "right") {
          tensor.position.x = bounds.right - ctx.tensorWidth(tensor) / 2;
        } else if (mode === "top") {
          tensor.position.y = bounds.top + ctx.tensorHeight(tensor) / 2;
        } else if (mode === "middle") {
          tensor.position.y = centerY;
        } else if (mode === "bottom") {
          tensor.position.y = bounds.bottom - ctx.tensorHeight(tensor) / 2;
        }
      });
    }, statusLabels[mode] || "Aligned tensors.");
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
    if (action === "left" || action === "right" || action === "top" || action === "middle" || action === "bottom") {
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
    if (mode === "chain") {
      const adjacency = buildTensorAdjacency(tensorIds);
      const orderedIds = isPathComponent(tensorIds, adjacency)
        ? buildPathOrder(tensorIds, adjacency)
        : sortTensorIdsByPosition(tensorIds);
      return centerLayoutPositions(
        buildChainLocalPositions(orderedIds),
        orderedIds,
        {
          x: (targetCenter.left + targetCenter.right) / 2,
          y: (targetCenter.top + targetCenter.bottom) / 2,
        }
      );
    }
    if (mode === "grid") {
      const orderedIds = sortTensorIdsByPosition(tensorIds);
      return centerLayoutPositions(
        buildGridLocalPositions(orderedIds),
        orderedIds,
        {
          x: (targetCenter.left + targetCenter.right) / 2,
          y: (targetCenter.top + targetCenter.bottom) / 2,
        }
      );
    }
    if (mode === "tree") {
      return centerPackedComponentLayouts(
        buildConnectedComponents(tensorIds, buildTensorAdjacency(tensorIds)).map(
          (componentIds) =>
            buildTreeComponentLayout(
              componentIds,
              buildTensorAdjacency(componentIds),
              componentIds.includes(state.primarySelectionId)
                ? state.primarySelectionId
                : null
            )
        ),
        tensorIds,
        {
          x: (targetCenter.left + targetCenter.right) / 2,
          y: (targetCenter.top + targetCenter.bottom) / 2,
        }
      );
    }
    return null;
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
          buildGridLocalPositions(sortTensorIdsByPosition(componentIds))
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

  function buildTreeComponentLayout(componentIds, adjacency, preferredRootId = null) {
    const sortedIds = sortTensorIdsByPosition(componentIds);
    const rootId =
      preferredRootId && componentIds.includes(preferredRootId)
        ? preferredRootId
        : sortedIds[0];
    const levels = buildBreadthFirstLevels(rootId, adjacency);
    const levelHeights = levels.map((levelIds) =>
      Math.max(...levelIds.map((tensorId) => ctx.tensorHeight(ctx.findTensorById(tensorId))))
    );
    const rowLayouts = levels.map((levelIds) =>
      buildHorizontalRowPositions(
        [...levelIds].sort((leftId, rightId) => {
          const leftTensor = ctx.findTensorById(leftId);
          const rightTensor = ctx.findTensorById(rightId);
          if (!leftTensor || !rightTensor) {
            return 0;
          }
          if (leftTensor.position.x !== rightTensor.position.x) {
            return leftTensor.position.x - rightTensor.position.x;
          }
          return leftTensor.position.y - rightTensor.position.y;
        })
      )
    );
    const maxRowWidth = Math.max(...rowLayouts.map((layout) => layout.width));
    const positions = {};
    let top = 0;
    rowLayouts.forEach((layout, levelIndex) => {
      const offsetX = (maxRowWidth - layout.width) / 2;
      const rowHeight = levelHeights[levelIndex];
      Object.entries(layout.positions).forEach(([tensorId, position]) => {
        positions[tensorId] = {
          x: position.x + offsetX,
          y: top + rowHeight / 2,
        };
      });
      top += rowHeight + LAYOUT_VERTICAL_GAP;
    });
    return buildComponentLayoutFromLocalPositions(componentIds, positions);
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
    const endpoints = tensorIds.filter(
      (tensorId) => (adjacency.get(tensorId) || []).length <= 1
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
    applyReflowLayoutAction,
    arrangeSelectedTensors,
    distributeSelectedTensors,
    reflowLastImportedTensors,
    snapSelectedTensorsToGrid,
  };
}

export function createLayoutAlgorithmPositionSupport({ ctx, layoutMetrics }) {
  const {
    LAYOUT_HORIZONTAL_GAP,
    LAYOUT_VERTICAL_GAP,
    LAYOUT_COMPONENT_GAP,
    LAYOUT_NON_OVERLAP_GAP,
  } = layoutMetrics;

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

  function buildAlignedTensorPositions(tensors, mode, { bounds, centerX, centerY }) {
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
    const maxLayerWidth = Math.max(0, ...layerRows.map((layerRow) => layerRow.width));
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

  return {
    getTensorHalfSize,
    computeAxisBounds,
    computeTensorBounds,
    computePositionBounds,
    buildAlignedTensorPositions,
    buildAlignedNonOverlappingAxisPositions,
    buildHorizontalRowPositions,
    buildLayerRowPositions,
    buildChainLocalPositions,
    buildGridLocalPositions,
    buildLayeredLocalPositions,
    buildComponentLayoutFromLocalPositions,
    centerLayoutPositions,
    centerPackedComponentLayouts,
    buildTreeSubtreeLayout,
  };
}

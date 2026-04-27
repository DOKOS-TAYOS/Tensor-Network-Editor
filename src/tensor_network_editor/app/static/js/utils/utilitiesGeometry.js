export function createUtilityGeometryBindings({
  ctx,
  state,
  constants,
  runtime,
}) {
  const {
    TENSOR_WIDTH,
    TENSOR_HEIGHT,
    MIN_TENSOR_WIDTH,
    MIN_TENSOR_HEIGHT,
    INDEX_RADIUS,
    INDEX_PADDING,
  } = constants;

  const TENSOR_BASE_Z_INDEX = 10;
  const EDGE_Z_INDEX = 100;
  const PORT_BASE_Z_INDEX = 200;
  const INDEX_LABEL_BASE_Z_INDEX = 230;

  function isIndexConnected(indexId) {
    if (typeof runtime.findConnectionByIndexId === "function") {
      return Boolean(runtime.findConnectionByIndexId(indexId));
    }
    if (typeof runtime.findEdgeByIndexId === "function" && runtime.findEdgeByIndexId(indexId)) {
      return true;
    }
    return Boolean(
      typeof runtime.findHyperedgeByIndexId === "function" &&
        runtime.findHyperedgeByIndexId(indexId)
    );
  }

  function isTensorActiveForPorts(tensorId) {
    return Boolean(
      tensorId &&
        ((Array.isArray(state.selectionIds) && state.selectionIds.includes(tensorId)) ||
          (state.activeTensorDrag &&
            Array.isArray(state.activeTensorDrag.tensorIds) &&
            state.activeTensorDrag.tensorIds.includes(tensorId)))
    );
  }

  function isIndexElevated(indexId, tensorId = null) {
    return Boolean(
      isIndexConnected(indexId) ||
        state.pendingIndexId === indexId ||
        isTensorActiveForPorts(tensorId) ||
        (Array.isArray(state.selectionIds) && state.selectionIds.includes(indexId))
    );
  }

  function getLayeredPortZIndex(indexId, indexPosition, tensorRank, tensorId = null) {
    if (isIndexElevated(indexId, tensorId)) {
      return PORT_BASE_Z_INDEX + tensorRank * 10 + indexPosition;
    }
    return TENSOR_BASE_Z_INDEX + tensorRank + 0.2 + indexPosition / 1000;
  }

  function getLayeredIndexLabelZIndex(indexId, indexPosition, tensorRank, tensorId = null) {
    if (isIndexElevated(indexId, tensorId)) {
      return INDEX_LABEL_BASE_Z_INDEX + tensorRank * 10 + indexPosition;
    }
    return TENSOR_BASE_Z_INDEX + tensorRank + 0.24 + indexPosition / 1000;
  }

  function tensorWidth(tensor) {
    return Math.max(
      MIN_TENSOR_WIDTH,
      runtime.asFiniteNumber(tensor && tensor.size && tensor.size.width, TENSOR_WIDTH)
    );
  }

  function tensorHeight(tensor) {
    return Math.max(
      MIN_TENSOR_HEIGHT,
      runtime.asFiniteNumber(tensor && tensor.size && tensor.size.height, TENSOR_HEIGHT)
    );
  }

  function ensureTensorIndexOffsets(tensor) {
    const needsAutoLayout =
      tensor.indices.length > 0 &&
      tensor.indices.every(
        (index) => Math.abs(index.offset.x) < 0.001 && Math.abs(index.offset.y) < 0.001
      );

    tensor.indices.forEach((index, indexPosition) => {
      if (needsAutoLayout) {
        index.offset = defaultIndexOffsetForOrder(indexPosition, tensor);
      } else {
        index.offset = clampIndexOffset(index.offset, tensor);
      }
    });
  }

  function defaultIndexOffsetForOrder(indexPosition, tensor) {
    const scaleX = tensorWidth(tensor) / TENSOR_WIDTH;
    const scaleY = tensorHeight(tensor) / TENSOR_HEIGHT;
    const slot = constants.DEFAULT_INDEX_SLOTS[indexPosition];
    if (slot) {
      return clampIndexOffset(
        { x: slot.x * scaleX, y: slot.y * scaleY },
        tensor
      );
    }
    return clampIndexOffset(
      {
        x: (indexPosition % 2 === 0 ? -58 : 58) * scaleX,
        y: (-30 + Math.floor(indexPosition / 2) * 18) * scaleY,
      },
      tensor
    );
  }

  function clampIndexOffset(offset, tensor) {
    return {
      x: runtime.clamp(
        runtime.asFiniteNumber(offset.x, 0),
        -tensorWidth(tensor) / 2 + INDEX_RADIUS + INDEX_PADDING,
        tensorWidth(tensor) / 2 - INDEX_RADIUS - INDEX_PADDING
      ),
      y: runtime.clamp(
        runtime.asFiniteNumber(offset.y, 0),
        -tensorHeight(tensor) / 2 + INDEX_RADIUS + INDEX_PADDING,
        tensorHeight(tensor) / 2 - INDEX_RADIUS - INDEX_PADDING
      ),
    };
  }

  function indexAbsolutePosition(tensor, index) {
    const offset = clampIndexOffset(index.offset, tensor);
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
    if (typeof runtime.syncHyperedgeHubNodePositions === "function") {
      runtime.syncHyperedgeHubNodePositions(
        tensor.indices.map((index) => index.id)
      );
    }
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
  }

  function syncIndexLabelNodePosition(index, absolutePosition) {
    if (!state.cy) {
      return;
    }
    const labelElement = state.cy.getElementById(runtime.indexLabelNodeId(index.id));
    if (!labelElement || !labelElement.length) {
      return;
    }
    labelElement.position(runtime.indexLabelPosition(absolutePosition));
    labelElement.data("label", `${index.name} · ${index.dimension}`);
    labelElement.data(
      "textColor",
      runtime.shiftColor(
        runtime.getIndexColor(
          index,
          Boolean(
            typeof runtime.findConnectionByIndexId === "function"
              ? runtime.findConnectionByIndexId(index.id)
              : runtime.findEdgeByIndexId(index.id)
          )
        ),
        64
      )
    );
  }

  function normalizeHyperedgeHubOffset(hyperedge) {
    const hubOffset = {
      x: runtime.asFiniteNumber(hyperedge?.hub_offset?.x, 0),
      y: runtime.asFiniteNumber(hyperedge?.hub_offset?.y, 0),
    };
    if (hyperedge) {
      hyperedge.hub_offset = hubOffset;
    }
    return hubOffset;
  }

  function getAutomaticHyperedgeHubCenter(hyperedge) {
    const endpoints = Array.isArray(hyperedge?.endpoints) ? hyperedge.endpoints : [];
    const endpointPositions = endpoints
      .map((endpoint) =>
        typeof runtime.findIndexOwner === "function"
          ? runtime.findIndexOwner(endpoint.index_id)
          : null
      )
      .filter((owner) => owner && owner.tensor && owner.index)
      .map((owner) => indexAbsolutePosition(owner.tensor, owner.index));
    if (!endpointPositions.length) {
      return null;
    }
    const summed = endpointPositions.reduce(
      (accumulator, position) => ({
        x: accumulator.x + position.x,
        y: accumulator.y + position.y,
      }),
      { x: 0, y: 0 }
    );
    return {
      x: Math.round(summed.x / endpointPositions.length),
      y: Math.round(summed.y / endpointPositions.length),
    };
  }

  function getHyperedgeHubPosition(hyperedge) {
    const automaticCenter = getAutomaticHyperedgeHubCenter(hyperedge);
    if (!automaticCenter) {
      return null;
    }
    const hubOffset = normalizeHyperedgeHubOffset(hyperedge);
    return {
      x: automaticCenter.x + hubOffset.x,
      y: automaticCenter.y + hubOffset.y,
    };
  }

  function syncHyperedgeHubNodePosition(hyperedgeId) {
    if (!state.cy || typeof runtime.findHyperedgeById !== "function") {
      return null;
    }
    const hyperedge = runtime.findHyperedgeById(hyperedgeId);
    if (!hyperedge || typeof runtime.hyperedgeHubNodeId !== "function") {
      return null;
    }
    const hubPosition = getHyperedgeHubPosition(hyperedge);
    if (!hubPosition) {
      return null;
    }
    const hubElement = state.cy.getElementById(runtime.hyperedgeHubNodeId(hyperedge.id));
    if (hubElement && hubElement.length) {
      runWithHyperedgeHubSync(() => {
        hubElement.position(hubPosition);
      });
    }
    return hubPosition;
  }

  function syncHyperedgeHubNodePositions(indexIds = null) {
    if (!state.cy || !state.spec) {
      return;
    }
    const targetedHyperedgeIds = new Set();
    if (Array.isArray(indexIds) && indexIds.length) {
      indexIds.forEach((indexId) => {
        const hyperedge =
          typeof runtime.findHyperedgeByIndexId === "function"
            ? runtime.findHyperedgeByIndexId(indexId)
            : null;
        if (hyperedge?.id) {
          targetedHyperedgeIds.add(hyperedge.id);
        }
      });
    } else {
      (Array.isArray(state.spec.hyperedges) ? state.spec.hyperedges : []).forEach(
        (hyperedge) => {
          if (hyperedge?.id) {
            targetedHyperedgeIds.add(hyperedge.id);
          }
        }
      );
    }
    targetedHyperedgeIds.forEach((hyperedgeId) => {
      syncHyperedgeHubNodePosition(hyperedgeId);
    });
  }

  function runWithIndexSync(action) {
    state.syncingIndexPositions = true;
    try {
      action();
    } finally {
      state.syncingIndexPositions = false;
    }
  }

  function runWithTensorSync(action) {
    state.syncingTensorPositions = true;
    try {
      action();
    } finally {
      state.syncingTensorPositions = false;
    }
  }

  function runWithHyperedgeHubSync(action) {
    state.syncingHyperedgeHubPositions = true;
    try {
      action();
    } finally {
      state.syncingHyperedgeHubPositions = false;
    }
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
    const bend = runtime.clamp(distance * 0.18, 18, 60);
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

  function expandBounds(bounds, x, y) {
    bounds.x1 = Math.min(bounds.x1, x);
    bounds.y1 = Math.min(bounds.y1, y);
    bounds.x2 = Math.max(bounds.x2, x);
    bounds.y2 = Math.max(bounds.y2, y);
  }

  function computeDesignBounds(padding) {
    const bounds = {
      x1: Number.POSITIVE_INFINITY,
      y1: Number.POSITIVE_INFINITY,
      x2: Number.NEGATIVE_INFINITY,
      y2: Number.NEGATIVE_INFINITY,
    };
    const visibleTensors =
      typeof ctx.getVisibleTensors === "function" ? ctx.getVisibleTensors() : state.spec.tensors;
    const visibleEdges =
      typeof ctx.getVisibleEdges === "function" ? ctx.getVisibleEdges() : state.spec.edges;

    visibleTensors.forEach((tensor) => {
      expandBounds(
        bounds,
        tensor.position.x - tensorWidth(tensor) / 2,
        tensor.position.y - tensorHeight(tensor) / 2
      );
      expandBounds(
        bounds,
        tensor.position.x + tensorWidth(tensor) / 2,
        tensor.position.y + tensorHeight(tensor) / 2
      );
      tensor.indices.forEach((index) => {
        const absolutePosition = indexAbsolutePosition(tensor, index);
        expandBounds(bounds, absolutePosition.x - INDEX_RADIUS, absolutePosition.y - INDEX_RADIUS);
        expandBounds(bounds, absolutePosition.x + INDEX_RADIUS, absolutePosition.y + INDEX_RADIUS);
        expandBounds(bounds, absolutePosition.x + 50, absolutePosition.y + 42);
      });
    });

    visibleEdges.forEach((edge) => {
      const left = runtime.findIndexOwner(edge.leftIndexId || edge.left.index_id);
      const right = runtime.findIndexOwner(edge.rightIndexId || edge.right.index_id);
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

  function buildTensorRankById(tensorOrder) {
    return Object.fromEntries(
      (Array.isArray(tensorOrder) ? tensorOrder : []).map((tensorId, index) => [
        tensorId,
        index,
      ])
    );
  }

  function reconcileTensorOrder() {
    const tensorIds = state.spec ? state.spec.tensors.map((tensor) => tensor.id) : [];
    const activeTensorIds = new Set(tensorIds);
    const nextOrder = [];
    const seenTensorIds = new Set();
    (Array.isArray(state.tensorOrder) ? state.tensorOrder : []).forEach((tensorId) => {
      if (!activeTensorIds.has(tensorId) || seenTensorIds.has(tensorId)) {
        return;
      }
      seenTensorIds.add(tensorId);
      nextOrder.push(tensorId);
    });
    tensorIds.forEach((tensorId) => {
      if (seenTensorIds.has(tensorId)) {
        return;
      }
      seenTensorIds.add(tensorId);
      nextOrder.push(tensorId);
    });
    state.tensorOrder = nextOrder;
    state.tensorRankById = buildTensorRankById(nextOrder);
  }

  function tensorLayerRank(tensorId) {
    if (!Object.prototype.hasOwnProperty.call(state.tensorRankById, tensorId)) {
      reconcileTensorOrder();
    }
    return state.tensorRankById[tensorId] ?? 0;
  }

  function bringTensorToFront(tensorId) {
    if (!tensorId) {
      return;
    }
    reconcileTensorOrder();
    state.tensorOrder = state.tensorOrder.filter((id) => id !== tensorId);
    state.tensorOrder.push(tensorId);
    state.tensorRankById = buildTensorRankById(state.tensorOrder);
    applyTensorLayerData();
  }

  function applyTensorLayerData() {
    if (!state.cy) {
      return;
    }
    reconcileTensorOrder();
    state.tensorOrder.forEach((tensorId) => {
      const tensorRank = state.tensorRankById[tensorId] ?? 0;
      const tensorElement = state.cy.getElementById(tensorId);
      if (tensorElement && tensorElement.length) {
        tensorElement.data("zIndex", TENSOR_BASE_Z_INDEX + tensorRank);
      }
      const tensor = runtime.findTensorById(tensorId);
      if (!tensor) {
        return;
      }
      tensor.indices.forEach((index, indexPosition) => {
        const indexElement = state.cy.getElementById(index.id);
        if (indexElement && indexElement.length) {
          indexElement.data(
            "zIndex",
            getLayeredPortZIndex(index.id, indexPosition, tensorRank, tensor.id)
          );
        }
        const labelElement = state.cy.getElementById(runtime.indexLabelNodeId(index.id));
        if (labelElement && labelElement.length) {
          labelElement.data(
            "zIndex",
            getLayeredIndexLabelZIndex(index.id, indexPosition, tensorRank, tensor.id)
          );
        }
      });
    });
    state.cy.edges().forEach((edgeElement) => {
      edgeElement.data("zIndex", EDGE_Z_INDEX);
    });
  }

  return {
    tensorWidth,
    tensorHeight,
    ensureTensorIndexOffsets,
    defaultIndexOffsetForOrder,
    clampIndexOffset,
    indexAbsolutePosition,
    syncIndexNodePositions,
    syncSingleIndexNodePosition,
    syncIndexLabelNodePosition,
    normalizeHyperedgeHubOffset,
    getAutomaticHyperedgeHubCenter,
    getHyperedgeHubPosition,
    syncHyperedgeHubNodePosition,
    syncHyperedgeHubNodePositions,
    runWithIndexSync,
    runWithTensorSync,
    runWithHyperedgeHubSync,
    buildQuadraticCurve,
    quadraticPointAt,
    drawRoundRectPath,
    computeDesignBounds,
    expandBounds,
    reconcileTensorOrder,
    tensorLayerRank,
    bringTensorToFront,
    applyTensorLayerData,
  };
}
